# SPDX-License-Identifier: Apache-2.0
"""Explicit-lifetime, serialized host API over asynchronous CUDA batches.

An export is a caller-managed distributed lease: release ONLY after its one peer
has drained all accesses and unmapped the import. No TTL, implicit destructor,
background control service, arbitrary-pointer registration, or transport fallback.
"""
from dataclasses import dataclass, field
import base64
import math
import os
from pathlib import Path
import threading
import time
import uuid
from functools import wraps
from .api import (BatchHandle, BusyError, MemoryHandle, Opcode, RegionDescriptor,
                  TransferError, TransferRequest, TransferState, TransferStatus, checked_int)


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        if os.getpid() != self._pid:
            raise TransferError("CUDA engine cannot be reused after fork; use spawn")
        with self._lock:
            if self._mapping_guard_poisoned:
                raise BusyError("engine mapping guard quarantined after failed cleanup")
            return method(self, *args, **kwargs)
    return call


@dataclass
class _Memory:
    public: MemoryHandle
    native: int
    descriptor: RegionDescriptor | None = None
    exports: set = field(default_factory=set)
    active: set = field(default_factory=set)
    poisoned: bool = False
    owner: object = None


@dataclass
class _Batch:
    public: BatchHandle
    native: int
    memories: set
    status: TransferStatus


class NvlinkEngine:
    def __init__(self, device: int, library: str | None = None, *,
                 mapping_peer_device: int | None = None, _driver=None):
        """Optionally keep peer-mapping state alive using private backing memory.

        mapping_peer_device must be a distinct, caller-authorized LOCAL CUDA device.
        It consumes one allocation-granularity block there until close(). This is
        NOT a remote data mapping cache: every imported KV lease still unmaps.
        Default None preserves the zero-extra-device-allocation behavior.
        """
        checked_int(device, "device")
        if mapping_peer_device is not None:
            checked_int(mapping_peer_device, "mapping_peer_device")
            if mapping_peer_device == device:
                raise ValueError("mapping_peer_device must differ from device")
        if _driver is None:
            from ._driver import Driver
            library = library or os.environ.get("NVLINK_TRANSFER_LIB")
            if not library:
                cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                library = str(cache / "sglang/l1_transfer/libnvlink_transfer.so")
                if not Path(library).is_file():
                    raise ValueError("build l1_transfer/build-native.sh first, or set NVLINK_TRANSFER_LIB")
            _driver = Driver(library)
        self._driver = _driver
        self.device = device
        self.owner_id = uuid.uuid4().hex
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._memories = {}
        self._batches = {}
        self._exports = {}
        self._imported_leases = set()
        self._mapping_guard = None
        self._mapping_guard_poisoned = False
        if mapping_peer_device is not None:
            self._mapping_guard = self._driver.create_mapping_guard(device, mapping_peer_device)

    def _memory(self, memory):
        if not isinstance(memory, MemoryHandle):
            raise ValueError("expected MemoryHandle")
        item = self._memories.get(memory.region_id)
        if item is None or item.public is not memory:
            raise ValueError("foreign, forged or closed memory handle")
        if item.poisoned:
            raise BusyError("region quarantined after uncertain CUDA access")
        return item

    def _batch(self, batch):
        if not isinstance(batch, BatchHandle):
            raise ValueError("expected BatchHandle")
        item = self._batches.get(batch.batch_id)
        if item is None or item.public is not batch:
            raise ValueError("foreign, forged or released batch handle")
        return item

    @staticmethod
    def _range(memory, offset, length):
        checked_int(offset, "offset"); checked_int(length, "length", 1)
        if offset > memory.size or length > memory.size - offset:
            raise ValueError("range exceeds logical region size")

    def _track(self, native, size, descriptor=None):
        ptr, _, _ = self._driver.info(native)
        memory = MemoryHandle(uuid.uuid4().hex, size, self.device, descriptor is not None, ptr)
        self._memories[memory.region_id] = _Memory(memory, native, descriptor)
        return memory

    @locked
    def allocate_memory(self, size):
        checked_int(size, "size", 1)
        return self._track(self._driver.create(self.device, size), size)

    @locked
    def register_vmm(self, address, physical_size, *, owner):
        """Borrow a full DEVICE/FABRIC mapping from a trusted allocator.

        Caller supplies exact mapping base/size, freezes pool resize/reset and
        coordinates producer writes. Holding owner prevents GC, not explicit close.
        Unregister releases our alias/reference only, never the original pool VA.
        """
        checked_int(address, "address", 1)
        checked_int(physical_size, "physical_size", 1)
        if owner is None:
            raise ValueError("owner keepalive is required")
        memory = self._track(self._driver.register_vmm(self.device, address, physical_size), physical_size)
        self._memories[memory.region_id].owner = owner
        return memory

    @locked
    def export_memory(self, memory, access="read"):
        item = self._memory(memory)
        if memory.remote or item.active:
            raise BusyError("cannot export imported/in-flight memory")
        if access not in ("read", "write"):
            raise ValueError("access must be read or write")
        # One peer per export, many readers allowed; remote writes are exclusive.
        if item.exports and (access == "write" or any(self._exports[x][1] == "write" for x in item.exports)):
            raise BusyError("write export requires exclusive ownership")
        _, physical, granularity = self._driver.info(item.native)
        lease = uuid.uuid4().hex
        desc = RegionDescriptor(1, "nvlink-fabric", self.owner_id, memory.region_id,
                                lease, memory.size, physical, granularity, access,
                                self._driver.export(item.native))
        desc.validate()
        item.exports.add(lease); self._exports[lease] = (memory.region_id, access)
        return desc

    @locked
    def release_export(self, lease_id):
        """Caller attests peer has drained AND unregistered this import.

        Never call merely because of timeout, Failed status or disconnected peer.
        Unknown/repeated ACK is rejected; stale ACK cannot release another lease.
        """
        if lease_id not in self._exports:
            raise ValueError("unknown/already-released export lease")
        region_id, _ = self._exports[lease_id]
        item = self._memories[region_id]
        if item.poisoned:
            raise BusyError("quarantined export")
        item.exports.remove(lease_id); del self._exports[lease_id]

    @locked
    def import_memory(self, descriptor):
        if not isinstance(descriptor, RegionDescriptor):
            raise ValueError("expected RegionDescriptor")
        descriptor.validate()
        if descriptor.transport != "nvlink-fabric":
            raise ValueError("NvlinkEngine requires nvlink-fabric descriptors")
        try:
            raw = base64.b64decode(descriptor.transport_handle, validate=True)
        except (ValueError, TypeError):
            raise ValueError("invalid Fabric handle encoding") from None
        if len(raw) != 64:
            raise ValueError("CUDA Fabric handle must have 64 bytes")
        if descriptor.owner_id == self.owner_id:
            raise ValueError("remote descriptor must belong to a peer engine")
        lease_key = (descriptor.owner_id, descriptor.lease_id)
        if lease_key in self._imported_leases:
            raise ValueError("lease already imported by this engine; obtain a fresh export")
        native = self._driver.import_region(self.device, descriptor)
        memory = self._track(native, descriptor.size, descriptor)
        self._imported_leases.add(lease_key) # tombstone survives unregister
        return memory

    @locked
    def unregister_memory(self, memory):
        item = self._memory(memory)
        if item.exports or item.active:
            raise BusyError("memory has active exports or transfer batches")
        self._driver.close(item.native)
        del self._memories[memory.region_id]

    def _host_access(self, memory, offset, length):
        item = self._memory(memory); self._range(memory, offset, length)
        if memory.remote or item.exports or item.active:
            raise BusyError("host access requires idle, unexported local memory")
        return item

    @locked
    def upload(self, memory, data: bytes, offset=0):
        if not isinstance(data, bytes):
            raise TypeError("upload requires bytes")
        item = self._host_access(memory, offset, len(data))
        try:
            self._driver.upload(item.native, offset, data)
        except Exception:
            item.poisoned = True
            raise

    @locked
    def download(self, memory, length=None, offset=0):
        length = memory.size - offset if length is None else length
        item = self._host_access(memory, offset, length)
        try:
            return self._driver.download(item.native, offset, length)
        except Exception:
            item.poisoned = True
            raise

    @locked
    def submit_transfer(self, requests, *, producer_stream=None):
        """Copy ready remote data; optionally order after a local producer stream.

        None means the caller already established readiness. An integer is a
        CUDA stream handle in this engine's context (0 means the default stream).
        The stream only covers local work, never an unsynchronized remote writer.
        Keep it valid until batch release. Do not append overlapping writes after
        the readiness marker until the transfer completes.
        """
        if producer_stream is not None:
            checked_int(producer_stream, "producer_stream")
        requests = tuple(requests)
        if not requests:
            raise ValueError("empty transfer batch")
        ranges, touched, writes = [], set(), {}
        total = 0
        # Cache only the immediately preceding identity pair/opcode while holding
        # the engine lock. All requests remain alive in the tuple; identity (not
        # dataclass equality or region_id) is required to reject forged handles.
        # Nothing survives this call, so every later batch rechecks current state.
        last_local = last_remote = None
        last_opcode = object()
        # Validate the ENTIRE batch before entering CUDA: no partial writes on rejection.
        for req in requests:
            if not isinstance(req, TransferRequest):
                raise ValueError("expected typed TransferRequest/Opcode")
            if (req.local is not last_local or req.remote is not last_remote
                    or req.opcode is not last_opcode):
                if not isinstance(req.opcode, Opcode):
                    raise ValueError("expected typed TransferRequest/Opcode")
                local, remote = self._memory(req.local), self._memory(req.remote)
                if req.local.remote or not req.remote.remote:
                    raise ValueError("expected local and imported remote memory")
                if local.exports or local.active or remote.active:
                    raise BusyError("region exported or already used by another batch")
                read = req.opcode == Opcode.READ
                required = "read" if read else "write"
                if remote.descriptor.access != required:
                    raise ValueError("remote lease does not permit this operation")
                source, target = (remote, local) if read else (local, remote)
                source_native, target_native = source.native, target.native
                key = ((target.descriptor.owner_id, target.descriptor.region_id)
                       if target.descriptor else (self.owner_id, target.public.region_id))
                destinations = writes.setdefault(key, [])
                touched.update((req.local.region_id, req.remote.region_id))
                last_local, last_remote, last_opcode = req.local, req.remote, req.opcode
            # Offsets and lengths are request-specific: never reuse their checks.
            self._range(req.local, req.local_offset, req.length)
            self._range(req.remote, req.remote_offset, req.length)
            so, to = ((req.remote_offset, req.local_offset) if read else
                      (req.local_offset, req.remote_offset))
            destinations.append((to, to + req.length))
            ranges.append((source_native, so, target_native, to, req.length))
            total += req.length
        for intervals in writes.values():
            intervals.sort()
            if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
                raise ValueError("overlapping destination ranges")
        # If the native call raises unexpectedly after partial issue, conservatively quarantine.
        try:
            native = (self._driver.submit(ranges) if producer_stream is None else
                      self._driver.submit(ranges, producer_stream=producer_stream))
        except Exception:
            for key in touched:
                self._memories[key].poisoned = True
            raise
        public = BatchHandle(uuid.uuid4().hex)
        self._batches[public.batch_id] = _Batch(public, native, touched,
                                              TransferStatus(TransferState.IN_FLIGHT, total, None))
        for key in touched:
            self._memories[key].active.add(public.batch_id)
        return public

    @locked
    def get_transfer_status(self, batch):
        item = self._batch(batch)
        if item.status.state != TransferState.IN_FLIGHT:
            return item.status
        try:
            code, error = self._driver.poll(item.native)
            state = {0: TransferState.IN_FLIGHT, 1: TransferState.SUCCESS,
                     2: TransferState.FAILED_DRAINED, 3: TransferState.FAILED_UNCERTAIN}[code]
        except Exception as exc:
            state, error = TransferState.FAILED_UNCERTAIN, str(exc)
        item.status = TransferStatus(state, item.status.requested_bytes,
                                     item.status.requested_bytes if state == TransferState.SUCCESS else None,
                                     error=error)
        if state != TransferState.IN_FLIGHT:
            for key in item.memories:
                memory = self._memories[key]
                if state == TransferState.FAILED_UNCERTAIN:
                    memory.poisoned = True
        return item.status

    @locked
    def wait(self, batch, timeout=None):
        if timeout is not None and (not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and non-negative")
        if os.environ.get("NVT_NATIVE_WAIT", "1") != "0" and hasattr(self._driver, "wait"):
            item = self._batch(batch)
            if item.status.state == TransferState.IN_FLIGHT:
                try:
                    self._driver.wait(item.native, timeout)
                except Exception as exc:
                    item.status = TransferStatus(TransferState.FAILED_UNCERTAIN,
                                                 item.status.requested_bytes, None, error=str(exc))
                    for key in item.memories:
                        self._memories[key].poisoned = True
            result = self.get_transfer_status(batch)
            if result.state == TransferState.IN_FLIGHT:
                raise TimeoutError("transfer still in flight; buffers must not be released")
            return result
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.get_transfer_status(batch)
            if result.state != TransferState.IN_FLIGHT:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                # Observation only. Batch and regions remain owned and in flight.
                raise TimeoutError("transfer still in flight; buffers must not be released")
            time.sleep(0.0001)

    @locked
    def release_batch(self, batch):
        item = self._batch(batch)
        if not self.get_transfer_status(batch).drained:
            raise BusyError("batch not safely drained")
        try:
            self._driver.release_batch(item.native)
        except Exception as exc:
            # Cleanup errors must never permit an uncertain handle to be retried
            # or its memory to be recycled by a later close/unregister call.
            item.status = TransferStatus(TransferState.FAILED_UNCERTAIN,
                                         item.status.requested_bytes, None, error=str(exc))
            for key in item.memories:
                self._memories[key].poisoned = True
            raise
        for key in item.memories:
            self._memories[key].active.remove(batch.batch_id)
        del self._batches[batch.batch_id]

    @locked
    def close(self):
        if self._batches or self._exports:
            raise BusyError("release batches and acknowledged exports before close")
        # Preflight before freeing anything.
        for item in self._memories.values():
            if item.active or item.poisoned:
                raise BusyError("engine has busy/quarantined memory")
        for item in list(self._memories.values()):
            self.unregister_memory(item.public)
        if self._mapping_guard is not None:
            try:
                self._driver.close(self._mapping_guard)
            except Exception:
                self._mapping_guard_poisoned = True
                raise  # retain the handle; no retry or silent recreation after uncertainty
            self._mapping_guard = None
