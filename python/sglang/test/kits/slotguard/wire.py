# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: TRY004
# Preserve the reference API's ValueError contract for malformed requests/handles.
"""SlotGuard wire-protocol fixtures: identities, hash chains and a CPU transport.

`CpuEngine` is a CPU byte model of the transport API with no native library.
An export is a caller-managed distributed lease: release ONLY after its one peer
has drained all accesses and unmapped the import. No TTL, implicit destructor,
background control service, arbitrary-pointer registration, or transport fallback.
"""

import base64
import hashlib
import math
import os
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from functools import wraps
from itertools import pairwise

from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer.hash_pin import HashLookupResult
from sglang.srt.mem_cache.l1_transfer.slot_guard import LeaseIdentity
from sglang.srt.mem_cache.l1_transfer.transport.api import (
    BatchHandle,
    BusyError,
    MemoryHandle,
    Opcode,
    RegionDescriptor,
    TransferError,
    TransferRequest,
    TransferState,
    TransferStatus,
    checked_int,
)


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
    native: int | None
    memories: set
    status: TransferStatus


class CpuEngine:
    def __init__(
        self,
        device: int,
        library: str | None = None,
        *,
        mapping_peer_device: int | None = None,
        _driver=None,
    ):
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
            raise ValueError("CPU fixture requires an explicit fake driver")
        self._driver = _driver
        self.device = device
        self.owner_id = uuid.uuid4().hex
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._memories = {}
        self._batches = {}
        self._exports = {}
        self._imported_leases = set()
        # Native handles whose Python tracking AND rollback failed. Retain owners
        # and retry close explicitly; never silently claim these imports drained.
        self._pending_cleanup = []
        self._mapping_guard = None
        self._mapping_guard_poisoned = False
        if mapping_peer_device is not None:
            self._mapping_guard = self._driver.create_mapping_guard(
                device, mapping_peer_device
            )

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
        checked_int(offset, "offset")
        checked_int(length, "length", 1)
        if offset > memory.size or length > memory.size - offset:
            raise ValueError("range exceeds logical region size")

    def _track(self, native, size, descriptor=None, owner=None):
        # A native mapping already exists; failed tracking must not orphan it.
        try:
            ptr, _, _ = self._driver.info(native)
            memory = MemoryHandle(
                uuid.uuid4().hex, size, self.device, descriptor is not None, ptr
            )
            self._memories[memory.region_id] = _Memory(
                memory, native, descriptor, owner=owner
            )
        except BaseException:
            try:
                self._driver.close(native)
            except BaseException:
                self._pending_cleanup.append((native, owner, descriptor))
                if descriptor is not None:
                    self._imported_leases.add(
                        (descriptor.owner_id, descriptor.lease_id)
                    )
                raise
            raise
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
        return self._track(
            self._driver.register_vmm(self.device, address, physical_size),
            physical_size,
            owner=owner,
        )

    @locked
    def export_memory(self, memory, access="read"):
        item = self._memory(memory)
        if memory.remote or item.active:
            raise BusyError("cannot export imported/in-flight memory")
        if access not in ("read", "write"):
            raise ValueError("access must be read or write")
        # One peer per export, many readers allowed; remote writes are exclusive.
        if item.exports and (
            access == "write"
            or any(self._exports[x][1] == "write" for x in item.exports)
        ):
            raise BusyError("write export requires exclusive ownership")
        _, physical, granularity = self._driver.info(item.native)
        lease = uuid.uuid4().hex
        desc = RegionDescriptor(
            1,
            "nvlink-fabric",
            self.owner_id,
            memory.region_id,
            lease,
            memory.size,
            physical,
            granularity,
            access,
            self._driver.export(item.native),
        )
        desc.validate()
        item.exports.add(lease)
        self._exports[lease] = (memory.region_id, access)
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
        item.exports.remove(lease_id)
        del self._exports[lease_id]

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
            raise ValueError(
                "lease already imported by this engine; obtain a fresh export"
            )
        native = self._driver.import_region(self.device, descriptor)
        memory = self._track(native, descriptor.size, descriptor)
        self._imported_leases.add(lease_key)  # tombstone survives unregister
        return memory

    @locked
    def unregister_memory(self, memory):
        item = self._memory(memory)
        if item.exports or item.active:
            raise BusyError("memory has active exports or transfer batches")
        self._driver.close(item.native)
        del self._memories[memory.region_id]

    def _host_access(self, memory, offset, length):
        item = self._memory(memory)
        self._range(memory, offset, length)
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
            if (
                req.local is not last_local
                or req.remote is not last_remote
                or req.opcode is not last_opcode
            ):
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
                key = (
                    (target.descriptor.owner_id, target.descriptor.region_id)
                    if target.descriptor
                    else (self.owner_id, target.public.region_id)
                )
                destinations = writes.setdefault(key, [])
                touched.update((req.local.region_id, req.remote.region_id))
                last_local, last_remote, last_opcode = req.local, req.remote, req.opcode
            # Offsets and lengths are request-specific: never reuse their checks.
            self._range(req.local, req.local_offset, req.length)
            self._range(req.remote, req.remote_offset, req.length)
            so, to = (
                (req.remote_offset, req.local_offset)
                if read
                else (req.local_offset, req.remote_offset)
            )
            destinations.append((to, to + req.length))
            ranges.append((source_native, so, target_native, to, req.length))
            total += req.length
        for intervals in writes.values():
            intervals.sort()
            if any(a[1] > b[0] for a, b in pairwise(intervals)):
                raise ValueError("overlapping destination ranges")
        # Allocate and install all Python ownership before DMA can begin. Even a
        # partially successful dict/set update can be rolled back safely here.
        public = BatchHandle(uuid.uuid4().hex)
        item = _Batch(
            public,
            None,
            touched,
            TransferStatus(TransferState.IN_FLIGHT, total, None),
        )
        uncertain = TransferStatus(
            TransferState.FAILED_UNCERTAIN,
            total,
            None,
            error="native submission failed after possible issue",
        )
        memories = tuple(self._memories[key] for key in touched)
        try:
            self._batches[public.batch_id] = item
            for memory in memories:
                memory.active.add(public.batch_id)
        except BaseException:
            for memory in memories:
                memory.active.discard(public.batch_id)
            self._batches.pop(public.batch_id, None)
            raise
        # Any interruption after entering the driver may follow partial issue.
        # Keep the prepared batch/ownership and quarantine every touched region;
        # do not poll a missing native handle or treat this as safe rollback.
        try:
            item.native = (
                self._driver.submit(ranges)
                if producer_stream is None
                else self._driver.submit(ranges, producer_stream=producer_stream)
            )
        except BaseException:
            item.status = uncertain
            for memory in memories:
                memory.poisoned = True
            raise
        return public

    @locked
    def get_transfer_status(self, batch):
        item = self._batch(batch)
        if item.status.state != TransferState.IN_FLIGHT:
            return item.status
        try:
            code, error = self._driver.poll(item.native)
            state = {
                0: TransferState.IN_FLIGHT,
                1: TransferState.SUCCESS,
                2: TransferState.FAILED_DRAINED,
                3: TransferState.FAILED_UNCERTAIN,
            }[code]
        except Exception as exc:  # noqa: BLE001 - any lost completion is uncertain
            state, error = TransferState.FAILED_UNCERTAIN, str(exc)
        item.status = TransferStatus(
            state,
            item.status.requested_bytes,
            item.status.requested_bytes if state == TransferState.SUCCESS else None,
            error=error,
        )
        if state != TransferState.IN_FLIGHT:
            for key in item.memories:
                memory = self._memories[key]
                if state == TransferState.FAILED_UNCERTAIN:
                    memory.poisoned = True
        return item.status

    @locked
    def wait(self, batch, timeout=None):
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.get_transfer_status(batch)
            if result.state != TransferState.IN_FLIGHT:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                # Observation only. Batch and regions remain owned and in flight.
                raise TimeoutError(
                    "transfer still in flight; buffers must not be released"
                )
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
            item.status = TransferStatus(
                TransferState.FAILED_UNCERTAIN,
                item.status.requested_bytes,
                None,
                error=str(exc),
            )
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
        while self._pending_cleanup:
            native, _, _ = self._pending_cleanup[-1]
            self._driver.close(native)
            self._pending_cleanup.pop()
        for item in list(self._memories.values()):
            self.unregister_memory(item.public)
        if self._mapping_guard is not None:
            try:
                self._driver.close(self._mapping_guard)
            except Exception:
                self._mapping_guard_poisoned = True
                raise  # retain the handle; no retry or silent recreation after uncertainty
            self._mapping_guard = None

    @staticmethod
    def plan_pages(source, target, source_pages, target_pages, **kwargs):
        from types import SimpleNamespace

        from sglang.srt.mem_cache.l1_transfer.transport.layout import plan_transfer

        return SimpleNamespace(
            ranges=plan_transfer(source, target, source_pages, target_pages, **kwargs),
            src_pages=tuple(source_pages),
        )

    @staticmethod
    def bind_pages(plan, source, target):
        from sglang.srt.mem_cache.l1_transfer.transport.layout import bind_regions

        class ByteRequests(list):
            pass

        requests = ByteRequests(bind_regions(plan.ranges, source, target, Opcode.READ))
        requests.requested_bytes = sum(r.length for r in requests)
        requests.num_buffers = len(set(r.buffer for r in plan.ranges))
        return requests


class FakeDriver:
    def __init__(self):
        self.memories = {}
        self.next = 0
        self.submits = 0
        self.state = 0
        self.closed = []
        self.raise_submit = False
        self.raise_poll = False

    def create(self, device, size):
        self.next += 1
        self.memories[self.next] = bytearray(size)
        return self.next

    def info(self, h):
        size = len(self.memories[h])
        return h * 4096, ((size + 4095) // 4096) * 4096, 4096

    def export(self, h):
        return base64.b64encode(h.to_bytes(64, "little")).decode()

    def import_region(self, device, desc):
        h = int.from_bytes(base64.b64decode(desc.transport_handle), "little")
        self.next += 1
        self.memories[self.next] = self.memories[h]
        return self.next

    def close(self, h):
        self.closed.append(h)
        del self.memories[h]

    def upload(self, h, offset, data):
        self.memories[h][offset : offset + len(data)] = data

    def download(self, h, offset, length):
        return bytes(self.memories[h][offset : offset + length])

    def submit(self, ranges):
        self.submits += 1
        if self.raise_submit:
            raise TransferError("unknown native submission error")
        for s, so, t, to, n in ranges:
            self.memories[t][to : to + n] = self.memories[s][so : so + n]
        return self.submits

    def poll(self, h):
        if self.raise_poll:
            raise TransferError("lost native completion state")
        return self.state, "injected" if self.state in (2, 3) else ""

    def release_batch(self, h):
        pass


class BlockedEvent:
    def __init__(self, fail=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def synchronize(self):
        self.entered.set()
        if not self.release.wait(2):
            raise RuntimeError("test event was not released")
        if self.fail:
            raise RuntimeError("uncertain CUDA fence")


class NoReadback:
    def cpu(self):
        raise AssertionError("CPU span metadata must not read GPU tensor")


@dataclass(frozen=True)
class ExportResponse:
    """Only the prepared Pin response fields that wire_codec.decode_export reads."""

    request: object
    lease_id: str
    regions: tuple
    buffer_bindings: tuple
    source_page_ids: tuple


class BorrowDriver(FakeDriver):
    def register_vmm(self, device, address, physical):
        return self.create(device, physical)


def identity(engine_id="target", **changes):
    return replace(
        p.EngineIdentity(
            engine_id=engine_id,
            incarnation=f"{engine_id}-boot-1",
            domain_id="nvlink-1",
            instance_id="instance-1",
            model_id="model-1",
            layout_digest="layout-1",
            tp_group_id=f"{engine_id}-tp",
            tp_size=2,
            tp_rank=0,
            shard_id="shard-0",
        ),
        **changes,
    )


def hashes(tokens, page_size, prior=None):
    """Native ROOT chain: prior digest then little-endian u32 page tokens."""
    output = []
    for start in range(0, len(tokens) // page_size * page_size, page_size):
        digest = hashlib.sha256()
        if prior is not None:
            digest.update(bytes.fromhex(prior))
        page = tokens[start : start + page_size]
        digest.update(struct.pack(f"<{len(page)}I", *page))
        prior = digest.hexdigest()
        output.append(prior)
    return tuple(output)


def keys(tokens, page_size):
    return tuple(
        int.from_bytes(bytes.fromhex(h)[:8], "big", signed=True)
        for h in hashes(tokens, page_size)
    )


def reference_native_hash(tokens, prior_digest=None, page_size=None):
    """hash_binding.cpp: SHA256(full parent digest + little-endian uint32 page)."""
    raw = tuple(tokens)
    hashes = []
    pages = (
        (raw,)
        if page_size is None
        else tuple(
            raw[start : start + page_size] for start in range(0, len(raw), page_size)
        )
    )
    for page in pages:
        # RadixKey bigram iteration yields the exact interleaved uint32 pairs.
        page = tuple(
            value
            for token in page
            for value in (token if isinstance(token, tuple) else (token,))
        )
        prior_digest = hashlib.sha256(
            (prior_digest or b"") + struct.pack(f"<{len(page)}I", *page)
        ).digest()
        hashes.append(prior_digest.hex())
    return hashes if page_size else hashes[-1]


def lookup_result(attempt=0, *, local=2, hit=6):
    """Target discovery result, including the native full-prefix hashes."""
    target = identity()
    tokens = tuple(range(7))
    q = p.LookupQuery(
        target,
        p.TransferId(target.incarnation, "rid", attempt),
        keys(tokens, 2),
        tokens,
        local,
        6,
        2,
        prefix_hashes=hashes(tokens, 2),
    )
    return p.LookupResult(q, identity("source"), hit, q.event_block_keys[: hit // 2])


def result(attempt=0, *, local=2, hit=6):
    """Compact hash Pin proof for the same interval."""
    return HashLookupResult.from_lookup(lookup_result(attempt, local=local, hit=hit))


def lease(attempt=0):
    return LeaseIdentity(
        identity("source").incarnation,
        identity().incarnation,
        result(attempt).transfer_id,
        f"nonce-{attempt}",
    )
