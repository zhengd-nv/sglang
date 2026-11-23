# SPDX-License-Identifier: Apache-2.0
"""SlotGuard's READ path on ``sglang.srt.mem_cache.kv_peer_read`` (the upstream reader).

Uses the v1 wire contract (``RegionDescriptor``, ``MemoryHandle``,
``BatchHandle``, ``TransferStatus``) so the control plane and the source side are unchanged.
Reads are page-indexed: :func:`plan_pages` plus :meth:`KvPeerEngine.bind_pages` replace
``plan_transfer`` plus ``bind_regions``, and ``submit_transfer`` takes their :class:`PageRead`.
READ only; no allocation, host upload/download or write exports.
The v1 descriptor has no field for the owner's fabric clique, so the reader's clique
check is skipped; a mismatched peer fails at import instead.
The reader keeps no owner identity or export leases; the engine keeps both.
"""

import base64
import threading
import uuid
from typing import Any

import msgspec
import numpy as np

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_peer_read import (
    KvPeerReader,
    PeerBuffer,
    PeerReadBusy,
    PeerRegionDescriptor,
    ReadGroup,
    ReadState,
    export_region,
)

from .api import (
    BatchHandle,
    BusyError,
    MemoryHandle,
    RegionDescriptor,
    TransferState,
    TransferStatus,
    checked_int,
)
from .layout import PoolLayout

_TRANSPORT = "nvlink-fabric"


class PagePlan(msgspec.Struct, frozen=True):
    """Validated same-layout page lists; pure, built before any import."""

    layout: PoolLayout
    src_pages: Any  # int64 numpy array
    dst_pages: Any
    source_capacity: int
    target_capacity: int


class PageRead(msgspec.Struct, frozen=True):
    groups: tuple[ReadGroup, ...]
    requested_bytes: int
    num_buffers: int
    # Local memories the read writes; the reader holds only their raw addresses.
    targets: tuple[MemoryHandle, ...]


def plan_pages(
    source: PoolLayout,
    target: PoolLayout,
    source_pages,
    target_pages,
    *,
    source_capacity: int,
    target_capacity: int,
) -> PagePlan:
    """The checks of ``plan_transfer`` without expanding pages into byte ranges."""
    source.validate()
    target.validate()
    if source != target:
        raise ValueError("same-layout transfer requires identical layout contracts")
    checked_int(source_capacity, "source_capacity")
    checked_int(target_capacity, "target_capacity")
    src, dst = tuple(source_pages), tuple(target_pages)
    if len(src) != len(dst):
        raise ValueError("page lists must have equal lengths")
    for pages, capacity in ((src, source_capacity), (dst, target_capacity)):
        for page in pages:
            if checked_int(page, "page") >= capacity:
                raise ValueError("page outside pool capacity")
    if len(set(dst)) != len(dst):
        raise ValueError("duplicate target pages")
    return PagePlan(
        layout=source,
        src_pages=np.asarray(src, dtype=np.int64),
        dst_pages=np.asarray(dst, dtype=np.int64),
        source_capacity=source_capacity,
        target_capacity=target_capacity,
    )


class _Vmm:
    """Driver calls for borrowing a pool allocation; tests substitute a fake."""

    def __init__(self):
        from cuda.bindings import driver

        self._drv = driver

    def retain(self, address: int, physical_size: int) -> tuple[Any, int]:
        from sglang.srt.utils.cuda_vmm_utils import (
            check_drv,
            get_allocation_granularity,
        )

        drv = self._drv
        err, base, size = drv.cuMemGetAddressRange(drv.CUdeviceptr(address))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemGetAddressRange: {err}")
        if int(base) != address or int(size) != physical_size:
            raise ValueError("register_vmm needs the exact mapping base and size")
        handle = check_drv(
            drv.cuMemRetainAllocationHandle(address), "cuMemRetainAllocationHandle"
        )
        try:
            prop = check_drv(
                drv.cuMemGetAllocationPropertiesFromHandle(handle),
                "cuMemGetAllocationPropertiesFromHandle",
            )
            return handle, get_allocation_granularity(prop)
        except BaseException:
            # The caller never sees this handle, so nothing else can release it.
            check_drv(drv.cuMemRelease(handle), "cuMemRelease")
            raise

    def release(self, handle) -> None:
        from sglang.srt.utils.cuda_vmm_utils import check_drv

        check_drv(self._drv.cuMemRelease(handle), "cuMemRelease")


class _Memory:
    __slots__ = ("public", "native", "descriptor", "granularity", "exports", "owner")

    def __init__(self, public, native, *, descriptor=None, granularity=0, owner=None):
        self.public = public
        self.native = native  # local: retained allocation handle; remote: PeerRegion
        self.descriptor = descriptor  # None for local memory (serving's readback)
        self.granularity = granularity
        self.exports: set[str] = set()
        self.owner = owner


class KvPeerEngine:
    """Owner and peer reader for one SlotGuard device."""

    def __init__(
        self,
        device: int,
        *,
        num_blocks: int | None = None,
        _export=None,
        _reader=None,
        _vmm=None,
    ):
        checked_int(device, "device")
        if num_blocks is None:
            num_blocks = envs.SGLANG_L1_KV_PEER_BLOCKS.get()
        self.device = device
        self._export = _export if _export is not None else export_region
        self._reader = (
            _reader
            if _reader is not None
            else KvPeerReader(device, num_blocks=num_blocks)
        )
        self._vmm = _vmm if _vmm is not None else _Vmm()
        # Peers refuse to import a lease whose owner id is their own.
        self.owner_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._memories: dict[str, _Memory] = {}
        self._batches: dict[
            str, tuple[BatchHandle, Any, int, tuple[MemoryHandle, ...]]
        ] = {}
        self._exports: dict[str, str] = {}  # lease_id -> region_id

    def _memory(self, memory) -> _Memory:
        if not isinstance(memory, MemoryHandle):
            raise ValueError("expected MemoryHandle")
        item = self._memories.get(memory.region_id)
        if item is None or item.public is not memory:
            raise ValueError("foreign, forged or closed memory handle")
        return item

    def _batch(self, batch):
        if not isinstance(batch, BatchHandle):
            raise ValueError("expected BatchHandle")
        entry = self._batches.get(batch.batch_id)
        if entry is None or entry[0] is not batch:
            raise ValueError("foreign, forged or released batch handle")
        return entry

    # ---- source side

    def register_vmm(self, address, physical_size, *, owner):
        checked_int(address, "address", 1)
        checked_int(physical_size, "physical_size", 1)
        if owner is None:
            raise ValueError("owner keepalive is required")
        with self._lock:
            handle, granularity = self._vmm.retain(address, physical_size)
            public = MemoryHandle(
                uuid.uuid4().hex, physical_size, self.device, False, address
            )
            self._memories[public.region_id] = _Memory(
                public, handle, granularity=granularity, owner=owner
            )
            return public

    def export_memory(self, memory, access="read"):
        if access != "read":
            raise ValueError("KvPeerEngine exports are read-only")
        with self._lock:
            item = self._memory(memory)
            if memory.remote:
                raise BusyError("cannot export imported memory")
            peer = self._export(item.native, memory.size, device=self.device)
            lease_id = uuid.uuid4().hex
            desc = RegionDescriptor(
                1,
                _TRANSPORT,
                self.owner_id,
                memory.region_id,
                lease_id,
                memory.size,
                memory.size,
                item.granularity,
                "read",
                base64.b64encode(peer.fabric_handle).decode(),
            )
            desc.validate()
            item.exports.add(lease_id)
            self._exports[lease_id] = memory.region_id
            return desc

    def release_export(self, lease_id):
        with self._lock:
            if lease_id not in self._exports:
                raise ValueError("unknown/already-released export lease")
            region_id = self._exports.pop(lease_id)
            self._memories[region_id].exports.discard(lease_id)

    # ---- target side

    def import_memory(self, descriptor):
        if not isinstance(descriptor, RegionDescriptor):
            raise ValueError("expected RegionDescriptor")
        descriptor.validate()
        if descriptor.transport != _TRANSPORT or descriptor.access != "read":
            raise ValueError("KvPeerEngine imports read-only nvlink-fabric leases")
        try:
            raw = base64.b64decode(descriptor.transport_handle, validate=True)
        except (ValueError, TypeError):
            raise ValueError("invalid Fabric handle encoding") from None
        if descriptor.owner_id == self.owner_id:
            raise ValueError("remote descriptor must belong to a peer engine")
        peer = PeerRegionDescriptor(
            version=1, size=descriptor.physical_size, fabric_handle=raw
        )
        with self._lock:
            region = self._reader.import_region(peer)
            public = MemoryHandle(
                uuid.uuid4().hex, descriptor.size, self.device, True, region.address
            )
            self._memories[public.region_id] = _Memory(
                public, region, descriptor=descriptor
            )
            return public

    plan_pages = staticmethod(plan_pages)

    def bind_pages(self, plan: PagePlan, source, target) -> PageRead:
        """One group over every buffer; each buffer must sit in one span on both sides."""
        buffers, targets = [], {}
        with self._lock:
            for layout in plan.layout.buffers:
                src = self._single_span(
                    source, layout.name, plan.source_capacity, layout
                )
                dst = self._single_span(
                    target, layout.name, plan.target_capacity, layout
                )
                if not src.memory.remote or dst.memory.remote:
                    raise ValueError("binding locality disagrees with READ")
                targets[dst.memory.region_id] = dst.memory
                buffers.append(
                    PeerBuffer(
                        src_region=self._memory(src.memory).native,
                        src_offset=src.region_offset,
                        dst_base=self._memory(dst.memory).public.address
                        + dst.region_offset,
                        page_bytes=layout.page_bytes,
                        dst_num_pages=plan.target_capacity,
                    )
                )
        group = ReadGroup(
            buffers=tuple(buffers), src_pages=plan.src_pages, dst_pages=plan.dst_pages
        )
        page_bytes = sum(b.page_bytes for b in plan.layout.buffers)
        return PageRead(
            groups=(group,),
            requested_bytes=len(plan.src_pages) * page_bytes,
            num_buffers=len(buffers),
            targets=tuple(targets.values()),
        )

    @staticmethod
    def _single_span(bindings, name, capacity, layout):
        try:
            spans = tuple(bindings[name])
        except KeyError:
            raise ValueError(f"missing binding: {name}") from None
        if len(spans) != 1:
            raise ValueError("KvPeerReader needs each buffer in one region span")
        span = spans[0]
        if (
            span.buffer_offset != 0
            or span.length != capacity * layout.page_bytes
            or span.region_offset + span.length > span.memory.size
        ):
            raise ValueError("binding does not cover the buffer inside its region")
        return span

    def submit_transfer(self, read: PageRead):
        if not isinstance(read, PageRead):
            raise ValueError("KvPeerEngine submits PageRead from bind_pages")
        with self._lock:
            for memory in read.targets:
                self._memory(memory)  # unregistered since bind_pages: refuse
            try:
                native = self._reader.read(read.groups)
            except PeerReadBusy as exc:
                raise BusyError(str(exc)) from exc
            public = BatchHandle(uuid.uuid4().hex)
            self._batches[public.batch_id] = (
                public,
                native,
                read.requested_bytes,
                read.targets,
            )
            return public

    def get_transfer_status(self, batch):
        with self._lock:
            _, native, requested, _ = self._batch(batch)
            status = self._reader.status(native)
        state = TransferState(status.state.value)
        return TransferStatus(
            state,
            requested,
            requested if status.state == ReadState.SUCCESS else None,
            error=status.error,
        )

    def wait(self, batch, timeout=None):
        with self._lock:
            _, native, _, _ = self._batch(batch)
        self._reader.wait(native, timeout=timeout)
        return self.get_transfer_status(batch)

    def release_batch(self, batch):
        with self._lock:
            _, native, _, _ = self._batch(batch)
            try:
                self._reader.release(native)
            except PeerReadBusy as exc:
                raise BusyError(str(exc)) from exc
            del self._batches[batch.batch_id]

    # ---- both

    def unregister_memory(self, memory):
        with self._lock:
            item = self._memory(memory)
            if memory.remote:
                try:
                    self._reader.unmap(item.native)
                except PeerReadBusy as exc:
                    raise BusyError(str(exc)) from exc
            else:
                if item.exports:
                    raise BusyError("memory has active exports")
                if any(memory in entry[3] for entry in self._batches.values()):
                    raise BusyError("memory is written by an unreleased read")
                self._vmm.release(item.native)
            del self._memories[memory.region_id]

    def close(self):
        with self._lock:
            if self._batches or self._exports:
                raise BusyError("release batches and acknowledged exports before close")
            for item in list(self._memories.values()):
                self.unregister_memory(item.public)


def verify_pages(
    engine: KvPeerEngine, read: PageRead, *, tensor_factory=None, torch_module=None
):
    """Test-only byte oracle for a drained PageRead; run before release and unmap."""
    if torch_module is None:
        import torch as torch_module
    if tensor_factory is None:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        tensor_factory = tensor_from_pointer
    chunk = 1024  # bounds the gathered copies to chunk * page_bytes per side
    for group in read.groups:
        src_pages = torch_module.as_tensor(group.src_pages, dtype=torch_module.int64)
        dst_pages = torch_module.as_tensor(group.dst_pages, dtype=torch_module.int64)
        for buf in group.buffers:
            pb = buf.page_bytes
            src_pages_needed = (buf.src_region.size - buf.src_offset) // pb
            src = tensor_factory(
                buf.src_region.address + buf.src_offset,
                src_pages_needed * pb,
                device_id=engine.device,
            ).view(-1, pb)
            dst = tensor_factory(
                buf.dst_base, buf.dst_num_pages * pb, device_id=engine.device
            ).view(-1, pb)
            for start in range(0, len(src_pages), chunk):
                s = src_pages[start : start + chunk].to(src.device)
                d = dst_pages[start : start + chunk].to(dst.device)
                if not torch_module.equal(src[s], dst[d]):
                    raise ValueError("GPU byte oracle mismatch")
    return True
