# SPDX-License-Identifier: Apache-2.0
"""Copy KV pages between this GPU and a peer allocation over NVLink (CUDA FABRIC handles).

The owner exports one FABRIC allocation per lease (:class:`KvPeerExporter`): a peer
GPU's KV pool, or host memory on a NUMA node (EGM, made with
``cuda_vmm_utils.create_host_numa_allocation``). Its control plane hands the
descriptor bytes to one reader. The reader (:class:`KvPeerReader`) imports it
read-only and gathers pages into its own, borrowed, destination buffers with a
page-index kernel on a private stream. A lease the owner exported writable may be
imported writable; the reader then also scatters pages of its own buffers into it,
on a second private stream, so a long write never delays a read.

Contract:

- A batch is validated as a whole before anything is enqueued: page bounds on
  both sides, destination overlap within the batch (ValueError) and against
  this reader's in-flight batches (PeerReadBusy), and that the local buffers are
  memory of this reader's device at both ends and not an imported region.
  A rejected batch never touches memory.
- Groups of a batch complete in order, one event each. A group that completed
  stays completed when a later group fails; ``wait_group`` blocks until one has.
- A failed batch is FAILED_DRAINED when the batch's last recorded group event
  synchronizes cleanly, FAILED_UNCERTAIN otherwise. Only SUCCESS and
  FAILED_DRAINED batches can be released; an uncertain batch, including one
  whose enqueue failed without draining, pins its regions and destination
  pages for the life of the process. A sticky CUDA error fail-stops the
  reader; it also leaves the process's CUDA context unusable.
- ``wait(timeout)`` only stops waiting; it never cancels or releases anything.
- The owner releases a lease only after the reader has released every batch on
  the region, unmapped it and acknowledged; never because of a timeout.
- Writes share an allocation: any number of readers, in any processes, may write
  it at once, and keeping their pages apart is the callers' job (a store's slot
  reservations). A reader only rejects a write that would share a destination
  byte with another page of the batch or with one of its own writes in flight.
- Source pages (peer pages of a read, local pages of a write) must not be
  rewritten while a copy is in flight, and borrowed local buffers must stay
  allocated until release; both are the callers' job.
"""

from __future__ import annotations

import enum
import functools
import os
import threading
import time
import uuid
from typing import Any, Optional, Sequence

import msgspec
import numpy as np

# Local bases, peer offsets, strides and page sizes must be multiples of this.
PAGE_ALIGN_BYTES = 128

_DESCRIPTOR_VERSION = 1
# Replays of this many most recent leases are rejected; arbitrary.
_MAX_LEASE_TOMBSTONES = 1 << 16
# Arbitrary: one buffer tuple per (layer group, source region) of a few pools.
_MAX_BUFFER_SETS = 2048
# Arbitrary: recycled events of successful batches kept for later groups.
_MAX_POOLED_EVENTS = 4096
_FABRIC_HANDLE_BYTES = 64


class ReadState(str, enum.Enum):
    IN_FLIGHT = "IN_FLIGHT"
    SUCCESS = "SUCCESS"
    FAILED_DRAINED = "FAILED_DRAINED"
    FAILED_UNCERTAIN = "FAILED_UNCERTAIN"


class PeerReadError(RuntimeError):
    pass


class PeerReadBusy(PeerReadError):
    """The object is still in use (in-flight, exported, or quarantined)."""


class PeerRegionDescriptor(msgspec.Struct, frozen=True):
    """What a reader needs to map one exported allocation. Opaque to the control plane."""

    version: int
    owner_id: str
    region_id: str
    lease_id: str
    size: int
    fabric_handle: bytes
    # (cluster_uuid, clique_id) of the owner GPU; None when NVML could not tell.
    fabric_clique: Optional[tuple[bytes, int]] = None
    # The owner lets this lease's reader map it writable and write it.
    writable: bool = False

    def validate(self) -> None:
        if self.version != _DESCRIPTOR_VERSION:
            raise ValueError(f"unsupported descriptor version {self.version}")
        if not (self.owner_id and self.region_id and self.lease_id):
            raise ValueError("descriptor identities must be non-empty")
        if self.size <= 0:
            raise ValueError("descriptor size must be positive")
        if len(self.fabric_handle) != _FABRIC_HANDLE_BYTES:
            raise ValueError("FABRIC handle must be 64 bytes")

    def to_bytes(self) -> bytes:
        self.validate()
        return msgspec.msgpack.encode(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> PeerRegionDescriptor:
        try:
            desc = msgspec.msgpack.decode(data, type=cls)
        except msgspec.DecodeError as error:
            raise ValueError(f"malformed peer region descriptor: {error}") from None
        desc.validate()
        return desc


class PeerRegion:
    """Reader-side handle of an imported region."""

    __slots__ = ("region_id", "size", "address", "writable")

    def __init__(self, region_id: str, size: int, address: int, writable=False):
        self.region_id = region_id
        self.size = size
        self.address = address
        self.writable = writable


class PeerBuffer(msgspec.Struct, frozen=True):
    """One per-layer buffer of a read: page ``p`` lives at ``src_offset + p * src_stride``
    in ``src_region`` and goes to ``dst_base + p * page_bytes`` on the reader."""

    src_region: PeerRegion
    src_offset: int
    dst_base: int
    page_bytes: int
    # Capacity of the destination buffer in pages; bounds dst_pages.
    dst_num_pages: int
    # Bytes between source pages; 0 means page_bytes. A page object of several
    # buffers is read with its size here and each buffer's place in src_offset.
    src_stride: int = 0


class ReadGroup(msgspec.Struct, frozen=True):
    """Buffers that share one page list and complete together (typically a layer group).

    ``src_pages`` / ``dst_pages`` are equal-length int64 page ids on the host
    (numpy array, CPU tensor or sequence); every buffer copies the same pages.
    """

    buffers: tuple[PeerBuffer, ...]
    src_pages: Any
    dst_pages: Any


class PeerWriteBuffer(msgspec.Struct, frozen=True):
    """One per-layer buffer of a write: page ``p`` at ``src_base + p * page_bytes`` on the
    reader goes to ``dst_offset + p * dst_stride`` in ``dst_region``, a writable import."""

    dst_region: PeerRegion
    dst_offset: int
    src_base: int
    page_bytes: int
    # Capacity of the source buffer in pages; bounds src_pages.
    src_num_pages: int
    # Bytes between destination pages; 0 means page_bytes.
    dst_stride: int = 0
    # Pages of the destination buffer; 0 means as many as fit in the region.
    dst_num_pages: int = 0


class WriteGroup(msgspec.Struct, frozen=True):
    """The write counterpart of :class:`ReadGroup`: ``src_pages`` are local page ids,
    ``dst_pages`` page ids in the destination buffers."""

    buffers: tuple[PeerWriteBuffer, ...]
    src_pages: Any
    dst_pages: Any


class ReadStatus(msgspec.Struct, frozen=True):
    state: ReadState
    groups_done: int
    num_groups: int
    error: str = ""


class ReadBatch:
    __slots__ = ("batch_id",)

    def __init__(self, batch_id: str):
        self.batch_id = batch_id


# ===== CUDA access =====
# Everything below talks to the GPU through this object, so the contract tests can
# substitute a fake. One launch per distinct (page size, peer stride) of a group.
# Enqueueing a group (launch + record) allocates nothing, never syncs the host and
# never compiles, so it can also run as a deferred, gated chunk (a TransferChain
# ChunkOp) on a stream the reader does not own, once the batch's upload has landed;
# drain is judged by the group events.


class _CudaOps:
    def __init__(self, *, num_blocks: int):
        from cuda.bindings import driver

        self._drv = driver
        self._num_blocks = num_blocks
        self._kernels = {}  # load -> compiled copy, set by warmup()
        self.not_ready = int(driver.CUresult.CUDA_ERROR_NOT_READY)
        self._sticky = {
            int(getattr(driver.CUresult, name))
            for name in (
                "CUDA_ERROR_ILLEGAL_ADDRESS",
                "CUDA_ERROR_LAUNCH_FAILED",
                "CUDA_ERROR_ILLEGAL_INSTRUCTION",
                "CUDA_ERROR_MISALIGNED_ADDRESS",
                "CUDA_ERROR_INVALID_ADDRESS_SPACE",
                "CUDA_ERROR_INVALID_PC",
                "CUDA_ERROR_HARDWARE_STACK_ERROR",
                "CUDA_ERROR_ASSERT",
                "CUDA_ERROR_ECC_UNCORRECTABLE",
                "CUDA_ERROR_NVLINK_UNCORRECTABLE",
                "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            )
        }

    def fabric_clique(self, device: int):
        from sglang.srt.utils.cuda_vmm_utils import get_gpu_fabric_clique

        return get_gpu_fabric_clique(device)

    def export_handle(self, handle) -> bytes:
        from sglang.srt.utils.cuda_vmm_utils import export_fabric_handle

        return export_fabric_handle(handle)

    def map_import(
        self, fabric_handle: bytes, size: int, device: int, *, read_only: bool
    ) -> int:
        from sglang.srt.utils.cuda_vmm_utils import map_fabric_import

        return map_fabric_import(fabric_handle, size, device, read_only=read_only)

    def unmap_import(self, address: int, size: int) -> None:
        from sglang.srt.utils.cuda_vmm_utils import unmap_fabric_import

        unmap_fabric_import(address, size)

    def pointer_device(self, pointer: int) -> Optional[int]:
        drv = self._drv
        attr = drv.CUpointer_attribute
        err, kind = drv.cuPointerGetAttribute(
            attr.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer
        )
        if err != drv.CUresult.CUDA_SUCCESS or int(kind) != int(
            drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
        ):
            return None
        err, ordinal = drv.cuPointerGetAttribute(
            attr.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer
        )
        return int(ordinal) if err == drv.CUresult.CUDA_SUCCESS else None

    def new_stream(self, device: int):
        import torch

        return torch.cuda.Stream(device=device)

    def wait_stream(self, stream, producer) -> None:
        """Make ``stream`` wait for ``producer``, a torch stream or event."""
        import torch

        if isinstance(producer, torch.cuda.Event):
            stream.wait_event(producer)
        else:
            stream.wait_stream(producer)

    def warmup(self, stream, *, load: bool) -> None:
        """Compile and load one direction's copy now: a first launch must not happen inside a group."""
        import torch

        from sglang.kernels.ops.kvcache.peer_page_gather import peer_page_kernel

        self._kernels[load] = peer_page_kernel(load=load, num_blocks=self._num_blocks)
        with torch.cuda.stream(stream):
            buf = torch.zeros(
                2 * PAGE_ALIGN_BYTES, dtype=torch.uint8, device=stream.device
            )
            ptrs = torch.tensor(
                [buf.data_ptr(), buf.data_ptr() + PAGE_ALIGN_BYTES],
                dtype=torch.int64,
                device=stream.device,
            ).view(torch.uint64)
            page = torch.zeros(1, dtype=torch.int64, device=stream.device)
            self.launch(
                (
                    load,
                    PAGE_ALIGN_BYTES,
                    PAGE_ALIGN_BYTES,
                    ptrs[:1],
                    ptrs[1:],
                    page,
                    page,
                )
            )
        stream.synchronize()

    def current(self, stream):
        import torch

        return torch.cuda.stream(stream)

    def stage(self, stream, meta: np.ndarray, spans):
        """Upload one batch's metadata (int64) on ``stream``; return what must stay alive
        until it has landed, and a device view per ``(offset, length, is_pointer)`` span."""
        import torch

        host = torch.from_numpy(meta).pin_memory()
        with torch.cuda.stream(stream):
            dev = host.to(stream.device, non_blocking=True)
        views = [
            dev[offset : offset + length].view(torch.uint64)
            if pointer
            else dev[offset : offset + length]
            for offset, length, pointer in spans
        ]
        return (host, dev), views

    def launch(self, view) -> None:
        """Enqueue one copy on the current stream: peer -> local when ``load``."""
        (
            load,
            page_bytes,
            peer_stride,
            peer_ptrs,
            local_ptrs,
            peer_pages,
            local_pages,
        ) = view
        self._kernels[load](
            peer_ptrs,
            local_ptrs,
            peer_pages,
            local_pages,
            peer_stride,
            page_bytes,
            page_bytes,
        )

    def new_event(self, stream):
        """An event in ``stream``'s context, whatever context is current."""
        drv = self._drv
        err, ctx = drv.cuStreamGetCtx(stream.cuda_stream)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError(f"cuStreamGetCtx: {err}")
        if drv.cuCtxPushCurrent(ctx)[0] != drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError("cuCtxPushCurrent failed")
        try:
            err, event = drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DISABLE_TIMING)
        finally:
            drv.cuCtxPopCurrent()
        if err != drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError(f"cuEventCreate: {err}")
        return event

    def record(self, event, stream) -> None:
        err = self._drv.cuEventRecord(event, stream.cuda_stream)[0]
        if err != self._drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError(f"cuEventRecord: {err}")

    def query_event(self, event) -> int:
        return int(self._drv.cuEventQuery(event)[0])

    def sync_event(self, event) -> int:
        """Block until ``event`` has fired; cuda-python releases the GIL meanwhile."""
        return int(self._drv.cuEventSynchronize(event)[0])

    def destroy_event(self, event) -> None:
        self._drv.cuEventDestroy(event)

    def is_sticky(self, code: int) -> bool:
        return code in self._sticky


# ===== Owner side =====


def _checked_int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value < 2**63:
        raise ValueError(f"{name} must be in [{minimum}, 2**63)")
    return value


def _guarded(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        if os.getpid() != self._pid:
            raise PeerReadError("peer KV objects cannot be used across fork")
        with self._lock:
            return method(self, *args, **kwargs)

    return call


class KvPeerExporter:
    """Hands out leases on the owner's FABRIC allocations.

    The allocation stays the caller's; it must not be freed or remapped while
    :meth:`outstanding` reports a lease on it. Writable leases are not exclusive:
    an allocation may have any number of them, alongside read leases.
    """

    def __init__(self, device: int, *, _ops=None):
        self.device = _checked_int(device, "device")
        self.owner_id = uuid.uuid4().hex
        self._ops = _ops if _ops is not None else _CudaOps(num_blocks=1)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._leases: dict[str, str] = {}  # lease_id -> region_id
        self._clique = self._ops.fabric_clique(self.device)

    @_guarded
    def export(
        self, handle, size: int, *, region_id: str, writable: bool = False
    ) -> PeerRegionDescriptor:
        """One lease for one reader on the allocation ``handle`` of ``size`` bytes.

        Export every lease of one allocation under the same ``region_id``: a reader
        tells two of its writes to the same allocation apart by (owner, region_id).
        """
        size = _checked_int(size, "size", 1)
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("region_id must be a non-empty string")
        if not isinstance(writable, bool):
            raise ValueError("writable must be a bool")
        clique = self._clique
        desc = PeerRegionDescriptor(
            version=_DESCRIPTOR_VERSION,
            owner_id=self.owner_id,
            region_id=region_id,
            lease_id=uuid.uuid4().hex,
            size=size,
            fabric_handle=self._ops.export_handle(handle),
            fabric_clique=None
            if clique is None
            else (bytes(clique[0]), int(clique[1])),
            writable=writable,
        )
        desc.validate()
        self._leases[desc.lease_id] = region_id
        return desc

    @_guarded
    def release_export(self, lease_id: str) -> None:
        """Call only after the reader acknowledged release and unmap of this lease."""
        if self._leases.pop(lease_id, None) is None:
            raise ValueError("unknown or already released lease")

    @_guarded
    def outstanding(self, region_id: Optional[str] = None) -> int:
        if region_id is None:
            return len(self._leases)
        return sum(r == region_id for r in self._leases.values())


# ===== Reader side =====
# Destinations are keyed (space, base, page_bytes, stride): page p of a buffer covers
# [base + p * stride, + page_bytes) in its space, which is None for this device's
# memory and (owner_id, region_id) of the export for an imported allocation.


class _Region:
    __slots__ = ("public", "desc", "batches", "writable")

    def __init__(self, public: PeerRegion, desc: PeerRegionDescriptor, writable: bool):
        self.public = public
        self.desc = desc
        self.batches: set[str] = set()
        self.writable = (
            writable  # the mapping's access; the public flag is the caller's copy
        )


class _BufferSet:
    """A validated ``ReadGroup.buffers`` or ``WriteGroup.buffers`` tuple.

    Pools keep their buffers for their lifetime, so callers that reuse the tuple
    pay the per-buffer checks (alignment, local device) and the upload of the
    address tables once.
    """

    __slots__ = (
        "buffers",
        "load",
        "regions",
        "launches",
        "dst_ends",
        "src_limit",
        "dst_limit",
        "tables",
    )

    def __init__(
        self, *, buffers, load, regions, launches, dst_ends, src_limit, dst_limit
    ):
        self.buffers = buffers  # the caller's tuple; identity is the cache key
        self.load = load  # True: peer -> local (read); False: local -> peer (write)
        self.regions = regions  # the regions the buffers read or write, by region id
        self.launches = launches  # [(page_bytes, peer_stride, peer_bases, local_bases)]
        # destination key -> end of its last page. Two writers can share a destination
        # byte only through the same key, or through keys _first_overlap reports.
        self.dst_ends = dst_ends
        self.src_limit = src_limit  # source page ids must be below this
        self.dst_limit = dst_limit  # destination page ids must be below this
        # Device views (peer_ptrs, local_ptrs) per launch once uploaded on this set's
        # stream; None until then, and again after an upload that may not have landed.
        self.tables = None


class _GroupPlan:
    __slots__ = ("buffer_set", "peer_pages", "local_pages", "dst_sorted")

    def __init__(self, buffer_set, peer_pages, local_pages, dst_sorted):
        self.buffer_set = buffer_set
        self.peer_pages = peer_pages
        self.local_pages = local_pages
        self.dst_sorted = dst_sorted


def _pack(plans):
    """Pack one batch's host metadata into one int64 array.

    Groups usually share their page arrays: each is uploaded once. A buffer set's
    address tables go up only while they are not resident yet. Returns the array,
    its spans ``(offset, length, is_pointer)``, per plan the span indices of its
    (peer pages, local pages), and the [(buffer set, [(peer span, local span)])]
    uploaded now.
    """
    parts, spans, placed = [], [], {}
    offset = 0

    def put(array, pointer: bool) -> int:
        nonlocal offset
        hit = placed.get(id(array))
        if hit is not None:
            return hit
        parts.append(array)
        spans.append((offset, array.size, pointer))
        offset += array.size
        placed[id(array)] = len(spans) - 1
        return placed[id(array)]

    pages, tables = [], {}
    for plan in plans:
        pages.append((put(plan.peer_pages, False), put(plan.local_pages, False)))
        buffer_set = plan.buffer_set
        if buffer_set.tables is None and id(buffer_set) not in tables:
            tables[id(buffer_set)] = (
                buffer_set,
                [(put(p, True), put(q, True)) for _, _, p, q in buffer_set.launches],
            )
    return np.concatenate(parts, dtype=np.int64), spans, pages, list(tables.values())


class _Shape:
    """A batch's destination keys: key -> its furthest end, key -> the groups writing it."""

    __slots__ = ("sets", "ends", "users", "shared", "distinct")

    def __init__(self, sets):
        self.sets = sets  # the groups' buffer sets, by identity
        self.ends: dict[tuple, int] = {}
        self.users: dict[tuple, list[int]] = {}
        for i, buffer_set in enumerate(sets):
            for key, end in buffer_set.dst_ends.items():
                self.users.setdefault(key, []).append(i)
                if end > self.ends.get(key, 0):
                    self.ends[key] = end
        self.shared = [users for users in self.users.values() if len(users) > 1]
        self.distinct = len({id(buffer_set) for buffer_set in sets})


class _Batch:
    __slots__ = (
        "public",
        "regions",
        "events",
        "recorded",
        "unrecorded",
        "keepalive",
        "destinations",
        "done",
        "state",
        "error",
        "waiters",
    )

    def __init__(self, public, regions, destinations):
        self.public = public
        self.regions = regions  # the _Regions this batch reads or writes
        self.events = []
        self.recorded = []  # recorded[i]: group i is enqueued and its event recorded
        self.unrecorded = False
        self.keepalive = []
        self.destinations = destinations  # [(buffer set, sorted dst pages)] per group
        self.done = 0
        self.state = ReadState.IN_FLIGHT
        self.error = ""
        self.waiters = 0  # threads blocked in wait_group on one of its events


def _lanes_apart(keys, stride: int) -> bool:
    """Whether the lanes [base % stride, + page_bytes) of ``keys`` are pairwise disjoint."""
    lanes = sorted((key[1] % stride, key[2]) for key in keys)
    for (start, width), (following, _) in zip(lanes, lanes[1:]):
        if start + width > following:
            return False
    last, width = lanes[-1]
    return last + width <= lanes[0][0] + stride


def _first_overlap(ends: dict):
    """Two destination keys whose pages may share a byte, or None.

    Keys of one space and stride whose lanes are pairwise disjoint interleave without
    touching (the buffers of a page object) and are checked as one span; any other
    two spans must not overlap. Spans are conservative: a false overlap is possible,
    a missed one is not.
    """
    spaces: dict = {}
    for key in ends:
        spaces.setdefault(key[0], []).append(key)
    for keys in spaces.values():
        spans, families = [], {}
        for key in keys:
            if key[3] > key[2]:
                families.setdefault(key[3], []).append(key)
            else:
                spans.append((key[1], ends[key], key))
        for stride, members in families.items():
            if _lanes_apart(members, stride):
                start = min(key[1] for key in members)
                spans.append((start, max(ends[key] for key in members), members[0]))
            else:
                spans.extend((key[1], ends[key], key) for key in members)
        spans.sort(key=lambda span: span[:2])
        previous = None
        for span in spans:
            if previous is not None and span[0] < previous[1]:
                return previous[2], span[2]
            if previous is None or span[1] > previous[1]:
                previous = span
    return None


def _pages_within(size: int, offset: int, page_bytes: int, stride: int) -> int:
    """How many pages of a buffer at ``offset`` with ``stride`` fit in ``size`` bytes."""
    room = size - offset - page_bytes
    return room // stride + 1 if room >= 0 else 0


def _page_info(value, name: str, cache: dict):
    """Validate a page-id array once per batch: ``(pages, sorted pages, max, has duplicates)``."""
    hit = cache.get(id(value))
    if hit is not None and hit[0] is value:
        return hit[1]
    pages = _as_pages(value, name)
    ordered = np.sort(pages)
    duplicates = ordered.size > 1 and bool((ordered[1:] == ordered[:-1]).any())
    info = (pages, ordered, int(ordered[-1]), duplicates)
    cache[id(value)] = (value, info)  # holds value, so its id stays unique
    return info


def _as_pages(value, name: str) -> np.ndarray:
    if hasattr(value, "is_cuda") and value.is_cuda:
        raise ValueError(f"{name} must be on the host; it is validated before launch")
    if hasattr(value, "numpy"):
        value = value.numpy()
    pages = np.asarray(value)
    if pages.ndim != 1 or pages.size == 0 or pages.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a non-empty 1-D integer array")
    pages = pages.astype(np.int64, copy=False)
    if int(pages.min()) < 0:
        raise ValueError(f"{name} has a negative page id")
    return pages


def _check_timeout(timeout) -> Optional[float]:
    """The monotonic deadline of ``timeout`` seconds, None for no timeout."""
    if timeout is None:
        return None
    if not (isinstance(timeout, (int, float)) and 0 <= timeout < float("inf")):
        raise ValueError("timeout must be finite and non-negative")
    return time.monotonic() + timeout


class KvPeerReader:
    """Imports peer regions on one device, reads pages out of them and writes pages
    into the ones imported writable.

    ``num_blocks`` is the copy kernel's CTA budget: more blocks copy faster out of a
    peer GPU but take SMs from whatever else runs on the device. On GB200, 64 blocks
    already reach the C2C limit (~210 GB/s) of a host-memory (EGM) source.
    """

    def __init__(self, device: int, *, num_blocks: int = 64, _ops=None):
        self.device = _checked_int(device, "device")
        _checked_int(num_blocks, "num_blocks", 1)
        self._ops = _ops if _ops is not None else _CudaOps(num_blocks=num_blocks)
        self._stream = self._ops.new_stream(self.device)
        self._ops.warmup(self._stream, load=True)
        self._write_stream = None  # made by the first writable import
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._regions: dict[str, _Region] = {}
        self._batches: dict[str, _Batch] = {}
        self._imported_leases: dict[tuple[str, str], None] = {}  # ordered set
        self._buffer_sets: dict[int, _BufferSet] = {}
        # Destinations of batches not yet known to be done: batch id -> groups, and
        # key -> [end, {batch id: [sorted destination pages of its groups]}].
        self._busy_groups: dict[str, list] = {}
        self._busy: dict[tuple, list] = {}
        self._last_shape: dict[
            bool, _Shape
        ] = {}  # by direction: loads and write-throughs interleave
        self._events: list = []  # recycled from released SUCCESS batches
        self._clique = self._ops.fabric_clique(self.device)
        self._fail_stop = ""

    # ---- regions

    @_guarded
    def import_region(
        self, desc: PeerRegionDescriptor, *, writable: bool = False
    ) -> PeerRegion:
        """Map the region read-only, or writable when the owner exported it so."""
        self._check_alive()
        if not isinstance(desc, PeerRegionDescriptor):
            raise ValueError("expected a PeerRegionDescriptor")
        desc.validate()
        if not isinstance(writable, bool):
            raise ValueError("writable must be a bool")
        if writable and not desc.writable:
            raise ValueError("the owner exported this lease read-only")
        lease_key = (desc.owner_id, desc.lease_id)
        if lease_key in self._imported_leases:
            raise ValueError("lease already imported; obtain a fresh export")
        mine = self._clique
        if (
            desc.fabric_clique is not None
            and mine is not None
            and tuple(desc.fabric_clique) != (bytes(mine[0]), int(mine[1]))
        ):
            raise PeerReadError("owner GPU is in a different NVLink fabric clique")
        if writable and self._write_stream is None:
            stream = self._ops.new_stream(self.device)
            self._ops.warmup(stream, load=False)
            self._write_stream = stream
        address = self._ops.map_import(
            desc.fabric_handle, desc.size, self.device, read_only=not writable
        )
        public = PeerRegion(uuid.uuid4().hex, desc.size, address, writable)
        self._regions[public.region_id] = _Region(public, desc, writable)
        self._imported_leases[lease_key] = None  # survives unmap: a lease maps once
        if len(self._imported_leases) > _MAX_LEASE_TOMBSTONES:
            del self._imported_leases[next(iter(self._imported_leases))]
        return public

    @_guarded
    def unmap(self, region: PeerRegion) -> None:
        """Unmap after every batch on the region is released; then the control plane may ACK."""
        item = self._region(region)
        if item.batches:
            raise PeerReadBusy("region has unreleased batches")
        self._ops.unmap_import(region.address, region.size)
        del self._regions[region.region_id]
        for key in [
            k for k, v in self._buffer_sets.items() if region.region_id in v.regions
        ]:
            del self._buffer_sets[key]

    # ---- batches

    @_guarded
    def read(self, groups: Sequence[ReadGroup], *, producer_stream=None) -> ReadBatch:
        """Enqueue ``groups`` in order; their buffers may read any imported regions.
        ``producer_stream`` (a torch stream or event of this device) is waited on first,
        for destinations its last user is still writing."""
        return self._submit(groups, producer_stream=producer_stream, load=True)

    @_guarded
    def write(self, groups: Sequence[WriteGroup], *, producer_stream=None) -> ReadBatch:
        """Enqueue ``groups`` in order on the write stream; their buffers may write any
        regions imported writable. ``producer_stream`` (a torch stream or event of this
        device) is waited on first, for source pages it is still writing. The batch is
        followed, waited for and released like a read."""
        return self._submit(groups, producer_stream=producer_stream, load=False)

    @_guarded
    def status(self, batch: ReadBatch) -> ReadStatus:
        item = self._batch(batch)
        self._poll(item)
        return ReadStatus(item.state, item.done, len(item.events), item.error)

    @_guarded
    def group_done(self, batch: ReadBatch, index: int) -> bool:
        item = self._batch(batch)
        if not 0 <= index < len(item.events):
            raise IndexError("group index out of range")
        self._poll(item)
        return index < item.done

    def wait(self, batch: ReadBatch, timeout: Optional[float] = None) -> ReadStatus:
        """Block until the batch leaves IN_FLIGHT. On timeout raise TimeoutError and keep everything."""
        deadline = _check_timeout(timeout)
        pause = 20e-6
        while True:
            result = self.status(batch)
            if result.state != ReadState.IN_FLIGHT:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("read still in flight; nothing may be released")
            time.sleep(pause)
            pause = min(pause * 2, 1e-3)

    def wait_group(
        self, batch: ReadBatch, index: int, timeout: Optional[float] = None
    ) -> ReadStatus:
        """Block until group ``index`` has landed or the batch has failed; the group has
        landed iff ``groups_done > index`` in the returned status.

        Without a timeout this blocks in the driver on the group's event, holding
        neither the reader's lock nor the GIL. With one it polls, and on timeout
        raises TimeoutError and keeps everything.
        """
        deadline = _check_timeout(timeout)
        if deadline is None:
            self._block_on_group(batch, index)
        pause = 20e-6
        while True:
            result = self._group_status(batch, index)
            if result.groups_done > index or result.state != ReadState.IN_FLIGHT:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("group still in flight; nothing may be released")
            time.sleep(pause)
            pause = min(pause * 2, 1e-3)

    @_guarded
    def release(self, batch: ReadBatch) -> None:
        item = self._batch(batch)
        self._poll(item)
        if item.state not in (ReadState.SUCCESS, ReadState.FAILED_DRAINED):
            raise PeerReadBusy(f"batch is {item.state.value}; it cannot be released")
        if item.waiters:
            raise PeerReadBusy("a thread is still waiting on the batch")
        if (
            item.state == ReadState.SUCCESS
            and len(self._events) + len(item.events) <= _MAX_POOLED_EVENTS
        ):
            self._events.extend(item.events)  # all fired; recorded again before reuse
        else:
            for event in item.events:
                self._ops.destroy_event(event)
        self._retire(item)
        for region in item.regions:
            region.batches.discard(batch.batch_id)
        del self._batches[batch.batch_id]

    @_guarded
    def close(self) -> None:
        if self._batches or self._regions:
            raise PeerReadBusy("release batches and unmap regions before close")
        while self._events:
            self._ops.destroy_event(self._events.pop())

    # ---- helpers

    def _check_alive(self) -> None:
        if self._fail_stop:
            raise PeerReadError(f"reader fail-stopped: {self._fail_stop}")

    def _region(self, region) -> _Region:
        item = (
            self._regions.get(region.region_id)
            if isinstance(region, PeerRegion)
            else None
        )
        if item is None or item.public is not region:
            raise ValueError("foreign, forged or unmapped region")
        return item

    def _batch(self, batch) -> _Batch:
        item = (
            self._batches.get(batch.batch_id) if isinstance(batch, ReadBatch) else None
        )
        if item is None or item.public is not batch:
            raise ValueError("foreign, forged or released batch")
        return item

    def _submit(self, groups, *, producer_stream, load: bool) -> ReadBatch:
        self._check_alive()
        groups = tuple(groups)
        if not groups:
            raise ValueError(f"empty {'read' if load else 'write'} batch")
        page_cache: dict = {}
        sets = [self._buffer_set(g, i, load=load) for i, g in enumerate(groups)]
        plans = [
            self._plan_group(g, i, bs, page_cache)
            for i, (g, bs) in enumerate(zip(groups, sets))
        ]
        self._check_destinations(plans)
        self._check_alive()  # polling in-flight batches above can fail-stop
        public = ReadBatch(uuid.uuid4().hex)
        regions = {rid: self._regions[rid] for bs in sets for rid in bs.regions}
        batch = _Batch(
            public,
            list(regions.values()),
            [(p.buffer_set, p.dst_sorted) for p in plans],
        )
        stream = self._stream if load else self._write_stream
        try:
            self._enqueue(batch, plans, producer_stream, stream)
        except BaseException:
            if batch.state == ReadState.FAILED_UNCERTAIN:
                self._admit(batch)  # pins its regions and destinations
            raise
        self._admit(batch)
        return public

    @_guarded
    def _group_status(self, batch: ReadBatch, index: int) -> ReadStatus:
        item = self._batch(batch)
        if not 0 <= index < len(item.events):
            raise IndexError("group index out of range")
        self._poll(item)
        return ReadStatus(item.state, item.done, len(item.events), item.error)

    def _block_on_group(self, batch: ReadBatch, index: int) -> None:
        """Block until group ``index``'s event fires, unless the group is known done."""
        if os.getpid() != self._pid:
            raise PeerReadError("peer KV objects cannot be used across fork")
        with self._lock:
            item = self._batch(batch)
            if not 0 <= index < len(item.events):
                raise IndexError("group index out of range")
            self._poll(item)
            if index < item.done or item.state != ReadState.IN_FLIGHT:
                return
            event = item.events[index]
            item.waiters += 1  # keeps release() from destroying or recycling it
        try:
            self._ops.sync_event(event)  # its outcome is read back by _poll
        finally:
            with self._lock:
                item.waiters -= 1

    def _plan_group(
        self, group, index: int, buffer_set: _BufferSet, page_cache: dict
    ) -> _GroupPlan:
        src_pages, _, src_max, _ = _page_info(group.src_pages, "src_pages", page_cache)
        dst_pages, dst_sorted, dst_max, dst_duplicates = _page_info(
            group.dst_pages, "dst_pages", page_cache
        )
        if src_pages.size != dst_pages.size:
            raise ValueError(f"group {index}: src_pages and dst_pages differ in length")
        if dst_duplicates:
            raise ValueError(f"group {index}: overlapping destination pages")
        src_side, dst_side = (
            ("region", "buffer") if buffer_set.load else ("buffer", "region")
        )
        if src_max >= buffer_set.src_limit:
            raise ValueError(f"group {index}: source page beyond its {src_side}")
        if dst_max >= buffer_set.dst_limit:
            raise ValueError(f"group {index}: destination page beyond the {dst_side}")
        if buffer_set.load:
            return _GroupPlan(buffer_set, src_pages, dst_pages, dst_sorted)
        return _GroupPlan(buffer_set, dst_pages, src_pages, dst_sorted)

    def _buffer_set(self, group, index: int, *, load: bool) -> _BufferSet:
        kind = ReadGroup if load else WriteGroup
        if not isinstance(group, kind) or not group.buffers:
            raise ValueError(f"group {index}: expected a {kind.__name__} with buffers")
        buffers = group.buffers
        cached = self._buffer_sets.get(id(buffers))
        if cached is not None and cached.buffers is buffers:
            if cached.load != load:
                raise ValueError(
                    f"group {index}: these buffers are for the other direction"
                )
            return cached  # unmap evicts sets that use the unmapped region
        if load:
            buffer_set = self._read_set(buffers, index)
        else:
            buffer_set = self._write_set(buffers, index)
        if _first_overlap(buffer_set.dst_ends) is not None:
            raise ValueError(f"group {index}: buffers overlap at the destination")
        if isinstance(buffers, tuple):  # a list could change under the cache
            if len(self._buffer_sets) >= _MAX_BUFFER_SETS:
                del self._buffer_sets[next(iter(self._buffer_sets))]
            self._buffer_sets[id(buffers)] = buffer_set
        return buffer_set

    def _read_set(self, buffers, index: int) -> _BufferSet:
        by_shape: dict[tuple[int, int], tuple[list, list]] = {}
        dst_ends, regions = {}, {}
        src_limit = dst_limit = 2**63
        for buf in buffers:
            if not isinstance(buf, PeerBuffer):
                raise ValueError(f"group {index}: expected PeerBuffer entries")
            region = self._region(buf.src_region)
            page_bytes = _checked_int(buf.page_bytes, "page_bytes", 1)
            src_offset = _checked_int(buf.src_offset, "src_offset")
            stride = _checked_int(buf.src_stride, "src_stride") or page_bytes
            dst_base = _checked_int(buf.dst_base, "dst_base", 1)
            capacity = _checked_int(buf.dst_num_pages, "dst_num_pages", 1)
            _check_geometry(index, page_bytes, stride, src_offset, dst_base)
            end = dst_base + capacity * page_bytes
            self._check_local(dst_base, end, index, "destination")
            key = (None, dst_base, page_bytes, page_bytes)
            if key in dst_ends:
                raise ValueError(f"group {index}: two buffers share a destination")
            src_limit = min(
                src_limit,
                _pages_within(region.desc.size, src_offset, page_bytes, stride),
            )
            dst_limit = min(dst_limit, capacity)
            regions[region.public.region_id] = None
            peers, locals_ = by_shape.setdefault((page_bytes, stride), ([], []))
            peers.append(region.public.address + src_offset)
            locals_.append(dst_base)
            dst_ends[key] = end
        return _BufferSet(
            buffers=buffers,
            load=True,
            regions=tuple(regions),
            launches=_launches(by_shape),
            dst_ends=dst_ends,
            src_limit=src_limit,
            dst_limit=dst_limit,
        )

    def _write_set(self, buffers, index: int) -> _BufferSet:
        by_shape: dict[tuple[int, int], tuple[list, list]] = {}
        dst_ends, regions = {}, {}
        src_limit = dst_limit = 2**63
        for buf in buffers:
            if not isinstance(buf, PeerWriteBuffer):
                raise ValueError(f"group {index}: expected PeerWriteBuffer entries")
            region = self._region(buf.dst_region)
            if not region.writable:
                raise ValueError(
                    f"group {index}: destination region is mapped read-only"
                )
            page_bytes = _checked_int(buf.page_bytes, "page_bytes", 1)
            dst_offset = _checked_int(buf.dst_offset, "dst_offset")
            stride = _checked_int(buf.dst_stride, "dst_stride") or page_bytes
            src_base = _checked_int(buf.src_base, "src_base", 1)
            capacity = _checked_int(buf.src_num_pages, "src_num_pages", 1)
            dst_pages = _checked_int(buf.dst_num_pages, "dst_num_pages")
            _check_geometry(index, page_bytes, stride, dst_offset, src_base)
            self._check_local(
                src_base, src_base + capacity * page_bytes, index, "source"
            )
            fit = _pages_within(region.desc.size, dst_offset, page_bytes, stride)
            if fit == 0 or dst_pages > fit:
                raise ValueError(f"group {index}: destination runs past its region")
            dst_pages = dst_pages or fit
            space = (region.desc.owner_id, region.desc.region_id)
            key = (space, dst_offset, page_bytes, stride)
            if key in dst_ends:
                raise ValueError(f"group {index}: two buffers share a destination")
            src_limit = min(src_limit, capacity)
            dst_limit = min(dst_limit, dst_pages)
            regions[region.public.region_id] = None
            peers, locals_ = by_shape.setdefault((page_bytes, stride), ([], []))
            peers.append(region.public.address + dst_offset)
            locals_.append(src_base)
            dst_ends[key] = dst_offset + (dst_pages - 1) * stride + page_bytes
        return _BufferSet(
            buffers=buffers,
            load=False,
            regions=tuple(regions),
            launches=_launches(by_shape),
            dst_ends=dst_ends,
            src_limit=src_limit,
            dst_limit=dst_limit,
        )

    def _check_local(self, start: int, end: int, index: int, what: str) -> None:
        """[start, end) must be memory of this device and outside every imported region."""
        for pointer in (start, end - 1):
            if self._ops.pointer_device(pointer) != self.device:
                raise ValueError(
                    f"group {index}: {what} is not memory of device {self.device}"
                )
        if any(
            start < r.public.address + r.public.size and r.public.address < end
            for r in self._regions.values()
        ):
            raise ValueError(f"group {index}: {what} is an imported region")

    def _check_destinations(self, plans) -> None:
        """Reject any destination byte two writers could share.

        Within the batch it is the caller's error (ValueError); against a batch
        still in flight it is PeerReadBusy. Writers can share a byte without a
        partial overlap only through the same destination key, where they
        conflict exactly on shared page ids.
        """
        overlap = {}  # (id(pages a), id(pages b)) -> pages intersect

        def shared(a, b) -> bool:
            key = (id(a), id(b))
            if key not in overlap:
                overlap[key] = np.intersect1d(a, b, assume_unique=True).size > 0
            return overlap[key]

        shape = self._shape_of(plans)
        for users in shape.shared:  # the groups that write one key
            for i, a in enumerate(users):
                for b in users[i + 1 :]:
                    if shared(plans[a].dst_sorted, plans[b].dst_sorted):
                        raise ValueError(
                            "overlapping destination pages within the batch"
                        )
        self._check_new_geometry(shape)
        busy = self._busy
        conflicts = {}  # batch ids in flight that write one of our pages; ordered
        for key, users in shape.users.items() if busy else ():
            entry = busy.get(key)
            for batch_id, pages in entry[1].items() if entry is not None else ():
                if batch_id not in conflicts and any(
                    shared(plans[i].dst_sorted, p) for i in users for p in pages
                ):
                    conflicts[batch_id] = None
        for batch_id in conflicts:
            self._poll(
                self._batches[batch_id]
            )  # it may have finished since anyone looked
            if batch_id in self._busy_groups:
                raise PeerReadBusy("destination pages of a batch in flight")

    def _shape_of(self, plans) -> _Shape:
        """The batch's destination keys, checked alone; layer-wise loads repeat their
        buffer sets, so the last shape is reused while they do."""
        sets = [plan.buffer_set for plan in plans]
        load = sets[0].load
        shape = self._last_shape.get(load)
        if shape is not None and shape.sets == sets:
            return shape
        shape = _Shape(sets)
        # Alone first: an uncertain batch never retires, so a self-overlap found
        # against it would stay PeerReadBusy instead of the caller's ValueError.
        # One buffer set was checked alone when it was validated.
        if shape.distinct > 1 and _first_overlap(shape.ends) is not None:
            raise ValueError("overlapping destination ranges within the batch")
        self._last_shape[load] = shape
        return shape

    def _check_new_geometry(self, shape: _Shape, *, polled: bool = False) -> None:
        """Partial overlaps with batches in flight can only come from a destination key
        not in use yet, or one in use whose range this batch extends to more pages."""
        busy = self._busy
        fresh = {}
        for key, end in shape.ends.items() if busy else ():
            entry = busy.get(key)
            if entry is None or end > entry[0]:
                fresh[key] = end
        if not fresh:
            return
        spaces = {key[0] for key in fresh}
        merged = {k: v[0] for k, v in busy.items() if k[0] in spaces}
        if not merged:
            return
        for key, end in fresh.items():
            merged[key] = max(end, merged.get(key, 0))
        if _first_overlap(merged) is None:
            return
        if not polled:  # batches may have finished since anyone looked
            for batch_id in list(self._busy_groups):
                self._poll(self._batches[batch_id])
            return self._check_new_geometry(shape, polled=True)
        raise PeerReadBusy("destination overlaps one of a batch in flight")

    def _admit(self, batch: _Batch) -> None:
        batch_id = batch.public.batch_id
        self._batches[batch_id] = batch
        for region in batch.regions:
            region.batches.add(batch_id)
        self._busy_groups[batch_id] = batch.destinations
        busy = self._busy
        for buffer_set, pages in batch.destinations:
            for key, end in buffer_set.dst_ends.items():
                entry = busy.get(key)
                if entry is None:
                    busy[key] = [end, {batch_id: [pages]}]
                    continue
                # The end is kept until the key's last batch retires: neighbours may
                # see a false PeerReadBusy meanwhile, never a missed overlap.
                if end > entry[0]:
                    entry[0] = end
                entry[1].setdefault(batch_id, []).append(pages)

    def _retire(self, batch: _Batch) -> None:
        """Its destinations are no longer written: drop them from the in-flight view."""
        batch_id = batch.public.batch_id
        busy = self._busy
        for buffer_set, _ in self._busy_groups.pop(batch_id, None) or ():
            for key in buffer_set.dst_ends:
                entry = busy.get(key)  # gone once another group of the batch dropped it
                if entry is not None and entry[1].pop(batch_id, None) is not None:
                    if not entry[1]:
                        del busy[key]

    def _enqueue(self, batch: _Batch, plans, producer_stream, stream) -> None:
        """Everything that may allocate or sync happens here, before any group is enqueued."""
        ops = self._ops
        meta, spans, pages, tables = _pack(plans)
        batch.recorded = [False] * len(plans)
        uploaded = []
        with ops.current(stream):
            try:
                for _ in plans:  # kept one by one so a failure destroys the rest
                    batch.events.append(
                        self._events.pop() if self._events else ops.new_event(stream)
                    )
                if producer_stream is not None:
                    ops.wait_stream(stream, producer_stream)
                keepalive, views = ops.stage(stream, meta, spans)
                batch.keepalive.append(keepalive)
                for buffer_set, table_spans in tables:
                    buffer_set.tables = [(views[p], views[q]) for p, q in table_spans]
                    uploaded.append(buffer_set)
                for index, (plan, (peer, local)) in enumerate(zip(plans, pages)):
                    self._enqueue_group(
                        stream, batch, index, _views(plan, views[peer], views[local])
                    )
                    batch.keepalive.append(plan.buffer_set.tables)
            except BaseException as error:
                for buffer_set in uploaded:  # the upload may never have run
                    buffer_set.tables = None
                code = self._drain(batch)
                if code != 0:
                    self._enter_fail_stop(
                        f"enqueue failed ({error!r}), drain returned {code}"
                    )
                    batch.state = ReadState.FAILED_UNCERTAIN
                    batch.error = f"enqueue failed without draining: {error!r}"
                    error.add_note("the read did not drain; the reader is fail-stopped")
                else:
                    for event in batch.events:
                        ops.destroy_event(event)
                raise

    def _enqueue_group(self, stream, batch: _Batch, index: int, views) -> None:
        """Launch one group on ``stream``, which must be current, and record its event,
        even when a launch raised, so the event covers whatever did get enqueued."""
        try:
            for view in views:
                self._ops.launch(view)
        finally:
            try:
                self._ops.record(batch.events[index], stream)
            except BaseException:
                batch.unrecorded = True  # launches no event covers: drain unknowable
                raise
            batch.recorded[index] = True

    def _drain(self, batch: _Batch) -> int:
        """Wait for everything this batch enqueued; 0 means it has stopped touching memory."""
        if batch.unrecorded:
            return -1
        recorded = [i for i, done in enumerate(batch.recorded) if done]
        return self._ops.sync_event(batch.events[recorded[-1]]) if recorded else 0

    def _poll(self, batch: _Batch) -> None:
        while batch.state == ReadState.IN_FLIGHT and batch.done < len(batch.events):
            if not batch.recorded[batch.done]:
                return  # not enqueued yet
            code = self._ops.query_event(batch.events[batch.done])
            if code == self._ops.not_ready:
                return
            if code == 0:
                batch.done += 1
                continue
            for buffer_set, _ in batch.destinations:  # re-upload before reuse
                buffer_set.tables = None
            drained = self._drain(batch)
            batch.error = f"group {batch.done}: CUDA error {code}"
            if drained != 0 or self._ops.is_sticky(code):
                self._enter_fail_stop(batch.error)
            else:
                batch.state = ReadState.FAILED_DRAINED
                self._retire(batch)
            return
        if batch.state == ReadState.IN_FLIGHT:
            batch.state = ReadState.SUCCESS
            batch.keepalive.clear()
            self._retire(batch)

    def _enter_fail_stop(self, reason: str) -> None:
        self._fail_stop = reason
        for batch in self._batches.values():
            if batch.state == ReadState.IN_FLIGHT:
                batch.state = ReadState.FAILED_UNCERTAIN
                batch.error = batch.error or f"reader fail-stopped: {reason}"


def _check_geometry(
    index: int, page_bytes: int, stride: int, peer_offset: int, local_base: int
) -> None:
    if (page_bytes | stride | peer_offset | local_base) % PAGE_ALIGN_BYTES:
        raise ValueError(
            f"group {index}: offsets, bases, strides and page sizes must be "
            f"{PAGE_ALIGN_BYTES}-byte aligned"
        )
    if stride < page_bytes:
        raise ValueError(f"group {index}: the peer stride is shorter than a page")


def _launches(by_shape):
    """One launch per (page size, peer stride): [(page_bytes, stride, peer, local bases)]."""
    return [
        (page_bytes, stride, np.asarray(peers, np.int64), np.asarray(locals_, np.int64))
        for (page_bytes, stride), (peers, locals_) in by_shape.items()
    ]


def _views(plan: _GroupPlan, peer_pages, local_pages):
    """The launch arguments of one group, from its set's resident tables."""
    buffer_set = plan.buffer_set
    return [
        (
            buffer_set.load,
            page_bytes,
            stride,
            peer_ptrs,
            local_ptrs,
            peer_pages,
            local_pages,
        )
        for (page_bytes, stride, _, _), (peer_ptrs, local_ptrs) in zip(
            buffer_set.launches, buffer_set.tables
        )
    ]
