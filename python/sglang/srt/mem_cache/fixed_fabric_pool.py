# SPDX-License-Identifier: Apache-2.0
"""Fixed startup GLM DSA backing. No resize, staging pool, or allocation fallback.

The CUDA array interface owner is retained by PyTorch's storage deleter (and
therefore by tensor views). Never use non-owning pointer storage for this pool.
"""

import logging
from math import prod
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)
_QUARANTINE = []
_MAX_BYTES = 2**63 - 1


class BufferSlice(NamedTuple):
    allocation_index: int
    offset: int
    nbytes: int


def _plan_buffers(shapes, itemsize, coalesce):
    """Pure byte planning: reject malformed/overflowing layouts before CUDA."""
    if type(coalesce) is not bool:
        raise ValueError("coalesce must be a bool")
    if type(itemsize) is not int or itemsize <= 0:
        raise ValueError("dtype itemsize must be a positive integer")
    try:
        shapes = [tuple(shape) for shape in shapes]
    except TypeError as exc:
        raise ValueError("fixed buffers require integer shapes") from exc
    if not shapes or any(
        not shape or any(type(d) is not int or d <= 0 for d in shape)
        for shape in shapes
    ):
        raise ValueError("fixed buffers require nonempty positive integer shapes")
    slices, end = [], 0
    for i, shape in enumerate(shapes):
        nbytes = prod(shape) * itemsize
        offset = (end + 255) // 256 * 256 if coalesce else 0
        if nbytes > _MAX_BYTES or offset + nbytes > _MAX_BYTES:
            raise ValueError("fixed buffer byte extent exceeds signed 64-bit range")
        slices.append(BufferSlice(0 if coalesce else i, offset, nbytes))
        end = offset + nbytes
    return shapes, slices, end


def validate_glm_geometry(
    *,
    size,
    page_size,
    layer_num,
    kv_lora_rank,
    qk_rope_head_dim,
    kv_cache_dim,
    index_head_dim,
    dtype,
    index_kpool,
    index_kpool_compress,
    tail_extra_slots,
    skip_topk_layers,
    index_buf_size,
    dcp_replicated,
    enable_memory_saver,
):
    """Fail before allocating any tensor for unsupported physical layouts."""
    if (
        type(size) is not int
        or size <= 0
        or size % 64
        or page_size != 64
        or layer_num != 78
        or kv_lora_rank != 512
        or qk_rope_head_dim != 64
        or kv_cache_dim != 576
        or index_head_dim != 128
        or dtype != torch.float8_e4m3fn
        or index_kpool != 1
        or index_kpool_compress
        or tail_extra_slots
        or (
            skip_topk_layers is not None
            and (len(skip_topk_layers) != layer_num or any(skip_topk_layers))
        )
        or index_buf_size not in (None, size)
        or dcp_replicated
        or enable_memory_saver
    ):
        raise ValueError(
            "fixed FABRIC pool requires full 78-layer GLM raw FP8 MLA "
            "576 + DSA 128/FP32-scale, page64; no compression, "
            "skip-topk, DCP, MTP, or memory saver"
        )


class _FabricAllocation:
    def __init__(self, nbytes, device_id):
        from sglang.srt.utils import cuda_vmm_utils as vmm

        self.device_id, self.nbytes = device_id, nbytes
        self.reservation = None
        with torch.cuda.device(device_id):
            drv = vmm._get_cuda_driver()
            fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            # Explicit: unlike the ordinary VMM allocator, never probe POSIX/NONE
            # and never omit RDMA registration capability for the existing P->D path.
            prop = vmm.make_device_allocation_prop(
                device_id, handle_types=fabric, gpu_direct_rdma=True
            )
            self.granularity = vmm.get_allocation_granularity(prop)
            self.size = vmm.align_up(nbytes, self.granularity)
            if self.size > _MAX_BYTES:
                raise ValueError(
                    "rounded FABRIC allocation exceeds signed 64-bit range"
                )
            self.reservation = vmm.VmmReservation(
                self.size, prop, device_id, alignment=self.granularity
            )
            self.base = self.reservation.base
            handle = self.reservation.map(0, self.size, retain_handle=True)
            vmm.check_drv(
                drv.cuMemExportToShareableHandle(handle, fabric, 0),
                "cuMemExportToShareableHandle(fixed pool)",
            )
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
            "strides": None,
        }

    def __del__(self):
        reservation = getattr(self, "reservation", None)
        if reservation is None:
            return
        try:
            # External memory is outside torch's caching allocator and stream
            # recording. Drain before releasing even when the last view dies.
            with torch.cuda.device(self.device_id):
                torch.cuda.synchronize(self.device_id)
                reservation.close()
            self.reservation = None
        except Exception:
            # Uncertain GPU completion must leak, never free in-flight memory.
            _QUARANTINE.append(self)
            logger.exception("fixed FABRIC allocation quarantined during teardown")


class FixedFabricBufferOwner:
    """Own fixed buffers; close drops references, while live views retain memory."""

    def __init__(self, *, device, dtype, shapes, coalesce=False):
        self.tensors, self.allocations, self.buffer_slices = [], [], []
        self.closed = False
        shapes, slices, total = _plan_buffers(shapes, dtype.itemsize, coalesce)
        self.coalesced = coalesce
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("fixed FABRIC backing requires CUDA")
        self.device_id = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        self.dtype = dtype
        try:
            if coalesce:
                allocation = _FabricAllocation(total, self.device_id)
                self.allocations.append(allocation)
                root = torch.as_tensor(allocation, device=f"cuda:{self.device_id}")
                for shape, span in zip(shapes, slices):
                    self.tensors.append(
                        root.narrow(0, span.offset, span.nbytes)
                        .view(dtype)
                        .reshape(shape)
                        .zero_()
                    )
                self.buffer_slices = slices
                return
            for shape, span in zip(shapes, slices):
                allocation = _FabricAllocation(span.nbytes, self.device_id)
                self.allocations.append(allocation)
                tensor = torch.as_tensor(allocation, device=f"cuda:{self.device_id}")
                self.tensors.append(tensor.view(dtype).reshape(shape).zero_())
            self.buffer_slices = slices
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        self.tensors.clear()
        self.allocations.clear()
        self.buffer_slices.clear()
