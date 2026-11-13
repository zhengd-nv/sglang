# SPDX-License-Identifier: Apache-2.0
"""Page copies between this GPU and a peer mapping (NVLink), on the page-row kernel.

The peer side, an imported FABRIC mapping of a peer GPU's KV pool or of host memory
on a NUMA node (EGM), is the kernel's "host" side: an address table with its own
page stride, so a peer page can sit inside a larger page object. The local side is
contiguous pages of ``page_bytes``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Measured on GB300 NVLink with GLM-5.2 pages (36864 / 8448 B): 4, 8 and 16 vectors
# per lane tie; 64 blocks reach 95% of a contiguous copy.
_VECS_PER_LANE = 16
_BLOCK_SIZE = 512


@cache_once
def _jit_peer_page_module(*, load: bool, num_blocks: int) -> Module:
    # page_row's "host" side is just an address table; here it is the peer mapping.
    args = make_cpp_args(load, _VECS_PER_LANE, _BLOCK_SIZE, num_blocks)
    return load_jit(
        "page_row",
        *args,
        cuda_files=["kvcacheio/page_row.cuh"],
        cuda_wrappers=[("run", f"&PageRowKernel<{args}>::run")],
    )


def peer_page_kernel(*, load: bool, num_blocks: int) -> Callable:
    """The compiled copy, peer -> local when ``load`` and local -> peer otherwise.

    Call it as ``run(peer_ptrs, local_ptrs, peer_pages, local_pages, peer_stride,
    page_bytes, page_bytes)`` on the current stream; the arguments are those of
    :func:`gather_pages`. Each (``load``, ``num_blocks``) compiles its own module.
    """
    return _jit_peer_page_module(load=load, num_blocks=num_blocks).run


def gather_pages(
    *,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
    page_bytes: int,
    num_blocks: int,
    src_stride: Optional[int] = None,
) -> None:
    """Copy page ``src_pages[i]`` of every buffer to page ``dst_pages[i]`` on the current stream.

    ``src_ptrs`` / ``dst_ptrs`` are CUDA uint64 ``[buffers]`` page-0 addresses; page
    ``p`` of a source buffer starts at ``ptr + p * src_stride`` (``page_bytes`` when
    None), of a destination buffer at ``ptr + p * page_bytes``. Source addresses may
    be a peer mapping. Pages are validated, same-device CUDA int32/int64 ids. Bases,
    ``page_bytes`` and the stride must be 128-byte aligned. ``num_blocks`` caps the
    CTAs taken from other work; each value compiles its own module.
    """
    if src_pages.numel() == 0 or src_ptrs.numel() == 0:
        return
    peer_page_kernel(load=True, num_blocks=num_blocks)(
        src_ptrs,
        dst_ptrs,
        src_pages,
        dst_pages,
        page_bytes if src_stride is None else src_stride,
        page_bytes,
        page_bytes,
    )


def scatter_pages(
    *,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
    page_bytes: int,
    num_blocks: int,
    dst_stride: Optional[int] = None,
) -> None:
    """The write direction of :func:`gather_pages`: local page ``src_pages[i]`` (stride
    ``page_bytes``) of every buffer to page ``dst_pages[i]`` of the peer buffer, which
    starts at ``ptr + p * dst_stride``. The peer mapping must be writable."""
    if src_pages.numel() == 0 or src_ptrs.numel() == 0:
        return
    peer_page_kernel(load=False, num_blocks=num_blocks)(
        dst_ptrs,
        src_ptrs,
        dst_pages,
        src_pages,
        page_bytes if dst_stride is None else dst_stride,
        page_bytes,
        page_bytes,
    )
