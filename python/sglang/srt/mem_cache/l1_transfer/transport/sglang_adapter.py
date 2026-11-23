# SPDX-License-Identifier: Apache-2.0
"""Strict adapters for MHA post-capture and fixed startup GLM MLA/DSA pools.

Pool classes are imported lazily and nothing is allocated here. The caller
freezes pool/page ownership, including explicit close/reset/resize, for the
entire registration lifetime. Private allocator metadata is version-specific;
unknown contracts fail closed.
"""

from itertools import pairwise
from math import prod

from .api import checked_int
from .layout import BufferLayout, PoolLayout, RegionSpan


class SGLangPoolAdapter:
    def __init__(self, pool, engine, *, tp_rank=None, tp_size=None, shard_id=None):
        try:
            if getattr(pool, "fixed_fabric_pool", False):
                capacity, layout, buffers = self._prepare_glm(
                    pool, engine.device, tp_rank, tp_size, shard_id
                )
                # Snapshot tensors: even an explicit pool clear cannot invalidate
                # borrowed backing while registration/export resources are alive.
                owner = (
                    pool,
                    tuple(pool.kv_buffer),
                    tuple(pool.index_k_with_scale_buffer),
                )
            else:
                capacity, layout, buffers = self._prepare(pool, engine.device)
                owner = pool
        except (AttributeError, TypeError) as exc:
            raise ValueError("unsupported or incomplete pool metadata") from exc
        self.pool, self.engine = pool, engine
        self.capacity, self.layout = capacity, layout
        self.memories, self.bindings = [], {}
        registered = {}
        try:
            for name, pieces in buffers:
                spans = []
                for offset, length, address, physical_size, region_offset in pieces:
                    if address not in registered:
                        handle = engine.register_vmm(
                            address, physical_size, owner=owner
                        )
                        registered[address] = handle
                        self.memories.append(handle)
                    spans.append(
                        RegionSpan(offset, length, registered[address], region_offset)
                    )
                self.bindings[name] = tuple(spans)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _prepare_glm(pool, device, tp_rank, tp_size, shard_id):
        """Every MLA and index layer is mandatory; TP identity is explicit.

        This describes one rank only. It does not coordinate ownership, Pin,
        import completion, or failure across ranks; serving must supply that.
        """
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        checked_int(tp_size, "tp_size", 1)
        checked_int(tp_rank, "tp_rank")
        if tp_rank >= tp_size or not isinstance(shard_id, str) or not shard_id:
            raise ValueError("matching rank/shard identity is required")
        if type(pool) is not DSATokenToKVPool:
            raise ValueError("unsupported GLM raw MLA/DSA layout")
        if pool.use_fp4_indexer:
            raise ValueError("FP4 indexer is unsupported; GLM needs FP8 index keys")
        size = checked_int(pool.size, "pool size", 1)
        if (
            size % 64
            or pool.page_size != 64
            or pool.layer_num != 78
            or pool.start_layer != 0
            or pool.kv_lora_rank != 512
            or pool.qk_rope_head_dim != 64
            or pool.kv_cache_dim != 576
            or str(pool.dtype) != "torch.float8_e4m3fn"
            or str(pool.store_dtype) != "torch.uint8"
            or pool.dsa_kv_cache_store_fp8
            or pool.dcp_replicated
            or pool.index_head_dim != 128
            or pool.index_kpool != 1
            or pool.index_kpool_compress
            or pool.tail_extra_slots
            or pool.index_page_size != 64
            or pool.index_buf_size != size
            or len(pool.skip_topk_layers) != 78
            or any(pool.skip_topk_layers)
        ):
            raise ValueError("unsupported GLM raw MLA/DSA layout")
        capacity = size // 64 + 1  # page 0 is PAD, not a request-owned page
        layouts, buffers, extents = [], [], []
        for kind, owner, tensors, infos, shape, page_bytes, ordering in (
            (
                "mla",
                pool._fixed_fabric_owner,
                pool.kv_buffer,
                pool.get_contiguous_buf_infos(),
                (size + 64, 1, 576),
                64 * 576,
                "raw-fp8-e4m3:token[nope512,rope64]",
            ),
            (
                "dsa_index",
                pool.index_key_cache._fixed_fabric_owner,
                pool.index_k_with_scale_buffer,
                pool.get_state_buf_infos(),
                (capacity, 8448),
                8448,
                "page[key-fp8[64,128],scale-fp32[64]]",
            ),
        ):
            if (
                owner is None
                or owner.closed
                or owner.allocation is None
                or checked_int(owner.device_id, "owner device") != device
                or str(owner.dtype) != "torch.uint8"
            ):
                raise ValueError("missing or closed fixed FABRIC owner")
            ptrs, sizes, items = infos
            if any(
                len(x) != 78
                for x in (ptrs, sizes, items, tensors, owner.tensors)
                + (owner.buffer_slices,)
            ):
                raise ValueError("every GLM MLA/index layer must have backing")
            logical = checked_int(capacity * page_bytes, "logical extent", 1)
            stride = checked_int((logical + 255) // 256 * 256, "slice stride", 1)
            root_extent = checked_int(77 * stride + logical, "root extent", 1)
            allocation = owner.allocation
            base = checked_int(allocation.base, "allocation base", 1)
            gran = checked_int(allocation.granularity, "granularity", 1)
            physical = checked_int(allocation.size, "physical size", 1)
            checked_int(base + physical, "allocation end", 1)
            # The allocation maps [base, base + size) once when built and unmaps
            # only on teardown, which clears its reservation.
            reservation = allocation.reservation
            if (
                checked_int(allocation.device_id, "allocation device") != device
                or allocation.nbytes != root_extent
                or base % gran
                or base % 256
                or physical != (root_extent + gran - 1) // gran * gran
                or reservation is None
                or reservation.base != base
                or reservation.size != physical
            ):
                raise ValueError("GLM physical root backing disagrees")
            extents.append((base, base + physical))
            for layer, (
                tensor,
                owned_tensor,
                span,
                ptr,
                extent,
                item,
            ) in enumerate(
                zip(tensors, owner.tensors, owner.buffer_slices, ptrs, sizes, items)
            ):
                checked_int(ptr, "buffer pointer", 1)
                checked_int(extent, "buffer extent", 1)
                checked_int(item, "page bytes", 1)
                offset = layer * stride
                if (
                    checked_int(span.offset, "slice offset") != offset
                    or checked_int(span.nbytes, "slice extent", 1) != logical
                ):
                    raise ValueError("GLM slice is not canonical")
                if (
                    tensor is not owned_tensor
                    or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous()
                    or str(tensor.dtype) != "torch.uint8"
                    or tensor.data_ptr() != ptr
                    or tensor.nbytes != extent
                    or base + offset != ptr
                    or extent != logical
                    or item != page_bytes
                ):
                    raise ValueError("GLM tensor/owner/physical backing disagree")
                name = f"layer{layer}.{kind}"
                layouts.append(
                    BufferLayout(
                        name,
                        page_bytes,
                        f"GLM-DSA-v1:{ordering}:tp={tp_size}:rank={tp_rank}:shard={shard_id}",
                    )
                )
                buffers.append((name, [(0, extent, base, physical, offset)]))
        extents.sort()
        if any(a[1] > b[0] for a, b in pairwise(extents)):
            raise ValueError("overlapping GLM physical allocations")
        layout = PoolLayout("GLM-MLA:fp8_e4m3;DSA:fp8_e4m3+fp32", 64, tuple(layouts))
        layout.validate()
        return capacity, layout, buffers

    @staticmethod
    def _prepare(pool, device):
        """Validate the complete pool before registering even the first mapping."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if type(pool) is not MHATokenToKVPool or pool.use_hnd:
            raise ValueError("only MHATokenToKVPool NHD supported")
        if pool.kv_cache_layout != "nhd":
            raise ValueError("only NHD supported")
        size = checked_int(pool.size, "pool size", 1)
        page_size = checked_int(pool.page_size, "page_size", 1)
        if size % page_size:
            raise ValueError("pool capacity must contain whole pages")
        owner = pool._post_capture_owner
        if owner is None or owner._final_num_tokens is None:
            raise ValueError("requires finalized post-capture VMM backing")
        if (
            checked_int(owner._final_num_tokens, "finalized tokens", 1) != size
            or checked_int(owner.page_size, "owner page_size", 1) != page_size
            or owner.store_dtype != pool.store_dtype
            or checked_int(owner.device_id, "owner device") != device
        ):
            raise ValueError("finalized owner and pool metadata disagree")
        checked_int(pool.start_layer, "start_layer")
        itemsize = checked_int(pool.store_dtype.itemsize, "dtype itemsize", 1)
        arena = owner._arena
        if arena is None or arena._closed or arena._allocation._closed:
            raise ValueError("pool backing is closed")
        base = checked_int(arena.base, "arena base", 1)
        arena_end = checked_int(
            base + checked_int(arena.reserved, "arena size", 1), "arena end"
        )
        granularity = checked_int(arena.granularity, "mapping granularity", 1)
        mappings = []
        for address, length, _ in arena._allocation._mappings:
            checked_int(address, "mapping address", 1)
            checked_int(length, "mapping length", 1)
            end = checked_int(address + length, "mapping end")
            if (
                address % granularity
                or length % granularity
                or address < base
                or end > arena_end
            ):
                raise ValueError("invalid physical mapping extent/alignment")
            mappings.append((address, length))
        mappings.sort()
        if any(a + length > b for (a, length), (b, _) in pairwise(mappings)):
            raise ValueError("overlapping physical mappings")

        ptrs, sizes, items = pool.get_contiguous_buf_infos()
        descs, specs = pool._kv_buffer_descs, owner._specs
        if (
            not descs
            or len({len(descs), len(ptrs), len(sizes), len(items), len(specs)}) != 1
        ):
            raise ValueError("pool metadata arrays must have equal nonzero arity")
        # Includes the physical padding page. Generic transport can address page 0;
        # serving PrivateSlotView forbids it when restoring request-owned KV slots.
        capacity = size // page_size + 1
        layouts, buffers, extents = [], [], []
        for desc, spec, ptr, extent, item in zip(descs, specs, ptrs, sizes, items):
            checked_int(ptr, "buffer pointer", 1)
            checked_int(extent, "buffer extent", 1)
            checked_int(item, "page bytes", 1)
            shape = tuple(desc.shape)
            if len(shape) != 3:
                raise ValueError("NHD descriptor must have three axes")
            for dimension in shape:
                checked_int(dimension, "shape dimension", 1)
            row_bytes = checked_int(desc.row_bytes, "row bytes", 1)
            if (
                checked_int(desc.tokens_per_row, "tokens per row", 1) != 1
                or shape[0] < size + page_size
                or row_bytes != prod(shape[1:]) * itemsize
                or item != page_size * row_bytes
                or extent != checked_int(capacity * item, "full buffer extent", 1)
            ):
                raise ValueError(
                    "buffer layout/extent disagrees with padded NHD capacity"
                )
            if any(
                getattr(desc, field) != getattr(spec.desc, field)
                for field in ("name", "shape", "row_bytes", "tokens_per_row")
            ):
                raise ValueError("pool descriptor disagrees with finalized owner")
            offset = checked_int(spec.offset, "owner buffer offset")
            if ptr != base + offset or offset % granularity:
                raise ValueError("buffer pointer is not the owned allocation base")
            end = checked_int(ptr + extent, "buffer end")
            extents.append((ptr, end))
            layouts.append(
                BufferLayout(
                    desc.name,
                    item,
                    f"NHD:shape={shape[1:]}:layer_start={pool.start_layer}",
                )
            )
            cursor, pieces = ptr, []
            for address, length in mappings:
                start, stop = max(ptr, address), min(end, address + length)
                if start >= stop:
                    continue
                if start != cursor:
                    raise ValueError("unbacked hole in pool buffer")
                pieces.append(
                    (start - ptr, stop - start, address, length, start - address)
                )
                cursor = stop
            if cursor != end:
                raise ValueError("incomplete pool backing")
            buffers.append((desc.name, pieces))
        extents.sort()
        if any(a[1] > b[0] for a, b in pairwise(extents)):
            raise ValueError("overlapping pool buffers")
        layout = PoolLayout(str(pool.store_dtype), page_size, tuple(layouts))
        layout.validate()
        return capacity, layout, buffers

    def close(self):
        # Refuse live exports/batches via engine; never close/free the original pool.
        while self.memories:
            self.engine.unregister_memory(self.memories[-1])
            self.memories.pop()
