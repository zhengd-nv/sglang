# SPDX-License-Identifier: Apache-2.0
"""Version-specific adapter for finalized SGLang MHA NHD post-capture pools.

No framework import here. Caller freezes pool/page ownership for the entire session.
Private VMM metadata is intentional: fail closed on unsupported pool types.
"""
from .layout import BufferLayout, PoolLayout, RegionSpan


class SGLangPoolAdapter:
    def __init__(self, pool, engine):
        if type(pool).__name__ != 'MHATokenToKVPool' or pool.use_hnd:
            raise ValueError('only MHATokenToKVPool NHD supported')
        owner = pool._post_capture_owner
        if owner is None or owner._final_num_tokens is None:
            raise ValueError('requires finalized post-capture VMM backing')
        if pool.kv_cache_layout != 'nhd':
            raise ValueError('only NHD supported')
        self.pool, self.engine = pool, engine
        self.memories = []; self.bindings = {}
        ptrs, sizes, items = pool.get_contiguous_buf_infos()
        descs = pool._kv_buffer_descs
        self.capacity = pool.size // pool.page_size + 1 # page IDs are 1..N; page 0 is padding
        self.layout = PoolLayout(str(pool.store_dtype), pool.page_size, tuple(
            BufferLayout(d.name, item, f'NHD:shape={tuple(d.shape[1:])}:layer_start={pool.start_layer}')
            for d,item in zip(descs,items)))
        self.layout.validate()
        mappings = sorted(owner._arena._allocation._mappings, key=lambda m:m[0])
        registered = {}
        try:
            for d, ptr, size in zip(descs,ptrs,sizes):
                spans=[]; cursor=ptr
                for address,length,_ in mappings:
                    start, end = max(ptr,address), min(ptr+size,address+length)
                    if start >= end: continue
                    if start != cursor: raise ValueError('unbacked hole in pool buffer')
                    if address not in registered:
                        h=engine.register_vmm(address,length,owner=pool)
                        registered[address]=h; self.memories.append(h)
                    spans.append(RegionSpan(start-ptr,end-start,registered[address],start-address))
                    cursor=end
                if cursor != ptr+size: raise ValueError('incomplete pool backing')
                self.bindings[d.name]=tuple(spans)
        except BaseException:
            self.close()
            raise

    def close(self):
        # Refuse live exports/batches via engine; do not close/free the pool.
        while self.memories:
            self.engine.unregister_memory(self.memories[-1])
            self.memories.pop()
