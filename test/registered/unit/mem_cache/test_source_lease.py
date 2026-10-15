# SPDX-License-Identifier: Apache-2.0
"""Real Python tree + paged allocator + VMM pool lifecycle tests (CUDA)."""
import unittest
from dataclasses import replace
import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def make_cache(pool):
    allocator = PagedTokenToKVPoolAllocator(pool.size, pool.page_size, pool.dtype,
                                            pool.device, pool, False)
    cache = UnifiedRadixCache(CacheInitParams(disable=False, req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator, page_size=pool.page_size,
        tree_components=(ComponentType.FULL,), tree_core_backend='python'))
    return cache, allocator


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA VMM')
class SourceLeaseTests(unittest.TestCase):
    def setUp(self):
        self.pool = MHATokenToKVPool(64, 4, torch.float16, 2, 8, 1, 'cuda:0', False,
                                    kv_cache_layout='nhd', post_capture_active=True)
        self.pool._finalize_backing_tokens(64)
        self.cache, self.allocator = make_cache(self.pool)
        self.indices = self.allocator.alloc(16)
        self.key = RadixKey(list(range(16)))
        self.cache.insert(InsertParams(key=self.key, value=self.indices))

    def acquire(self, key=None, ready=None):
        return self.cache.acquire_source_lease(self.key if key is None else key,
            fingerprint='test-model:fp16:nhd:tp1', ready=ready or torch.cuda.synchronize)

    def tearDown(self):
        for item in list(self.cache.source_leases._active.values()):
            self.cache.release_source_lease(item[0], drained=True)
        self.cache.reset()
        self.pool._clear_buffers()

    def test_pin_evict_release(self):
        lease = self.acquire()
        self.assertEqual(lease.token_indices, tuple(self.indices.tolist()))
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 0)
        self.assertTrue(self.cache.release_source_lease(lease, drained=True))
        self.assertFalse(self.cache.release_source_lease(lease, drained=True))
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 16)

    def test_guards_and_timeout(self):
        lease = self.acquire()
        for operation in [self.cache.reset, self.allocator.clear, self.pool._clear_buffers,
                          lambda:self.pool._finalize_backing_tokens(32),
                          lambda:self.allocator.free(self.indices),
                          lambda:self.allocator.free_page_ids(self.indices[::4]//4),
                          lambda:self.cache.release_source_lease(lease, drained=False)]:
            with self.assertRaises(RuntimeError): operation()
        self.allocator.free_group_begin()
        with self.assertRaises(RuntimeError): self.allocator.free(self.indices)
        self.allocator.free_group_end()
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 0)

    def test_split_multiple_leases_and_stale(self):
        first = self.acquire()
        second = self.acquire(self.key[:8]) # split while ancestor path is pinned
        self.cache.release_source_lease(first, drained=True)
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.cache.match_prefix(MatchPrefixParams(key=self.key[:8])).device_prefix_len, 8)
        self.cache.release_source_lease(second, drained=True)
        self.cache.reset()
        with self.assertRaises(ValueError): self.cache.release_source_lease(first, drained=True)

    def test_flush_and_foreign_thread(self):
        from types import SimpleNamespace
        from concurrent.futures import ThreadPoolExecutor
        from sglang.srt.managers.scheduler import Scheduler
        lease = self.acquire()
        # Early refusal must happen before touching any scheduler/pool state.
        self.assertFalse(Scheduler.flush_cache(SimpleNamespace(tree_cache=self.cache)))
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaises(RuntimeError):
                executor.submit(self.cache.release_source_lease, lease, drained=True).result()
        self.assertTrue(self.cache.source_leases.busy)

    def test_ready_failure_and_miss(self):
        def failure(): raise RuntimeError('producer failed')
        with self.assertRaises(RuntimeError): self.acquire(ready=failure)
        self.assertFalse(self.cache.source_leases.busy)
        with self.assertRaises(ValueError): self.acquire(RadixKey([999]*4))
        self.assertFalse(self.cache.source_leases.busy)
        lease=self.acquire()
        with self.assertRaises(ValueError): self.cache.release_source_lease(replace(lease,epoch=999),drained=True)
        with self.assertRaises(ValueError): self.cache.release_source_lease(replace(lease,fingerprint='other'),drained=True)


if __name__ == '__main__': unittest.main()
