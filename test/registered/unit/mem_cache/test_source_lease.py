# SPDX-License-Identifier: Apache-2.0
"""Real Python tree + paged allocator + VMM pool lifecycle tests (CUDA)."""
import unittest
from array import array
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
        self.key = RadixKey(array('q', range(16)))
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
        self.assertEqual(tuple(lease.token_indices.tolist()), tuple(self.indices.tolist()))
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

    def test_page_key_lease_is_depth_checked_split_safe_and_evictable(self):
        """A target's own page keys must name the source prefix: only [L, H) is leased,
        a key at another depth is refused, a later split keeps the prefix findable,
        and after eviction the key misses instead of returning a stale node."""
        from sglang.srt.mem_cache.source_index import prefix_page_keys
        self.cache.source_leases.enable_key_index()
        keys = prefix_page_keys(list(range(16)), 4)
        def by_key(key, end, start):
            return self.cache.acquire_source_lease_by_key(key, end=end, start=start,
                fingerprint='test-model:fp16:nhd:tp1', ready=torch.cuda.synchronize)
        lease = by_key(keys[-1], 16, 8)
        self.assertEqual(tuple(lease.token_indices.tolist()), tuple(self.indices[8:16].tolist()))
        with self.assertRaises(ValueError): by_key(keys[1], 16, 8)  # key of page 2 ends at 8
        branch = self.allocator.alloc(4)
        self.cache.insert(InsertParams(key=RadixKey(array('q', [*range(8), 90, 91, 92, 93])),
                                       value=torch.cat((self.indices[:8], branch))))
        inner = by_key(keys[1], 8, 0)  # page 2 now ends a split-off node
        self.assertEqual(tuple(inner.token_indices.tolist()), tuple(self.indices[:8].tolist()))
        for item in (lease, inner):
            self.cache.release_source_lease(item, drained=True)
        self.cache.evict(EvictParams(num_tokens=64))
        with self.assertRaises(ValueError): by_key(keys[-1], 16, 8)
        self.assertFalse(self.cache.source_leases.busy)

    def test_page_key_lease_keeps_the_prefix_recent(self):
        """A prefix leased by page key counts as used, as a token match would make it;
        otherwise a hot shared prefix ages out under unrelated traffic and every
        restore of it falls back."""
        from sglang.srt.mem_cache.source_index import prefix_page_keys
        index = self.cache.source_leases.enable_key_index()
        other = RadixKey(array('q', range(100, 116)))
        self.cache.insert(InsertParams(key=other, value=self.allocator.alloc(16)))
        key = prefix_page_keys(list(range(16)), 4)[-1]
        lease = self.cache.acquire_source_lease_by_key(key, end=16, start=0,
            fingerprint='test-model:fp16:nhd:tp1', ready=torch.cuda.synchronize)
        self.cache.release_source_lease(lease, drained=True)
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=16)).num_tokens_evicted, 16)
        index.find(key, 16)  # the leased prefix survived; the newer, unused one went
        with self.assertRaises(ValueError):
            index.find(prefix_page_keys(list(range(100, 116)), 4)[-1], 16)

    def test_tree_lock_serializes_worker_leases_with_scheduler_calls(self):
        """A network worker leases between scheduler cache calls: it waits while the
        scheduler holds the tree access lock, and bypassing the lock is refused."""
        import time
        from concurrent.futures import ThreadPoolExecutor
        from sglang.srt.mem_cache.source_index import prefix_page_keys
        from sglang.srt.mem_cache.tree_access_lock import install_tree_access_lock
        self.cache.source_leases.enable_key_index(ready_stream=torch.cuda.current_stream())
        lock = install_tree_access_lock(self.cache)
        key = prefix_page_keys(list(range(16)), 4)[-1]
        args = dict(end=16, start=0, fingerprint='test-model:fp16:nhd:tp1', ready=lambda: None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            bypass = executor.submit(self.cache.source_leases.acquire_by_key, key, **args)
            with self.assertRaises(RuntimeError): bypass.result(timeout=5)
            with lock:  # the scheduler is inside a cache call
                future = executor.submit(self.cache.acquire_source_lease_by_key, key, **args)
                time.sleep(0.05)
                self.assertFalse(future.done())
            lease = future.result(timeout=5)
        self.assertEqual(tuple(lease.token_indices.tolist()), tuple(self.indices.tolist()))
        self.cache.release_source_lease(lease, drained=True)

    def step_lease(self):
        from sglang.srt.mem_cache.source_index import prefix_page_keys
        self.cache.source_leases.enable_key_index()
        self.cache.source_leases.enable_step_protection()
        return self.cache.acquire_source_lease_by_key(prefix_page_keys(list(range(16)), 4)[-1],
            end=16, start=0, fingerprint='test-model:fp16:nhd:tp1', ready=lambda: None, protection='step')

    def test_step_lease_is_invalid_once_a_page_was_freed_before_its_lock(self):
        """An optimistic (c') lease holds nothing until the next step: eviction may free
        and reuse its pages, and then neither the step lock nor the release may call
        the read intact, even after the same page IDs come back."""
        lease = self.step_lease()
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 16)
        reused = self.allocator.alloc(16)
        self.assertEqual(sorted(reused.tolist()), sorted(self.indices.tolist()))
        self.cache.source_leases.apply_step_locks()
        self.assertTrue(self.cache.release_source_lease(lease, drained=True))
        self.assertFalse(self.cache.source_leases.lease_valid(lease))
        self.assertEqual(self.cache.source_leases.step_log[lease.lease_id]['outcome'], 'stale_at_lock')

    def test_step_lock_protects_the_rest_of_the_read(self):
        """Once the step lock applies, eviction cannot touch the pages and the read stays valid."""
        lease = self.step_lease()
        self.cache.source_leases.apply_step_locks()
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 0)
        self.assertTrue(self.cache.release_source_lease(lease, drained=True))
        self.assertTrue(self.cache.source_leases.lease_valid(lease))
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 16)

    def test_tp_step_commands_change_locks_only_when_applied(self):
        """With TP>1 every rank must lock and unlock at the same step: a lease only queues a
        lock command, release only queues an unlock, and nothing changes the tree until a
        command is applied; a lock command for pages freed meanwhile locks nothing."""
        self.cache.source_leases.broadcast_steps = True
        lease = self.step_lease()
        manager = self.cache.source_leases
        [lock] = manager.drain_step_commands()
        self.assertEqual((lock.kind, lock.pages), ('lock', lease.page_indices))
        manager.apply_step_command(lock)
        self.assertTrue(self.cache.release_source_lease(lease, drained=True))
        self.assertTrue(manager.lease_valid(lease))
        [unlock] = manager.drain_step_commands()
        self.assertEqual(unlock.kind, 'unlock')
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 0)  # still locked
        manager.apply_step_command(unlock)
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted, 16)
        manager.apply_step_command(lock)  # replayed after the pages were freed
        self.assertFalse(manager._held)

    def test_tp_lease_refreshes_recency_only_at_the_step(self):
        """Recency decides eviction, so on TP>1 a lease taken off the step on rank 0 must not
        touch it; the lock command refreshes it on every rank at the same step instead."""
        from sglang.srt.mem_cache.source_index import prefix_page_keys
        manager = self.cache.source_leases
        index = manager.enable_key_index()
        manager.enable_step_protection()
        manager.broadcast_steps = True
        key = prefix_page_keys(list(range(16)), 4)[-1]
        node = index.find(key, 16)
        before = node.last_access_time
        lease = self.cache.acquire_source_lease_by_key(key, end=16, start=0,
            fingerprint='test-model:fp16:nhd:tp1', ready=lambda: None, protection='step')
        self.assertEqual(node.last_access_time, before)
        [lock] = manager.drain_step_commands()
        manager.apply_step_command(lock)
        self.assertGreater(node.last_access_time, before)
        self.cache.release_source_lease(lease, drained=True)
        for command in manager.drain_step_commands():
            manager.apply_step_command(command)

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
