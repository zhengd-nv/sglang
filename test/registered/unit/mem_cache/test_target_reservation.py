# SPDX-License-Identifier: Apache-2.0
"""Real FULL tree target ownership, including competing canonical insertions."""
import unittest
from types import SimpleNamespace
import torch
from test_source_lease import make_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams
from sglang.srt.mem_cache.target_reservation import TargetReservation


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA VMM')
class TargetTests(unittest.TestCase):
    def setUp(self):
        self.pool = MHATokenToKVPool(64,4,torch.float16,2,8,1,'cuda:0',False,
                                    kv_cache_layout='nhd',post_capture_active=True)
        self.pool._finalize_backing_tokens(64)
        self.cache,self.allocator = make_cache(self.pool)
        self.key = RadixKey(list(range(16)))

    def tearDown(self):
        for r in list(self.cache.source_leases._targets):
            if r.state == 'PRIVATE': r.discard()
            else: r.release()
        self.cache.reset(); self.pool._clear_buffers()

    def seed(self, n):
        indices = self.allocator.alloc(n)
        self.cache.insert(InsertParams(key=self.key[:n],value=indices))
        return indices

    def test_partial_hit_and_publication(self):
        prefix = self.seed(8)
        r = TargetReservation(self.cache,self.key)
        self.assertEqual(r.prefix_len,8); self.assertEqual(len(r.private),8)
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted,0)
        expected = torch.cat((prefix,r.private))
        self.assertTrue(torch.equal(r.publish(),expected))
        self.assertTrue(torch.equal(r.publish(),expected))
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted,0)
        r.release(); r.release()
        self.assertEqual(self.cache.evict(EvictParams(num_tokens=64)).num_tokens_evicted,16)
        self.assertEqual(self.allocator.available_size(),64)

    def test_competing_full_prefix_frees_duplicate_once(self):
        prefix = self.seed(8)
        r = TargetReservation(self.cache,self.key)
        competitor = self.allocator.alloc(8)
        canonical = torch.cat((prefix,competitor))
        self.cache.insert(InsertParams(key=self.key,value=canonical,prev_prefix_len=8))
        self.assertTrue(torch.equal(r.publish(),canonical))
        self.assertEqual(self.allocator.available_size(),48)
        r.release()
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.allocator.available_size(),64)
        self.allocator._debug_check_no_duplicate_pages()

    def test_competing_partial_prefix_adopts_only_tail(self):
        r = TargetReservation(self.cache,self.key)
        competing = self.seed(8)
        expected = torch.cat((competing,r.private[8:]))
        self.assertTrue(torch.equal(r.publish(),expected))
        self.assertEqual(self.allocator.available_size(),48)
        r.release(); self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.allocator.available_size(),64)

    def test_full_hit_does_not_allocate_or_free_canonical(self):
        canonical = self.seed(16)
        r = TargetReservation(self.cache,self.key)
        self.assertEqual(r.prefix_len,16); self.assertEqual(len(r.private),0)
        self.assertTrue(torch.equal(r.publish(),canonical))
        r.release(); self.assertEqual(self.allocator.available_size(),48)

    def test_guards_and_discard(self):
        from sglang.srt.managers.scheduler import Scheduler
        r = TargetReservation(self.cache,self.key)
        for op in (self.cache.reset,self.allocator.clear,self.pool._clear_buffers,
                   lambda:self.pool._finalize_backing_tokens(32),lambda:self.allocator.free(r.private)):
            with self.assertRaises(RuntimeError): op()
        self.assertFalse(Scheduler.flush_cache(SimpleNamespace(tree_cache=self.cache)))
        r.discard(); r.discard()
        self.assertFalse(self.cache.source_leases.busy)
        self.assertEqual(self.allocator.available_size(),64)

    def test_last_page_and_allocation_failure(self):
        held = self.allocator.alloc(60)
        r = TargetReservation(self.cache,self.key[:4])
        self.assertEqual(r.pages,(16,))
        with self.assertRaises(MemoryError): TargetReservation(self.cache,self.key)
        self.assertEqual(len(self.cache.source_leases._targets),1)
        r.discard(); self.allocator.free(held)
        self.assertEqual(self.allocator.available_size(),64)

    def test_unaligned_key_rejected(self):
        with self.assertRaises(ValueError): TargetReservation(self.cache,self.key[:7])
        self.assertFalse(self.cache.source_leases.busy)
        self.assertEqual(self.allocator.available_size(),64)


if __name__ == '__main__': unittest.main()
