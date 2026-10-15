import unittest
import weakref
import gc
from types import SimpleNamespace as NS
from sglang.srt.mem_cache.l1_transfer.transport import NvlinkEngine, BusyError
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from test_contract import FakeDriver


class BorrowDriver(FakeDriver):
    def register_vmm(self, device, address, physical):
        return self.create(device,physical)


class AdapterTests(unittest.TestCase):
    def pool(self):
        pool=type('MHATokenToKVPool',(),{})()
        pool.use_hnd=False; pool.kv_cache_layout='nhd'
        pool.size=16; pool.page_size=4; pool.store_dtype='float16'; pool.start_layer=0
        pool._kv_buffer_descs=[NS(name='k0',shape=(20,2,4))]
        pool.get_contiguous_buf_infos=lambda:([1000],[320],[64])
        pool._post_capture_owner=NS(_final_num_tokens=16,_arena=NS(_allocation=NS(
            _mappings=[(1000,128,1),(1128,192,2)])))
        return pool

    def test_split_and_owner_lifetime(self):
        engine=NvlinkEngine(0,_driver=BorrowDriver()); pool=self.pool()
        adapter=SGLangPoolAdapter(pool,engine)
        self.assertEqual([s.length for s in adapter.bindings['k0']],[128,192])
        h=adapter.memories[0]; desc=engine.export_memory(h)
        with self.assertRaises(BusyError): engine.unregister_memory(h)
        engine.release_export(desc.lease_id)
        ref=weakref.ref(pool)
        del pool; gc.collect(); self.assertIsNotNone(ref())
        adapter.close(); self.assertFalse(engine._memories)
        del adapter; gc.collect(); self.assertIsNone(ref())
        engine.close()

    def test_last_real_page_is_inside_capacity(self):
        from sglang.srt.mem_cache.l1_transfer.transport.layout import plan_transfer
        engine=NvlinkEngine(0,_driver=BorrowDriver())
        adapter=SGLangPoolAdapter(self.pool(),engine)
        self.assertEqual(adapter.capacity,5)
        plan=plan_transfer(adapter.layout,adapter.layout,[4],[4],
                           source_capacity=adapter.capacity,target_capacity=adapter.capacity)
        self.assertEqual(plan[0].target_offset+plan[0].length,320)
        adapter.close();engine.close()

    def test_partial_registration_cleanup(self):
        pool=self.pool(); pool._post_capture_owner._arena._allocation._mappings.pop()
        engine=NvlinkEngine(0,_driver=BorrowDriver())
        with self.assertRaises(ValueError): SGLangPoolAdapter(pool,engine)
        self.assertFalse(engine._memories)

    def test_reject_unsupported(self):
        engine=NvlinkEngine(0,_driver=BorrowDriver())
        for attr,value in [('use_hnd',True),('_post_capture_owner',None),('kv_cache_layout','hnd')]:
            pool=self.pool(); setattr(pool,attr,value)
            with self.assertRaises(ValueError): SGLangPoolAdapter(pool,engine)
        with self.assertRaises(ValueError): engine.register_vmm(100,128,owner=None)
