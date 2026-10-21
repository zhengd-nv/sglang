# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from sglang.srt.mem_cache.l1_transfer.peer.peer import plan_remote_regions
from sglang.srt.mem_cache.l1_transfer.transport.api import MemoryHandle
from sglang.srt.mem_cache.l1_transfer.transport.layout import BufferRange, RegionSpan


class RegionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.desc = [SimpleNamespace(region_id=f'{i:032x}', size=64) for i in range(3)]
        self.bindings = {'k': [[0,64,0,0],[64,64,1,0],[128,64,2,0]]}
        self.target = MemoryHandle('target',192,0,False,1000)
        self.target_bindings = {'k': (RegionSpan(0,192,self.target),)}

    def test_small_read_does_not_import_unrelated_regions(self):
        needed, reqs = plan_remote_regions(self.desc, self.bindings,
            [BufferRange('k',80,0,16)], self.target_bindings)
        self.assertEqual(needed,[self.desc[1]])
        self.assertEqual(len(reqs),1)
        self.assertEqual((reqs[0].remote_offset,reqs[0].local_offset,reqs[0].length),(16,0,16))
        self.assertIs(reqs[0].local,self.target)

    def test_crossing_boundary_keeps_both_regions_and_offsets(self):
        needed, reqs = plan_remote_regions(self.desc,self.bindings,
            [BufferRange('k',48,32,48)],self.target_bindings)
        self.assertEqual(needed,self.desc[:2])
        self.assertEqual([(r.remote_offset,r.local_offset,r.length) for r in reqs],[(48,32,16),(0,48,32)])

    def test_destination_alias_is_still_rejected_before_import(self):
        with self.assertRaisesRegex(ValueError,'overlapping'):
            plan_remote_regions(self.desc,self.bindings,
                [BufferRange('k',0,0,32),BufferRange('k',128,16,32)],self.target_bindings)

    def test_invalid_unused_binding_is_not_ignored(self):
        for index in (-1,3,True):
            bad={'k': [[0,64,0,0],[64,64,1,0],[128,64,index,0]]}
            with self.assertRaises(ValueError):
                plan_remote_regions(self.desc,bad,[BufferRange('k',0,0,16)],self.target_bindings)

    def test_duplicate_region_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            plan_remote_regions([self.desc[0]]*3,self.bindings,
                [BufferRange('k',0,0,16)],self.target_bindings)

if __name__=='__main__': unittest.main()
