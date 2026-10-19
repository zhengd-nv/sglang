import random
import unittest
from dataclasses import replace
from sglang.srt.mem_cache.l1_transfer.transport.api import MemoryHandle, Opcode
from sglang.srt.mem_cache.l1_transfer.transport.layout import (BufferLayout, PoolLayout, RegionSpan,
                                    plan_transfer, bind_regions)

LAYOUT = PoolLayout('float16', 16, (BufferLayout('0.k', 37, 'NHD:heads=2:dim=4:rank=0'),
                                    BufferLayout('0.v', 19, 'NHD:heads=2:dim=2:rank=0')))


class LayoutTests(unittest.TestCase):
    def plan(self, src, dst, layout=LAYOUT):
        return plan_transfer(LAYOUT, layout, src, dst, source_capacity=16, target_capacity=16)

    def test_coalescing(self):
        p = self.plan([2, 3, 7], [8, 9, 12])
        self.assertEqual([(r.source_offset, r.target_offset, r.length) for r in p],
                         [(74,296,74),(259,444,37),(38,152,38),(133,228,19)])
        self.assertEqual(len(self.plan([1,2,3], [5,7,9])), 6)
        self.assertEqual(len(self.plan([1,3,5], [5,6,7])), 6)
        self.assertEqual(self.plan([], []), ())

    def test_invalid(self):
        for s,t in [([], [0]), ([0], []), ([16],[0]), ([-1],[0]), ([True],[0]), ([0,1],[2,2])]:
            with self.subTest(s=s,t=t), self.assertRaises(ValueError): self.plan(s,t)
        for layout in [replace(LAYOUT,dtype='int8'), replace(LAYOUT,page_size=8),
                       replace(LAYOUT,buffers=LAYOUT.buffers[::-1])]:
            with self.assertRaises(ValueError): self.plan([0],[0],layout)

    def test_random_byte_oracle_and_boundaries(self):
        rng = random.Random(13)
        for opcode in Opcode:
            for iteration in range(40):
                source, target, arrays, expected = {}, {}, {}, {}
                for b in LAYOUT.buffers:
                    size = 16*b.page_bytes
                    for side, bindings in [('s',source),('t',target)]:
                        cut = rng.randrange(1,size)
                        spans=[]
                        for i,(start,end) in enumerate(((0,cut),(cut,size))):
                            key=f'{side}-{b.name}-{i}'
                            remote = (side=='s') == (opcode==Opcode.READ)
                            h=MemoryHandle(key,end-start+14,0,remote,0)
                            spans.append(RegionSpan(start,end-start,h,7))
                            arrays[key]=bytearray(rng.randbytes(h.size) if side=='s' else b'\xa5'*h.size)
                            expected[key]=bytearray(arrays[key])
                        bindings[b.name]=spans
                sp=rng.choices(range(16),k=8); tp=rng.sample(range(16),8)
                def locate(bindings, name, offset):
                    for span in bindings[name]:
                        if span.buffer_offset <= offset < span.buffer_offset+span.length:
                            return span.memory.region_id,span.region_offset+offset-span.buffer_offset
                # Independent per-byte logical-page oracle; never uses planner ranges.
                for b in LAYOUT.buffers:
                    for s,t in zip(sp,tp):
                        for byte in range(b.page_bytes):
                            sk,so=locate(source,b.name,s*b.page_bytes+byte)
                            tk,to=locate(target,b.name,t*b.page_bytes+byte)
                            expected[tk][to]=arrays[sk][so]
                requests=bind_regions(self.plan(sp,tp),source,target,opcode)
                for r in requests:
                    a,ao,b,bo = ((r.remote,r.remote_offset,r.local,r.local_offset) if opcode==Opcode.READ
                                 else (r.local,r.local_offset,r.remote,r.remote_offset))
                    arrays[b.region_id][bo:bo+r.length]=arrays[a.region_id][ao:ao+r.length]
                self.assertEqual(arrays,expected)

    def test_invalid_bindings(self):
        p=self.plan([0],[0]); local=MemoryHandle('l',1024,0,False,0); remote=replace(local,region_id='r',remote=True)
        s={b.name:[RegionSpan(0,16*b.page_bytes,remote)] for b in LAYOUT.buffers}
        t={b.name:[RegionSpan(0,16*b.page_bytes,local)] for b in LAYOUT.buffers}
        with self.assertRaises(ValueError): bind_regions(p,s,t,Opcode.READ) # destination alias
        for bad in [[],[RegionSpan(1,100,local)],[RegionSpan(0,1025,local)]]:
            with self.assertRaises(ValueError): bind_regions(p,s,{'0.k':bad},Opcode.READ)
        with self.assertRaises(ValueError): bind_regions(p,s,t,Opcode.WRITE)


if __name__ == '__main__': unittest.main()
