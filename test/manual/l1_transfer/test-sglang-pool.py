#!/usr/bin/env python3
"""Two-process real SGLang MHA pool test; fixed pages, no scheduler/model."""
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import socket
import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.l1_transfer.transport import NvlinkEngine, Opcode, RegionDescriptor, TransferState
from sglang.srt.mem_cache.l1_transfer.transport.layout import PoolLayout, BufferLayout, RegionSpan, plan_transfer, bind_regions
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.srt.mem_cache.l1_transfer.peer import RestoreTransaction, RestoreState
from test_source_lease import make_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams, MatchPrefixParams

spec=importlib.util.spec_from_file_location('gpu_helpers',Path(__file__).with_name('test-gpu.py'))
helpers=importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def run(root, rank, device):
    torch.cuda.set_device(device)
    channel=helpers.FileChannel(root/'control',rank)
    engine=NvlinkEngine(device)
    # >2 MiB each: prefix backing plus finalize creates multiple physical extents.
    pool=MHATokenToKVPool(size=4096,page_size=16,dtype=torch.float16,
                         head_num=8,head_dim=64,v_head_dim=32,layer_num=2,
                         device=f'cuda:{device}',enable_memory_saver=False,
                         enable_alt_stream=False,kv_cache_layout='nhd',post_capture_active=True)
    pool._finalize_backing_tokens(4096)
    adapter=SGLangPoolAdapter(pool,engine)
    cache, allocator = make_cache(pool)
    tensors=pool.k_buffer+pool.v_buffer
    rows=[]
    for opcode, outcome in ((Opcode.READ, 'success'), (Opcode.READ, 'failed_drained'),
                            (Opcode.READ, 'cancel'), (Opcode.WRITE, 'success')):
        # Unique token/head/dim pattern, exactly representable in fp16.
        originals=[]
        for i,t in enumerate(tensors):
            data=((torch.arange(t.numel(),device=t.device).reshape(t.shape)*17+i*31+rank*97)%2003).to(t.dtype)
            t.copy_(data); originals.append(data.cpu())
        torch.cuda.synchronize()
        src=[1,2,126,127,128,129,240]; dst=[3,4,125,126,127,128,250]
        allocator.clear()
        assert allocator.alloc(4096) is not None
        source_rank = 0 if opcode == Opcode.READ else 1
        lease = None
        restore_state = None
        target_published = False
        key = RadixKey(list(range(len(src)*16)))
        if rank == source_rank:
            indices = torch.tensor([p*16+i for p in src for i in range(16)], device=pool.device)
            key = RadixKey(list(range(len(indices))))
            cache.insert(InsertParams(key=key, value=indices))
            lease = cache.acquire_source_lease(key, fingerprint='test:fp16:nhd:tp1', ready=torch.cuda.synchronize)
            assert list(lease.page_indices) == src
            assert cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted == 0
            helpers.must_raise(RuntimeError, allocator.clear)
        if rank==0:
            exports=[engine.export_memory(h,'read' if opcode==Opcode.READ else 'write') for h in adapter.memories]
            indices={h.region_id:i for i,h in enumerate(adapter.memories)}
            channel.send({'layout':dataclasses.asdict(adapter.layout),
                          'descriptors':[d.to_dict() for d in exports],
                          'bindings':{name:[[s.buffer_offset,s.length,indices[s.memory.region_id],s.region_offset] for s in spans]
                                      for name,spans in adapter.bindings.items()}})
            assert channel.recv()['unmapped']
            for d in exports: engine.release_export(d.lease_id)
            if opcode == Opcode.READ:
                cache.release_source_lease(lease, drained=True)
                assert cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted == len(src)*16
                lease = None
                channel.send({'source_released': True})
        else:
            msg=channel.recv(); raw=msg['layout']
            remote_layout=PoolLayout(raw['dtype'],raw['page_size'],tuple(BufferLayout(**b) for b in raw['buffers']))
            imported=[engine.import_memory(RegionDescriptor.from_dict(d)) for d in msg['descriptors']]
            bindings={name:tuple(RegionSpan(a,n,imported[i],off) for a,n,i,off in spans) for name,spans in msg['bindings'].items()}
            source,target=(bindings,adapter.bindings) if opcode==Opcode.READ else (adapter.bindings,bindings)
            plan=plan_transfer(remote_layout,adapter.layout,src,dst,source_capacity=256,target_capacity=256)
            requests=bind_regions(plan,source,target,opcode)
            if opcode == Opcode.READ:
                target_indices = torch.tensor([p*16+i for p in dst for i in range(16)], device=pool.device)
                assert cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len == 0
                def publish():
                    nonlocal target_published
                    # Validate bytes BEFORE publication, independently of planner.
                    torch.cuda.synchronize()
                    for i, t in enumerate(tensors):
                        expected = originals[i].clone()
                        peer = ((torch.arange(t.numel()).reshape(t.shape)*17+i*31)%2003).to(t.dtype)
                        for a,b in zip(src,dst): expected[b*16:(b+1)*16] = peer[a*16:(a+1)*16]
                        assert torch.equal(t.cpu(), expected)
                    cache.insert(InsertParams(key=key, value=target_indices))
                    assert cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len == len(src)*16
                    target_published = True
                def discard():
                    assert cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len == 0
                    allocator.free(target_indices)
                    assert allocator.available_size() == len(src)*16
                # Inject FAILED_DRAINED only AFTER actual DMA success/drain.
                # This validates discard ownership, not a real CUDA device fault.
                class DrainFailure:
                    def __getattr__(self, name): return getattr(engine, name)
                    def wait(self, batch, timeout):
                        status = engine.wait(batch, timeout)
                        assert status.state == TransferState.SUCCESS
                        return dataclasses.replace(status, state=TransferState.FAILED_DRAINED,
                                                   completed_bytes=None, error='injected after drain')
                def release_source():
                    channel.send({'unmapped':True})
                    assert channel.recv()['source_released']
                tx = RestoreTransaction(DrainFailure() if outcome == 'failed_drained' else engine,
                    requests, imported, publish=publish, discard=discard,
                    release_source=release_source,
                    keepalive=(pool, adapter, cache, target_indices))
                if outcome == 'cancel':
                    tx.cancel(); tx.cancel()
                else:
                    tx.start(); tx.wait(30); tx.wait(0)
                restore_state = tx.state.value
                assert tx.state == (RestoreState.PUBLISHED if outcome == 'success' else RestoreState.DISCARDED)
                assert tx.source_released
            else:
                batch=engine.submit_transfer(requests)
                result=engine.wait(batch,30); assert result.state==TransferState.SUCCESS,result
                engine.release_batch(batch)
                for h in imported: engine.unregister_memory(h)
                channel.send({'unmapped':True})
        torch.cuda.synchronize()
        target_rank=1 if opcode==Opcode.READ else 0
        for i,t in enumerate(tensors):
            expected=originals[i].clone()
            if rank==target_rank and outcome != 'cancel':
                peer=((torch.arange(t.numel()).reshape(t.shape)*17+i*31+(1-rank)*97)%2003).to(t.dtype)
                for s,d in zip(src,dst): expected[d*16:(d+1)*16]=peer[s*16:(s+1)*16]
            assert torch.equal(t.cpu(),expected),(rank,opcode,i)
        channel.send({'verified':True}); assert channel.recv()['verified']
        if lease is not None:
            cache.release_source_lease(lease, drained=True)
            assert cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted == len(src)*16
        if target_published:
            assert cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted == len(src)*16
            assert allocator.available_size() == len(src)*16
        cache.reset()
        rows.append({'outcome':outcome, 'restore_state':restore_state, 'target_published':target_published,
                     'source_lease_checked': rank == source_rank, 'opcode':opcode.value,'pass':True,'regions':len(adapter.memories),
                     'buffer_spans':[len(s) for s in adapter.bindings.values()]})
    adapter.close(); engine.close()
    # Unregister must not free pool backing: original tensors remain writable.
    for t in tensors: t.fill_(7)
    torch.cuda.synchronize()
    assert all(bool((t==7).all()) for t in tensors)
    helpers.atomic(root/f'result-{rank}.json',{'pass':True,'rank':rank,'host':socket.gethostname(),
                                             'cases':rows,'pool_survives_unregister':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True); p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    try: run(a.output,a.rank,a.device)
    except BaseException:
        import traceback
        helpers.atomic(a.output/'control'/f'error-{a.rank}.json',traceback.format_exc()); raise
