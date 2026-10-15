#!/usr/bin/env python3
"""Two-node socket control + NVLink READ + real FULL tree target ownership."""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import socket

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, EvictParams
from test_source_lease import make_cache
from sglang.srt.mem_cache.l1_transfer.transport import NvlinkEngine
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.srt.mem_cache.l1_transfer.peer.control import ModelIdentity, SocketRPC, RemoteRejected, digest, serve_once, send, receive
from sglang.srt.mem_cache.l1_transfer.peer.peer import SourceEndpoint, PeerReceiver

spec=importlib.util.spec_from_file_location('gpu_helpers',Path(__file__).with_name('test-gpu.py'))
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def run(root,rank):
    torch.cuda.set_device(0)
    channel=helpers.FileChannel(root/'bootstrap',rank)
    engine=NvlinkEngine(0)
    pool=MHATokenToKVPool(4096,16,torch.float16,8,64,2,'cuda:0',False,
                         v_head_dim=32,kv_cache_layout='nhd',post_capture_active=True)
    pool._finalize_backing_tokens(4096)
    adapter=SGLangPoolAdapter(pool,engine)
    cache,allocator=make_cache(pool)
    tensors=pool.k_buffer+pool.v_buffer
    tokens=list(range(115));key=RadixKey(tokens[:112])
    identity=ModelIdentity('synthetic-weight-revision',digest({'mha':8}),digest({'rope':'test'}),
                           'none',digest({'context':'test-peer'}))
    expected=[((torch.arange(112*t[0].numel(),device=t.device).reshape(112,*t.shape[1:])*17+i*31)%2003).to(t.dtype)
              for i,t in enumerate(tensors)]
    def fill(indices,start=0):
        for t,e in zip(tensors,expected):t[indices]=e[start:start+len(indices)]
        torch.cuda.synchronize()
    def seed(n):
        indices=allocator.alloc(n);fill(indices)
        cache.insert(InsertParams(key=key[:n],value=indices));return indices
    if rank==0:
        seed(112)
        endpoint=SourceEndpoint(cache,adapter,identity,torch.cuda.synchronize)
        class Server:
            stopped=False
            def dispatch(self,request):
                if request.get('operation')=='stop':
                    assert not endpoint.active and not cache.source_leases.busy
                    self.stopped=True
                    return {'released':len(endpoint.released)}
                return endpoint.dispatch(request)
        server=Server()
        with socket.socket() as listener:
            listener.bind(('0.0.0.0',0));listener.listen(1);listener.settimeout(120)
            channel.send({'host':socket.gethostname(),'port':listener.getsockname()[1]})
            conn,_=listener.accept()
            with conn:
                conn.settimeout(120)
                while not server.stopped:serve_once(conn,server)
        assert cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted==112
        helpers.atomic(root/'result-0.json',{'pass':True,'released_sessions':len(endpoint.released),
                       'host':socket.gethostname()})
    else:
        address=channel.recv()
        rows=[]
        with socket.create_connection((address['host'],address['port']),timeout=60) as conn:
            rpc=SocketRPC(conn)
            for case,prefix,competitor in [('partial',32,0),('duplicate_full',32,112),
                                            ('duplicate_partial',0,64),('full_local',112,0),('cancel',32,0),('last_page',96,0)]:
                cache.reset()
                for t in tensors:t.fill_(-7)
                if prefix:seed(prefix)
                held = allocator.alloc(4096-112) if case=='last_page' else None
                receiver=PeerReceiver(cache,adapter,identity,rpc)
                session=receiver.prepare(tokens)
                reservation=session['reservation'];tx=session['transaction']
                assert reservation.prefix_len==prefix
                if case=='last_page':assert reservation.pages==(256,)
                if competitor:
                    other=allocator.alloc(competitor-prefix);fill(other,prefix)
                    canonical=torch.cat((reservation.indices[:prefix],other))
                    cache.insert(InsertParams(key=key[:competitor],value=canonical,prev_prefix_len=prefix))
                if tx:
                    # An altered receipt must not release the active source.
                    receipt=session['offer']['receipt']
                    with_error={**receipt,'epoch':receipt['epoch']+1}
                    try:rpc.call('release',receipt=with_error,unmapped=True)
                    except RemoteRejected:pass
                    else:raise AssertionError('altered receipt accepted')
                    # Replay the original acquire ID: same lease, no extra pin/export.
                    send(conn,{'version':1,'id':receipt['request_id'],'operation':'acquire',
                               'tokens':tokens[:112],'contract':receiver.contract})
                    replay=receive(conn);assert replay['ok'] and replay['result']==session['offer']
                    if case=='cancel':tx.cancel();tx.cancel()
                    else:tx.start();tx.wait(30);tx.wait(0)
                    assert rpc.call('release',receipt=receipt,unmapped=True)['released']
                if case!='cancel':
                    for t,e in zip(tensors,expected):assert torch.equal(t[reservation.indices],e),(case,rank)
                    assert cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted==0
                copied=0 if tx is None or case=='cancel' else sum(r.length for r in tx.requests)
                if tx and case!='cancel':assert copied==(112-prefix)*sum(t[0].numel()*t.element_size() for t in tensors)
                receiver.finish_consumer();receiver.finish_consumer()
                evicted=cache.evict(EvictParams(num_tokens=4096)).num_tokens_evicted
                assert evicted==(prefix if case=='cancel' else 112),(case,evicted)
                if held is not None:allocator.free(held)
                assert allocator.available_size()==4096,(case,allocator.available_size())
                allocator._debug_check_no_duplicate_pages()
                rows.append({'case':case,'pass':True,'local_tokens':prefix,'transfer_bytes':copied,
                             'complete_tokens':112,'untransferred_tail_tokens':3})
            # Explicit pre-acquire mismatch must discard receiver reservation.
            bad=PeerReceiver(cache,adapter,replace(identity,weights_revision='different'),rpc)
            try:bad.prepare(tokens)
            except RemoteRejected:pass
            else:raise AssertionError('model mismatch accepted')
            assert bad.pending is None and allocator.available_size()==4096
            assert not cache.source_leases.busy
            response=rpc.call('stop')
            assert response['released']==5
        helpers.atomic(root/'result-1.json',{'pass':True,'host':socket.gethostname(),'cases':rows,
                                            'identity_rejection':True,'released_sessions':5})
    cache.reset();adapter.close();engine.close();pool._clear_buffers()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--rank',required=True,type=int);a=p.parse_args()
    try:run(a.output,a.rank)
    except BaseException:
        import traceback
        helpers.atomic(a.output/'bootstrap'/f'error-{a.rank}.json',traceback.format_exc());raise
