#!/usr/bin/env python3
"""Synthetic same-layout FABRIC READ microbenchmark, not model-pool integration.

One owner and one reader process. Immutable source, exclusive target, hot lease.
Full-allocation byte oracle (including holes) runs outside measurement.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import ctypes
import gc
import os
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import time

from sglang.srt.mem_cache.l1_transfer.transport import NvlinkEngine, Opcode, RegionDescriptor, TransferState, MemoryHandle
from sglang.srt.mem_cache.l1_transfer.transport.layout import BufferLayout, PoolLayout, RegionSpan, plan_transfer, bind_regions

MIB = 1 << 20


def cases():
    for size in (32768, MIB, 32*MIB, 256*MIB):
        yield dict(name=f'raw-{size}', model='raw', layout='contiguous', widths=[size], useful_page=size, page_tokens=1, pages=1, run=1)
    models = [
        ('glm53-bf16', [73728]*78 + [8448]*21, [78*73728,21*8448], 5928192,64),
        ('glm53-fp8-scaled', [41984]*78 + [8448]*21, [78*41984,21*8448],3452160,64),
        ('dsv4-flash-v4', [37440]*21+[8448]*21+[1728]*20, [37440*21,8448*21,1728*20],21*(64*584+64*132)+20*2*584,256),
        ('dsv4-pro-v4', [37440]*30+[8448]*30+[1728]*31, [37440*30,8448*30,1728*31],30*(64*584+64*132)+31*2*584,256),
    ]
    for model, layered, grouped, useful, page_tokens in models:
        for layout,widths in [('layer-first',layered),('page-first-grouped',grouped)]:
            for run in (1,4,16):
                yield dict(name=f'{model}-{layout}-run{run}',model=model,layout=layout,widths=widths,useful_page=useful,page_tokens=page_tokens,pages=16,run=run)


def geometry(case):
    n,run=case['pages'],case['run']
    # Gaps at each run boundary; reverse target run order, preserving order inside runs.
    starts=list(range(1,1+2*n,2*run))
    src=[s+j for s in starts for j in range(run)]
    dst=[s+j for s in reversed(starts) for j in range(run)]
    capacity=2*n+2
    offsets=[]; end=0
    for width in case['widths']:
        offsets.append(end); end+=capacity*width
    return src,dst,capacity,offsets,end


def payload(size):
    # Prime period avoids power-of-two page/slot aliases; generation is untimed.
    block=random.Random(20261016).randbytes(1048573)
    return (block*((size+len(block)-1)//len(block)))[:size]


def recv(pipe):
    if not pipe.poll(600):
        raise TimeoutError('peer did not acknowledge within 600 seconds')
    return pipe.recv()


def create_te(device):
    # Pin the CUDA Fabric transport, never silently compare with TCP/RDMA/IPC.
    for key in ('MC_FORCE_TCP', 'MC_FORCE_HCA', 'MC_FORCE_SHM', 'MC_USE_NVLINK_IPC', 'MC_USE_TENT', 'MC_INTRANODE_NVLINK'):
        if os.environ.get(key):
            raise RuntimeError(f'incompatible TE environment: {key}')
    os.environ['MC_FORCE_MNNVL']='1'
    os.environ['MC_USE_TEV1']='1'
    from mooncake.engine import TransferEngine
    cuda=ctypes.CDLL('libcudart.so.13')
    assert cuda.cudaSetDevice(device)==0
    te=TransferEngine()
    assert te.initialize('127.0.0.1','P2PHANDSHAKE','nvlink','')==0
    return te


def owner(pipe, backend):
    engine=NvlinkEngine(0)
    while True:
        msg=recv(pipe)
        if msg is None: break
        memory=engine.allocate_memory(msg['size'])
        engine.upload(memory,payload(msg['size']))
        desc=engine.export_memory(memory,'read')
        message={'descriptor':desc.to_dict()}
        if backend=='mooncake':
            te=create_te(0)
            assert te.register_memory(memory.address,memory.size)==0
            message.update(endpoint=f'127.0.0.1:{te.get_rpc_port()}',address=memory.address)
        pipe.send(message)
        assert recv(pipe)=={'unmapped':desc.lease_id}
        if backend=='mooncake':
            assert te.unregister_memory(memory.address)==0
            del te
            gc.collect()
        engine.release_export(desc.lease_id)
        engine.unregister_memory(memory)
        pipe.send('released')
    engine.close()
    pipe.send('closed')


def run_case(engine,pipe,case,args):
    src,dst,capacity,offsets,slot_size=geometry(case)
    copied=case['pages']*sum(case['widths'])
    # Small raw cases intentionally expose repeated-address/cache-hot API overhead.
    slots=1 if case['model']=='raw' and copied<MIB else max(1,math.ceil(args.working_mib*MIB/copied))
    size=slots*slot_size
    pipe.send({'size':size})
    message=recv(pipe)
    desc=RegionDescriptor.from_dict(message['descriptor'])
    start=time.perf_counter_ns()
    if args.backend=='mooncake':
        te=create_te(1)
        # Metadata handle only; TE imports the Fabric mapping on the first untimed copy.
        remote=MemoryHandle(desc.region_id,desc.size,0,True,message['address'])
    else:
        remote=engine.import_memory(desc)
    import_ns=time.perf_counter_ns()-start
    local=engine.allocate_memory(size)
    engine.upload(local,b'\xa5'*size)
    if args.backend=='mooncake':
        assert te.register_memory(local.address,local.size)==0
    layout=PoolLayout('encoded-bytes',case['page_tokens'],tuple(BufferLayout(str(i),w,case['layout']) for i,w in enumerate(case['widths'])))
    start=time.perf_counter_ns()
    plan=plan_transfer(layout,layout,src,dst,source_capacity=capacity,target_capacity=capacity)
    plan_ns=time.perf_counter_ns()-start
    start=time.perf_counter_ns()
    bindings=lambda memory: {str(i):[RegionSpan(0,capacity*w,memory,offsets[i])] for i,w in enumerate(case['widths'])}
    requests=bind_regions(plan,bindings(remote),bindings(local),Opcode.READ)
    batches=[tuple(replace(r,local_offset=r.local_offset+slot*slot_size,remote_offset=r.remote_offset+slot*slot_size) for r in requests) for slot in range(slots)]
    bind_ns=time.perf_counter_ns()-start
    assert sum(r.length for r in requests)==copied
    original=payload(size)
    expected=bytearray(b'\xa5'*size)
    # Oracle uses individual page mappings, never planner ranges.
    for slot in range(slots):
        for offset,width in zip(offsets,case['widths']):
            base=slot*slot_size+offset
            for s,t in zip(src,dst):
                expected[base+t*width:base+(t+1)*width]=original[base+s*width:base+(s+1)*width]
    del original

    def transfer(requests):
        t0=time.perf_counter_ns()
        if args.backend=='mooncake':
            # List adaptation and pybind conversion are INSIDE the common API boundary.
            buffers=[r.local.address+r.local_offset for r in requests]
            addresses=[r.remote.address+r.remote_offset for r in requests]
            lengths=[r.length for r in requests]
            t1=time.perf_counter_ns()
            ret=te.batch_transfer_sync_read(message['endpoint'],buffers,addresses,lengths,'nvlink')
            t2=time.perf_counter_ns()
            assert ret==0, f'TE failure {ret}; do not release possibly in-flight memory'
            # TE sync API includes batch recycling. No fake submit/wait decomposition.
            return {'batch_ns':t2-t0,'adapt_ns':t1-t0,'sync_call_ns':t2-t1,
                    'submit_ns':None,'wait_ns':None,'release_ns':None,'iteration_ns':t2-t0}
        batch=engine.submit_transfer(requests)
        t1=time.perf_counter_ns()
        status=engine.wait(batch,timeout=30)
        t2=time.perf_counter_ns()
        assert status.state==TransferState.SUCCESS, status
        assert status.completed_bytes==copied
        engine.release_batch(batch)
        t3=time.perf_counter_ns()
        return {'batch_ns':t2-t0,'submit_ns':t1-t0,'wait_ns':t2-t1,'release_ns':t3-t2,'iteration_ns':t3-t0}

    for batch in batches: transfer(batch)
    assert engine.download(local)==expected, 'preflight data/sentinel mismatch'
    samples=[]
    for rnd in range(args.rounds):
        for i in range(args.warmup): transfer(batches[i%slots])
        for i in range(args.samples):
            slot=(rnd*args.samples+i)%slots
            samples.append(dict(round=rnd,slot=slot,**transfer(batches[slot])))
    assert engine.download(local)==expected, 'post-measurement data/sentinel mismatch'
    digest=hashlib.sha256(expected).hexdigest()
    del expected
    start=time.perf_counter_ns()
    if args.backend=='mooncake':
        assert te.unregister_memory(local.address)==0
        # Successful copies are drained; destroy TE's cached remote mappings before ACK.
        del te
        gc.collect()
    else:
        engine.unregister_memory(remote)
    unmap_ns=time.perf_counter_ns()-start
    pipe.send({'unmapped':desc.lease_id}); assert recv(pipe)=='released'
    engine.unregister_memory(local)
    median=statistics.median(s['batch_ns'] for s in samples)
    return dict(**case,backend=args.backend,pass_correctness=True,sha256=digest,source_pages=src,target_pages=dst,
                B_copy=copied,B_useful=case['pages']*case['useful_page'],N_copy=len(requests),
                range_histogram=dict(Counter(r.length for r in requests)),slots=slots,
                selected_working_set_bytes=slots*copied,allocation_bytes_per_endpoint=size,
                physical_region_bytes=desc.physical_size,granularity=desc.granularity,
                import_ns=import_ns,unmap_ns=unmap_ns,plan_ns=plan_ns,bind_all_slots_ns=bind_ns,
                median_batch_us=median/1000,copy_GBps=copied/median,
                useful_GBps=case['pages']*case['useful_page']/median,samples=samples)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--working-mib',type=int,default=256)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--list-only',action='store_true')
    parser.add_argument('--backend',choices=['native','mooncake'],default='native')
    parser.add_argument('--case',help='optional single-case correctness smoke')
    args=parser.parse_args()
    todo=list(cases()); random.Random(20261016).shuffle(todo)
    if args.case:
        todo=[c for c in todo if c['name']==args.case]
        assert todo, 'unknown case'
    if args.list_only:
        for c in todo:
            s,t,cap,offsets,size=geometry(c)
            print(c['name'],c['pages']*sum(c['widths']),len(c['widths'])*c['pages']//c['run'])
        return
    args.output.mkdir(parents=True,exist_ok=True)
    context=mp.get_context('spawn'); a,b=context.Pipe()
    process=context.Process(target=owner,args=(b,args.backend)); process.start()
    engine=NvlinkEngine(1)
    try:
        for case in todo:
            result=run_case(engine,a,case,args)
            (args.output/(case['name']+'.json')).write_text(json.dumps(result,indent=2))
            print(case['name'],f"{result['copy_GBps']:.3f} GB/s",flush=True)
        engine.close(); a.send(None); assert recv(a)=='closed'
        process.join(30); assert process.exitcode==0
        (args.output/'complete.json').write_text(json.dumps({'pass':True,'cases':len(todo),'args':vars(args)},default=str,indent=2))
    finally:
        if process.is_alive():
            # Job/process teardown, never a lease-release assertion after failed drain.
            process.terminate(); process.join(30)


if __name__=='__main__': main()
