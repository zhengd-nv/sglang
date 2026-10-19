#!/usr/bin/env python3
"""Hardware correctness, two independent processes; SGLang-packaged CUDA Driver transport.

Only descriptors/acks use shared files. Payloads use DEVICE/FABRIC mappings.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback

from sglang.srt.mem_cache.l1_transfer.transport import (BusyError, NvlinkEngine, Opcode, RegionDescriptor,
                             TransferRequest, TransferState)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


class FileChannel:
    def __init__(self, root, rank):
        self.root = Path(root); self.rank = rank; self.tx = 0; self.rx = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def send(self, value):
        atomic(self.root / f"{self.rank}-{self.tx}.json", value); self.tx += 1

    def recv(self):
        path = self.root / f"{1-self.rank}-{self.rx}.json"
        deadline = time.monotonic() + 180
        while not path.exists():
            errors = list(self.root.glob("error-*.json"))
            if errors:
                raise RuntimeError(errors[0].read_text())
            if time.monotonic() > deadline: raise TimeoutError(str(path))
            time.sleep(.02)
        self.rx += 1
        return json.loads(path.read_text())


def pattern(size, seed):
    block = bytes((i*37 + seed*13) % 251 for i in range(4096))
    return (block * ((size+4095)//4096))[:size]


def planned_spec():
    from sglang.srt.mem_cache.l1_transfer.transport.layout import BufferLayout, PoolLayout
    layout = PoolLayout('uint8', 16, tuple(BufferLayout(f'{layer}.{kind}', width, f'NHD:{width}')
                        for layer in range(3) for kind,width in (('k',37),('v',19))))
    return layout, [2,3,7,8,12], [8,9,2,4,14]


def planned_requests(local, remote, opcode):
    from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan, plan_transfer, bind_regions
    layout, src, dst = planned_spec()
    source_memory, target_memory = (remote, local) if opcode == Opcode.READ else (local, remote)
    source, target = {}, {}
    base = 17
    for buffer in layout.buffers:
        size = 16 * buffer.page_bytes
        # Differing binding boundaries, including boundaries inside a page.
        for bindings, memory, cut in ((source,source_memory,3*buffer.page_bytes+5),
                                       (target,target_memory,9*buffer.page_bytes+3)):
            bindings[buffer.name] = [RegionSpan(0,cut,memory,base),
                                     RegionSpan(cut,size-cut,memory,base+cut)]
        base += size + 23
    plan = plan_transfer(layout,layout,src,dst,source_capacity=16,target_capacity=16)
    return bind_regions(plan,source,target,opcode)


def cases():
    # Offsets explicitly exercise odd bytes, padding boundaries, and random fragmentation.
    yield "odd-boundaries", 65539, [(0, 65538, 1), (65538, 0, 1), (31, 79, 119)]
    yield "contiguous", 2*1024*1024+137, [(17, 53, 2*1024*1024)]
    rng = random.Random(9017)
    source_pages = rng.sample(range(256), 128)
    target_pages = rng.sample(range(256), 128)
    yield "fragmented-128", 256*4096+37, [(s*4096+7, t*4096+13, 2017) for s,t in zip(source_pages,target_pages)]
    # Three layers, K/V split, NHD-like slots with holes on both sides.
    yield "layered-kv-ranges", 6*32768+19, [(layer*32768+s*128,layer*32768+t*128,128)
                                           for layer in range(6) for s,t in ((2,21),(7,3),(19,13),(31,42))]

    # Oracle uses original individual pages, not the planner's coalesced ranges.
    layout, src, dst = planned_spec()
    base = 17; ranges = []
    for buffer in layout.buffers:
        width = buffer.page_bytes
        ranges.extend((base+s*width,base+t*width,width) for s,t in zip(src,dst))
        base += 16*width+23
    yield "planned-same-layout", base+17, ranges


def must_raise(error_type, function):
    try: function()
    except error_type: return
    raise AssertionError(f"expected {error_type.__name__}")


def snapshot(path):
    # Read-only, never reset system counters. C2C/setup may appear; these are supporting evidence.
    value = {}
    for name, command in (("nvlink", ["nvidia-smi","nvlink","-gt","d"]),):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        value[name] = {"rc":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
    sysfs = os.environ.get("NVT_SYSFS_ROOT")
    value["rdma"] = ({str(p):int(p.read_text()) for p in
                      Path(sysfs).glob("class/infiniband/*/ports/*/counters/port_*_data")}
                     if sysfs else {})
    atomic(path, value)


def owner(channel, device, out):
    engine = NvlinkEngine(device)
    rows = []
    for idx, (name, size, ranges) in enumerate(cases()):
        for opcode in (Opcode.READ, Opcode.WRITE):
            memory = engine.allocate_memory(size)
            original = pattern(size, idx+1)
            engine.upload(memory, original)
            descriptor = engine.export_memory(memory, "read" if opcode == Opcode.READ else "write")
            must_raise(BusyError, lambda: engine.unregister_memory(memory))
            must_raise(BusyError, lambda: engine.upload(memory,b"X"))
            channel.send({"descriptor":descriptor.to_dict(),"case":name,"opcode":opcode.value,
                          "host":socket.gethostname(),"device":device})
            ready = channel.recv(); assert ready["ready"]
            # The importer is mapped and idle; validation and setup are outside this window.
            snapshot(out / f"{name}-{opcode.value}-owner-before.json")
            channel.send({"go":True})
            done = channel.recv(); assert done["drained"]
            snapshot(out / f"{name}-{opcode.value}-owner-after.json")
            channel.send({"observed":True})
            ack = channel.recv(); assert ack["unmapped"] and ack["lease_id"] == descriptor.lease_id
            engine.release_export(descriptor.lease_id)
            actual = engine.download(memory)
            expected = bytearray(original)
            if opcode == Opcode.WRITE:
                local_data = pattern(size, idx+31)
                for source, target, length in ranges:
                    expected[target:target+length] = local_data[source:source+length]
            assert actual == expected, (name, opcode, "owner mismatch")
            engine.unregister_memory(memory)
            rows.append({"case":name,"opcode":opcode.value,"bytes":sum(r[2] for r in ranges),
                         "ranges":len(ranges),"sha256":hashlib.sha256(actual).hexdigest(),"pass":True})
            channel.send({"verified":True})
    engine.close()
    atomic(out / "owner-result.json", {"pass":True,"cases":rows,"host":socket.gethostname(),"device":device})


def consumer(channel, device, out):
    engine = NvlinkEngine(device); rows=[]
    for idx, (name,size,ranges) in enumerate(cases()):
        for opcode in (Opcode.READ, Opcode.WRITE):
            message = channel.recv(); assert message["case"] == name and message["opcode"] == opcode.value
            desc = RegionDescriptor.from_dict(message["descriptor"])
            remote = engine.import_memory(desc); local = engine.allocate_memory(size)
            initial = b"\xa5"*size if opcode == Opcode.READ else pattern(size,idx+31)
            engine.upload(local,initial)
            must_raise(ValueError, lambda: engine.import_memory(desc))
            requests=[]
            for source,target,length in ranges:
                lo,ro = (target,source) if opcode == Opcode.READ else (source,target)
                requests.append(TransferRequest(opcode,local,remote,lo,ro,length))
            if name == "planned-same-layout":
                requests = planned_requests(local, remote, opcode)
            bad = TransferRequest(opcode,local,remote,0,size,1)
            must_raise(ValueError, lambda: engine.submit_transfer([requests[0],bad]))
            assert engine.download(local)==initial, "validation caused partial local write"
            channel.send({"ready":True}); assert channel.recv()["go"]
            snapshot(out / f"{name}-{opcode.value}-consumer-before.json")
            batch = engine.submit_transfer(requests)
            must_raise(BusyError, lambda: engine.unregister_memory(local))
            must_raise(BusyError, lambda: engine.unregister_memory(remote))
            result = engine.wait(batch,timeout=30)
            assert result.state == TransferState.SUCCESS and result.completed_bytes == sum(r[2] for r in ranges), result
            engine.release_batch(batch)
            snapshot(out / f"{name}-{opcode.value}-consumer-after.json")
            channel.send({"drained":True}); assert channel.recv()["observed"]
            actual = engine.download(local)
            expected = bytearray(initial)
            if opcode == Opcode.READ:
                source_data = pattern(size,idx+1)
                for source,target,length in ranges: expected[target:target+length]=source_data[source:source+length]
            assert actual == expected, (name,opcode,"consumer mismatch/sentinel changed")
            engine.unregister_memory(remote); engine.unregister_memory(local)
            must_raise(ValueError, lambda: engine.import_memory(desc))
            channel.send({"unmapped":True,"lease_id":desc.lease_id})
            assert channel.recv()["verified"]
            rows.append({"case":name,"opcode":opcode.value,"bytes":result.completed_bytes,
                         "ranges":len(ranges),"pass":True,"peer_host":message["host"],"device":device})
    engine.close()
    forbidden = [name for name in sys.modules if name.split(".")[0] in ("mooncake","nixl","mori")]
    assert not forbidden, forbidden
    atomic(out / "consumer-result.json", {"pass":True,"cases":rows,"forbidden_imports":forbidden,
                                         "host":socket.gethostname(),"device":device})


def run_worker(root, rank, device):
    root=Path(root); out=root/f"rank-{rank}"; out.mkdir(parents=True,exist_ok=True)
    channel=FileChannel(root/"control",rank)
    try:
        (owner if rank==0 else consumer)(channel,device,out)
    except BaseException:
        atomic(root/"control"/f"error-{rank}.json",traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--rank",type=int,choices=(0,1))
    parser.add_argument("--device",type=int,default=0)
    parser.add_argument("--local",action="store_true")
    args=parser.parse_args()
    if args.local:
        context=mp.get_context("spawn")
        children=[context.Process(target=run_worker,args=(args.output,rank,rank)) for rank in (0,1)]
        for child in children: child.start()
        try:
            deadline=time.monotonic()+240
            for child in children:
                child.join(max(0,deadline-time.monotonic()))
                if child.exitcode != 0: raise RuntimeError(f"GPU test child exit: {child.exitcode}")
        finally:
            for child in children:
                if child.is_alive(): child.terminate()
                child.join()
    else:
        if args.rank is None: parser.error("provide --rank or --local")
        run_worker(args.output,args.rank,args.device)
    print("PASS",args.output,flush=True)


if __name__=="__main__": main()
