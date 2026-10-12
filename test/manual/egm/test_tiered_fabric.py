# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Two processes/nodes, bidirectional mixed HBM/DRAM reads through one API.

Rendezvous files contain metadata only. KV bytes are never written to disk or
sent through sockets. Run both ranks with a shared fresh --output directory.
"""
import _bootstrap

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import torch
from sglang.srt.mem_cache.storage.egm import peer
from sglang.srt.mem_cache.storage.egm.peer import PeerClient, PeerServer
from sglang.srt.mem_cache.storage.egm.region import Region
from sglang.srt.mem_cache.storage.egm.store import EGMStore


def rejects(fn, error=Exception):
    try:
        fn()
    except error:
        return
    raise AssertionError('operation unexpectedly succeeded')


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def wait(path, root):
    deadline = time.monotonic() + 600
    while not path.exists():
        if list(root.glob('failed-*.json')):
            raise RuntimeError('peer test failed; retain provider resources until exit')
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(.1)
    return json.loads(path.read_text())


def barrier(root, name, rank):
    atomic(root / f'{name}-{rank}.json', True)
    wait(root / f'{name}-{1-rank}.json', root)


def counters(dest):
    dest.mkdir(exist_ok=True)
    for name, command in [('nvlink', ['nvidia-smi', 'nvlink', '-gt', 'd']),
                          ('topology', ['nvidia-smi', 'topo', '-m'])]:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30)
        (dest / f'{name}.txt').write_text(result.stdout)
        (dest / f'{name}.rc').write_text(str(result.returncode))
    values = {}
    for path in Path('/sys/class/infiniband').glob('*/ports/*/counters/port_*_data'):
        try:
            values[str(path)] = int(path.read_text())
        except (OSError, ValueError):
            pass
    (dest / 'rdma.json').write_text(json.dumps(values, indent=2))


def main(args):
    rank, root = args.rank, Path(args.output)
    out = root / f'rank-{rank}'
    out.mkdir(exist_ok=True)
    owner_device = rank * 2 if args.local else 0
    reader_device = owner_device + 1
    torch.cuda.set_device(owner_device)
    secret = (root / 'control-secret').read_text().strip()
    n = 8 << 20
    store = EGMStore(2*n, hbm_capacity=2*n, device=owner_device)
    for index, tier in enumerate(('hbm', 'dram')):
        data = ((torch.arange(n, device=f'cuda:{owner_device}', dtype=torch.int64)
                 + rank * 17 + index * 41) % 251).to(torch.uint8)
        torch.cuda.synchronize(owner_device)
        assert store.put(tier, [(data.data_ptr(), n)], device=True, tier=tier)
        rejects(lambda: store.put(tier, [(data.data_ptr(), n)], tier='dram' if tier == 'hbm' else 'hbm'), ValueError)
        assert not store.put('overflow-'+tier, [(data.data_ptr(), 2*n)], tier=tier)
        del data
    # Verify actual allocation kind, including rejection of forged tier metadata.
    forged = store.regions['hbm'].descriptor()
    forged.update(tier='dram', kind='HOST_NUMA_FABRIC')
    rejects(lambda: Region(descriptor=forged, device=reader_device), RuntimeError)
    server = PeerServer(store, (socket.gethostbyname(socket.gethostname()), 0), secret=secret)
    atomic(root / f'endpoint-{rank}.json', {'address': server.path, 'host': socket.gethostname(),
                                           'owner_gpu': owner_device, 'reader_gpu': reader_device})
    remote = wait(root / f'endpoint-{1-rank}.json', root)
    assert args.local or remote['host'] != socket.gethostname(), 'cross-node test requires distinct hosts'
    client = PeerClient(remote['address'], device=reader_device, secret=secret)
    rejects(lambda: PeerClient(remote['address'], secret='wrong').exists('hbm'), RuntimeError)
    assert all(client.exists(t) for t in ('hbm', 'dram'))
    assert not client.exists('missing')
    torch.cuda.set_device(reader_device)
    dst = torch.full((n+32,), 0xA5, dtype=torch.uint8, device=f'cuda:{reader_device}')
    torch.cuda.synchronize(reader_device)
    assert not client.get('missing', [(dst.data_ptr()+16, n, 0)], device=True)
    checks = []
    with client.reader(['hbm', 'dram']) as reader:
        barrier(root, 'pinned', rank)
        rejects(store.clear, RuntimeError)
        rejects(store.close, RuntimeError)
        for index, tier in enumerate(('hbm', 'dram')):
            # Separate traffic windows for each direction and tier.
            for active in (0, 1):
                phase = f'{tier}-reader-{active}'
                barrier(root, phase+'-start', rank)
                counters(out / (phase+'-before'))
                barrier(root, phase+'-measured', rank)
                if rank == active:
                    expected = ((torch.arange(n, dtype=torch.int64) + (1-rank)*17 + index*41) % 251).to(torch.uint8)
                    for _ in range(4):
                        dst.fill_(0xA5)
                        torch.cuda.synchronize(reader_device)
                        assert reader.get(tier, [(dst.data_ptr()+16, n//2, 0),
                                                (dst.data_ptr()+16+n//2, n-n//2, n//2)], device=True)
                        actual = dst.cpu()
                        assert torch.equal(actual[16:-16], expected)
                        assert actual[:16].eq(0xA5).all() and actual[-16:].eq(0xA5).all()
                    rejects(lambda: reader.get(tier, [(dst.data_ptr(), n+1, 0)], True), ValueError)
                    checks.append({'tier': tier, 'bytes': n*4, 'read_api': 'reader.get', 'status': 'PASS'})
                barrier(root, phase+'-copied', rank)
                counters(out / (phase+'-after'))
                barrier(root, phase+'-done', rank)
    barrier(root, 'released', rank)
    assert not store._leases
    barrier(root, 'release-checked', rank)
    # Fail after a successful HBM read, before DRAM copy submission. One pin
    # must protect BOTH source tiers, including gaps between get calls.
    original = peer.Region
    class InjectFailure(original):
        def copy(self, *values):
            if self.tier == 'dram':
                raise RuntimeError('injected pre-copy DRAM failure')
            return super().copy(*values)
    peer.Region = InjectFailure
    try:
        def failed_read():
            with client.reader(['hbm', 'dram']) as reader:
                reader.get('hbm', [(dst.data_ptr()+16, n, 0)], True)
                reader.get('dram', [(dst.data_ptr()+16, n, 0)], True)
        rejects(failed_read, RuntimeError)
    finally:
        peer.Region = original
    assert len(client.quarantined) == 1
    barrier(root, 'quarantined', rank)
    rejects(store.clear, RuntimeError)
    rejects(store.close, RuntimeError)
    barrier(root, 'quarantine-checked', rank)
    lease, mappings = client.quarantined.pop()
    # Test-only recovery: preceding HBM copy drained and injected DRAM copy
    # performed no DMA. Never do this after an actual unknown CUDA failure.
    for mapping in mappings.values():
        mapping.close()
    client.rpc('release', token=lease['token'], epoch=lease['epoch'])
    barrier(root, 'recovered', rank)
    assert not store._leases
    old_epoch = store.epoch
    store.clear()
    assert store.epoch != old_epoch and not store.exists('hbm') and not store.exists('dram')
    store.close()
    server.close()
    result = {'status': 'PASS', 'host': socket.gethostname(), 'peer': remote,
              'owner_gpu': owner_device, 'reader_gpu': reader_device, 'checks': checks,
              'mixed_pin_failure': 'PASS', 'source': peer.__file__, 'torch': torch.__version__,
              'control_bytes': client.control_bytes,
              'transport': 'CUDA FABRIC + cuMemcpyDtoD; TCP metadata only'}
    assert sum(client.control_bytes.values()) < n // 100
    atomic(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=(0,1), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        atomic(Path(args.output) / f'failed-{args.rank}.json', repr(error))
        raise
