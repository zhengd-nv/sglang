# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import _bootstrap  # Optional installed-runtime overlay for manual tests.

import argparse
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile

import torch
from sglang.srt.mem_cache.storage.egm.region import Region
from sglang.srt.mem_cache.storage.egm.store import EGMStore
from sglang.srt.mem_cache.storage.egm.peer import PeerClient, PeerServer


def rejects(fn, error=Exception):
    try:
        fn()
    except error:
        return
    raise AssertionError("expected explicit rejection")


def reader(path, queue):
    try:
        client = PeerClient(path)
        dst = torch.full((260,), 0xA5, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        assert client.get("pattern", [(dst.data_ptr() + 2, 256, 0)], device=True)
        actual = dst.cpu()
        assert torch.equal(actual[2:258], torch.arange(256, dtype=torch.uint8))
        assert actual[:2].eq(0xA5).all() and actual[258:].eq(0xA5).all()
        host = torch.empty(256, dtype=torch.uint8)
        assert client.get("pattern", [(host.data_ptr(), 256, 0)])
        assert torch.equal(host, torch.arange(256, dtype=torch.uint8))
        rejects(lambda: client.get("pattern", [(host.data_ptr(), 256, 1)]), ValueError)
        assert not client.get("missing", [(host.data_ptr(), 256, 0)])
        queue.put({"status": "PASS", "pid": os.getpid()})
    except BaseException as e:
        queue.put({"status": "FAIL", "error": repr(e)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {}
    store = EGMStore(4 << 20, numa=0)
    src = torch.arange(256, dtype=torch.uint8)
    assert store.put("pattern", [(src.data_ptr(), src.nbytes)])
    host = torch.zeros_like(src)
    assert store.get("pattern", [(host.data_ptr(), host.nbytes, 0)])
    assert torch.equal(src, host)
    rejects(lambda: store.get("pattern", [(host.data_ptr(), 256, -1)]), ValueError)
    rejects(lambda: store.get("pattern", [(host.data_ptr(), 257, 0)]), ValueError)
    rejects(lambda: store.put("pattern", [(src.data_ptr(), 1)]), ValueError)
    assert not store.put("too-large", [(src.data_ptr(), 8 << 20)])
    assert not store.exists("too-large")
    lease = store.acquire(["pattern"])
    rejects(store.clear, RuntimeError)
    rejects(store.close, RuntimeError)
    rejects(lambda: store.release(lease["token"], "wrong-epoch"), ValueError)
    store.release(lease["token"], lease["epoch"])
    rejects(lambda: store.release(lease["token"], lease["epoch"]), ValueError)
    results["region"] = {"status": "PASS", "allocation_bytes": store.region.allocation_bytes,
                         "granularity": store.region.granularity, "kind": "HOST_NUMA_FABRIC"}
    with tempfile.TemporaryDirectory(prefix="egm-peer-") as tmp:
        server = PeerServer(store, str(Path(tmp) / "control.sock"))
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        child = ctx.Process(target=reader, args=(server.path, queue))
        child.start()
        child.join(120)
        if child.is_alive():
            child.terminate(); child.join()
            raise RuntimeError("consumer timeout; provider pins retained")
        result = queue.get(timeout=5)
        assert child.exitcode == 0 and result["status"] == "PASS", result
        results["peer_fabric"] = result
        with PeerClient(server.path).reader(["pattern"]) as pinned:
            assert pinned.get("pattern", [(host.data_ptr(), 128, 0)])
            rejects(store.clear, RuntimeError)
            rejects(store.close, RuntimeError)
            assert pinned.get("pattern", [(host.data_ptr() + 128, 128, 128)])
        assert torch.equal(host, src)
        # A transfer error is not permission to release the source pin.
        import sglang.srt.mem_cache.storage.egm.peer as peer
        original = peer.Region
        class InjectFailure(original):
            def copy(self, *args):
                raise RuntimeError("injected copy error")
        peer.Region = InjectFailure
        client = PeerClient(server.path)
        try:
            rejects(lambda: client.get("pattern", [(host.data_ptr(), 256, 0)]), RuntimeError)
            assert len(client.quarantined) == 1
            rejects(store.clear, RuntimeError)
            rejects(store.close, RuntimeError)
            abandoned, mapping = client.quarantined.pop()
            # Injection ran no DMA; explicit test-only recovery is safe.
            for region in mapping.values():
                region.close()
            store.release(abandoned["token"], abandoned["epoch"])
        finally:
            peer.Region = original
        results["failed_reader_pin"] = {"status": "PASS", "reclaim": "refused until explicit safe recovery"}
        server.close()
    store.clear()
    assert not store.exists("pattern")
    store.close(); store.close()
    rejects(lambda: store.exists("pattern"), RuntimeError)
    # Repeat complete native lifecycles, including a different NUMA node.
    for numa in (0, 1):
        for _ in range(8):
            with Region(4097, numa) as r:
                r.copy(1, src.data_ptr(), 256, 0)
                r.copy(1, host.data_ptr(), 256, 1)
                assert torch.equal(src, host)
                rejects(lambda: r.copy(4096, host.data_ptr(), 2, 1), ValueError)
    results["lifecycle"] = {"status": "PASS", "cycles": 16, "numa": [0, 1]}
    test_hicache(results)
    test_linker(results)
    test_linker(results, "hbm")
    Path(args.output, "egm-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def test_hicache(results):
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, PoolName, PoolTransfer
    from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
    # Actual SGLang HostPool metadata method with small controlled CPU storage;
    # avoid allocating the server's default multi-GB HostPool just for geometry.
    for layout in ("layer_first", "page_first"):
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout, pool.page_size, pool.size = layout, 2, 16
        pool.layer_num, pool.head_num, pool.head_dim = 2, 2, 8
        pool.dtype = torch.bfloat16
        dims = (2, 2, 16, 2, 8) if layout == "layer_first" else (2, 16, 2, 2, 8)
        pool.kv_buffer = torch.arange(1024, dtype=torch.float32).reshape(dims).to(pool.dtype)
        cfg = HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
            attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
            enable_storage_metrics=False, is_page_first_layout=layout != "layer_first", model_name="fixture",
            extra_config={"backend_name": "egm", "module_path": "sglang.srt.mem_cache.storage.egm.hicache", "class_name": "EGMHiCacheStorage", "interface_v1": True, "capacity_bytes": 4 << 20})
        backend = StorageBackendFactory.create_backend("dynamic", cfg, pool)
        backend.register_mem_pool_host(pool)
        saved = pool.kv_buffer.clone()
        indices = torch.tensor([4, 5, 0, 1, 10, 11])
        keys = ["page2", "page0", "page5"]
        assert backend.batch_set_v1(keys, indices) == [True] * 3
        assert backend.batch_exists(["page2", "missing", "page0"]) == 1
        pool.kv_buffer.fill_(-1)
        assert backend.batch_get_v1(keys, indices) == [True] * 3
        ptrs, sizes = pool.get_page_buffer_meta(indices)
        for ptr, n in zip(ptrs, sizes):
            offset = ptr - pool.kv_buffer.data_ptr()
            expected = saved.view(torch.uint8).flatten()[offset:offset+n]
            actual = pool.kv_buffer.view(torch.uint8).flatten()[offset:offset+n]
            assert torch.equal(expected, actual)
        before = pool.kv_buffer.clone()
        assert backend.batch_get_v1(["missing"], indices[:2]) == [False]
        assert torch.equal(before, pool.kv_buffer)
        rejects(lambda: backend.batch_get_v1(["page0"], torch.tensor([1, 2])), ValueError)
        t = PoolTransfer(name=PoolName.KV, keys=keys, host_indices=indices)
        assert backend.batch_get_v2([t])["kv"] == [True] * 3
        assert backend.batch_exists_v2(keys, [t]).kv_hit_pages == 3
        rejects(lambda: backend.register_mem_host_pool_v2(pool, "swa"), ValueError)
        backend.close()
        # Fixture has no registration; stop HostPool destructor from treating it as one.
        pool.pin_memory = False
    results["hicache"] = {"status": "PASS", "layouts": ["layer_first", "page_first"], "factory": "dynamic", "api": ["v1", "v2"]}


def test_linker(results, tier="dram"):
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
    from sglang.srt.mem_cache.storage.egm.linker import EGMDirectLinker
    k = [torch.arange(256, device="cuda", dtype=torch.float32).reshape(16, 2, 8) + i*1000 for i in range(2)]
    v = [x + 10000 for x in k]
    saved = [x.clone() for x in k + v]
    entry = DevicePoolEntry(name=PoolName.KV, indices_from_pool=PoolName.KV, device_pool=None,
        components=[k, v], layer_mapping={0: 0, 1: 1}, page_size=2, rows_are_pages=False, packed=True)
    store = EGMStore(4 << 20 if tier == "dram" else 0,
                     hbm_capacity=4 << 20 if tier == "hbm" else 0, default_tier=tier)
    linker = EGMDirectLinker(entry, 2, store, "test-model")
    indices = torch.tensor([4, 5, 0, 1], device="cuda")
    transfer = PoolTransfer(name=PoolName.KV, keys=["p2", "p0"], device_indices=indices)
    assert linker.offload([transfer]) and linker.pop_completed_offload()
    assert linker.lookup("req", [transfer]) == [1, 2]
    assert linker.load("cancel", [transfer]) and linker.cancel_queued_load("cancel")
    assert not linker.cancel_queued_load("cancel")
    for t in k + v:
        t.fill_(-1)
    assert linker.load("req", [transfer])
    index = linker.start_layer_wise_loading()
    linker.layer_done_counter.set_consumer(index)
    for layer in range(2):
        linker.layer_done_counter.wait_until(layer)
        linker.layer_done_counter.wait_until(layer)  # K/V may each wait.
    assert linker.pop_completed_load() == ["req"]
    for actual, expected in zip(k + v, saved):
        assert torch.equal(actual[indices], expected[indices])
        assert actual[2:4].eq(-1).all()
    linker.reset(); linker.close(); store.close()
    results["direct_linker" if tier == "dram" else "direct_linker_hbm"] = {"status": "PASS", "mode": "synchronous FULL TP1", "layers": 2}

    # Lookup-to-read miss fails before forward, and cannot be reset into service.
    store = EGMStore(4 << 20 if tier == "dram" else 0,
                     hbm_capacity=4 << 20 if tier == "hbm" else 0, default_tier=tier)
    linker = EGMDirectLinker(entry, 2, store, "failure-model")
    linker.load("missing", [transfer])
    rejects(linker.start_layer_wise_loading, RuntimeError)
    rejects(linker.reset, RuntimeError)
    rejects(lambda: linker.load("next", [transfer]), RuntimeError)
    linker.close(); store.close()
    results["linker_failure" if tier == "dram" else "linker_failure_hbm"] = {"status": "PASS", "recovery": "Engine restart required"}


if __name__ == "__main__":
    main()
