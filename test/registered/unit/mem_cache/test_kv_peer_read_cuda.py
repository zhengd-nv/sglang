"""Peer KV page reads between two processes over a CUDA FABRIC export, byte-checked."""

import multiprocessing as mp
import unittest

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cuda_ci(est_time=40, stage="base-c", runner_config="2-gpu-gb300")

LAYERS = 3
PAGE_BYTES = (1024, 384)  # two buffer classes per layer, like MLA + DSA index
CAPACITY = 64
REGION_BYTES = LAYERS * CAPACITY * sum(PAGE_BYTES)
SENTINEL = 0xEE


def _src_offset(layer: int, cls: int) -> int:
    # Class-major, then layer: [cls0 l0 .. l2][cls1 l0 .. l2], CAPACITY pages each.
    return (cls * LAYERS * PAGE_BYTES[0] + layer * PAGE_BYTES[cls]) * CAPACITY


def _pattern(offsets: torch.Tensor) -> torch.Tensor:
    return ((offsets * 7 + offsets // 4096) % 251).to(torch.uint8)


# EGM page objects, as a store lays them out in host memory: slot s holds one page of
# every buffer at the buffer's item offset (latent of every layer, then indexer K of
# some), and GAP bytes after each item that nothing may write.
EGM_ITEMS = [(layer, 1024) for layer in range(LAYERS)] + [(0, 384), (2, 384)]
EGM_GAP = 128
EGM_ITEM_OFF = [
    i * EGM_GAP + sum(size for _, size in EGM_ITEMS[:i]) for i in range(len(EGM_ITEMS))
]
EGM_OBJECT = EGM_ITEM_OFF[-1] + EGM_ITEMS[-1][1] + EGM_GAP
EGM_SLOTS = 96


def _egm_local_page(item: int, page: int, device) -> torch.Tensor:
    """What a writer's local page holds: distinct for every (item, page)."""
    size = EGM_ITEMS[item][1]
    offsets = torch.arange(size, device=device) + (item * CAPACITY + page) * 4099 + 17
    return _pattern(offsets)


def _egm_owner(conn) -> None:
    from sglang.srt.mem_cache.kv_peer_read import KvPeerExporter
    from sglang.srt.utils.cuda_vmm_utils import (
        create_host_numa_allocation,
        get_device_host_numa_id,
        release_host_numa_allocation,
        tensor_from_pointer,
    )

    torch.cuda.set_device(0)
    numa = get_device_host_numa_id(0)
    if numa < 0:
        conn.send(None)
        return
    slab = create_host_numa_allocation(EGM_SLOTS * EGM_OBJECT, numa, devices=(0,))
    view = tensor_from_pointer(slab.address, slab.size, device_id=0)
    view.copy_(_pattern(torch.arange(slab.size, device="cuda", dtype=torch.int64)))
    torch.cuda.synchronize()
    exporter = KvPeerExporter(0)
    leases = [
        exporter.export(slab.handle, slab.size, region_id="egm"),
        exporter.export(slab.handle, slab.size, region_id="egm", writable=True),
    ]
    conn.send([numa] + [d.to_bytes() for d in leases])
    writes = conn.recv()  # the reader's ACK with what it wrote: [(local page, slot)]
    expected = _pattern(torch.arange(slab.size, device="cuda", dtype=torch.int64))
    for page, slot in writes:
        for item, (_, size) in enumerate(EGM_ITEMS):
            at = slot * EGM_OBJECT + EGM_ITEM_OFF[item]
            expected[at : at + size] = _egm_local_page(item, page, "cuda")
    torch.cuda.synchronize()
    intact = bool(torch.equal(view, expected))  # written items, and nothing else
    for desc in leases:
        exporter.release_export(desc.lease_id)
    del view
    release_host_numa_allocation(slab)
    conn.send(intact)


def _owner(conn) -> None:
    from cuda.bindings import driver as drv

    from sglang.srt.mem_cache.kv_peer_read import KvPeerExporter
    from sglang.srt.utils.cuda_vmm_utils import (
        align_up,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
        make_rw_access_desc,
        tensor_from_pointer,
    )

    torch.cuda.set_device(0)
    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = make_device_allocation_prop(0, handle_types=int(fabric))
    size = align_up(REGION_BYTES, get_allocation_granularity(prop))
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = int(check_drv(drv.cuMemAddressReserve(size, 0, 0, 0), "reserve"))
    check_drv(drv.cuMemMap(va, size, 0, handle, 0), "map")
    check_drv(drv.cuMemSetAccess(va, size, [make_rw_access_desc(0)], 1), "access")
    view = tensor_from_pointer(va, size, device_id=0)
    view.copy_(_pattern(torch.arange(size, device="cuda", dtype=torch.int64)))
    torch.cuda.synchronize()

    exporter = KvPeerExporter(0)
    # Two leases on one allocation: the reader maps it twice, as two regions.
    descs = [exporter.export(handle, size, region_id="kv-pool") for _ in range(2)]
    conn.send([d.to_bytes() for d in descs])
    assert conn.recv() == "unmapped"  # the reader's ACK: drained, released, unmapped
    for desc in descs:
        exporter.release_export(desc.lease_id)
    assert exporter.outstanding() == 0
    check_drv(drv.cuMemUnmap(va, size), "unmap")
    check_drv(drv.cuMemAddressFree(va, size), "free")
    check_drv(drv.cuMemRelease(handle), "release")
    conn.send("ok")


class TestKvPeerReadCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.cuda_vmm_utils import get_gpu_fabric_clique

        if torch.cuda.device_count() >= 2 and get_gpu_fabric_clique(0) is not None:
            return
        reason = "needs two GPUs in an NVLink fabric clique"
        # The registered runner has one: a skip there hides a broken setup.
        if is_in_ci():
            raise RuntimeError(reason)
        raise unittest.SkipTest(reason)

    def _destinations(self):
        return [
            [
                torch.full((CAPACITY, pb), SENTINEL, dtype=torch.uint8, device="cuda:1")
                for pb in PAGE_BYTES
            ]
            for _ in range(LAYERS)
        ]

    def _group(self, regions, dst, layers, src_pages, dst_pages):
        from sglang.srt.mem_cache.kv_peer_read import PeerBuffer, ReadGroup

        buffers = tuple(
            PeerBuffer(
                src_region=regions[layer % len(regions)],
                src_offset=_src_offset(layer, cls),
                dst_base=dst[layer][cls].data_ptr(),
                page_bytes=pb,
                dst_num_pages=CAPACITY,
            )
            for layer in layers
            for cls, pb in enumerate(PAGE_BYTES)
        )
        return ReadGroup(buffers=buffers, src_pages=src_pages, dst_pages=dst_pages)

    def _check(self, dst, src_pages, dst_pages, layers):
        copied = set(dst_pages.tolist())
        for layer in layers:
            for cls, pb in enumerate(PAGE_BYTES):
                for sp, dp in zip(src_pages.tolist(), dst_pages.tolist()):
                    base = _src_offset(layer, cls) + sp * pb
                    want = _pattern(torch.arange(base, base + pb, device="cuda:1"))
                    self.assertTrue(
                        torch.equal(dst[layer][cls][dp], want), (layer, cls, sp, dp)
                    )
                untouched = [p for p in range(CAPACITY) if p not in copied]
                self.assertTrue(bool((dst[layer][cls][untouched] == SENTINEL).all()))

    def test_egm_page_objects_read_by_layer_and_written(self):
        """Host-memory (HOST_NUMA) slab of page objects: reads take each buffer's item
        at the object stride, layer by layer; writes put local pages into slots
        and touch no other byte, which the owner checks in its own mapping."""
        from sglang.srt.mem_cache.kv_peer_read import (
            KvPeerReader,
            PeerBuffer,
            PeerRegionDescriptor,
            PeerWriteBuffer,
            ReadGroup,
            ReadState,
            WriteGroup,
        )

        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        owner = ctx.Process(target=_egm_owner, args=(child,))
        owner.start()
        try:
            hello = conn.recv()
            if hello is None:
                reason = "GPU 0 reports no host NUMA node"
                if is_in_ci():
                    raise RuntimeError(reason)
                self.skipTest(reason)
            numa, read_lease, write_lease = hello
            reader = KvPeerReader(1, num_blocks=16)
            ro = reader.import_region(PeerRegionDescriptor.from_bytes(read_lease))
            rw = reader.import_region(
                PeerRegionDescriptor.from_bytes(write_lease), writable=True
            )
            local = [
                torch.full(
                    (CAPACITY, size), SENTINEL, dtype=torch.uint8, device="cuda:1"
                )
                for _, size in EGM_ITEMS
            ]
            torch.cuda.synchronize(1)
            layer_buffers = [
                tuple(
                    PeerBuffer(
                        src_region=ro,
                        src_offset=EGM_ITEM_OFF[item],
                        dst_base=local[item].data_ptr(),
                        page_bytes=size,
                        dst_num_pages=CAPACITY,
                        src_stride=EGM_OBJECT,
                    )
                    for item, (layer, size) in enumerate(EGM_ITEMS)
                    if layer == want
                )
                for want in range(LAYERS)
            ]
            rng = np.random.default_rng(1)
            slots = rng.choice(EGM_SLOTS, 24, replace=False)
            pages = rng.choice(CAPACITY, 24, replace=False)
            batch = reader.read([ReadGroup(b, slots, pages) for b in layer_buffers])
            for layer in range(LAYERS):
                status = reader.wait_group(batch, layer)
                self.assertGreater(status.groups_done, layer)
                for item, (owner_layer, size) in enumerate(EGM_ITEMS):
                    if owner_layer != layer:
                        continue
                    for slot, page in zip(slots.tolist(), pages.tolist()):
                        at = slot * EGM_OBJECT + EGM_ITEM_OFF[item]
                        want = _pattern(torch.arange(at, at + size, device="cuda:1"))
                        self.assertTrue(torch.equal(local[item][page], want))
                    untouched = [p for p in range(CAPACITY) if p not in set(pages)]
                    self.assertTrue(bool((local[item][untouched] == SENTINEL).all()))
            self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
            reader.release(batch)

            for item in range(len(EGM_ITEMS)):
                for page in range(CAPACITY):
                    local[item][page].copy_(_egm_local_page(item, page, "cuda:1"))
            src_pages = rng.choice(CAPACITY, 16, replace=False)
            dst_slots = rng.choice(EGM_SLOTS, 16, replace=False)
            writes = WriteGroup(
                tuple(
                    PeerWriteBuffer(
                        dst_region=rw,
                        dst_offset=EGM_ITEM_OFF[item],
                        src_base=local[item].data_ptr(),
                        page_bytes=size,
                        src_num_pages=CAPACITY,
                        dst_stride=EGM_OBJECT,
                    )
                    for item, (_, size) in enumerate(EGM_ITEMS)
                ),
                src_pages,
                dst_slots,
            )
            # The local pages were filled on the current stream: wait for it.
            batch = reader.write([writes], producer_stream=torch.cuda.current_stream(1))
            self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
            reader.release(batch)
            for region in (ro, rw):
                reader.unmap(region)
            reader.close()
            conn.send(list(zip(src_pages.tolist(), dst_slots.tolist())))
            self.assertTrue(conn.recv(), f"slab on NUMA {numa} differs from the writes")
        finally:
            owner.join(timeout=60)
            if owner.is_alive():
                owner.kill()
        self.assertEqual(owner.exitcode, 0)

    def test_export_import_read_release(self):
        from sglang.srt.mem_cache.kv_peer_read import (
            KvPeerReader,
            PeerRegionDescriptor,
            ReadState,
        )

        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        owner = ctx.Process(target=_owner, args=(child,))
        owner.start()
        try:
            # GPU 0 stays current: the reader must not depend on the caller's device.
            torch.cuda.set_device(0)
            reader = KvPeerReader(1, num_blocks=16)
            regions = [
                reader.import_region(PeerRegionDescriptor.from_bytes(d))
                for d in conn.recv()
            ]
            rng = np.random.default_rng(0)

            # Two groups (layers 0-1, layer 2), random pages, layers alternating between
            # the two regions; the rest keep the sentinel.
            dst = self._destinations()
            # No producer stream: the caller vouches the targets are ready.
            torch.cuda.synchronize(1)
            src_pages = rng.choice(CAPACITY, 20, replace=False)
            dst_pages = rng.choice(CAPACITY, 20, replace=False)
            groups = [
                self._group(regions, dst, [0, 1], src_pages, dst_pages),
                self._group(regions, dst, [2], src_pages, dst_pages),
            ]
            batch = reader.read(groups)
            self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
            self.assertTrue(reader.group_done(batch, 0))
            self._check(dst, src_pages, dst_pages, range(LAYERS))
            reader.release(batch)

            # Every page, reversed, behind a producer stream that still writes the targets.
            dst = self._destinations()
            producer = torch.cuda.Stream(device=1)
            with torch.cuda.stream(producer):
                torch.cuda._sleep(10_000_000)
                for layer in dst:
                    for buf in layer:
                        buf.fill_(SENTINEL)
            pages = np.arange(CAPACITY)
            batch = reader.read(
                [
                    self._group(
                        regions[:1], dst, range(LAYERS), pages, pages[::-1].copy()
                    )
                ],
                producer_stream=producer,
            )
            self.assertEqual(reader.wait(batch, timeout=30).state, ReadState.SUCCESS)
            self._check(dst, pages, pages[::-1].copy(), range(LAYERS))
            reader.release(batch)

            for region in regions:
                reader.unmap(region)
            reader.close()
            conn.send("unmapped")
            self.assertEqual(conn.recv(), "ok")
        finally:
            owner.join(timeout=60)
            if owner.is_alive():
                owner.kill()
        self.assertEqual(owner.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
