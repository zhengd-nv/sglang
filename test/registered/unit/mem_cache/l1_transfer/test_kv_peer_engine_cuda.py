"""KvPeerEngine between two processes on real FABRIC allocations, byte-checked.

The CPU contract test fakes the driver; this one covers the borrowed-allocation
lookup, the v1 descriptor round trip, the read-only import and the byte oracle.
"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cuda_ci(est_time=40, stage="base-c", runner_config="2-gpu-gb300")

CAPACITY = 64
PAGE_BYTES = (36864, 8448)  # GLM-5.2 MLA and DSA index pages
SENTINEL = 0xEE


def _offsets():
    first = CAPACITY * PAGE_BYTES[0]
    return (0, (first + 255) // 256 * 256)


def _fabric_allocation(device):
    from cuda.bindings import driver as drv

    from sglang.srt.utils.cuda_vmm_utils import (
        align_up,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
        make_rw_access_desc,
    )

    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = make_device_allocation_prop(device, handle_types=int(fabric))
    need = _offsets()[1] + CAPACITY * PAGE_BYTES[1]
    size = align_up(need, get_allocation_granularity(prop))
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = int(check_drv(drv.cuMemAddressReserve(size, 0, 0, 0), "reserve"))
    check_drv(drv.cuMemMap(va, size, 0, handle, 0), "map")
    check_drv(drv.cuMemSetAccess(va, size, [make_rw_access_desc(device)], 1), "access")
    return handle, va, size


def _free_allocation(handle, va, size):
    from cuda.bindings import driver as drv

    from sglang.srt.utils.cuda_vmm_utils import check_drv

    check_drv(drv.cuMemUnmap(va, size), "unmap")
    check_drv(drv.cuMemAddressFree(va, size), "free")
    check_drv(drv.cuMemRelease(handle), "release")


def _layout():
    from sglang.srt.mem_cache.l1_transfer.transport.layout import (
        BufferLayout,
        PoolLayout,
    )

    return PoolLayout(
        "test",
        64,
        tuple(BufferLayout(f"b{i}", pb, "test") for i, pb in enumerate(PAGE_BYTES)),
    )


def _bindings(memory):
    from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan

    return {
        f"b{i}": (RegionSpan(0, CAPACITY * pb, memory, off),)
        for i, (pb, off) in enumerate(zip(PAGE_BYTES, _offsets()))
    }


def _owner(conn) -> None:
    from sglang.srt.mem_cache.l1_transfer.transport.kv_peer_engine import (
        KvPeerEngine,
    )
    from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

    torch.cuda.set_device(0)
    handle, va, size = _fabric_allocation(0)
    view = tensor_from_pointer(va, size, device_id=0)
    view.copy_(torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda:0"))
    torch.cuda.synchronize()
    engine = KvPeerEngine(0)
    keepalive = object()
    try:
        engine.register_vmm(va + 4096, size - 4096, owner=keepalive)
    except ValueError:
        pass
    else:
        raise AssertionError("an interior pointer must not register")
    memory = engine.register_vmm(va, size, owner=keepalive)
    desc = engine.export_memory(memory)
    conn.send(desc.to_dict())
    assert conn.recv() == "unmapped"
    engine.release_export(desc.lease_id)
    engine.close()  # releases the retained allocation handle
    _free_allocation(handle, va, size)
    conn.send("ok")


class TestKvPeerEngineCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils.cuda_vmm_utils import get_gpu_fabric_clique

        if torch.cuda.device_count() >= 2 and get_gpu_fabric_clique(0) is not None:
            return
        reason = "needs two GPUs in an NVLink fabric clique"
        if is_in_ci():
            raise RuntimeError(reason)
        raise unittest.SkipTest(reason)

    def test_slotguard_read_round_trip(self):
        from sglang.srt.mem_cache.l1_transfer.transport.api import (
            RegionDescriptor,
            TransferState,
        )
        from sglang.srt.mem_cache.l1_transfer.transport.kv_peer_engine import (
            KvPeerEngine,
            verify_pages,
        )
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        owner = ctx.Process(target=_owner, args=(child,))
        owner.start()
        try:
            torch.cuda.set_device(1)
            handle, va, size = _fabric_allocation(1)
            tensor_from_pointer(va, size, device_id=1).fill_(SENTINEL)
            torch.cuda.synchronize(1)
            engine = KvPeerEngine(1, num_blocks=16)
            local = engine.register_vmm(va, size, owner=object())
            remote = engine.import_memory(RegionDescriptor.from_dict(conn.recv()))
            layout = _layout()
            src_pages, dst_pages = [5, 1, 63, 2, 40], [3, 4, 9, 63, 0]
            plan = engine.plan_pages(
                layout,
                layout,
                src_pages,
                dst_pages,
                source_capacity=CAPACITY,
                target_capacity=CAPACITY,
            )
            read = engine.bind_pages(plan, _bindings(remote), _bindings(local))
            batch = engine.submit_transfer(read)
            status = engine.wait(batch, timeout=30)
            self.assertIs(status.state, TransferState.SUCCESS)
            self.assertTrue(verify_pages(engine, read))
            # The oracle must see a single wrong byte in a copied page.
            dst = tensor_from_pointer(va, size, device_id=1)
            dst[_offsets()[1] + 63 * PAGE_BYTES[1] + 7] ^= 1
            with self.assertRaises(ValueError):
                verify_pages(engine, read)
            untouched = tensor_from_pointer(
                va + 7 * PAGE_BYTES[0], PAGE_BYTES[0], device_id=1
            )
            self.assertTrue(bool((untouched == SENTINEL).all()))
            engine.release_batch(batch)
            engine.unregister_memory(remote)
            engine.close()
            _free_allocation(handle, va, size)
            conn.send("unmapped")
            self.assertEqual(conn.recv(), "ok")
        finally:
            owner.join(timeout=60)
            if owner.is_alive():
                owner.kill()
        self.assertEqual(owner.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
