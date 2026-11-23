"""KvPeerEngine between two processes on real FABRIC allocations, byte-checked.

The CPU contract test fakes the driver; this one covers the borrowed-allocation
lookup, the v1 descriptor round trip, the read-only import and the byte oracle.
"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.slotguard.target import (
    FABRIC_CAPACITY,
    FABRIC_PAGE_BYTES,
    FABRIC_SENTINEL,
    fabric_allocation,
    fabric_bindings,
    fabric_layout,
    fabric_offsets,
    free_fabric_allocation,
    require_fabric_pair,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-c", runner_config="2-gpu-gb300")


def _owner(conn) -> None:
    from sglang.srt.mem_cache.l1_transfer.transport.kv_peer_engine import (
        KvPeerEngine,
    )
    from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

    torch.cuda.set_device(0)
    handle, va, size = fabric_allocation(0)
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
    free_fabric_allocation(handle, va, size)
    conn.send("ok")


class TestKvPeerEngineCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        require_fabric_pair()

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
        # Without our copy of the child end, an owner that dies early gives EOF.
        child.close()
        try:
            torch.cuda.set_device(1)
            handle, va, size = fabric_allocation(1)
            tensor_from_pointer(va, size, device_id=1).fill_(FABRIC_SENTINEL)
            torch.cuda.synchronize(1)
            engine = KvPeerEngine(1, num_blocks=16)
            local = engine.register_vmm(va, size, owner=object())
            self.assertTrue(conn.poll(60), "owner sent no descriptor")
            remote = engine.import_memory(RegionDescriptor.from_dict(conn.recv()))
            layout = fabric_layout()
            src_pages, dst_pages = [5, 1, 63, 2, 40], [3, 4, 9, 63, 0]
            plan = engine.plan_pages(
                layout,
                layout,
                src_pages,
                dst_pages,
                source_capacity=FABRIC_CAPACITY,
                target_capacity=FABRIC_CAPACITY,
            )
            read = engine.bind_pages(
                plan, fabric_bindings(remote), fabric_bindings(local)
            )
            batch = engine.submit_transfer(read)
            status = engine.wait(batch, timeout=30)
            self.assertIs(status.state, TransferState.SUCCESS)
            self.assertTrue(verify_pages(engine, read))
            # The oracle must see a single wrong byte in a copied page.
            dst = tensor_from_pointer(va, size, device_id=1)
            dst[fabric_offsets()[1] + 63 * FABRIC_PAGE_BYTES[1] + 7] ^= 1
            with self.assertRaises(ValueError):
                verify_pages(engine, read)
            untouched = tensor_from_pointer(
                va + 7 * FABRIC_PAGE_BYTES[0], FABRIC_PAGE_BYTES[0], device_id=1
            )
            self.assertTrue(bool((untouched == FABRIC_SENTINEL).all()))
            engine.release_batch(batch)
            engine.unregister_memory(remote)
            engine.close()
            free_fabric_allocation(handle, va, size)
            conn.send("unmapped")
            self.assertTrue(conn.poll(60), "owner did not finish")
            self.assertEqual(conn.recv(), "ok")
        finally:
            owner.join(timeout=60)
            if owner.is_alive():
                owner.kill()
        self.assertEqual(owner.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
