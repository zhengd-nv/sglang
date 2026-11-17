"""One real network shard: async lookup -> HTTP Pin -> FABRIC read -> Unpin.

The TP vote is repeated locally here; the separate Gloo test checks four ranks.
KVCM returns a controlled candidate. All source lease, HTTP, CUDA import/copy,
producer fences and cleanup are real. This is not a GLM deployment gate.
"""

import multiprocessing as mp
import time
import unittest
from array import array
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_kv_peer_engine_cuda as engine_cuda_test
import test_slotguard_allocator as allocator_fixture
import torch
from test_kv_peer_engine_cuda import (
    CAPACITY,
    PAGE_BYTES,
    SENTINEL,
    _bindings,
    _fabric_allocation,
    _free_allocation,
    _layout,
    _offsets,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-c", runner_config="2-gpu-gb300")
TOKEN = "test-only-slotguard-linker-token"


def identity(name):
    from sglang.srt.mem_cache.l1_transfer.protocol import EngineIdentity

    return EngineIdentity(
        name,
        name + "-boot",
        "fabric",
        "instance",
        "model",
        "layout",
        name + "-tp",
        4,
        0,
        "shard-0",
    )


def adapter(engine, memory):
    from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import (
        SGLangPoolAdapter,
    )

    # Only pool geometry is synthetic; memory is a real registered FABRIC map.
    out = object.__new__(SGLangPoolAdapter)
    out.engine, out.capacity, out.layout = engine, CAPACITY, _layout()
    out.memories, out.bindings = [memory], _bindings(memory)
    return out


def source_process(conn):
    from sglang.srt.mem_cache.l1_transfer.control_http import HTTPControlGateway
    from sglang.srt.mem_cache.l1_transfer.control_profile import ControlProfile
    from sglang.srt.mem_cache.l1_transfer.metadata import root_event_keys
    from sglang.srt.mem_cache.l1_transfer.slot_guard import SlotGuard
    from sglang.srt.mem_cache.l1_transfer.slotguard_catalog import (
        Publication,
        PublicationCatalog,
        PublicationScope,
    )
    from sglang.srt.mem_cache.l1_transfer.slotguard_service import SlotGuardService
    from sglang.srt.mem_cache.l1_transfer.transport.kv_peer_engine import KvPeerEngine
    from sglang.srt.mem_cache.l1_transfer.transport_role_arbiter import (
        TransportRoleArbiter,
    )
    from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

    torch.cuda.set_device(0)
    handle, va, size = _fabric_allocation(0)
    view = tensor_from_pointer(va, size, device_id=0)
    view.zero_()
    for i, (pb, off) in enumerate(zip(PAGE_BYTES, _offsets())):
        for page in (1, 2, 3):
            view[off + page * pb : off + (page + 1) * pb].fill_(32 + page + i)
    torch.cuda.synchronize(0)
    engine = KvPeerEngine(0)
    pool = adapter(engine, engine.register_vmm(va, size, owner=view))
    guard = SlotGuard(
        1,
        identity("source").incarnation,
        max_remote_pinned_pages=32,
        max_active_leases=16,
        max_pending_reclaim_pages=32,
        shared_headroom_credit=True,
    )
    # An empty first epoch initializes the shared headroom credit.
    guard.begin_epoch(0)
    guard.install_epoch(0, ())
    guard.finish_epoch(0, verified_headroom_pages=32)
    refs = guard.register_allocation((1, 2, 3))
    catalog = PublicationCatalog(guard)
    catalog.submit(
        Publication(
            PublicationScope("model", "layout", 64, 1),
            None,
            tuple(tuple(range(start, start + 64)) for start in (0, 64, 128)),
            root_event_keys(tuple(range(192)), 64),
            refs,
            NS(ready=lambda: True),
        )
    )
    catalog.progress()
    arbiter = TransportRoleArbiter()
    service = SlotGuardService(
        guard=guard,
        catalog=catalog,
        local_identity=identity("source"),
        exact_peers=(identity("target"),),
        engine=engine,
        adapter=pool,
        arbiter=arbiter,
    )
    gateway = HTTPControlGateway(
        identity("source"),
        (identity("target"),),
        bind_address=("127.0.0.1", 0),
        bearer_token=TOKEN,
        trusted_private_domain=True,
        control_profile=ControlProfile.SLOTGUARD_HASH_V3,
        slotguard_service=service,
    )
    gateway.start()
    conn.send(f"http://127.0.0.1:{gateway.address[1]}")
    assert conn.poll(90) and conn.recv() == "drained"
    deadline = time.monotonic() + 5
    while service.external_snapshot().owners:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    gateway.close()
    service.close()
    arbiter.close()
    pool.close()
    engine.close()
    _free_allocation(handle, va, size)
    conn.send("released")


class TestSlotGuardLinkerCuda(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        engine_cuda_test.TestKvPeerEngineCuda.setUpClass()

    def test_network_read_and_release(self):
        from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
        from sglang.srt.mem_cache.l1_transfer.control_http import (
            HTTPPrefixControlClient,
        )
        from sglang.srt.mem_cache.l1_transfer.control_profile import ControlProfile
        from sglang.srt.mem_cache.l1_transfer.directory import (
            EngineDirectory,
            EngineDirectoryEntry,
        )
        from sglang.srt.mem_cache.l1_transfer.kvcm_client import KVCMClient
        from sglang.srt.mem_cache.l1_transfer.nvlink_backend import (
            NVLinkTransferBackend,
        )
        from sglang.srt.mem_cache.l1_transfer.protocol import LookupResult
        from sglang.srt.mem_cache.l1_transfer.role_gate import RoleGate
        from sglang.srt.mem_cache.l1_transfer.slotguard_linker_transport import (
            SlotGuardLinkerTransport,
        )
        from sglang.srt.mem_cache.l1_transfer.tp_group import SchedulerCollective
        from sglang.srt.mem_cache.l1_transfer.transport.kv_peer_engine import (
            KvPeerEngine,
        )
        from sglang.srt.mem_cache.l1_transfer.transport_role_arbiter import (
            TransportRoleArbiter,
        )
        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            PrefetchContext,
        )
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        source = ctx.Process(target=source_process, args=(child,))
        source.start()
        try:
            self.assertTrue(conn.poll(90), "source did not start")
            endpoint = conn.recv()
            torch.cuda.set_device(1)
            handle, va, size = _fabric_allocation(1)
            view = tensor_from_pointer(va, size, device_id=1)
            view.fill_(SENTINEL)
            engine = KvPeerEngine(1, num_blocks=16)
            pool = adapter(engine, engine.register_vmm(va, size, owner=view))
            directory = EngineDirectory(
                identity("target"),
                (EngineDirectoryEntry("source", identity("source"), endpoint),),
            )
            client = HTTPPrefixControlClient(
                directory,
                bearer_token=TOKEN,
                trusted_private_domain=True,
                control_profile=ControlProfile.SLOTGUARD_HASH_V3,
            )
            kvcm = KVCMClient(None, directory)
            kvcm.lookup = lambda query: LookupResult(
                query, identity("source"), 192, query.event_block_keys
            )
            backend = NVLinkTransferBackend(
                engine,
                pool,
                directory,
                kvcm,
                client,
                RoleGate(),
                group_coordinated=True,
                control_profile=ControlProfile.SLOTGUARD_HASH_V3,
                transport_arbiter=TransportRoleArbiter(),
                async_unpin=True,
                verify_bytes=True,
                rpc_timeout_s=10,
            )
            transport = SlotGuardLinkerTransport(
                backend,
                SchedulerCollective(lambda value: [value] * 4, 0),
                forward_stream=torch.cuda.current_stream(1),
            )
            context = PrefetchContext(
                request=CacheRequestHandle("cuda-read", 0),
                token_ids=array("q", range(193)).tobytes(),
                local_hit_tokens=64,
                max_restore_tokens=192,
                page_size=64,
            )
            lookup = transport.submit_lookup(context)
            transport.lookups[context.request].job.result(10)
            transport.poll()
            self.assertEqual(lookup.result(0), [1, 2])
            future = transport.prefetch(
                context.request,
                NS(device_indices=torch.arange(256, 384, device="cuda:1")),
            )
            deadline = time.monotonic() + 45
            while not future.done():
                transport.poll()
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.001)
            self.assertTrue(future.result())
            for i, (pb, off) in enumerate(zip(PAGE_BYTES, _offsets())):
                for src, dst in ((2, 4), (3, 5)):
                    self.assertTrue(
                        bool(
                            (
                                view[off + dst * pb : off + (dst + 1) * pb]
                                == 32 + src + i
                            ).all()
                        )
                    )
                self.assertTrue(
                    bool((view[off + 7 * pb : off + 8 * pb] == SENTINEL).all())
                )
            transport.release(context.request)
            deadline = time.monotonic() + 15
            while True:
                try:
                    transport.close()
                    break
                except RuntimeError:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.001)
            client.close()
            pool.close()
            engine.close()
            _free_allocation(handle, va, size)
            conn.send("drained")
            self.assertTrue(conn.poll(10))
            self.assertEqual(conn.recv(), "released")
        finally:
            source.join(5)
            if source.is_alive():
                source.kill()
                source.join(5)
        self.assertEqual(source.exitcode, 0)


class TestCudaAllocatorIdentity(CustomTestCase):
    def test_cuda_free_uses_no_device_readback_or_global_synchronize(self):
        alloc = allocator_fixture.allocator(pages=4)
        # Construct the real CUDA allocator with the same CPU ledger contract.
        alloc = allocator_fixture.SlotGuardPagedTokenToKVPoolAllocator(
            16,
            4,
            torch.float16,
            "cuda",
            None,
            True,
            slot_guard=alloc.slot_guard,
        )
        indices, span, refs = allocator_fixture.owned(alloc, 2)
        alloc.slot_guard.try_pin(allocator_fixture.identity(), refs[:1])
        with (
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
            patch.object(
                torch.cuda,
                "synchronize",
                side_effect=AssertionError("device-wide wait"),
            ),
        ):
            allocator_fixture.epoch(alloc)
            alloc.free(indices, page_span=span)
            self.assertEqual(alloc.conservation_counts(), (0, 3, 1, 4))
        torch.cuda.synchronize()
        self.assertEqual(set(alloc.get_all_free_pages().cpu().tolist()), {2, 3, 4})


if __name__ == "__main__":
    unittest.main()
