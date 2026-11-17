"""Four-process TP bridge with real SlotGuard services and backend workers.

Only CUDA memory/driver and KVCM discovery are substituted. Owner collectives,
Pin/Unpin, producer handoff, role arbitration and backend drain/retire are real.
"""

import importlib
import tempfile
import time
import unittest
from array import array
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_protocol import PACKAGE, register_cpu_ci
from test_slotguard_linker_transport import Handle

from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def worker(rank, rendezvous, cases=("success", "abort")):
    import torch
    import torch.distributed as dist
    from test_slotguard_backend import Rank, wait_source_quiescent

    t = importlib.import_module(f"{PACKAGE}.slotguard_linker_transport")
    a = importlib.import_module(f"{PACKAGE}.transport_role_arbiter")
    g = importlib.import_module(f"{PACKAGE}.tp_group")
    l = importlib.import_module(f"{PACKAGE}.role_gate")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        for case in cases:
            fixture = Rank(rank)
            backend = fixture.backend
            backend._gate.release(l.EngineRole.INBOUND, fixture.result.transfer_id)
            backend.transport_arbiter = a.TransportRoleArbiter()
            backend._kvcm.lookup = lambda query: replace(fixture.result, query=query)
            controller = t.SlotGuardLinkerTransport(
                backend,
                g.SchedulerCollective.from_process_group(dist.group.WORLD, rank),
            )
            query = fixture.result.query
            context = NS(
                request=Handle(query.transfer_id.rid, query.transfer_id.attempt),
                token_ids=array("q", query.prefix_token_ids).tobytes(),
                local_hit_tokens=query.local_hit_tokens,
                max_restore_tokens=query.max_restore_tokens,
                page_size=query.page_size,
                extra_key=None,
                cache_salt=None,
                is_bigram=False,
            )
            with patch.object(
                t,
                "event_keys_and_hashes",
                return_value=(query.event_block_keys, query.prefix_hashes),
            ):
                lookup = controller.submit_lookup(context)
                controller.lookups[context.request].job.result(3)
            controller.poll()
            lookup.result(0)
            if rank == 2:
                if case == "role_busy":
                    assert backend._gate.acquire(
                        l.EngineRole.OUTBOUND, fixture.result.transfer_id
                    )
                elif case == "enqueue_refused":
                    backend._max_records = 0
                elif case == "source_paused":
                    fixture.source.pause_for_maintenance(1)
            if case == "peer_disagreement" and rank == 2:
                lookup = controller.lookups[context.request]
                lookup.result = replace(
                    lookup.result,
                    source=replace(
                        lookup.result.source, incarnation="another-peer-boot"
                    ),
                )
            start = 257 if case == "bad_pages" and rank == 2 else 256
            future = controller.prefetch(
                context.request, NS(device_indices=torch.arange(start, start + 128))
            )
            if case in ("role_busy", "peer_disagreement"):
                assert future is None
                assert not controller.disabled
                if rank == 2 and case == "role_busy":
                    backend._gate.release(
                        l.EngineRole.OUTBOUND, fixture.result.transfer_id
                    )
            else:
                assert future is not None
            # Only one rank cancels. Every rank must still enter identical
            # rendezvous and report the same terminal publication decision.
            if case == "abort" and rank == 2:
                controller.cancel(context.request)
            deadline = time.monotonic() + 10
            while future is not None and not future.done():
                controller.poll()
                assert time.monotonic() < deadline, "transport did not converge"
                time.sleep(0.001)
            if future is not None:
                assert future.result() is (case == "success"), case
            if case in (
                "role_busy",
                "peer_disagreement",
                "enqueue_refused",
                "bad_pages",
            ):
                assert not fixture.client.calls, (case, fixture.client.calls)
            assert backend._gate.role is l.EngineRole.IDLE
            assert backend.transport_arbiter.owner_snapshot().owners == 0
            controller.release(context.request)
            deadline = time.monotonic() + 3
            while True:
                try:
                    controller.close()
                    break
                except RuntimeError:
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
            # The controller has already retired the transfer and closed the
            # backend. Only source and driver fixture resources remain.
            wait_source_quiescent(fixture.source)
            fixture.adapter.close()
            fixture.engine.close()
            fixture.base.tearDown()
            dist.barrier()
    finally:
        dist.destroy_process_group()


class GlooTest(CustomTestCase):
    def test_tp4_success_and_single_rank_abort_converge(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="l1-linker-gloo-") as directory:
            mp.spawn(
                worker, args=(str(Path(directory) / "store"),), nprocs=4, join=True
            )

    def test_single_rank_refusals_release_all_owners(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="l1-linker-refusal-") as directory:
            mp.spawn(
                worker,
                args=(
                    str(Path(directory) / "store"),
                    (
                        "role_busy",
                        "peer_disagreement",
                        "enqueue_refused",
                        "source_paused",
                        "bad_pages",
                    ),
                ),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
