"""Four real target rendezvous + independent source services; CPU driver only."""

import importlib
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from unittest.mock import patch

import test_slotguard_service as fixture
from test_nvlink_backend import pool
from test_protocol import PACKAGE, identity, p, register_cpu_ci

from sglang.test.test_utils import CustomTestCase

b = importlib.import_module(f"{PACKAGE}.nvlink_backend")
d = importlib.import_module(f"{PACKAGE}.directory")
k = importlib.import_module(f"{PACKAGE}.kvcm_client")
l = importlib.import_module(f"{PACKAGE}.role_gate")
g = importlib.import_module(f"{PACKAGE}.tp_group")
profiles = importlib.import_module(f"{PACKAGE}.control_profile")
V3 = profiles.ControlProfile.SLOTGUARD_HASH_V3
v = fixture.v
register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class DirectClient:
    def __init__(self, service):
        self.service = service
        self.hold_pin = self.hold_ack = False
        self.held = self.held_ack = None
        self.calls = []

    def pin(self, endpoint, request):
        self.calls.append(("pin", request))
        out = Future()

        def done(future):
            value = future.result()
            if self.hold_pin:
                self.held = (out, value)
            else:
                out.set_result(value)

        self.service.submit(request).add_done_callback(done)
        return out

    def unpin(self, endpoint, request):
        self.calls.append(("unpin", request))
        out = Future()

        def done(future):
            value = future.result()
            if self.hold_ack:
                self.held_ack = (out, value)
            else:
                out.set_result(value)

        self.service.submit(request).add_done_callback(done)
        return out


def wait_source_quiescent(source, *, timeout_s=2):
    """ACK can wake a caller before the service releases accepted-work ownership.

    Read the authoritative proof before fixture resource teardown. Never issue
    another Unpin, mutate ownership, or accept a nonempty service on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            source.assert_quiescent()
            return
        except fixture.svc.ServiceBusy:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.001, remaining))


class Rank:
    def __init__(self, rank):
        self.base = fixture.ServiceTest()
        self.base.setUp()
        self.base.service.close()
        source = identity("source", tp_size=4, tp_rank=rank)
        target = identity(tp_size=4, tp_rank=rank)
        self.base.service = fixture.svc.SlotGuardService(
            guard=self.base.g,
            catalog=self.base.catalog,
            local_identity=source,
            exact_peers=(target,),
            engine=self.base.engine,
            adapter=self.base.adapter,
            arbiter=self.base.arbiter,
        )
        self.source = self.base.service
        initial = self.base.lookup()
        self.result = replace(
            initial, source=source, query=replace(initial.query, target=target)
        )
        self.driver = self.base.engine._driver
        self.driver.state = 1
        self.engine = fixture.NvlinkEngine(1, _driver=self.driver)
        self.adapter = fixture.SGLangPoolAdapter(pool(1), self.engine)
        directory = d.EngineDirectory(
            target, (d.EngineDirectoryEntry("source", source, "http://source:1234"),)
        )
        self.client = DirectClient(self.source)
        self.gate = l.RoleGate()
        self.gate.acquire(l.EngineRole.INBOUND, self.result.transfer_id)
        self.identity = None
        self.backend = b.NVLinkTransferBackend(
            self.engine,
            self.adapter,
            directory,
            k.KVCMClient(None, directory),
            self.client,
            self.gate,
            control_profile=V3,
            group_coordinated=True,
            rpc_timeout_s=0.5,
        )
        self.slots = p.PrivateSlotView(
            self.result.transfer_id, 64, 192, 64, tuple(range(256, 384)), (4, 5)
        )

    def status(self):
        return self.backend.poll_prefetch(self.result.transfer_id)

    def close(self):
        status = self.status()
        if status is p.PrefetchStatus.ACKED:
            status = self.backend.cancel_prefetch(self.result.transfer_id)
        if status in (p.PrefetchStatus.READY, p.PrefetchStatus.FAILED_SAFE):
            self.backend.retire(self.result.transfer_id)
            self.backend.close()
        else:
            # Local test reconciliation uses the authoritative source query.
            # It does not retire or reuse the quarantined backend's slots.
            for kind, req in self.client.calls:
                if kind == "pin":
                    state = self.source.query(req.identity)
                    if state is not None and state.state == "PREPARED":
                        self.source.submit(
                            v.UnpinReq(
                                req.reference,
                                state.lease_id,
                                True,
                                True,
                                req.protocol_version,
                                req.capability,
                            )
                        ).result(2)
        wait_source_quiescent(self.source)
        self.adapter.close()
        self.engine.close()
        self.base.tearDown()


class SourceQuiescenceFixtureTest(CustomTestCase):
    def test_future_done_callback_can_precede_accepted_work_release(self):
        base = fixture.ServiceTest()
        base.setUp()
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        try:
            response = base.service.submit(base.pin()).result(2)
            original = base.service._unpin
            worker_entered, dispatch = threading.Event(), threading.Event()

            def unpin(request):
                worker_entered.set()
                self.assertTrue(dispatch.wait(2))
                return original(request)

            with patch.object(base.service, "_unpin", side_effect=unpin):
                future = base.service.submit(base.unpin(response))
                self.assertTrue(worker_entered.wait(2))

                def done(_future):
                    entered.set()
                    self.assertTrue(release.wait(2))

                future.add_done_callback(done)
                dispatch.set()
                future.result(2)
                self.assertTrue(entered.wait(2))
                self.assertEqual(base.service.owner_snapshot().requests, 1)
                with self.assertRaises(fixture.svc.ServiceBusy):
                    wait_source_quiescent(base.service, timeout_s=0.01)

                waiter = threading.Thread(
                    target=lambda: (wait_source_quiescent(base.service), finished.set())
                )
                waiter.start()
                self.assertFalse(finished.wait(0.01))
                release.set()
                waiter.join(2)
                self.assertFalse(waiter.is_alive())
                self.assertTrue(finished.is_set())
                self.assertEqual(base.service.owner_snapshot().requests, 0)
        finally:
            release.set()
            wait_source_quiescent(base.service)
            base.tearDown()

    def test_live_lease_times_out_and_unrelated_errors_are_not_suppressed(self):
        base = fixture.ServiceTest()
        base.setUp()
        try:
            response = base.service.submit(base.pin()).result(2)
            with self.assertRaises(fixture.svc.ServiceBusy):
                wait_source_quiescent(base.service, timeout_s=0.01)
            self.assertEqual(base.service.owner_snapshot().leases, 1)
            with (
                patch.object(
                    base.service,
                    "assert_quiescent",
                    side_effect=RuntimeError("fixture defect"),
                ),
                self.assertRaisesRegex(RuntimeError, "fixture defect"),
            ):
                wait_source_quiescent(base.service)
        finally:
            base.service.submit(base.unpin(response)).result(2)
            wait_source_quiescent(base.service)
            base.tearDown()


class SlotGuardBackendTest(CustomTestCase):
    def setUp(self):
        self.ranks = [Rank(rank) for rank in range(4)]
        self.decisions = []

    def tearDown(self):
        for rank in self.ranks:
            rank.close()

    def start(self):
        for rank in self.ranks:
            rank.backend.prefetch(rank.result, rank.slots, identity=rank.identity)

    def tick(self):
        snapshots = [
            rank.backend.group_snapshot(rank.result.transfer_id) for rank in self.ranks
        ]
        decision = g.worker_decision(snapshots)
        if decision == "QUARANTINE":
            return
        if decision:
            if not self.decisions or self.decisions[-1] != decision:
                self.decisions.append(decision)
            if decision in ("COPY", "ABORT"):
                self.assertEqual([r.driver.submits for r in self.ranks], [0] * 4)
                self.assertTrue(all(s[0] == "PIN" for s in snapshots))
            if decision == "RELEASE":
                self.assertFalse(
                    any(
                        kind == "unpin"
                        for rank in self.ranks
                        for kind, _ in rank.client.calls
                    )
                )
                self.assertTrue(
                    all(
                        not rank.backend._records[rank.result.transfer_id].imports
                        for rank in self.ranks
                    )
                )
            for rank in self.ranks:
                rank.backend.group_permit(rank.result.transfer_id, decision)
        statuses = [rank.status() for rank in self.ranks]
        if all(status is p.PrefetchStatus.ACKED for status in statuses):
            for rank in self.ranks:
                rank.backend.group_commit_ready(rank.result.transfer_id)

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.tick()
            if predicate():
                return
            time.sleep(0.001)
        self.fail(
            str(
                [
                    (r.status(), r.backend.group_snapshot(r.result.transfer_id))
                    for r in self.ranks
                ]
            )
        )

    def assert_rejected_cohort(self):
        self.start()
        self.until(
            lambda: all(r.status() is p.PrefetchStatus.FAILED_SAFE for r in self.ranks)
        )
        self.assertIn("ABORT", self.decisions)
        self.assertNotIn("COPY", self.decisions)
        self.assertEqual([r.driver.submits for r in self.ranks], [0] * 4)
        self.assertEqual(
            [sum(kind == "unpin" for kind, _ in r.client.calls) for r in self.ranks],
            [1, 1, 1, 0],
        )
        self.assertTrue(
            all(r.base.g.owner_counts().active_leases == 0 for r in self.ranks)
        )

    def test_three_prepared_one_miss_aborts_before_import_and_dma(self):
        # An unpublished source still validates full tokens/keys, then returns MISS.
        self.ranks[3].base.catalog.lookup = lambda request: None
        self.assert_rejected_cohort()

    def test_three_prepared_one_busy_aborts_before_import_and_dma(self):
        self.ranks[3].source.fence_pins(1)
        self.assert_rejected_cohort()

    def test_all_prepared_copy_drain_unmap_ack_then_all_ready(self):
        self.ranks[3].client.hold_ack = True
        self.start()
        self.until(lambda: self.ranks[3].client.held_ack is not None)
        self.assertEqual([r.driver.submits for r in self.ranks], [1] * 4)
        self.assertNotIn(p.PrefetchStatus.READY, [r.status() for r in self.ranks])
        pending, ack = self.ranks[3].client.held_ack
        self.ranks[3].client.hold_ack = False
        pending.set_result(ack)
        self.until(
            lambda: all(r.status() is p.PrefetchStatus.READY for r in self.ranks)
        )
        self.assertEqual(self.decisions, ["PIN", "COPY", "RELEASE"])
        self.assertTrue(
            all(r.base.g.owner_counts().active_leases == 0 for r in self.ranks)
        )
        self.assertTrue(
            all(
                not r.backend._records[r.result.transfer_id].imports for r in self.ranks
            )
        )

    def test_lost_pin_reply_is_unknown_and_quarantines_all_without_dma(self):
        self.ranks[3].client.hold_pin = True
        self.start()
        self.until(
            lambda: any(r.status() is p.PrefetchStatus.QUARANTINED for r in self.ranks)
        )
        self.assertEqual(
            g.worker_decision(
                [r.backend.group_snapshot(r.result.transfer_id) for r in self.ranks]
            ),
            "QUARANTINE",
        )
        self.assertEqual([r.driver.submits for r in self.ranks], [0] * 4)
        self.assertTrue(
            all(r.base.g.owner_counts().active_leases == 1 for r in self.ranks)
        )
        self.assertFalse(
            any(kind == "unpin" for r in self.ranks for kind, _ in r.client.calls)
        )
        for rank in self.ranks:
            self.assertIsNotNone(rank.backend._live)


if __name__ == "__main__":
    unittest.main()
