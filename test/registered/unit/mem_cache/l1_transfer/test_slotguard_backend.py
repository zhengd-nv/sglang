"""Four real target rendezvous + independent source services; CPU driver only."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer import slotguard_catalog as c
from sglang.srt.mem_cache.l1_transfer import slotguard_receipts as receipts
from sglang.srt.mem_cache.l1_transfer import slotguard_service as svc
from sglang.srt.mem_cache.l1_transfer import tp_group as g
from sglang.srt.mem_cache.l1_transfer.observation import TransferObserver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.source import Ready, ServiceFixture
from sglang.test.kits.slotguard.target import BackendCohort, wait_source_quiescent
from sglang.test.kits.slotguard.wire import keys
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class SourceQuiescenceFixtureTest(CustomTestCase):
    def test_future_done_callback_can_precede_accepted_work_release(self):
        base = ServiceFixture()
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
                with self.assertRaises(svc.ServiceBusy):
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
        base = ServiceFixture()
        base.setUp()
        try:
            response = base.service.submit(base.pin()).result(2)
            with self.assertRaises(svc.ServiceBusy):
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


class SlotGuardBackendTest(BackendCohort, CustomTestCase):
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


class ReceiptChainTest(BackendCohort, CustomTestCase):
    """Source and target receipts of one TP4 transfer, read back from the observers."""

    def setUp(self):
        BackendCohort.setUp(self)
        self.directory = tempfile.TemporaryDirectory()
        self.observers = []
        for index, rank in enumerate(self.ranks):
            source = TransferObserver(
                str(Path(self.directory.name) / f"source{index}"),
                f"source{index}",
                capacity=64,
                tp_rank=index,
                tp_group_id="source-tp",
            )
            target = TransferObserver(
                str(Path(self.directory.name) / f"target{index}"),
                f"target{index}",
                capacity=64,
                tp_rank=index,
                tp_group_id="target-tp",
            )
            self.observers.extend((source, target))
            rank.source.observer = source
            rank.backend.observer = target
            catalog = c.PublicationCatalog(
                rank.base.g, event_keys=keys, observer=source
            )
            ready = Ready()
            ready.value = True
            catalog.submit(
                c.Publication(
                    c.PublicationScope("model-1", "layout-1", 64, 1),
                    None,
                    tuple(tuple(range(start, start + 64)) for start in (0, 64, 128)),
                    keys(tuple(range(192)), 64),
                    rank.base.refs,
                    ready,
                )
            )
            catalog.progress()
            rank.base.catalog = rank.source.catalog = catalog

    def tearDown(self):
        try:
            BackendCohort.tearDown(self)
        finally:
            for observer in self.observers:
                observer.close()
            self.directory.cleanup()

    def events(self, rank, source=False):
        observer = self.observers[rank * 2 + (0 if source else 1)]
        return [
            event
            for event in observer.snapshot()["events"]
            if event["phase"].startswith("slotguard_")
        ]

    def test_full_ordered_chain_exact_same_rank_binding_and_all_four_ready(self):
        self.ranks[3].client.hold_ack = True
        owner = threading.get_ident()
        real_digest = receipts.ordered_page_digest

        def off_owner(pages):
            self.assertNotEqual(threading.get_ident(), owner)
            return real_digest(pages)

        with patch.object(receipts, "ordered_page_digest", side_effect=off_owner):
            self.start()
            self.until(lambda: self.ranks[3].client.held_ack is not None)
            self.assertFalse(
                any(
                    e["phase"] == "slotguard_group_ready"
                    for i in range(4)
                    for e in self.events(i)
                )
            )
            future, ack = self.ranks[3].client.held_ack
            self.ranks[3].client.hold_ack = False
            future.set_result(ack)
            self.until(
                lambda: all(r.status() is p.PrefetchStatus.READY for r in self.ranks)
            )
        phases = [
            "target_pin_validated",
            "group_copy",
            "target_dma_drained",
            "target_imports_unmapped",
            "target_unpin_ack",
            "group_ready",
        ]
        binding = (
            "profile",
            "control_version",
            "data_abi_version",
            "pool_epoch",
            "page_count",
            "page_set_digest",
            "content_digest",
            "identity_digest",
            "lease_digest",
            "source_rank",
            "target_rank",
            "rid",
            "attempt",
        )
        for rank in range(4):
            events = self.events(rank)
            self.assertEqual(
                [e["phase"] for e in events], ["slotguard_" + p for p in phases]
            )
            prepared = next(
                e
                for e in self.events(rank, True)
                if e["phase"] == "slotguard_source_prepared"
            )
            self.assertEqual(prepared["pin_generation"], 0)
            for event in events:
                self.assertEqual(
                    {k: event[k] for k in binding}, {k: prepared[k] for k in binding}
                )
                self.assertNotIn("pin_generation", event)
            self.assertEqual(events[1]["ranks"], 4)
            self.assertEqual(events[-1]["decision"], "READY")
        for observer in self.observers:
            state = observer.snapshot()
            self.assertEqual((state["errors"], state["dropped"]), (0, 0))

    def test_three_prepared_one_miss_never_emits_copy_or_dma(self):
        self.ranks[3].source.catalog.lookup = lambda result: None
        self.assert_rejected_cohort()
        for rank in range(4):
            self.assertFalse(
                any(
                    e["phase"]
                    in (
                        "slotguard_group_copy",
                        "slotguard_target_dma_drained",
                        "slotguard_group_ready",
                    )
                    for e in self.events(rank)
                )
            )

    def test_lost_unpin_ack_retains_accepted_unknown_owner_after_unmap(self):
        self.ranks[3].client.hold_ack = True
        self.start()
        self.until(lambda: self.ranks[3].status() is p.PrefetchStatus.QUARANTINED)
        state = self.ranks[3].backend.owner_snapshot()
        self.assertEqual(
            (state.accepted, state.unknown, state.imports, state.batches), (1, 1, 0, 0)
        )
        self.assertFalse(
            any(
                e["phase"] in ("slotguard_target_unpin_ack", "slotguard_group_ready")
                for e in self.events(3)
            )
        )
        self.assertTrue(all(r.backend.owner_snapshot().owners for r in self.ranks))


if __name__ == "__main__":
    unittest.main()
