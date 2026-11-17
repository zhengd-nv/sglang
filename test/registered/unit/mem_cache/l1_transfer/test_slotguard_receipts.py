"""Bounded receipt schema and CPU fake transport evidence, never native proof."""

import importlib
import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_slotguard_backend as backend_fixture
import test_slotguard_service as service_fixture
from test_protocol import register_cpu_ci

from sglang.test.test_utils import CustomTestCase

TransferObserver = importlib.import_module(
    f"{backend_fixture.PACKAGE}.observation"
).TransferObserver
receipts = importlib.import_module(f"{backend_fixture.PACKAGE}.slotguard_receipts")


register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class ReceiptSchemaTest(CustomTestCase):
    def test_explicit_api_keeps_emit_allowlist_and_counts_bad_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = TransferObserver(
                str(Path(directory) / "trace"), "engine", capacity=2
            )
            try:
                observer.emit("legacy", page_set_digest="a" * 64)
                self.assertEqual(observer.snapshot()["errors"], 1)
                observer.emit_slotguard("invented", payload={"arbitrary": object()})
                self.assertEqual(observer.snapshot()["errors"], 2)
                self.assertFalse(observer.snapshot()["events"])
            finally:
                observer.close()

    def test_strict_scalar_bounds_and_same_ring_loss_counters(self):
        from test_slotguard_catalog import Ready, c, guard, keys

        from sglang.srt.mem_cache.l1_transfer.slotguard_receipts import (
            publication_receipt,
        )

        g = guard()
        refs = g.register_allocation((1,))
        publication = c.Publication(
            c.PublicationScope("model", "layout", 64, 1),
            None,
            (tuple(range(64)),),
            keys(tuple(range(64)), 64),
            refs,
            Ready(),
        )
        fields = publication_receipt(publication)
        with tempfile.TemporaryDirectory() as directory:
            observer = TransferObserver(
                str(Path(directory) / "trace"), "engine", capacity=2
            )
            try:
                observer.emit_slotguard("publication_ready", **fields)
                event = observer.snapshot()["events"][0]
                self.assertEqual(event["page_count"], 1)
                self.assertTrue(event["ready_proof"])
                observer.emit_slotguard(
                    "publication_ready", **{**fields, "page_count": True}
                )
                observer.emit_slotguard(
                    "publication_ready", **{**fields, "page_set_digest": "x" * 64}
                )
                observer.emit_slotguard("publication_ready", **{**fields, "extra": []})
                self.assertEqual(observer.snapshot()["errors"], 3)
                with patch.object(
                    observer._queue, "put_nowait", side_effect=queue.Full
                ):
                    observer.emit_slotguard("publication_ready", **fields)
                self.assertEqual(observer.snapshot()["dropped"], 1)
                self.assertEqual(observer.snapshot()["seq"], 2)
                self.assertEqual(observer.snapshot()["capacity"], 2)
                self.assertEqual(observer.snapshot()["first_seq"], 1)
                self.assertFalse(observer.snapshot()["good"])
                observer.close()
                observer.emit_slotguard("publication_ready", **fields)
                self.assertEqual(observer.snapshot()["dropped"], 2)
            finally:
                if not observer.snapshot()["closed"]:
                    observer.close()

    def test_order_and_every_page_generation_are_bound(self):
        from sglang.srt.mem_cache.l1_transfer.slot_guard import PageRef
        from sglang.srt.mem_cache.l1_transfer.slotguard_receipts import (
            ordered_page_digest,
        )

        refs = (PageRef(1, 1, 1), PageRef(1, 2, 1))
        digest = ordered_page_digest(refs)
        for changed in (
            refs[::-1],
            refs[:1],
            (refs[0], PageRef(1, 2, 2)),
            (PageRef(2, 1, 1), refs[1]),
        ):
            self.assertNotEqual(digest, ordered_page_digest(changed))


class ReceiptChainTest(CustomTestCase):
    start = backend_fixture.SlotGuardBackendTest.start
    tick = backend_fixture.SlotGuardBackendTest.tick
    until = backend_fixture.SlotGuardBackendTest.until
    assert_rejected_cohort = backend_fixture.SlotGuardBackendTest.assert_rejected_cohort

    def setUp(self):
        backend_fixture.SlotGuardBackendTest.setUp(self)
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
            catalog = service_fixture.c.PublicationCatalog(
                rank.base.g, event_keys=service_fixture.keys, observer=source
            )
            ready = service_fixture.Ready()
            ready.value = True
            catalog.submit(
                service_fixture.c.Publication(
                    service_fixture.c.PublicationScope("model-1", "layout-1", 64, 1),
                    None,
                    tuple(tuple(range(start, start + 64)) for start in (0, 64, 128)),
                    service_fixture.keys(tuple(range(192)), 64),
                    rank.base.refs,
                    ready,
                )
            )
            catalog.progress()
            rank.base.catalog = rank.source.catalog = catalog

    def tearDown(self):
        try:
            backend_fixture.SlotGuardBackendTest.tearDown(self)
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
                lambda: all(
                    r.status() is backend_fixture.p.PrefetchStatus.READY
                    for r in self.ranks
                )
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
        self.until(
            lambda: (
                self.ranks[3].status() is backend_fixture.p.PrefetchStatus.QUARANTINED
            )
        )
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


class PublicationReceiptTest(CustomTestCase):
    def test_pending_and_uncertain_producer_never_emit_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = TransferObserver(str(Path(directory) / "trace"), "source")
            try:
                g = service_fixture.guard()
                refs = g.register_allocation((1, 2))
                catalog = service_fixture.c.PublicationCatalog(
                    g, event_keys=service_fixture.keys, observer=observer
                )
                ready = service_fixture.Ready()
                publication = service_fixture.c.Publication(
                    service_fixture.c.PublicationScope("model", "layout", 64, 1),
                    None,
                    (tuple(range(64)),),
                    service_fixture.keys(tuple(range(64)), 64),
                    refs[:1],
                    ready,
                )
                catalog.submit(publication)
                catalog.progress()
                self.assertFalse(observer.snapshot()["events"])
                ready.value = True
                catalog.progress()
                catalog.progress()
                self.assertEqual(len(observer.snapshot()["events"]), 1)
                uncertain = service_fixture.Ready()
                uncertain.ready = lambda: (_ for _ in ()).throw(
                    RuntimeError("query unknown")
                )
                catalog.submit(
                    replace(publication, pages=refs[1:], producer_ready=uncertain)
                )
                catalog.progress()
                self.assertEqual(catalog.snapshot().uncertain, 1)
                self.assertEqual(len(observer.snapshot()["events"]), 1)
            finally:
                observer.close()


class SourceFenceReceiptTest(CustomTestCase):
    def test_close_seal_stops_controls_without_changing_ordinary_seal_replays(self):
        base = service_fixture.ServiceTest()
        base.setUp()
        try:
            response = base.service.submit(base.pin()).result(2)
            request = base.unpin(response)
            base.service.submit(request).result(2)
            base.service.fence_pins(1)
            self.assertTrue(base.service.seal_if_empty(1))
            self.assertIs(
                base.service.submit(request).result(2).status,
                service_fixture.v.UnpinStatus.ALREADY_RELEASED,
            )
            self.assertTrue(base.service.seal_if_empty(1, close_admission=True))
            self.assertIs(
                base.service.submit(request).result(2).status,
                service_fixture.v.UnpinStatus.BUSY,
            )
            self.assertIs(
                base.service.submit(base.pin(1)).result(2).status,
                service_fixture.v.PinStatus.BUSY,
            )
            self.assertEqual(base.service.owner_snapshot().requests, 0)
        finally:
            base.tearDown()

    def test_export_completion_after_fence_reports_acquisition_generation(self):
        base = service_fixture.ServiceTest()
        base.setUp()
        with tempfile.TemporaryDirectory() as directory:
            observer = TransferObserver(str(Path(directory) / "trace"), "source")
            base.service.observer = observer
            entered, proceed = threading.Event(), threading.Event()
            original = base.engine.export_memory

            def export(memory):
                entered.set()
                self.assertTrue(proceed.wait(2))
                return original(memory)

            try:
                with patch.object(base.engine, "export_memory", side_effect=export):
                    future = base.service.submit(base.pin())
                    self.assertTrue(entered.wait(2))
                    base.service.fence_pins(7)
                    self.assertFalse(base.service.seal_if_empty(7))
                    proceed.set()
                    response = future.result(2)
                prepared = next(
                    e
                    for e in observer.snapshot()["events"]
                    if e["phase"] == "slotguard_source_prepared"
                )
                self.assertEqual(prepared["pin_generation"], 0)
                self.assertEqual(base.service.owner_snapshot().generation, 7)
                base.service.submit(base.unpin(response)).result(2)
                self.assertTrue(base.service.seal_if_empty(7))
            finally:
                proceed.set()
                observer.close()
                base.tearDown()

    def test_source_close_racing_accepted_pin_never_drops_owner(self):
        for _ in range(12):
            base = service_fixture.ServiceTest()
            base.setUp()
            barrier = threading.Barrier(2)
            outcome = {}

            def close(base=base, barrier=barrier, outcome=outcome):
                barrier.wait()
                try:
                    base.service.close()
                    outcome["closed"] = True
                except service_fixture.svc.ServiceBusy:
                    outcome["closed"] = False

            thread = threading.Thread(target=close)
            thread.start()
            barrier.wait()
            response = base.service.submit(base.pin()).result(2)
            thread.join(2)
            self.assertFalse(thread.is_alive())
            if response.status is service_fixture.v.PinStatus.PREPARED:
                self.assertFalse(outcome["closed"])
                self.assertEqual(base.service.owner_snapshot().leases, 1)
                base.service.submit(base.unpin(response)).result(2)
            else:
                self.assertIs(response.status, service_fixture.v.PinStatus.BUSY)
            base.tearDown()


if __name__ == "__main__":
    unittest.main()
