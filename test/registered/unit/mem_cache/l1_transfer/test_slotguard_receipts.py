"""Bounded receipt schema and CPU fake transport evidence, never native proof."""

import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import slotguard_catalog as c
from sglang.srt.mem_cache.l1_transfer import slotguard_protocol as v
from sglang.srt.mem_cache.l1_transfer import slotguard_service as svc
from sglang.srt.mem_cache.l1_transfer.observation import TransferObserver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.ledger import guard
from sglang.test.kits.slotguard.source import Ready, ServiceFixture
from sglang.test.kits.slotguard.wire import keys
from sglang.test.test_utils import CustomTestCase

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


class PublicationReceiptTest(CustomTestCase):
    def test_pending_and_uncertain_producer_never_emit_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = TransferObserver(str(Path(directory) / "trace"), "source")
            try:
                g = guard()
                refs = g.register_allocation((1, 2))
                catalog = c.PublicationCatalog(g, event_keys=keys, observer=observer)
                ready = Ready()
                publication = c.Publication(
                    c.PublicationScope("model", "layout", 64, 1),
                    None,
                    (tuple(range(64)),),
                    keys(tuple(range(64)), 64),
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
                uncertain = Ready()
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
        base = ServiceFixture()
        base.setUp()
        try:
            response = base.service.submit(base.pin()).result(2)
            request = base.unpin(response)
            base.service.submit(request).result(2)
            base.service.fence_pins(1)
            self.assertTrue(base.service.seal_if_empty(1))
            self.assertIs(
                base.service.submit(request).result(2).status,
                v.UnpinStatus.ALREADY_RELEASED,
            )
            self.assertTrue(base.service.seal_if_empty(1, close_admission=True))
            self.assertIs(
                base.service.submit(request).result(2).status,
                v.UnpinStatus.BUSY,
            )
            self.assertIs(
                base.service.submit(base.pin(1)).result(2).status,
                v.PinStatus.BUSY,
            )
            self.assertEqual(base.service.owner_snapshot().requests, 0)
        finally:
            base.tearDown()

    def test_export_completion_after_fence_reports_acquisition_generation(self):
        base = ServiceFixture()
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
        base = ServiceFixture()
        base.setUp()
        entered, proceed = threading.Event(), threading.Event()
        original = base.engine.export_memory

        def export(memory):
            entered.set()
            self.assertTrue(proceed.wait(2))
            return original(memory)

        try:
            with patch.object(base.engine, "export_memory", side_effect=export):
                future = base.service.submit(base.pin())
                # The Pin holds guard pages and is mid-export when close arrives.
                self.assertTrue(entered.wait(2))
                with self.assertRaises(svc.ServiceBusy):
                    base.service.close()
                proceed.set()
                response = future.result(2)
            self.assertIs(response.status, v.PinStatus.PREPARED)
            self.assertEqual(base.service.owner_snapshot().leases, 1)
            base.service.submit(base.unpin(response)).result(2)
        finally:
            proceed.set()
            base.tearDown()


if __name__ == "__main__":
    unittest.main()
