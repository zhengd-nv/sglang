"""CPU publication and lifetime certificates; no tree or scheduler fixture."""

import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import slot_guard as s
from sglang.srt.mem_cache.l1_transfer import slotguard_catalog as c
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.ledger import guard, seed_credit
from sglang.test.kits.slotguard.source import Ready
from sglang.test.kits.slotguard.wire import (
    hashes,
    keys,
    lease,
    result,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


K = keys(tuple(range(6)), 2)


class CatalogTest(CustomTestCase):
    def setUp(self):
        self.guard = guard()
        self.refs = self.guard.register_allocation((1, 2, 3))
        self.catalog = c.PublicationCatalog(
            self.guard, max_records=8, max_pending=2, event_keys=keys
        )
        self.scope = c.PublicationScope("model-1", "layout-1", 2, 1)

    def publish(self, ready=None):
        ready = ready or Ready()
        return self.catalog.submit(
            c.Publication(
                self.scope,
                None,
                ((0, 1), (2, 3), (4, 5)),
                K,
                self.refs,
                ready,
            )
        )

    def test_incomplete_never_visible_then_exact_nonzero_interval(self):
        ready = Ready()
        self.publish(ready)
        self.assertEqual(self.catalog.progress(), 0)
        self.assertIsNone(self.catalog.lookup(result()))
        ready.value = True
        self.assertEqual(self.catalog.progress(), 1)
        self.assertEqual(self.catalog.lookup(result()), self.refs[1:])

    def test_tokens_namespace_keys_layout_and_pool_epoch_fail_closed(self):
        ready = Ready()
        ready.value = True
        self.publish(ready)
        self.catalog.progress()
        r = result()
        with self.assertRaises(ValueError):
            replace(r, cache_salt="different")
        other = hashes((0, 9, 2, 3, 4, 5), 2)[-1]
        self.assertIsNone(self.catalog.lookup(replace(r, prefix_hash=other)))
        self.assertIsNone(
            self.catalog.lookup(
                replace(
                    r,
                    source=replace(r.source, layout_digest="other"),
                    target=replace(r.target, layout_digest="other"),
                )
            )
        )

    def test_late_free_cannot_publish_or_revive_generation(self):
        ready = Ready()
        self.publish(ready)
        self.guard.on_free(self.refs)
        self.guard.begin_epoch(1)
        self.guard.install_epoch(1, ())
        self.guard.drain_reclaim(1)
        self.guard.finish_epoch(1)
        newer = self.guard.register_allocation((1, 2, 3))
        ready.value = True
        self.assertEqual(self.catalog.progress(), 0)
        self.assertIsNone(self.catalog.lookup(result()))
        with self.assertRaises(s.PageUnavailable):
            self.guard.try_pin(lease(), self.refs[1:])
        self.assertNotEqual(newer, self.refs)

    def test_pending_failure_retains_uncertainty_and_bound(self):
        class Failed:
            def ready(self):
                raise RuntimeError("event status unknown")

        self.publish(Failed())
        self.catalog.progress()
        self.assertEqual(self.catalog.snapshot().uncertain, 1)
        self.publish()
        with self.assertRaises(c.CatalogBusy):
            self.publish()
        self.assertEqual(self.catalog.snapshot().pending, 2)

    def test_never_ready_head_cannot_starve_a_ready_publication(self):
        """A never-ready pending head must not hide ready pages queued behind it."""
        ready = Ready()
        ready.value = True
        for pages, certificate in ((self.refs[:1], Ready()), (self.refs[1:2], ready)):
            self.catalog.submit(
                c.Publication(self.scope, None, ((0, 1),), K[:1], pages, certificate)
            )
        self.assertEqual(self.catalog.progress(limit=1), 0)
        self.assertEqual(self.catalog.progress(limit=1), 1)
        self.assertEqual(self.catalog.snapshot().pending, 1)

    def test_over_limit_pin_is_refused_before_the_catalog_walk(self):
        """A Pin over the guard's page limit must not walk the chain under the lock."""
        ready = Ready()
        ready.value = True
        self.publish(ready)
        self.catalog.progress()

        class WalkForbidden:
            def __enter__(self):
                raise AssertionError("over-limit Pin entered the catalog walk")

            def __exit__(self, *exc):
                return False

        pin = result()
        with (
            patch.object(self.guard, "max_pin_pages", pin.restore_pages - 1),
            patch.object(self.catalog, "_lock", WalkForbidden()),
        ):
            self.assertIsNone(self.catalog.lookup(pin))

    def test_dirty_suffix_chain_and_bounded_eviction(self):
        ready = Ready()
        ready.value = True
        first = self.catalog.submit(
            c.Publication(self.scope, None, ((0, 1),), K[:1], self.refs[:1], ready)
        )
        self.catalog.progress()
        self.catalog.submit(
            c.Publication(
                self.scope, first, ((2, 3), (4, 5)), K[1:], self.refs[1:], ready
            )
        )
        self.catalog.progress()
        self.assertEqual(self.catalog.lookup(result()), self.refs[1:])
        for _ in range(6):
            self.publish(ready)
            self.catalog.progress()
        self.assertLessEqual(self.catalog.snapshot().records, 8)

    def test_pending_parent_can_receive_dirty_suffix_without_wait(self):
        ready = Ready()
        first = self.catalog.submit(
            c.Publication(self.scope, None, ((0, 1),), K[:1], self.refs[:1], ready)
        )
        self.catalog.submit(
            c.Publication(
                self.scope, first, ((2, 3), (4, 5)), K[1:], self.refs[1:], ready
            )
        )
        self.assertIsNone(self.catalog.lookup(result()))
        ready.value = True
        self.catalog.progress()
        self.assertEqual(self.catalog.lookup(result()), self.refs[1:])

    def test_receipt_returns_each_page_id_and_pending_internal_parent(self):
        ready = Ready()
        receipt = self.catalog.submit_with_receipt(
            c.Publication(
                self.scope, None, ((0, 1), (2, 3)), K[:2], self.refs[:2], ready
            )
        )
        self.assertEqual(len(receipt.page_cids), 2)
        self.assertEqual(receipt.terminal_cid, receipt.page_cids[-1])
        # Use the explicit first-page CID while its whole chunk is still pending.
        suffix = self.catalog.submit_with_receipt(
            c.Publication(
                self.scope,
                receipt.page_cids[0],
                ((2, 3), (4, 5)),
                K[1:],
                self.refs[1:],
                ready,
            )
        )
        self.assertTrue(self.catalog.contains(suffix.terminal_cid, self.scope))
        self.assertFalse(
            self.catalog.contains(
                suffix.terminal_cid, replace(self.scope, pool_epoch=2)
            )
        )
        ready.value = True
        self.catalog.progress()
        self.assertEqual(self.catalog.lookup(result()), self.refs[1:])

    def test_free_after_publication_is_catalog_miss(self):
        ready = Ready()
        ready.value = True
        self.publish(ready)
        self.catalog.progress()
        self.guard.on_free(self.refs)
        self.assertIsNone(self.catalog.lookup(result()))

    def test_catalog_token_budget_bounds_pending_and_committed_chunks(self):
        catalog = c.PublicationCatalog(
            self.guard, max_records=64, max_pending=4, max_token_ids=6, event_keys=keys
        )
        ready = Ready()
        publication = c.Publication(
            self.scope, None, ((0, 1), (2, 3), (4, 5)), K, self.refs, ready
        )
        catalog.submit(publication)
        with self.assertRaises(c.CatalogBusy):
            catalog.submit(publication)
        ready.value = True
        catalog.progress()
        catalog.submit(publication)
        catalog.progress()
        self.assertLessEqual(catalog.snapshot().token_ids, 6)

    def test_native_defaults_reject_salt_or_extra_key_before_hashing(self):
        catalog = c.PublicationCatalog(self.guard)
        ready = Ready()
        ready.value = True
        for scope in (
            replace(self.scope, cache_salt="salt"),
            replace(self.scope, extra_key="lora"),
        ):
            publication = c.Publication(
                scope, None, ((0, 1),), (1,), self.refs[:1], ready
            )
            with self.assertRaises(ValueError):
                catalog.submit(publication)
            with self.assertRaises(ValueError):
                replace(
                    result(), cache_salt=scope.cache_salt, extra_key=scope.extra_key
                )
        self.assertEqual(catalog.snapshot().records, 0)

    def test_same_content_new_physical_pages_does_not_rebind_active_grant(self):
        ready = Ready()
        ready.value = True
        self.publish(ready)
        self.catalog.progress()
        old_pages = self.catalog.lookup(result())
        self.guard.try_pin(lease(), old_pages)
        newer = self.guard.register_allocation((4, 5, 6))
        self.catalog.submit(
            c.Publication(self.scope, None, ((0, 1), (2, 3), (4, 5)), K, newer, ready)
        )
        self.catalog.progress()
        self.assertEqual(self.catalog.lookup(result()), newer[1:])
        self.guard.release(lease(), old_pages)
        self.assertEqual(self.guard.owner_counts().active_leases, 0)


class CreditTest(CustomTestCase):
    def test_cumulative_unique_credit_no_refund_and_atomic_finish(self):
        g = guard(shared_headroom_credit=True)
        refs = g.register_allocation((1, 2, 3))
        g.publish(refs)
        with self.assertRaises(s.QuotaExceeded):
            g.try_pin(lease(), refs[:1])
        seed_credit(g, 2)
        g.try_pin(lease(), refs[:2])
        g.try_pin(lease(1), refs[1:2])
        with self.assertRaises(s.QuotaExceeded):
            g.try_pin(lease(2), refs[2:])
        g.release(lease(), refs[:2])
        with self.assertRaises(s.QuotaExceeded):
            g.try_pin(lease(2), refs[2:])
        g.begin_epoch(1)
        g.install_epoch(1, refs[1:2])
        with self.assertRaises(s.EpochError):
            g.finish_epoch(1)
        self.assertEqual(g.owner_counts().frozen_epoch, 1)
        g.finish_epoch(1, verified_headroom_pages=1)
        g.try_pin(lease(2), refs[2:])
        self.assertEqual(g.owner_counts().credit_remaining, 0)

    def test_epoch_freeze_cannot_unfreeze_without_new_zero_credit(self):
        g = guard(shared_headroom_credit=True)
        refs = g.register_allocation((1,))
        g.publish(refs)
        seed_credit(g, 1)
        g.begin_epoch(1)
        g.install_epoch(1, ())
        before, after = threading.Event(), threading.Event()
        errors = []

        def pin():
            try:
                g.try_pin(lease(), refs)
            except s.PinsFrozen:
                before.set()
            after.wait(2)
            try:
                g.try_pin(lease(), refs)
            except s.QuotaExceeded:
                errors.append("no credit")

        worker = threading.Thread(target=pin)
        worker.start()
        self.assertTrue(before.wait(2))
        g.finish_epoch(1, verified_headroom_pages=0)
        after.set()
        worker.join(2)
        self.assertEqual(errors, ["no credit"])
        self.assertEqual(g.owner_counts().active_leases, 0)


if __name__ == "__main__":
    unittest.main()
