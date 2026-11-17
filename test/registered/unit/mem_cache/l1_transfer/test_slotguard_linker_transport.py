"""Owner lifecycle tests with real role gates and controlled transport drain."""

import importlib
import threading
import unittest
from array import array
from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_protocol import PACKAGE, identity, p, register_cpu_ci
from test_slotguard_catalog import lease

from sglang.test.test_utils import CustomTestCase

t = importlib.import_module(f"{PACKAGE}.slotguard_linker_transport")
a = importlib.import_module(f"{PACKAGE}.transport_role_arbiter")
l = importlib.import_module(f"{PACKAGE}.role_gate")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


@dataclass(frozen=True)
class Handle:
    rid: str
    attempt_id: int


class Backend:
    def __init__(self):
        self._directory = NS(target=identity(tp_size=4))
        self._gate = l.RoleGate()
        self.transport_arbiter = a.TransportRoleArbiter()
        self.phase = ("ENQUEUED", True)
        self.status = p.PrefetchStatus.PIN_PENDING
        self.retired = self.cancelled = False
        self.accepted = False
        self.unknown = 0
        self.permits = []
        self.thread = None

    def lookup(self, query):
        import threading

        self.thread = threading.get_ident()
        hit = query.max_restore_tokens
        return p.LookupResult(
            query,
            identity("source", tp_size=4) if hit else None,
            hit,
            query.event_block_keys[: hit // query.page_size],
        )

    def prefetch(self, result, slots, *, identity):
        self.accepted = True
        self.result, self.slots = result, slots

    def poll_prefetch(self, ticket):
        return self.status

    def group_snapshot(self, ticket):
        return self.phase

    def group_permit(self, ticket, decision):
        self.permits.append(decision)
        if decision == "PIN":
            self.phase = ("PIN", True)
        elif decision == "COPY":
            self.phase = ("COPY", False)
        elif decision in ("DROP", "ABORT"):
            self.phase = ("DRAINED", False)
        elif decision == "RELEASE":
            self.phase = ("RELEASE", False)
            self.status = (
                p.PrefetchStatus.FAILED_SAFE
                if self.cancelled
                else p.PrefetchStatus.ACKED
            )

    def group_commit_ready(self, ticket):
        self.status = p.PrefetchStatus.READY

    def cancel_prefetch(self, ticket):
        self.cancelled = True
        if self.status is p.PrefetchStatus.ACKED:
            self.status = p.PrefetchStatus.FAILED_SAFE

    def retire(self, ticket):
        assert self.status in (p.PrefetchStatus.READY, p.PrefetchStatus.FAILED_SAFE)
        self.retired = True
        self.accepted = False

    def owner_snapshot(self):
        return NS(
            unknown=self.unknown,
            working=0,
            imports=0,
            batches=0,
            accepted=int(self.accepted),
            queued=0,
        )

    def close(self):
        pass


class TransportTest(CustomTestCase):
    def setUp(self):
        self.backend = Backend()
        self.controller = t.SlotGuardLinkerTransport(
            self.backend, NS(gather=lambda value: [value] * 4)
        )
        self.context = NS(
            request=Handle("req", 1),
            token_ids=array("q", range(129)).tobytes(),
            local_hit_tokens=0,
            max_restore_tokens=128,
            page_size=64,
            extra_key=None,
            cache_salt=None,
            is_bigram=False,
        )
        self.hashes = patch.object(
            t, "event_keys_and_hashes", return_value=((1, 2), ("a" * 64, "b" * 64))
        )
        self.hashes.start()
        self.addCleanup(self.hashes.stop)
        self.addCleanup(lambda: self.controller._lookup_worker.shutdown(wait=True))
        self.addCleanup(lambda: self.controller._prepare_worker.shutdown(wait=True))

    def lookup(self, context=None):
        context = context or self.context
        future = self.controller.submit_lookup(context)
        self.controller.lookups[context.request].job.result(2)
        self.controller.poll()
        return future

    def read(self, context, *, indices=None, expected_pages=None):
        indices = tuple(range(256, 384)) if indices is None else indices
        fence = NS(materialize=lambda: indices)
        with patch.object(t.DeferredSlots, "capture", return_value=fence):
            future = self.controller.prefetch(
                context.request,
                NS(
                    device_indices=indices,
                    page_span=None
                    if expected_pages is None
                    else NS(page_ids=expected_pages),
                ),
            )
        self.controller.active.prepare.result(2)
        return future

    def start(self, *, indices=None, expected_pages=None):
        self.assertEqual(self.lookup().result(0), [1, 2])
        import threading

        self.assertNotEqual(self.backend.thread, threading.get_ident())
        return self.read(self.context, indices=indices, expected_pages=expected_pages)

    def advance_to_copy(self):
        for _ in range(3):
            self.controller.poll()
        self.assertEqual(self.backend.phase[0], "COPY")

    def test_success_waits_for_drain_and_common_ready(self):
        future = self.start()
        self.advance_to_copy()
        self.assertFalse(future.done())
        self.backend.phase = ("DRAINED", True)
        self.controller.poll()
        self.controller.poll()
        self.assertTrue(future.result())
        self.assertTrue(self.backend.retired)
        self.assertEqual(self.backend.permits, ["PIN", "COPY", "RELEASE"])
        self.assertEqual(self.backend.transport_arbiter.owner_snapshot().owners, 0)
        self.controller.release(self.context.request)
        self.controller.close()

    def test_peer_disagreement_releases_roles_before_read_and_allows_retry(self):
        self.lookup().result(0)

        def disagree(value):
            other = list(value)
            other[2] = ("another-peer",)
            return [value, tuple(other), value, value]

        with (
            patch.object(self.controller.collective, "gather", side_effect=disagree),
            patch.object(t.DeferredSlots, "capture") as capture,
        ):
            self.assertIsNone(
                self.controller.prefetch(
                    self.context.request, NS(device_indices=tuple(range(256, 384)))
                )
            )
            capture.assert_not_called()
        self.assertFalse(self.controller.disabled)
        self.assertIsNone(self.controller.active)
        self.assertFalse(self.backend.accepted)
        self.assertEqual(self.backend.permits, [])
        self.assertEqual(self.backend._gate.role, l.EngineRole.IDLE)
        self.assertEqual(self.backend.transport_arbiter.owner_snapshot().owners, 0)
        self.controller.release(self.context.request)
        future = self.start()
        self.advance_to_copy()
        self.backend.phase = ("DRAINED", True)
        self.controller.poll()
        self.controller.poll()
        self.assertTrue(future.result())
        self.controller.release(self.context.request)
        self.controller.close()

    def test_identity_or_role_vote_failure_still_quarantines(self):
        self.lookup().result(0)
        for field, replacement in ((0, "other-request"), (1, 2), (3, 64), (5, True)):
            with self.subTest(field=field):

                def disagree(value):
                    other = list(value)
                    other[field] = replacement
                    return [value, tuple(other), value, value]

                with (
                    patch.object(
                        self.controller.collective, "gather", side_effect=disagree
                    ),
                    patch.object(t.DeferredSlots, "capture") as capture,
                ):
                    with self.assertRaises(t.LinkerQuarantined):
                        self.controller.prefetch(
                            self.context.request,
                            NS(device_indices=tuple(range(256, 384))),
                        )
                    capture.assert_not_called()
                self.assertTrue(self.controller.disabled)
                self.assertFalse(self.backend.accepted)
                self.assertEqual(self.backend._gate.role, l.EngineRole.INBOUND)
                self.assertEqual(
                    self.backend.transport_arbiter.owner_snapshot().owners, 1
                )

    def test_abort_keeps_writer_and_roles_until_drain(self):
        future = self.start()
        self.advance_to_copy()
        self.controller.cancel(self.context.request)
        self.assertFalse(future.cancel())
        self.controller.poll()
        self.assertFalse(future.done())
        self.assertEqual(self.backend.transport_arbiter.owner_snapshot().owners, 1)
        with self.assertRaises(RuntimeError):
            self.controller.reset()
        self.backend.phase = ("DRAINED", False)
        self.controller.poll()
        self.controller.poll()
        self.assertFalse(future.result())
        self.controller.release(self.context.request)
        self.controller.close()

    def test_uncertain_disables_and_preserves_future_and_roles(self):
        future = self.start()
        self.advance_to_copy()
        self.backend.status = p.PrefetchStatus.QUARANTINED
        with self.assertRaises(t.LinkerQuarantined):
            self.controller.poll()
        self.assertTrue(self.controller.disabled)
        self.assertFalse(future.done())
        self.assertEqual(self.backend.transport_arbiter.owner_snapshot().owners, 1)
        with self.assertRaises(RuntimeError):
            self.controller.close()
        self.assertIsNone(self.controller.submit_lookup(self.context))

    def test_prepare_validation_failure_is_safe_after_fence_drain(self):
        future = self.start(indices=tuple(range(257, 385)))
        self.controller.poll()
        self.assertFalse(future.result())
        self.assertFalse(self.backend.accepted)
        self.controller.release(self.context.request)
        self.controller.close()

    def test_guarded_page_identity_mismatch_never_reaches_backend(self):
        future = self.start(expected_pages=(6, 7))
        self.controller.poll()
        self.assertFalse(future.result())
        self.assertFalse(self.backend.accepted)
        self.controller.release(self.context.request)
        self.controller.close()

    def test_unsupported_request_features_are_rejected_before_lookup(self):
        for feature in (
            "session",
            "multimodal_inputs",
            "input_embeds",
            "dllm",
            "beam_group",
            "return_logprob",
            "holds_kv",
        ):
            with self.subTest(feature=feature):
                self.context.features = frozenset((feature,))
                self.assertIsNone(self.controller.submit_lookup(self.context))
        self.assertIsNone(self.backend.thread)
        self.assertFalse(self.controller.lookups)
        self.controller.close()

    def test_same_rid_attempts_have_independent_lookup_ownership(self):
        old = self.lookup()
        self.assertEqual(old.result(0), [1, 2])
        second = NS(**vars(self.context))
        second.request = Handle("req", 2)
        new = self.lookup(second)
        self.controller.cancel(self.context.request)
        self.controller.release(self.context.request)
        self.assertIn(second.request, self.controller.lookups)
        self.assertFalse(self.controller.lookups[second.request].cancelled)
        self.controller.poll()
        self.assertEqual(new.result(0), [1, 2])
        self.controller.release(second.request)
        self.controller.close()

    def test_reused_rid_and_handle_gets_a_fresh_wire_attempt(self):
        self.lookup().result(0)
        ticket = self.controller.lookups[self.context.request].result.transfer_id
        self.controller.release(self.context.request)
        self.lookup().result(0)
        next_ticket = self.controller.lookups[self.context.request].result.transfer_id
        self.assertGreater(next_ticket.attempt, ticket.attempt)
        self.controller.release(self.context.request)
        self.controller.close()

    def test_hits_queue_behind_the_inbound_read_and_misses_pass(self):
        """Concurrent hits restore one after another instead of falling back."""
        first = self.start()
        second, miss = NS(**vars(self.context)), NS(**vars(self.context))
        second.request, miss.request = Handle("second", 1), Handle("miss", 1)
        miss.max_restore_tokens = 0
        queued, passed = self.lookup(second), self.lookup(miss)
        self.assertEqual(passed.result(0), [])
        self.assertFalse(queued.done())
        self.advance_to_copy()
        self.backend.phase = ("DRAINED", True)
        self.controller.poll()
        self.assertFalse(queued.done())
        self.controller.poll()
        self.assertTrue(first.result(0))
        self.assertEqual(queued.result(0), [1, 2])
        self.controller.release(self.context.request)
        self.controller.release(miss.request)
        self.backend.retired, self.backend.phase = False, ("ENQUEUED", True)
        self.backend.status = p.PrefetchStatus.PIN_PENDING
        self.read(second)
        self.advance_to_copy()
        self.backend.phase = ("DRAINED", True)
        self.controller.poll()
        self.controller.poll()
        self.assertTrue(self.backend.retired)
        self.controller.release(second.request)
        self.controller.close()

    def test_released_turn_passes_to_the_next_hit(self):
        """A hit that never reads (rejected, cancelled) must not block later hits."""
        self.lookup().result(0)
        second = NS(**vars(self.context))
        second.request = Handle("second", 1)
        queued = self.lookup(second)
        self.assertFalse(queued.done())
        self.controller.release(self.context.request)
        self.controller.poll()
        self.assertEqual(queued.result(0), [1, 2])
        self.controller.cancel(second.request)
        self.controller.release(second.request)
        self.assertIsNone(self.controller.reserved)
        self.controller.close()

    def test_async_unpin_unknown_owner_blocks_close_after_ready(self):
        future = self.start()
        self.advance_to_copy()
        self.backend.phase = ("DRAINED", True)
        self.controller.poll()
        self.controller.poll()
        self.assertTrue(future.result())
        self.backend.unknown = 1
        with self.assertRaises(t.LinkerQuarantined):
            self.controller.poll()
        self.assertTrue(self.controller.disabled)
        self.controller.release(self.context.request)
        with self.assertRaises(RuntimeError):
            self.controller.close()


class ArbiterTest(CustomTestCase):
    def test_many_outbound_exclusive_inbound_replay_and_close(self):
        arb = a.TransportRoleArbiter(max_owners=2)
        self.assertTrue(arb.acquire(a.TransportRole.OUTBOUND, lease()))
        self.assertTrue(arb.acquire(a.TransportRole.OUTBOUND, lease()))
        self.assertTrue(arb.acquire(a.TransportRole.OUTBOUND, lease(1)))
        self.assertFalse(arb.acquire(a.TransportRole.INBOUND, lease(2)))
        self.assertFalse(arb.acquire(a.TransportRole.OUTBOUND, lease(2)))
        with self.assertRaises(a.RoleConflict):
            arb.close()
        arb.release(a.TransportRole.OUTBOUND, lease())
        arb.release(a.TransportRole.OUTBOUND, lease(1))
        self.assertTrue(arb.acquire(a.TransportRole.INBOUND, lease(2)))
        self.assertFalse(arb.acquire(a.TransportRole.OUTBOUND, lease()))
        arb.release(a.TransportRole.INBOUND, lease(2))
        arb.release(a.TransportRole.INBOUND, lease(2))
        self.assertEqual(arb.owner_snapshot().owners, 0)
        arb.close()

    def test_race_reserves_role_atomically(self):
        arb = a.TransportRoleArbiter(max_owners=4)
        barrier = threading.Barrier(3)
        results = []

        def acquire(role, owner):
            barrier.wait()
            results.append(arb.acquire(role, owner))

        threads = [
            threading.Thread(target=acquire, args=(role, lease(i)))
            for i, role in enumerate(
                (a.TransportRole.INBOUND, a.TransportRole.OUTBOUND)
            )
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(2)
        self.assertEqual(sorted(results), [False, True])


if __name__ == "__main__":
    unittest.main()
