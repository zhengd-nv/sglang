"""Pin byte ownership from enqueue through record/replay/callback completion."""

import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_slotguard_service as fixture
from test_protocol import register_cpu_ci

from sglang.test.test_utils import CustomTestCase

v, w = fixture.v, fixture.w
register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class AdmissionBytesTest(CustomTestCase):
    def setUp(self):
        self.base = fixture.ServiceTest()
        self.base.setUp()
        self.service = self.base.service

    def tearDown(self):
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.base.tearDown()

    def wait(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.001)
        self.fail("service ownership did not settle")

    def hold_pin_workers(self):
        proceed = threading.Event()
        entered = [threading.Event(), threading.Event()]

        def hold(event):
            event.set()
            proceed.wait(2)

        tasks = [self.service._pins.submit(hold, event) for event in entered]
        self.assertTrue(all(event.wait(1) for event in entered))
        return proceed, tasks

    @staticmethod
    def size(request):
        return request._wire_size + len(request._wire_body)

    def test_tiny_budget_rejects_before_executor_retains_complete_pin(self):
        proceed, tasks = self.hold_pin_workers()
        request = self.base.pin()
        self.service.max_retained_bytes = 1
        try:
            future = self.service.submit(request)
            self.assertTrue(future.done(), "queued full Pin bypassed byte admission")
            self.assertEqual(future.result().status, v.PinStatus.BUSY)
            self.assertEqual(self.service.owner_snapshot().requests, 0)
            self.assertEqual(self.service.owner_snapshot().retained_bytes, 0)
        finally:
            proceed.set()
            for task in tasks:
                task.result(2)

    def test_fresh_queued_duplicates_cancel_and_unpin_have_exact_separate_owners(self):
        request = self.base.pin()
        response = self.service.submit(request).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        baseline = self.service.owner_snapshot().retained_bytes
        size = self.size(request)
        self.service.max_requests = 8
        self.service.max_retained_bytes = baseline + 2 * size
        proceed, tasks = self.hold_pin_workers()
        try:
            first = self.service.submit(w.decode(w.encode(request), v.PinReq))
            second = self.service.submit(w.decode(w.encode(request), v.PinReq))
            self.assertEqual(
                self.service.owner_snapshot().retained_bytes, baseline + 2 * size
            )
            self.assertTrue(second.cancel())
            self.assertEqual(
                self.service.submit(self.base.pin(1)).result(1).status, v.PinStatus.BUSY
            )
            self.assertEqual(
                self.service.submit(self.base.unpin(response)).result(1).status,
                v.UnpinStatus.RELEASED,
            )
            self.wait(lambda: self.service.owner_snapshot().requests == 2)
            binding = len(w.reference_bytes(request.reference))
            self.assertEqual(
                self.service.owner_snapshot().retained_bytes, binding + 2 * size
            )
        finally:
            proceed.set()
            for task in tasks:
                task.result(2)
        self.assertEqual(first.result(2).status, v.PinStatus.MISS)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, binding)

    def test_executor_submit_failure_and_pre_record_exception_refund_admission(self):
        request = self.base.pin()
        with (
            patch.object(
                self.service._pins, "submit", side_effect=RuntimeError("submit failed")
            ),
            self.assertRaises(RuntimeError),
        ):
            self.service.submit(request)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, 0)
        self.assertEqual(self.service.owner_snapshot().requests, 0)
        with (
            patch.object(
                self.service, "_peer_status", side_effect=RuntimeError("worker failed")
            ),
            self.assertRaises(RuntimeError),
        ):
            self.service.submit(request).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, 0)
        self.assertEqual(self.service.owner_snapshot().records, 0)

    def test_executor_raise_after_queue_cannot_run_or_refund_twice(self):
        submit = self.service._pins.submit
        tasks = []

        def queued_then_failed(*args):
            tasks.append(submit(*args))
            raise RuntimeError("thread start failed after enqueue")

        with (
            patch.object(self.service._pins, "submit", side_effect=queued_then_failed),
            patch.object(
                self.service, "_pin", side_effect=AssertionError("rolled back work ran")
            ) as pin,
        ):
            with self.assertRaises(RuntimeError):
                self.service.submit(self.base.pin())
            tasks[0].result(2)
            pin.assert_not_called()
        self.assertEqual(self.service.owner_snapshot().retained_bytes, 0)
        self.assertEqual(self.service.owner_snapshot().requests, 0)
        self.assertEqual(self.service.owner_snapshot().records, 0)

    def test_peer_rejection_and_record_conflict_drop_only_their_submission_charge(self):
        request = self.base.pin()
        # A page size that does not match the source layout is a peer rejection.
        incompatible = replace(request, result=replace(request.result, page_size=32))
        self.assertEqual(
            self.service.submit(incompatible).result(2).status, v.PinStatus.INCOMPATIBLE
        )
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, 0)
        response = self.service.submit(request).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        baseline = self.service.owner_snapshot().retained_bytes
        conflict = replace(
            request, identity=replace(request.identity, nonce="conflict")
        )
        self.assertEqual(
            self.service.submit(conflict).result(2).status, v.PinStatus.INCOMPATIBLE
        )
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, baseline)
        self.service.submit(self.base.unpin(response)).result(2)

    def test_failed_unpin_keeps_prepared_record_charge_until_known_cleanup(self):
        response = self.service.submit(self.base.pin()).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        baseline = self.service.owner_snapshot().retained_bytes
        with patch.object(
            self.base.engine,
            "release_export",
            side_effect=RuntimeError("unknown cleanup"),
        ):
            self.assertEqual(
                self.service.submit(self.base.unpin(response)).result(2).status,
                v.UnpinStatus.BUSY,
            )
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, baseline)
        self.assertEqual(self.service.owner_snapshot().uncertain, 1)
        self.assertEqual(self.base.g.owner_counts().active_leases, 1)
        self.service.submit(self.base.unpin(response)).result(2)

    def test_new_pin_callback_keeps_charge_after_unpin_at_full_budget(self):
        proceed, tasks = self.hold_pin_workers()
        entered, done = threading.Event(), threading.Event()
        request = self.base.pin()

        def callback(future):
            entered.set()
            done.wait(2)

        future = self.service.submit(request)
        future.add_done_callback(callback)
        proceed.set()
        try:
            self.assertTrue(entered.wait(1))
            response = future.result(1)
            response_bytes = len(w.encode(response)) + len(response._wire_body)
            self.service.max_retained_bytes = (
                self.service.owner_snapshot().retained_bytes
            )
            self.assertEqual(
                self.service.submit(self.base.unpin(response)).result(1).status,
                v.UnpinStatus.RELEASED,
            )
            self.wait(lambda: self.service.owner_snapshot().requests == 1)
            self.assertEqual(
                self.service.owner_snapshot().retained_bytes,
                self.size(request)
                + response_bytes
                + len(w.reference_bytes(request.reference)),
            )
        finally:
            done.set()
            for task in tasks:
                task.result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)

    def test_cancelled_queued_pin_keeps_charge_then_transfers_to_live_record(self):
        proceed, tasks = self.hold_pin_workers()
        request = self.base.pin()
        size = self.size(request)
        try:
            pending = self.service.submit(request)
            self.assertTrue(pending.cancel())
            self.assertEqual(self.service.owner_snapshot().retained_bytes, size)
        finally:
            proceed.set()
            for task in tasks:
                task.result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        state = self.service.query(request.identity)
        self.assertEqual(state.state, "PREPARED")
        record = self.service._records[request.identity.transfer_id]
        self.assertEqual(record.request_bytes, size)
        self.assertEqual(
            self.service.owner_snapshot().retained_bytes,
            size + record.binding_bytes + record.response_bytes,
        )
        self.service.submit(
            v.UnpinReq(request.reference, state.lease_id, True, True)
        ).result(2)

    def test_replay_callback_holds_original_and_new_pin_after_unpin_terminal(self):
        request = self.base.pin()
        response = self.service.submit(request).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        duplicate = w.decode(w.encode(request), v.PinReq)
        self.assertIsNot(duplicate, request)
        proceed, tasks = self.hold_pin_workers()
        callback_entered, callback_done = threading.Event(), threading.Event()

        def callback(future):
            self.assertIs(future.result().request, request)
            callback_entered.set()
            callback_done.wait(2)

        replay = self.service.submit(duplicate)
        replay.add_done_callback(callback)
        proceed.set()
        try:
            self.assertTrue(callback_entered.wait(1))
            self.assertEqual(
                self.service.submit(self.base.unpin(response)).result(1).status,
                v.UnpinStatus.RELEASED,
            )
            self.wait(lambda: self.service.owner_snapshot().requests == 1)
            expected = (
                self.size(request)
                + self.size(duplicate)
                + len(w.encode(response))
                + len(response._wire_body)
                + len(w.reference_bytes(request.reference))
            )
            self.assertEqual(self.service.owner_snapshot().retained_bytes, expected)
        finally:
            callback_done.set()
            for task in tasks:
                task.result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(
            self.service.owner_snapshot().retained_bytes,
            len(w.reference_bytes(request.reference)),
        )

    def test_retained_prepared_cache_is_a_separate_fee_and_http_reuses_it(self):
        request = self.base.pin()
        response = self.service.submit(request).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        record = self.service._records[request.identity.transfer_id]
        body = response._wire_body
        self.assertIsNotNone(body)
        self.assertEqual(record.response_bytes, 2 * len(body))
        self.assertEqual(record.request_charge.response_bytes, 2 * len(body))
        self.assertEqual(
            self.service.owner_snapshot().retained_bytes,
            self.size(request) + record.binding_bytes + 2 * len(body),
        )
        with patch.object(w, "_body", side_effect=AssertionError("HTTP reencoded")):
            self.assertIs(w.encode(response), body)
            replay = self.service.submit(request).result(2)
            self.assertIs(replay, response)
            self.assertIs(w.encode(replay), body)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.service.submit(self.base.unpin(response)).result(2)

    def test_cache_fee_can_exhaust_completion_budget_without_leaking_owners(self):
        # Admit semantic metadata but leave no space for the additional cache.
        # A precise limit demonstrates the cache was included, not just any
        # completion charge exceeding a deliberately tiny budget.
        request = self.base.pin()
        response = self.service.submit(request).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        response_size = len(w.encode(response))
        self.service.submit(self.base.unpin(response)).result(2)
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        baseline = self.service.owner_snapshot().retained_bytes
        request = self.base.pin(1)
        binding = len(w.reference_bytes(request.reference))
        self.service.max_retained_bytes = (
            baseline + self.size(request) + binding + response_size + 128
        )
        self.assertEqual(
            self.service.submit(request).result(2).status, v.PinStatus.BUSY
        )
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        snapshot = self.service.owner_snapshot()
        self.assertEqual(snapshot.retained_bytes, baseline + binding)
        self.assertEqual(snapshot.owners, 0)
        self.assertFalse(self.base.engine._exports)

    def test_cache_budget_failure_unknown_cleanup_retains_guard_and_journal(self):
        request = self.base.pin()
        binding = len(w.reference_bytes(request.reference))
        self.service.max_retained_bytes = self.size(request) + binding + 1
        with patch.object(
            self.base.engine, "release_export", side_effect=RuntimeError("unknown")
        ):
            self.assertEqual(
                self.service.submit(request).result(2).status, v.PinStatus.BUSY
            )
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        record = self.service._records[request.identity.transfer_id]
        self.assertEqual(record.state, "UNCERTAIN")
        self.assertIsNone(record.response)
        self.assertEqual(record.response_bytes, 0)
        self.assertEqual(self.base.g.owner_counts().active_leases, 1)
        self.assertTrue(record.journal)
        # Pin byte exhaustion cannot deny the independent Unpin control path.
        self.service.max_retained_bytes = binding
        unpin = v.UnpinReq(request.reference, record.lease_id, True, True)
        self.assertEqual(
            self.service.submit(unpin).result(2).status, v.UnpinStatus.RELEASED
        )
        self.wait(lambda: self.service.owner_snapshot().requests == 0)
        self.assertEqual(self.service.owner_snapshot().retained_bytes, binding)


if __name__ == "__main__":
    unittest.main()
