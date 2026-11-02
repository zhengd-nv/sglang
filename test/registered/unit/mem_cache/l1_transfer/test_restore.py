# SPDX-License-Identifier: Apache-2.0
"""Joint page/region lifetime checks using the real transport API and fault driver."""
import unittest
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from sglang.srt.mem_cache.l1_transfer.transport import NvlinkEngine, Opcode, TransferRequest, BusyError
from sglang.srt.mem_cache.l1_transfer.peer import RestoreState, RestoreTransaction
from test_contract import FakeDriver


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeDriver()
        self.owner = NvlinkEngine(0, _driver=self.driver)
        self.receiver = NvlinkEngine(1, _driver=self.driver)
        self.source = self.owner.allocate_memory(64)
        self.owner.upload(self.source, bytes(range(64)))
        self.descriptor = self.owner.export_memory(self.source)
        self.remote = self.receiver.import_memory(self.descriptor)
        self.target = self.receiver.allocate_memory(64)
        self.receiver.upload(self.target, b'X'*64)
        self.events = []
        self.source_pinned = True
        self.target_private = True
        self.fault = None
        self.request = TransferRequest(Opcode.READ, self.target, self.remote, 4, 8, 12)
        self.tx = RestoreTransaction(self.receiver, [self.request], [self.remote],
            publish=self.publish, discard=self.discard, release_source=self.release)

    def publish(self):
        self.assertTrue(self.source_pinned)
        self.assertTrue(self.target_private)
        self.assertEqual(self.tx.unmapped_count, 1)
        self.assertEqual(self.receiver.download(self.target), b'X'*4 + bytes(range(8,20)) + b'X'*48)
        self.events.append('publish')
        self.target_private = False
        if self.fault == 'publish':
            raise RuntimeError('publication partially applied')
        if self.fault == 'reentry':
            self.tx.wait()

    def discard(self):
        self.assertTrue(self.source_pinned)
        self.assertTrue(self.target_private)
        self.events.append('discard')
        if self.fault == 'discard':
            raise RuntimeError('allocator failure')
        self.receiver.unregister_memory(self.target)
        self.target_private = False

    def release(self):
        self.assertEqual(self.tx.unmapped_count, 1)
        self.events.append('release')
        if self.fault == 'release':
            raise RuntimeError('ACK lost')
        self.owner.release_export(self.descriptor.lease_id)
        self.source_pinned = False

    def assert_quarantined(self, events=()):
        self.assertEqual(self.tx.state, RestoreState.QUARANTINED)
        self.assertTrue(self.source_pinned)
        self.assertEqual(self.events, list(events))
        for op in (self.tx.start, self.tx.wait, self.tx.cancel):
            with self.assertRaises(RuntimeError): op()
        self.assertEqual(self.events, list(events))

    def test_success_and_duplicate_completion(self):
        self.driver.state = 1
        self.tx.start()
        self.assertEqual(self.tx.wait(), RestoreState.PUBLISHED)
        self.assertEqual(self.tx.wait(), RestoreState.PUBLISHED)
        self.assertEqual(self.events, ['publish', 'release'])
        self.assertFalse(self.source_pinned)
        self.owner.close(); self.receiver.close()

    def test_timeout_then_success(self):
        self.tx.start()
        with self.assertRaises(TimeoutError): self.tx.wait(0)
        self.assertEqual(self.tx.state, RestoreState.IN_FLIGHT)
        self.assertEqual(self.events, [])
        self.assertTrue(self.source_pinned and self.target_private)
        with self.assertRaises(BusyError): self.receiver.unregister_memory(self.target)
        self.driver.state = 1
        self.assertEqual(self.tx.wait(), RestoreState.PUBLISHED)

    def test_drained_failure_discards_without_publish(self):
        self.driver.state = 2
        self.tx.start()
        self.assertEqual(self.tx.wait(), RestoreState.DISCARDED)
        self.assertEqual(self.tx.wait(), RestoreState.DISCARDED)
        self.assertEqual(self.events, ['discard', 'release'])
        self.owner.close(); self.receiver.close()

    def test_cancel_before_submit(self):
        self.assertEqual(self.tx.cancel(), RestoreState.DISCARDED)
        self.tx.cancel(); self.tx.wait()
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual(self.events, ['discard', 'release'])
        self.owner.close(); self.receiver.close()

    def test_inflight_cancel_retains_pages_until_drain_and_never_publishes(self):
        self.tx.start()
        for _ in range(2):
            self.assertEqual(self.tx.cancel(), RestoreState.IN_FLIGHT)
            self.assertEqual(self.tx.poll(), RestoreState.IN_FLIGHT)
        self.assertEqual(self.events, [])
        self.assertTrue(self.source_pinned and self.target_private)
        self.driver.state = 1
        self.assertEqual(self.tx.poll(), RestoreState.DISCARDED)
        self.assertEqual(self.events, ['discard', 'release'])
        self.tx.cancel(); self.tx.poll()
        self.assertEqual(self.events, ['discard', 'release'])
        self.owner.close(); self.receiver.close()

    def test_cancel_after_publish_waits_for_ack_without_discard(self):
        self.driver.state = 1
        self.tx._release_source = Mock(return_value=False)
        self.tx.start()
        self.assertEqual(self.tx.poll(), RestoreState.RELEASING)
        self.assertEqual(self.tx.cancel(), RestoreState.RELEASING)
        self.assertEqual(self.events, ['publish'])
        self.tx._release_source = self.release
        self.assertEqual(self.tx.poll(), RestoreState.PUBLISHED)
        self.assertEqual(self.events, ['publish', 'release'])
        self.owner.close(); self.receiver.close()

    def test_cancel_does_not_turn_uncertain_dma_into_safe_failure(self):
        self.tx.start(); self.tx.cancel()
        self.driver.state = 3
        self.tx.poll()
        self.assert_quarantined()

    def test_uncertain_retains_both_sides(self):
        self.driver.state = 3
        self.tx.start(); self.tx.wait()
        self.assert_quarantined()
        with self.assertRaises(BusyError): self.receiver.close()
        with self.assertRaises(BusyError): self.owner.close()

    def test_partial_submit_exception(self):
        self.driver.raise_submit = True
        with self.assertRaises(RuntimeError): self.tx.start()
        self.assert_quarantined()

    def test_poll_exception(self):
        self.driver.raise_poll = True
        self.tx.start(); self.tx.wait()
        self.assert_quarantined()

    def test_publish_exception_never_discards(self):
        self.fault = 'publish'; self.driver.state = 1
        self.tx.start()
        with self.assertRaises(RuntimeError): self.tx.wait()
        self.assert_quarantined(['publish'])
        self.assertFalse(self.target_private)  # partial adoption cannot be rolled back

    def test_discard_exception_never_releases_source(self):
        self.fault = 'discard'; self.driver.state = 2
        self.tx.start()
        with self.assertRaises(RuntimeError): self.tx.wait()
        self.assert_quarantined(['discard'])

    def test_lost_ack_does_not_republish(self):
        self.fault = 'release'; self.driver.state = 1
        self.tx.start()
        with self.assertRaises(RuntimeError): self.tx.wait()
        self.assertTrue(self.tx.published)
        self.assert_quarantined(['publish', 'release'])

    def test_unmap_failure_stops_before_publish(self):
        original = self.receiver.unregister_memory
        def fail(handle):
            if handle is self.remote: raise RuntimeError('unmap failed')
            original(handle)
        self.receiver.unregister_memory = fail
        self.driver.state = 1; self.tx.start()
        with self.assertRaises(RuntimeError): self.tx.wait()
        self.assert_quarantined()

    def test_release_batch_failure_stops_before_unmap(self):
        def fail(batch): raise RuntimeError('batch release failed')
        self.receiver.release_batch = fail
        self.driver.state = 1; self.tx.start()
        with self.assertRaises(RuntimeError): self.tx.wait()
        self.assertEqual(self.tx.unmapped_count, 0)
        self.assert_quarantined()

    def test_inconsistent_completion_is_not_publishable(self):
        self.driver.state = 1; self.tx.start()
        original = self.receiver.wait
        self.receiver.wait = lambda *args: replace(original(*args), completed_bytes=1)
        self.tx.wait()
        self.assert_quarantined()

    def test_reentry_and_foreign_thread_rejected(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaises(RuntimeError): executor.submit(self.tx.start).result()
        self.assertEqual(self.tx.state, RestoreState.PREPARED)
        self.fault = 'reentry'; self.driver.state = 1; self.tx.start()
        with self.assertRaises(RuntimeError): self.tx.wait()
        self.assert_quarantined(['publish'])

    def test_unused_import_is_also_unmapped(self):
        extra_source = self.owner.allocate_memory(64)
        extra_export = self.owner.export_memory(extra_source)
        extra_import = self.receiver.import_memory(extra_export)
        events = []
        tx = RestoreTransaction(self.receiver, [self.request], [self.remote, extra_import],
            publish=lambda: events.append('publish'), discard=lambda: events.append('discard'),
            release_source=lambda: events.append('release'))
        self.driver.state = 1; tx.start(); tx.wait()
        self.assertEqual(tx.unmapped_count, 2)
        self.assertEqual(events, ['publish', 'release'])
        self.owner.release_export(extra_export.lease_id)
        self.owner.release_export(self.descriptor.lease_id)
        self.owner.close(); self.receiver.close()

    def test_poll_does_not_wait_or_repeat_publication(self):
        self.receiver.wait = Mock(side_effect=AssertionError('blocking wait called'))
        self.tx.start()
        self.assertEqual(self.tx.poll(), RestoreState.IN_FLIGHT)
        self.assertEqual(self.events, [])
        release = Mock(return_value=False)
        self.tx._release_source = release
        self.driver.state = 1
        self.assertEqual(self.tx.poll(), RestoreState.RELEASING)
        self.assertEqual(self.events, ['publish'])
        self.assertEqual(self.tx.unmapped_count, 1)
        self.assertFalse(self.tx.source_released)
        self.assertEqual(self.tx.poll(), RestoreState.RELEASING)
        release.side_effect = self.release
        self.assertEqual(self.tx.poll(), RestoreState.PUBLISHED)
        self.assertEqual(self.tx.poll(), RestoreState.PUBLISHED)
        self.assertEqual(self.events, ['publish', 'release'])
        self.receiver.wait.assert_not_called()

    def test_optimistic_source_publishes_only_a_read_its_release_validates(self):
        """An optimistic source may reuse the pages mid-read: the target publishes only
        after the release ACK reports the read intact, and discards a stale read
        (a fallback, not a fault) even though the copy itself succeeded."""
        for valid, expected in ((True, ['release', 'publish']), (False, ['release', 'discard'])):
            events = []
            descriptor = self.owner.export_memory(self.source)
            remote = self.receiver.import_memory(descriptor)
            answers = iter((False, {'released': True, 'valid': valid}))
            def release():
                answer = next(answers)
                if answer is not False:
                    events.append('release')
                return answer
            tx = RestoreTransaction(self.receiver, [replace(self.request, remote=remote)], [remote],
                publish=lambda: events.append('publish'), discard=lambda: events.append('discard'),
                release_source=release, validate_on_release=True)
            self.driver.state = 1; tx.start()
            self.assertEqual(tx.poll(), RestoreState.RELEASING)
            self.assertEqual(events, [])  # ACK pending: nothing settled yet
            self.assertEqual(tx.poll(), RestoreState.PUBLISHED if valid else RestoreState.DISCARDED)
            self.assertEqual(events, expected)
            self.assertEqual(tx.stale, not valid)
            self.owner.release_export(descriptor.lease_id)
        missing = RestoreTransaction(self.receiver, [self.request], [self.remote], publish=self.publish,
            discard=self.discard, release_source=lambda: {'released': True}, validate_on_release=True)
        missing.start()
        with self.assertRaises(RuntimeError): missing.wait()  # an ACK without validity is a fault
        self.assertEqual(missing.state, RestoreState.QUARANTINED)
        self.assertEqual(self.events, [])  # neither published nor discarded

    def test_retained_import_is_reused_and_never_unmapped_by_a_restore(self):
        """With retain_imports a peer region is mapped once: each later lease of it reuses
        that mapping, a finished restore leaves it mapped, and a changed descriptor for the
        same region is refused rather than silently reusing the old mapping."""
        receiver = NvlinkEngine(1, _driver=self.driver, retain_imports=True)
        first = self.owner.export_memory(self.source)
        mapped = receiver.import_memory(first)
        second = self.owner.export_memory(self.source)
        self.assertIs(receiver.import_memory(second), mapped)
        target = receiver.allocate_memory(64)
        tx = RestoreTransaction(receiver, [TransferRequest(Opcode.READ, target, mapped, 0, 0, 8)], [mapped],
            publish=lambda: None, discard=lambda: None, release_source=lambda: {'released': True})
        self.driver.state = 1; tx.start()
        self.assertEqual(tx.wait(), RestoreState.PUBLISHED)
        self.assertEqual((tx.unmapped_count, tx.retained_count), (0, 1))
        self.assertTrue(receiver.retains(mapped))
        with self.assertRaises(ValueError):
            receiver.import_memory(replace(self.owner.export_memory(self.source), size=32))
        for d in (first, second):
            self.owner.release_export(d.lease_id)

    def test_concurrent_reads_share_regions_only_with_disjoint_destinations(self):
        """Several restores copy into one pool region at once: with concurrent_reads two
        READ batches may be in flight on the same local and remote regions if their
        destination bytes are disjoint; an overlap is refused, and without the option
        the second batch is refused as before."""
        for concurrent in (False, True):
            receiver = NvlinkEngine(1, _driver=self.driver, concurrent_reads=concurrent)
            remote = receiver.import_memory(self.owner.export_memory(self.source))
            target = receiver.allocate_memory(64)
            self.driver.state = 0  # keep batches in flight
            receiver.submit_transfer([TransferRequest(Opcode.READ, target, remote, 0, 0, 8)])
            disjoint = [TransferRequest(Opcode.READ, target, remote, 8, 8, 8)]
            if not concurrent:
                with self.assertRaises(BusyError): receiver.submit_transfer(disjoint)
                continue
            receiver.submit_transfer(disjoint)
            with self.assertRaises(BusyError):
                receiver.submit_transfer([TransferRequest(Opcode.READ, target, remote, 4, 16, 8)])

    def test_plan_contract(self):
        for requests, imports in (([], []), ([replace(self.request, opcode=Opcode.WRITE)], [self.remote]),
                                  ([self.request], []), ([self.request], [self.remote, self.remote])):
            with self.assertRaises(ValueError):
                RestoreTransaction(self.receiver, requests, imports, publish=self.publish,
                                   discard=self.discard, release_source=self.release)
        self.assertEqual(self.events, [])
        self.tx.cancel()


if __name__ == '__main__': unittest.main()
