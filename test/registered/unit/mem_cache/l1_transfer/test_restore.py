# SPDX-License-Identifier: Apache-2.0
"""Joint page/region lifetime checks using the real transport API and fault driver."""
import unittest
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
        with self.assertRaises(RuntimeError): self.tx.cancel()
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

    def test_plan_contract(self):
        for requests, imports in (([], []), ([replace(self.request, opcode=Opcode.WRITE)], [self.remote]),
                                  ([self.request], []), ([self.request], [self.remote, self.remote])):
            with self.assertRaises(ValueError):
                RestoreTransaction(self.receiver, requests, imports, publish=self.publish,
                                   discard=self.discard, release_source=self.release)
        self.assertEqual(self.events, [])
        self.tx.cancel()


if __name__ == '__main__': unittest.main()
