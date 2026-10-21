"""Owner-thread lifecycle faults; real SchedulerPeer, controlled network futures."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.l1_transfer.peer.scheduler import SchedulerPeer
from sglang.srt.mem_cache.l1_transfer.peer.restore import RestoreState
from sglang.srt.mem_cache.l1_transfer.peer.control import RemoteRejected

MODULE = 'sglang.srt.mem_cache.l1_transfer.peer.scheduler'


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.peer = p = object.__new__(SchedulerPeer)
        p.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(disable_overlap_schedule=True, page_size=16),
            _engine_paused=False, is_fully_idle=Mock(return_value=True),
            _add_request_to_queue=Mock(), finish_l1_waiting_request=Mock())
        p.cache = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(size=128))
        p.adapter = p.identity = p.rpc_executor = None
        p.engine = SimpleNamespace(device=0)
        p.source = SimpleNamespace(active={})
        p.request = p.receiver = p.service = p.phase = p.deadline = None
        p.cancel_reason = None
        p.stats = Mock(return_value={'target_active': False, 'pinned_pages': 0})
        self.req = SimpleNamespace(rid='r1', origin_input_ids=list(range(23)),
                                   sampling_params=SimpleNamespace(max_new_tokens=8))
        self.options = dict(source={'host':'source', 'port':1234}, prefix_tokens=16,
                            mode='nonblocking', wait_timeout_ms=30000)

    def start(self):
        self.peer.start_request(self.req, self.options)

    def acquiring(self):
        self.start()
        p = self.peer
        p.phase = 'WAIT_ACQUIRE'
        p.rpc = Mock()
        p.acquire_future = Future()
        p.receiver = Mock()
        p.receiver.pending = {'reservation': object()}
        p.steps = Mock()
        return p

    def transferring(self, state):
        p = self.acquiring()
        p.phase = 'WAIT_TRANSFER'
        tx = Mock()
        tx.poll.return_value = state
        p.session = dict(transaction=tx)
        return p, tx

    def assert_terminal(self, reason):
        p = self.peer
        self.assertIsNone(p.request)
        self.assertIsNone(p.phase)
        self.assertEqual(p.required_capacity, 0)
        self.assertEqual(p.request_result('r1')['reason'], reason)
        p.scheduler.finish_l1_waiting_request.assert_called_once_with(self.req, reason)
        p.scheduler._add_request_to_queue.assert_not_called()

    def test_capacity_wait_holds_nothing_and_does_not_block_ordinary_compute(self):
        self.start()
        p = self.peer
        p.scheduler.is_fully_idle.return_value = False
        with patch(MODULE + '.AsyncRPC') as rpc:
            p._pump()
        rpc.assert_not_called()
        self.assertFalse(p.blocks_compute)
        self.assertIsNone(p.receiver)
        self.assertTrue(p.allows_request(object()))
        p.cancel_request(); p.cancel_request()
        self.assert_terminal('CANCELLED')

    def test_capacity_deadline_never_contacts_source(self):
        self.start()
        self.peer.deadline = 0
        with patch(MODULE + '.AsyncRPC') as rpc:
            self.peer._pump()
        rpc.assert_not_called()
        self.assert_terminal('WAIT_TIMEOUT')

    def test_maximum_output_is_budgeted_before_any_lease(self):
        self.req.sampling_params.max_new_tokens = 120
        self.start()
        self.assert_terminal('CAPACITY_EXCEEDED')

    def test_second_pending_request_is_rejected_without_replacing_first(self):
        self.start()
        with self.assertRaises(ValueError):
            self.peer.start_request(self.req, self.options)
        self.assertIs(self.peer.request, self.req)

    def test_transient_allocation_failure_retries_without_retaining_receiver(self):
        self.start()
        p = self.peer
        def capacity_miss(*args, **kwargs):
            raise MemoryError('temporary pressure')
            yield  # generator matches the real preparation API
        receiver = Mock(pending=None)
        receiver.prepare_steps.side_effect = capacity_miss
        with patch(MODULE + '.AsyncRPC') as rpc, patch(MODULE + '.PeerReceiver', return_value=receiver):
            p._pump(); p._pump()
            self.assertEqual(rpc.return_value.close.call_count, 2)
            rpc.return_value.submit.assert_not_called()
        self.assertIsNone(p.receiver)
        self.assertEqual(p.phase, 'WAIT_CAPACITY')

    def test_partial_prepare_is_not_treated_as_capacity_miss(self):
        self.start()
        receiver = Mock(pending={'imports':['uncertain']})
        def broken(*args, **kwargs):
            raise MemoryError('tracking failure')
            yield
        receiver.prepare_steps.side_effect = broken
        with patch(MODULE + '.AsyncRPC'), patch(MODULE + '.PeerReceiver', return_value=receiver):
            with self.assertRaises(MemoryError):
                self.peer._pump()
        self.assertIs(self.peer.receiver, receiver)

    def test_cancel_acquire_waits_for_late_offer_then_does_not_submit_dma(self):
        p = self.acquiring()
        p.cancel_request(); p._pump()
        p.receiver.finish_consumer.assert_not_called()
        self.assertTrue(p.blocks_compute)
        tx = Mock()
        p.steps.send.side_effect = StopIteration({'transaction':tx})
        tx.poll.return_value = RestoreState.RELEASING
        p.acquire_future.set_result({'valid':'offer'})
        p._pump()
        tx.cancel.assert_called_once()
        tx.start.assert_not_called()
        p.scheduler._add_request_to_queue.assert_not_called()
        tx.poll.return_value = RestoreState.DISCARDED
        with patch(MODULE + '.torch.cuda.synchronize'):
            p._pump()
        self.assert_terminal('CANCELLED')

    def test_explicit_acquire_rejection_is_safe_failure(self):
        p = self.acquiring()
        p.acquire_future.set_exception(RemoteRejected('source miss'))
        p.steps.throw.side_effect = RemoteRejected('source miss')
        with patch(MODULE + '.torch.cuda.synchronize'):
            p._pump()
        self.assert_terminal('SOURCE_REJECTED')

    def test_lost_acquire_reply_retains_every_owner_even_after_cancel(self):
        p = self.acquiring()
        p.cancel_request()
        p.acquire_future.set_exception(ConnectionError('lost offer'))
        with self.assertRaises(ConnectionError):
            p._pump()
        self.assertIsNotNone(p.receiver)
        p.receiver.finish_consumer.assert_not_called()
        p.scheduler.finish_l1_waiting_request.assert_not_called()

    def test_drained_failure_reports_without_forward(self):
        p, tx = self.transferring(RestoreState.DISCARDED)
        with patch(MODULE + '.torch.cuda.synchronize'):
            p._pump()
        self.assert_terminal('TRANSFER_FAILED_DRAINED')

    def test_deadline_during_dma_waits_for_drain(self):
        p, tx = self.transferring(RestoreState.IN_FLIGHT)
        p.deadline = 0
        p._pump()
        tx.cancel.assert_called_once()
        p.receiver.finish_consumer.assert_not_called()
        tx.poll.return_value = RestoreState.DISCARDED
        with patch(MODULE + '.torch.cuda.synchronize'):
            p._pump()
        self.assert_terminal('WAIT_TIMEOUT')

    def test_cancel_after_publication_releases_consumer_only_after_ack(self):
        p, tx = self.transferring(RestoreState.RELEASING)
        p.phase = 'WAIT_RELEASE'
        p.cancel_request(); p._pump()
        p.receiver.finish_consumer.assert_not_called()
        tx.poll.return_value = RestoreState.PUBLISHED
        with patch(MODULE + '.torch.cuda.synchronize'):
            p._pump()
        self.assert_terminal('CANCELLED')

    def test_lost_ack_and_uncertain_dma_cannot_report_safe_failure(self):
        for status in (RestoreState.QUARANTINED, None):
            with self.subTest(status=status):
                self.setUp()
                p, tx = self.transferring(status)
                if status is None:
                    tx.poll.side_effect = ConnectionError('lost ACK')
                with self.assertRaises((RuntimeError, ConnectionError)):
                    p._pump()
                p.receiver.finish_consumer.assert_not_called()
                p.scheduler.finish_l1_waiting_request.assert_not_called()

    def test_reserved_headroom_excludes_unrelated_request_until_finish(self):
        p = self.acquiring()
        self.assertFalse(p.allows_request(object()))
        p.phase = 'READY'
        self.assertFalse(p.allows_request(object()))
        self.assertTrue(p.allows_request(self.req))

    def test_ready_queue_abort_releases_consumer_before_cache_finish(self):
        from sglang.srt.managers.scheduler import Scheduler
        events = []
        req = SimpleNamespace(finished_reason=None, cache_request_handle=object(), session=None,
                              release_mm_inputs_on_abort=lambda: events.append('mm'))
        req.finished = lambda: req.finished_reason is not None
        def release():
            self.assertTrue(req.finished())
            events.append('consumer')
            req._l1_consumer_release = None
        req._l1_consumer_release = release
        scheduler = SimpleNamespace(tree_cache=SimpleNamespace(
            finish=lambda *args: events.append('cache')))
        Scheduler._release_aborted_request(scheduler, req)
        self.assertEqual(events, ['consumer', 'mm', 'cache'])
        Scheduler._release_aborted_request(scheduler, req)
        self.assertEqual(events.count('consumer'), 1)

    def test_running_abort_syncs_before_releasing_pin(self):
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        p = self.acquiring()
        p.phase = 'READY'
        p.restored = {'state':'READY'}
        self.req.finished_reason = None
        self.req.to_finish = FINISH_ABORT('memory pressure')
        self.req.finished = lambda: False
        receiver = p.receiver
        with patch(MODULE + '.torch.cuda.synchronize') as sync:
            sync.side_effect = lambda *args: receiver.finish_consumer.assert_not_called()
            p.finish_request()
        receiver.finish_consumer.assert_called_once()
        self.assertEqual(p.request_result('r1')['reason'], 'GENERATION_ABORTED')
        self.assertEqual(p.request_result('r1')['state'], 'FAILED')

    def test_pump_quarantines_unknown_state_without_terminal_callback(self):
        p, tx = self.transferring(RestoreState.QUARANTINED)
        with patch(MODULE + '.os._exit', side_effect=RuntimeError('fail-stop')) as stop:
            with self.assertRaisesRegex(RuntimeError, 'fail-stop'):
                p.pump()
        stop.assert_called_once_with(70)
        p.receiver.finish_consumer.assert_not_called()
        p.scheduler.finish_l1_waiting_request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
