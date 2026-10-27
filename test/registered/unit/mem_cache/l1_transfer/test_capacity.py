from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.l1_transfer.peer.capacity import RequestCapacity
from sglang.srt.mem_cache.l1_transfer.peer.scheduler import SchedulerPeer


def req(rid, prompt=23, output=9, allocated=0):
    return NS(rid=rid, origin_input_ids=list(range(prompt)), customized_info=None,
              sampling_params=NS(max_new_tokens=output),
              kv=NS(kv_allocated_len=allocated), finished=lambda: False)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.cache = NS(token_to_kv_pool_allocator=NS(page_size=16, size=128,
                        available_size=Mock(return_value=64)), evictable_size=lambda: 0)
        self.c = RequestCapacity(self.cache)

    def test_full_output_rounding_and_no_cache_hit_discount(self):
        r = req('a', output=10)
        self.assertEqual(self.c.bound(r), 48)
        self.assertTrue(self.c.can_admit(r, 16))
        self.assertFalse(self.c.can_admit(r, 17))

    def test_multiple_admissions_charge_before_allocation(self):
        a, b, c = req('a'), req('b'), req('c')
        self.c.admit(a)
        self.assertTrue(self.c.can_admit(b))
        self.c.admit(b)
        self.assertFalse(self.c.can_admit(c))
        self.c.admit(a)
        self.assertEqual(self.c.remaining(), 64)

    def test_allocated_pages_and_future_growth_are_not_double_counted(self):
        a = req('a', allocated=17)
        self.c.admit(a)
        self.assertEqual(self.c.remaining(), 0)
        a.kv.kv_allocated_len = 16
        self.assertEqual(self.c.remaining(), 16)
        a.kv.kv_allocated_len = 0  # retraction preserves full entitlement
        self.assertEqual(self.c.remaining(), 32)
        a.finished = lambda: True
        self.c.prune()
        self.assertEqual(self.c.remaining(), 0)

    def test_waiting_restore_stops_new_admission_but_not_running_compute(self):
        p = object.__new__(SchedulerPeer)
        p.concurrent = True
        p.source = NS(active={})
        p.capacity, p.unadmitted = self.c, {}
        p.request, p.receiver, p.phase = req('shared'), None, 'WAIT_CAPACITY'
        running, fresh = req('running'), req('fresh')
        self.c.admit(running)
        self.assertFalse(p.blocks_compute)
        self.assertTrue(p.allows_request(running))
        self.assertFalse(p.allows_request(fresh))

    def test_private_prefix_only_reserves_unallocated_tail(self):
        p = object.__new__(SchedulerPeer)
        p.concurrent, p.capacity, p.phase, p.unadmitted = True, self.c, 'WAIT_TRANSFER', {}
        p.source = NS(active={})
        p.request, p.required_capacity = req('shared'), 32
        p.receiver = NS(pending={'reservation': NS(indices=range(16))})
        self.assertEqual(p.reserved_headroom(), 16)
        self.assertFalse(p.blocks_compute)
        self.assertTrue(p.allows_request(req('a')))
        self.c.admit(req('a'))
        self.assertFalse(p.allows_request(req('b')))
        p.note_admitted(p.request)
        self.assertEqual(p.reserved_headroom(), 0)

    def test_concurrent_finish_does_not_device_sync_unrelated_dma(self):
        p = object.__new__(SchedulerPeer)
        p.concurrent, p.receiver = True, Mock()
        p.engine, p.stats = NS(device=0), Mock(return_value={})
        with patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.torch.cuda.synchronize') as sync:
            p.finish()
        sync.assert_not_called()
        self.assertIsNone(p.receiver)

    def test_copy_waits_for_prior_schedule_stream_before_concurrent_forward(self):
        p = object.__new__(SchedulerPeer)
        p.concurrent = True
        p.cancel_reason, p.receiver = None, None
        p.scheduler = NS(schedule_stream=NS(cuda_stream=123))
        tx = Mock()
        p._prepared({'transaction': tx})
        tx.start.assert_called_once_with(producer_stream=123)
        self.assertEqual(p.phase, 'WAIT_TRANSFER')

    def test_busy_target_can_reserve_without_waiting_for_engine_idle(self):
        p = object.__new__(SchedulerPeer)
        p.concurrent, p.capacity, p.required_capacity, p.unadmitted = True, self.c, 32, {}
        p.scheduler = NS(_engine_paused=False, is_fully_idle=Mock(return_value=False))
        p.cache, p.adapter, p.identity, p.rpc_executor = self.cache, None, None, None
        p.payload = dict(host='source', port=1, tokens=list(range(16)))
        p.restore_id = 'test'
        p._prepared = Mock()
        receiver = Mock()
        receiver.prepare_steps.return_value = iter([('acquire', {})])
        with patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.AsyncRPC'), patch(
                'sglang.srt.mem_cache.l1_transfer.peer.scheduler.PeerReceiver', return_value=receiver):
            p._try_capacity()
        p.scheduler.is_fully_idle.assert_not_called()
        receiver.prepare_steps.assert_called_once_with(p.payload['tokens'], required_capacity=32,
                                                       headroom=0, private=True)
        self.assertEqual(p.phase, 'WAIT_ACQUIRE')

    def test_ready_request_frees_slot_but_keeps_growth_reserved_until_admission(self):
        p = object.__new__(SchedulerPeer)
        p.concurrent, p.capacity, p.unadmitted, p.consumers = True, self.c, {}, {}
        p.scheduler = NS(_add_request_to_queue=Mock())
        p.stats = Mock(return_value={})
        p.rpc, p.attempts, p.poll_counts, p.restore_id = Mock(), [], {}, 'id'
        shared = p.request = req('shared')
        p.required_capacity = 48
        reservation = NS(indices=range(16), prefix_len=0)
        receiver = p.receiver = Mock(pending=None)
        p.session = {'reservation': reservation, 'transaction': None}
        p._ready()
        self.assertIsNone(p.request)
        self.assertIsNone(p.receiver)
        p.scheduler._add_request_to_queue.assert_called_once_with(shared)
        self.assertEqual(p.reserved_headroom(), 32)
        self.assertTrue(p.allows_request(shared))
        self.assertFalse(p.allows_request(req('fresh', output=10)))  # 48 + 32 > 64
        p.note_admitted(shared)
        self.assertEqual(p.reserved_headroom(), 0)
        shared.finished_reason, shared.to_finish = 'done', None
        shared.finished = lambda: True
        shared._l1_consumer_release()
        receiver.finish_consumer.assert_called_once()
        self.assertEqual(p.request_result('shared')['state'], 'FINISHED')
        self.assertTrue(p.request_result('shared')['consumer_released'])

    def test_safe_rejection_preserves_active_request_and_prior_result(self):
        p = object.__new__(SchedulerPeer)
        active = p.request = req('active')
        p.receiver = object()
        p.scheduler = NS(finish_l1_waiting_request=Mock())
        p._record_result('earlier', {'state': 'FINISHED'})
        p.reject_request(req('second'), 'busy')
        self.assertIs(p.request, active)
        self.assertEqual(p.request_result('earlier')['state'], 'FINISHED')
        self.assertTrue(p.request_result('second')['consumer_released'])
        self.assertEqual(p.request_result('second')['reason'], 'busy')

    def test_beam_expansion_is_rejected_before_budget_admission(self):
        r = req('beam')
        r.sampling_params.beam_width = 2
        with self.assertRaisesRegex(ValueError, 'beam search'):
            self.c.can_admit(r)
        self.assertFalse(self.c.active)

    def test_non_generation_engine_is_rejected_before_transport_creation(self):
        args = NS(tp_size=1, pp_size=1, dp_size=1, disaggregation_mode='null',
                  speculative_algorithm=None, enable_lora=False, max_running_requests=4)
        scheduler = NS(server_args=args, is_generation=False)
        with patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.NvlinkEngine') as engine:
            with self.assertRaisesRegex(ValueError, 'generation Engine'):
                SchedulerPeer(scheduler, {}, concurrent=True)
        engine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
