# SPDX-License-Identifier: Apache-2.0
import socket
from queue import Queue
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.l1_transfer.peer.nonblocking import AsyncRPC, SourceService
from sglang.srt.mem_cache.l1_transfer.peer.scheduler import SchedulerPeer, parse_l1_config, scheduler_peer_rpc
from sglang.srt.mem_cache.l1_transfer.peer.restore import RestoreState
from sglang.srt.mem_cache.l1_transfer.peer.control import RemoteRejected
from sglang.srt.mem_cache.l1_transfer.peer.peer import SourceEndpoint


class NonblockingTests(unittest.TestCase):
    def test_shared_worker_survives_transaction_close(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = AsyncRPC('localhost', 1, executor=executor)
            second = AsyncRPC('localhost', 1, executor=executor)
            first.close()
            self.assertEqual(second.executor.submit(lambda: 42).result(timeout=2), 42)
            second.close()

    def test_idle_service_does_not_enter_unrelated_zmq_wait(self):
        from unittest.mock import patch
        from sglang.srt.managers.scheduler import Scheduler
        scheduler = SimpleNamespace(_l1_peer=SimpleNamespace(service=object(), poll_interval=0.0001),
                                    idle_sleeper=Mock())
        with patch('sglang.srt.managers.scheduler.time.sleep') as sleep:
            Scheduler.maybe_sleep_on_idle(scheduler)
        sleep.assert_called_once_with(0.0001)
        scheduler.idle_sleeper.maybe_sleep.assert_not_called()

    def peer(self):
        peer = object.__new__(SchedulerPeer)
        peer.service = None
        peer.source = SimpleNamespace(active={})
        peer.phase = 'WAIT_ACQUIRE'
        peer.deadline = None
        peer.cancel_reason = None
        peer.request = SimpleNamespace(rid='consumer')
        peer.queue, peer.consumers = [], {}
        peer.poll_counts = dict(WAIT_ACQUIRE=0, WAIT_TRANSFER=0, WAIT_RELEASE=0)
        peer.acquire_future = Future()
        peer.scheduler = SimpleNamespace(_add_request_to_queue=Mock())
        return peer

    def test_pending_acquire_yields_without_enqueue(self):
        peer = self.peer()
        for _ in range(4):
            peer._pump()
        self.assertTrue(peer.blocks_compute)
        self.assertEqual(peer.poll_counts['WAIT_ACQUIRE'], 4)
        peer.scheduler._add_request_to_queue.assert_not_called()

    def test_ready_requires_release_ack(self):
        peer = self.peer()
        peer.phase = 'WAIT_TRANSFER'
        tx = Mock()
        tx.poll.side_effect = [RestoreState.IN_FLIGHT, RestoreState.RELEASING, RestoreState.PUBLISHED]
        peer.session = {'transaction': tx}
        peer._ready = Mock()
        peer._pump(); peer._pump()
        self.assertEqual(peer.phase, 'WAIT_RELEASE')
        peer._ready.assert_not_called()
        peer._pump()
        peer._ready.assert_called_once()

    def test_stats_allowed_while_busy_but_mutation_rejected(self):
        peer = self.peer()
        peer.stats = Mock(return_value={'wait_state':'WAIT_ACQUIRE'})
        scheduler = SimpleNamespace(is_fully_idle=lambda:False, _l1_peer=peer)
        self.assertIn('WAIT_ACQUIRE', scheduler_peer_rpc(scheduler, 'stats', '{}'))
        with self.assertRaises(RuntimeError):
            scheduler_peer_rpc(scheduler, 'finish', '{}')

    def test_network_service_dispatches_on_owner_across_connections(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        service = SourceService(listener)
        for _ in range(2):
            rpc = AsyncRPC(*listener.getsockname())
            acquire = rpc.submit('acquire', {'tokens':[1]})
            kind, request, result = service.events.get(timeout=2)
            self.assertEqual(kind, 'request')
            self.assertEqual(request['operation'], 'acquire')
            self.assertFalse(acquire.done())
            result.set_result({'offer':1})
            self.assertEqual(acquire.result(timeout=2), {'offer':1})
            self.assertIsNone(rpc.call('release', receipt={}))
            kind, request, result = service.events.get(timeout=2)
            self.assertEqual(request['operation'], 'release')
            result.set_result({'released':True})
            self.assertEqual(rpc.release_future.result(timeout=2), {'released':True})
            self.assertEqual(service.events.get(timeout=2)[0], 'closed')
            rpc.close()
        # Listener belongs to the process-lifetime daemon; avoid closing a live
        # accept from another thread. It is reclaimed when the test process exits.

    def test_unreachable_source_is_a_safe_rejection(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        address = listener.getsockname()
        listener.close()  # nothing listens: connect is refused before any request is sent
        rpc = AsyncRPC(*address)
        with self.assertRaises(RemoteRejected):
            rpc.submit('acquire', {'tokens': [1]}).result(timeout=5)
        rpc.close()

    def test_service_accepts_overlapping_connections(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0)); listener.listen(4)
        service = SourceService(listener)
        rpcs = [AsyncRPC(*listener.getsockname()) for _ in range(2)]
        futures = [rpc.submit('acquire', {'tokens': [i]}) for i, rpc in enumerate(rpcs)]
        # Both acquires reach the owner before either is answered.
        pending = [service.events.get(timeout=2) for _ in rpcs]
        self.assertEqual(sorted(e[1]['tokens'] for e in pending), [[0], [1]])
        for _, request, result in pending:
            result.set_exception(RemoteRejected('miss'))
        for future in futures:
            with self.assertRaises(RemoteRejected):
                future.result(timeout=2)
        closed = [service.events.get(timeout=2) for _ in rpcs]
        self.assertEqual([(kind, lease) for kind, lease, _ in closed], [('closed', None)] * 2)
        for rpc in rpcs:
            rpc.close()

    def online_source(self, *, max_leases=2, max_lease_pages=3):
        source = object.__new__(SourceEndpoint)
        source.contract = {'model': 'm'}
        source.ready, source.max_leases, source.max_lease_pages = None, max_leases, max_lease_pages
        source.active, source.pending, source.released = {}, {}, {}
        source.adapter = SimpleNamespace(layout=SimpleNamespace(page_size=2), memories=[SimpleNamespace(region_id='r')],
                                         capacity=8, bindings={}, engine=Mock())
        source.adapter.engine.any_in_flight.return_value = False
        source.adapter.engine.export_memory.return_value = SimpleNamespace(to_dict=lambda: {'region': 'r'}, lease_id='x')
        def acquire_source_lease(key, *, fingerprint, ready):
            ready()
            return SimpleNamespace(page_indices=tuple(range(len(key) // 2)), incarnation='i', epoch=0, lease_id='l')
        source.cache = SimpleNamespace(source_leases=SimpleNamespace(_check_thread=lambda: None),
                                       acquire_source_lease=Mock(side_effect=acquire_source_lease))
        return source

    @staticmethod
    def acquire(rid, tokens=4):
        return dict(version=1, operation='acquire', id=rid * 32, contract={'model': 'm'}, tokens=list(range(tokens)))

    def test_online_source_offers_only_after_producer_event(self):
        source = self.online_source()
        event = Mock()
        event.query.return_value = False
        self.assertIsNone(source.dispatch_deferred(self.acquire('a'), record_producer=lambda: event))
        self.assertEqual(source.poll_ready(), [])
        source.adapter.engine.export_memory.assert_not_called()  # no descriptor before KV is written
        event.query.return_value = True
        source.adapter.engine.any_in_flight.return_value = True  # local copy still in flight
        self.assertEqual(source.poll_ready(), [])
        source.adapter.engine.any_in_flight.return_value = False
        [(rid, offer)] = source.poll_ready()
        self.assertEqual((rid, offer['pages']), ('a' * 32, [0, 1]))
        self.assertIn(rid, source.active)
        self.assertFalse(source.pending)

    def test_source_quota_rejects_before_locking(self):
        source = self.online_source()
        done = Mock(query=Mock(return_value=False))
        source.dispatch_deferred(self.acquire('a'), record_producer=lambda: done)
        with self.assertRaisesRegex(RemoteRejected, 'page quota'):
            source.dispatch_deferred(self.acquire('b'), record_producer=lambda: done)
        source.dispatch_deferred(self.acquire('c', tokens=2), record_producer=lambda: done)
        with self.assertRaisesRegex(RemoteRejected, 'lease quota'):
            source.dispatch_deferred(self.acquire('d', tokens=2), record_producer=lambda: done)
        self.assertEqual(source.cache.acquire_source_lease.call_count, 2)

    def test_closed_connection_checks_only_its_own_lease(self):
        peer = self.peer()
        peer.concurrent, peer.source_waiters = True, {}
        peer.source = SimpleNamespace(active={'live': 1}, pending={}, poll_ready=lambda: [])
        peer.service = SimpleNamespace(events=Queue())
        self.assertFalse(peer.blocks_compute)  # an online source keeps computing
        peer.service.events.put(('closed', 'released-earlier', None))
        peer.service.events.put(('closed', None, None))
        peer._serve_source_events()
        peer.service.events.put(('closed', 'live', None))
        with self.assertRaisesRegex(RuntimeError, 'outstanding lease'):
            peer._serve_source_events()

    def test_copy_into_leased_page_fail_stops_before_dma(self):
        peer = self.peer()
        peer.concurrent, peer.cancel_reason = True, None
        peer.scheduler = SimpleNamespace(schedule_stream=SimpleNamespace(cuda_stream=1))
        peer.source = SimpleNamespace(active={'r': {'lease': SimpleNamespace(page_indices=(3, 4))}}, pending={})
        peer.receiver = SimpleNamespace(pending={'reservation': SimpleNamespace(pages=(4,))})
        tx = Mock()
        with self.assertRaisesRegex(RuntimeError, 'overlaps'):
            peer._prepared({'transaction': tx})
        tx.start.assert_not_called()
        peer.receiver.pending['reservation'].pages = (5,)
        peer._prepared({'transaction': tx})
        tx.start.assert_called_once()

    def test_l1_config_accepts_file_and_rejects_unknown_keys(self):
        import json, tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            json.dump({'identity': {}, 'concurrent': True, 'service_port': 31000}, f); f.flush()
            self.assertEqual(parse_l1_config('@' + f.name)['service_port'], 31000)
        for text in ('{"concurrent": true}', '{"identity": {}, "source_max_lease": 2}', '[]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_l1_config(text)


if __name__ == '__main__':
    unittest.main()
