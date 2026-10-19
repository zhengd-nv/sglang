# SPDX-License-Identifier: Apache-2.0
import socket
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.l1_transfer.peer.nonblocking import AsyncRPC, SourceService
from sglang.srt.mem_cache.l1_transfer.peer.scheduler import SchedulerPeer, scheduler_peer_rpc
from sglang.srt.mem_cache.l1_transfer.peer.restore import RestoreState


class NonblockingTests(unittest.TestCase):
    def peer(self):
        peer = object.__new__(SchedulerPeer)
        peer.service = None
        peer.source = SimpleNamespace(active={})
        peer.phase = 'WAIT_ACQUIRE'
        peer.request = SimpleNamespace(rid='consumer')
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


if __name__ == '__main__':
    unittest.main()
