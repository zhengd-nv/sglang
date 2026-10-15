import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.l1_transfer.peer.control import RemoteRejected
from sglang.srt.mem_cache.l1_transfer.peer.scheduler import SchedulerPeer, scheduler_peer_rpc


class TestSchedulerPeer(unittest.TestCase):
    def test_busy_scheduler_rejects_before_mutation(self):
        scheduler = SimpleNamespace(is_fully_idle=lambda: False)
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            scheduler_peer_rpc(scheduler, 'init', '{}')
        self.assertFalse(hasattr(scheduler, '_l1_peer'))

    def peer(self):
        peer = object.__new__(SchedulerPeer)
        peer.cache = peer.adapter = peer.identity = None
        peer.receiver = None
        peer.stats = Mock(return_value={})
        return peer

    @patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.socket.create_connection')
    @patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.PeerReceiver')
    def test_local_hit_keeps_consumer_without_network(self, receiver_cls, connect):
        receiver = receiver_cls.return_value
        receiver.prepare.return_value = dict(reservation=SimpleNamespace(indices=[1]*16,prefix_len=16),transaction=None)
        peer = self.peer()
        result = peer.restore([1]*16)
        self.assertEqual(result['transferred_bytes'], 0)
        self.assertEqual(result['local_tokens'], 16)
        connect.assert_not_called()
        self.assertIs(peer.receiver, receiver)
        peer.finish()
        receiver.finish_consumer.assert_called_once()
        self.assertIsNone(peer.receiver)

    @patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.os._exit')
    @patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.PeerReceiver')
    def test_safe_rejection_does_not_kill_scheduler(self, receiver_cls, stop):
        receiver = receiver_cls.return_value
        receiver.pending = None
        receiver.prepare.side_effect = RemoteRejected('missing prefix')
        peer = self.peer()
        with self.assertRaises(RemoteRejected):
            peer.restore([1]*16)
        stop.assert_not_called()
        self.assertIsNone(peer.receiver)

    @patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.os._exit', side_effect=RuntimeError('fail-stop'))
    @patch('sglang.srt.mem_cache.l1_transfer.peer.scheduler.PeerReceiver')
    def test_uncertain_prepare_retains_ownership(self, receiver_cls, stop):
        receiver = receiver_cls.return_value
        receiver.pending = {'imports':['unconfirmed']}
        receiver.prepare.side_effect = ConnectionError('lost reply')
        peer = self.peer()
        with self.assertRaisesRegex(RuntimeError, 'fail-stop'):
            peer.restore([1]*16)
        stop.assert_called_once_with(70)
        self.assertIs(peer.receiver, receiver)
        receiver.finish_consumer.assert_not_called()


if __name__ == '__main__': unittest.main()
