"""Control-plane ordering tests. CUDA copies are covered by manual/test-pd.py."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from dataclasses import asdict
import numpy as np
from sglang.srt.mem_cache.l1_transfer.transport.api import MemoryHandle, TransferState
from sglang.srt.mem_cache.l1_transfer.transport.layout import BufferLayout, PoolLayout, RegionSpan

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nvlink.conn import NvlinkKVReceiver, NvlinkKVSender


class TestNvlinkCompletion(unittest.TestCase):
    def receiver(self, nonce='correct', frames=(b'ab',)):
        receiver = object.__new__(NvlinkKVReceiver)
        receiver.conclude_state = None
        receiver.bootstrap_room = 7
        receiver.nonce = 'correct'
        receiver.aux_index = 3
        receiver.exports = [SimpleNamespace(lease_id='lease')]
        receiver.kv_mgr = SimpleNamespace(
            message=Mock(return_value=({'nonce': nonce}, frames)),
            kv_args=SimpleNamespace(aux_item_lens=[2]),
            engine=SimpleNamespace(release_export=Mock()),
            update_status=Mock(), active_room=7,
        )
        return receiver

    @patch('sglang.srt.disaggregation.nvlink.conn.AuxDataCodec.deserialize_data_to_buffer')
    def test_success_releases_before_publishing(self, deserialize):
        receiver = self.receiver()
        events = []
        deserialize.side_effect = lambda *args: events.append('aux')
        receiver.kv_mgr.engine.release_export.side_effect = lambda *args: events.append('release')
        receiver.kv_mgr.update_status.side_effect = lambda *args: events.append('publish')
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertEqual(events, ['aux', 'release', 'publish'])
        self.assertEqual(receiver.exports, [])
        self.assertIsNone(receiver.kv_mgr.active_room)
        self.assertEqual(receiver.poll(), KVPoll.Success)
        receiver.kv_mgr.engine.release_export.assert_called_once_with('lease')

    @patch('sglang.srt.disaggregation.nvlink.conn.fail_stop', side_effect=RuntimeError('fail-stop'))
    def test_bad_completion_does_not_release(self, stop):
        for nonce, frames in [('stale', (b'ab',)), ('correct', (b'abc',))]:
            receiver = self.receiver(nonce, frames)
            with self.assertRaisesRegex(RuntimeError, 'fail-stop'):
                receiver.poll()
            receiver.kv_mgr.engine.release_export.assert_not_called()
            receiver.kv_mgr.update_status.assert_not_called()
            self.assertEqual(len(receiver.exports), 1)

    @patch('sglang.srt.disaggregation.nvlink.conn.fail_stop', side_effect=RuntimeError('fail-stop'))
    def test_abort_and_clear_never_recycle_published_target(self, stop):
        for method in ('abort', 'clear'):
            receiver = self.receiver()
            with self.assertRaisesRegex(RuntimeError, 'fail-stop'):
                getattr(receiver, method)()
            receiver.kv_mgr.engine.release_export.assert_not_called()
            self.assertEqual(len(receiver.exports), 1)


class TestNvlinkSender(unittest.TestCase):
    def sender(self, state=TransferState.SUCCESS):
        sender = object.__new__(NvlinkKVSender)
        sender.bootstrap_room, sender.num_kv_indices, sender.aux_index = 7, 1, 0
        sender.conclude_state = None
        sender._imports = []
        sender._remote_bindings = None
        sender._prepare_send_indices = Mock(return_value=(np.array([1]), slice(0, 1), True, False))
        sender._record_transfer_indices = Mock()
        source = MemoryHandle('source', 16, 0, False, 100)
        remote = MemoryHandle('import', 16, 0, True, 200)
        layout = PoolLayout('torch.bfloat16', 1, (BufferLayout('k', 4, 'NHD'),))
        descriptor = dict(version=1, transport='nvlink-fabric', owner_id='a'*32,
                          region_id='b'*32, lease_id='c'*32, size=16,
                          physical_size=16, granularity=16, access='write', transport_handle='opaque')
        import json
        header = dict(layout=json.loads(json.dumps(asdict(layout))), pages=[2], capacity=4,
                      regions=[descriptor], bindings={'k': [[0, 16, 'b'*32, 0]]},
                      endpoint='tcp://peer:1', nonce='nonce')
        events = []
        engine = Mock()
        engine.import_memory.return_value = remote
        engine.submit_transfer.return_value = 'batch'
        engine.wait.return_value = SimpleNamespace(state=state)
        engine.release_batch.side_effect = lambda *a: events.append('drain')
        engine.unregister_memory.side_effect = lambda *a: events.append('unmap')
        sender.kv_mgr = SimpleNamespace(
            adapter=SimpleNamespace(layout=layout, capacity=4, bindings={'k': (RegionSpan(0,16,source),)}),
            engine=engine, message=Mock(return_value=(header, [])),
            kv_args=SimpleNamespace(gpu_id=0, aux_data_ptrs=[100], aux_item_lens=[2]),
            send_message=Mock(side_effect=lambda *a: events.append('done')),
        )
        return sender, events

    @patch('sglang.srt.disaggregation.nvlink.conn.torch.cuda.synchronize')
    @patch('sglang.srt.disaggregation.nvlink.conn.AuxDataCodec.serialize_data_from_buffer', return_value=b'ab')
    def test_write_drains_and_unmaps_before_done(self, serialize, synchronize):
        sender, events = self.sender()
        sender.send(np.array([1]))
        self.assertEqual(events, ['drain', 'unmap', 'done'])
        request = sender.kv_mgr.engine.submit_transfer.call_args.args[0][0]
        self.assertEqual((request.local_offset, request.remote_offset, request.length), (4,8,4))
        self.assertEqual(sender.conclude_state, KVPoll.Success)

    @patch('sglang.srt.disaggregation.nvlink.conn.torch.cuda.synchronize')
    @patch('sglang.srt.disaggregation.nvlink.conn.AuxDataCodec.serialize_data_from_buffer', return_value=b'ab')
    def test_chunks_share_one_import_until_final_completion(self, serialize, synchronize):
        sender, events = self.sender()
        sender.num_kv_indices = 2
        sender.kv_mgr.message.return_value[0]['pages'] = [2, 3]
        sender._prepare_send_indices.return_value = (np.array([1]), slice(0, 1), False, False)
        sender.send(np.array([1]))
        self.assertEqual(events, ['drain'])
        self.assertEqual(len(sender._imports), 1)
        sender._prepare_send_indices.return_value = (np.array([2]), slice(1, 2), True, False)
        sender.send(np.array([2]))
        self.assertEqual(events, ['drain', 'drain', 'unmap', 'done'])
        sender.kv_mgr.engine.import_memory.assert_called_once()
        self.assertEqual(sender._imports, [])

    @patch('sglang.srt.disaggregation.nvlink.conn.torch.cuda.synchronize')
    @patch('sglang.srt.disaggregation.nvlink.conn.fail_stop', side_effect=RuntimeError('fail-stop'))
    def test_uncertain_write_never_unmaps_or_notifies_success(self, stop, synchronize):
        sender, events = self.sender(TransferState.FAILED_UNCERTAIN)
        with self.assertRaisesRegex(RuntimeError, 'fail-stop'):
            sender.send(np.array([1]))
        self.assertEqual(events, [])
        sender.kv_mgr.engine.release_batch.assert_not_called()
        self.assertIsNone(sender.conclude_state)


if __name__ == '__main__':
    unittest.main()
