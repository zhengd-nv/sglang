"""Optional diagnostics must survive a READY queue rejection."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.l1_transfer.peer import timing, peer as peer_module
from sglang.srt.mem_cache.l1_transfer.peer.nonblocking import AsyncRPC
from sglang.srt.mem_cache.l1_transfer.peer.restore import RestoreTransaction
from sglang.srt.mem_cache.l1_transfer.transport._driver import Driver
from sglang.srt.mem_cache.target_reservation import TargetReservation


class LifecycleTimingTests(unittest.TestCase):
    def test_ready_trace_survives_synchronous_consumer_cleanup(self):
        peer = SimpleNamespace(request=SimpleNamespace(rid='queue-rejected'),
                               session={'transaction': None}, source=Mock(),
                               cache=Mock(), engine=Mock(), start_request=Mock(),
                               finish=Mock(), finish_request=Mock())
        peer._ready = lambda: setattr(peer, 'request', None)
        peer._trace_request_start = (time.perf_counter_ns(), time.thread_time_ns())
        # install() instruments shared methods; restore every one after this test.
        shared = [(TargetReservation, n) for n in ('__init__', 'publish', 'release', '_set_pins')]
        shared += [(peer_module, n) for n in ('plan_transfer', 'bind_regions')]
        shared += [(Driver, n) for n in ('submit', 'import_region', 'close')]
        shared += [(AsyncRPC, '_call'), (RestoreTransaction, 'start')]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for obj, name in shared:
                stack.callback(setattr, obj, name, getattr(obj, name))
            stack.enter_context(patch.dict(os.environ, SGLANG_L1_TRACE_DIR=directory))
            timing.install(peer)
            try:
                peer._ready()
                peer._timing_output.flush()
                rows = [json.loads(line) for path in Path(directory).glob('*.jsonl')
                        for line in path.read_text().splitlines()]
                self.assertEqual([row['rid'] for row in rows], ['queue-rejected'] * 2)
                self.assertIsNone(peer.request)
            finally:
                peer._timing_output.close()


if __name__ == '__main__':
    unittest.main()
