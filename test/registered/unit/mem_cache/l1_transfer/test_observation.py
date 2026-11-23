import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import observation as o
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class ObservationTest(CustomTestCase):
    def test_live_miss_uses_bounded_separate_attempt_without_remote_proof(self):
        fields = dict(
            input_tokens=128000,
            prefill_start=115200,
            prefill_tokens=12800,
            cached_tokens=115200,
            cached_tokens_device=115200,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            retraction_count=0,
            cache_attempt=0,
            sharing_enabled=True,
            record_valid=False,
            record_status="miss",
            miss_attempt=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "miss.jsonl"
            observer = o.TransferObserver(
                path, "engine", tp_rank=0, tp_group_id="group"
            )
            observer.emit_destination_prefill("native", **fields)
            for altered in (
                {k: v for k, v in fields.items() if k != "miss_attempt"},
                dict(fields, miss_attempt=True),
                dict(fields, miss_attempt=2**63),
                dict(fields, transfer_attempt=3),
                dict(fields, record_valid=True),
            ):
                observer.emit_destination_prefill("native", **altered)
            observer.close()
            row = json.loads(path.read_text())
            self.assertFalse(row["record_valid"])
            self.assertEqual((row["record_status"], row["miss_attempt"]), ("miss", 3))
            self.assertNotIn("transfer_attempt", row)
            self.assertEqual(observer.snapshot()["errors"], 5)

    def test_explicit_destination_rid_is_bounded_and_uses_original_sink(self):
        fields = dict(
            input_tokens=128000,
            prefill_start=115200,
            prefill_tokens=12800,
            cached_tokens=115200,
            cached_tokens_device=115200,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            retraction_count=0,
            cache_attempt=0,
            sharing_enabled=False,
            record_valid=False,
            record_status="missing",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dispatch.jsonl"
            observer = o.TransferObserver(
                path, "engine", tp_rank=2, tp_group_id="group"
            )
            observer.emit_destination_prefill("native", **fields)
            observer.emit_destination_prefill("x" * 257, **fields)
            observer.emit_destination_prefill("native", **dict(fields, input_tokens=[]))
            observer.emit_destination_prefill(
                "native", **dict(fields, record_valid=True)
            )
            observer.close()
            snapshot = observer.snapshot()
            self.assertEqual((snapshot["written"], snapshot["errors"]), (1, 3))
            row = json.loads(path.read_text())
            self.assertEqual(row["phase"], "destination_prefill_dispatch")
            self.assertEqual(row["rid"], "native")
            self.assertEqual(row["tp_rank"], 2)
            self.assertNotIn("attempt", row)
            self.assertEqual(snapshot["events"], [row])
            observer.emit_destination_prefill("native", **fields)
            self.assertEqual(observer.snapshot()["dropped"], 1)

    def test_file_ring_and_payload_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            observer = o.TransferObserver(path, "engine", capacity=2)
            observer.emit("copy_done", NS(rid="rid", attempt=3), completed_bytes=64)
            observer.emit("bad", tokens=[123])
            observer.close()
            snapshot = observer.snapshot()
            self.assertEqual(snapshot["errors"], 1)
            self.assertEqual(snapshot["written"], 1)
            row = json.loads(path.read_text())
            self.assertEqual(row["rid"], "rid")
            self.assertNotIn("tokens", row)
            with self.assertRaises(FileExistsError):
                o.TransferObserver(path, "engine")

    def test_queue_loss_and_writer_error_visible_outside_sink(self):
        started, unblock = threading.Event(), threading.Event()

        class BlockedFile:
            def write(self, value):
                started.set()
                unblock.wait(2)
                raise OSError("no disk")

            def close(self):
                pass

        with patch("builtins.open", return_value=BlockedFile()):
            observer = o.TransferObserver("/tmp/unused-l1-trace", "engine", capacity=1)
        observer.emit("first")
        self.assertTrue(started.wait(1))
        observer.emit("second")
        observer.emit("third")
        self.assertEqual(observer.snapshot()["dropped"], 1)
        self.assertEqual(len(observer.snapshot()["events"]), 1)
        unblock.set()
        observer.close()
        self.assertEqual(observer.snapshot()["errors"], 2)

    def test_debug_endpoint_requires_bearer_and_is_opt_in(self):
        from sglang.srt.mem_cache.l1_transfer import control_http as h

        handler = object.__new__(h._Handler)
        handler.path = "/debug/l1-trace"
        handler.headers = NS(get_all=lambda *args: [])
        handler.server = NS(gateway=NS(_authorization="Bearer secret", observer=None))
        replies = []
        handler.send_error = lambda code: replies.append(code)
        handler._reply = lambda code, body: replies.append((code, json.loads(body)))
        handler.do_GET()
        self.assertEqual(replies.pop(), 401)
        handler.headers = NS(get_all=lambda *args: ["Bearer secret"])
        handler.do_GET()
        self.assertEqual(replies.pop(), 404)
        handler.server.gateway.observer = NS(snapshot=lambda: {"dropped": 4})
        handler.do_GET()
        self.assertEqual(replies.pop(), (200, {"dropped": 4}))


if __name__ == "__main__":
    unittest.main()
