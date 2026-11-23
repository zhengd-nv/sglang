"""Contracts of the local evidence harness; not part of the upstream package."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase


def harness_module():
    path = Path(__file__).resolve().parent / "test-prefix-sharing.py"
    spec = importlib.util.spec_from_file_location("prefix_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_fixture():
    domain = "10000009-0000-4000-8000-000000000009"
    return {
        "identities": [
            {
                "engine_id": name,
                "domain_id": domain,
                "instance_id": "instance",
                "tp_size": 1,
                "model_id": "model",
                "layout_digest": "layout",
            }
            for name in ("a", "b")
        ],
        "ips": ["10.0.0.1", "10.0.0.2"],
        "reporters": ["r1", "r2"],
        "control_port": 31081,
        "expected_layout": {"buffers": [{"layout": "NHD:shape=(4,128)"}]},
        "evidence": {
            "schema": 1,
            "revisions": {
                name: "a" * 40
                for name in (
                    "source_base",
                    "transport_reference",
                    "kvcm_adapter_reference",
                    "serving",
                )
            },
            "checkpoint_sha256": "b" * 64,
            "engines": [
                {
                    "engine_id": name,
                    "pod_uid": "pod-" + name,
                    "physical_host": "host-" + name,
                    "gpu_uuid": "GPU-" + domain[:-1] + name,
                    "fabric_cluster_uuid": domain,
                    "fabric_state": "Completed",
                    "fabric_status": "Success",
                    "serving_sha": "a" * 40,
                    "checkpoint_sha256": "b" * 64,
                }
                for name in ("a", "b")
            ],
        },
    }


class HarnessTest(CustomTestCase):
    def test_harness_refuses_missing_or_out_of_order_evidence(self):
        module = harness_module()
        with self.assertRaisesRegex(AssertionError, "missing transfer"):
            module.validate_trace([], "rid", 64, 4)
        phases = [
            "lookup",
            "reservation",
            "pin_start",
            "pin_reply",
            "import_done",
            "copy_submitted",
            "copy_done",
            "verify_done",
            "imports_released",
            "unpin_ack",
            "backend_state",
            "private_ready",
            "prefill_forward",
            "adopted",
            "checkpoint",
        ]
        rows = [{"phase": p, "rid": "rid", "attempt": 0} for p in phases]
        by_phase = {row["phase"]: row for row in rows}
        by_phase["copy_done"].update(
            status="SUCCESS", requested_bytes=64, completed_bytes=64
        )
        by_phase["copy_submitted"].update(requested_bytes=64)
        by_phase["pin_reply"].update(status="OK")
        by_phase["unpin_ack"].update(status="RELEASED")
        by_phase["backend_state"].update(status="READY")
        by_phase["verify_done"].update(verified=True)
        by_phase["private_ready"].update(remaining_prefill_tokens=1)
        by_phase["prefill_forward"].update(prefill_tokens=1)
        rows.insert(
            7,
            {
                "phase": "verify_buffer",
                "rid": "rid",
                "attempt": 0,
                "requested_bytes": 64,
                "verified": True,
                "buffers": 1,
            },
        )
        module.validate_trace(rows, "rid", 64, 1)
        for phase, field, value in (
            ("backend_state", "status", "QUARANTINED"),
            ("pin_reply", "status", "BUSY"),
            ("unpin_ack", "status", "INVALID"),
            ("verify_done", "verified", False),
        ):
            previous = by_phase[phase][field]
            by_phase[phase][field] = value
            with self.assertRaises(AssertionError):
                module.validate_trace(rows, "rid", 64, 1)
            by_phase[phase][field] = previous
        verifier = rows.pop(7)
        rows.append(verifier)
        with self.assertRaisesRegex(AssertionError, "outside"):
            module.validate_trace(rows, "rid", 64, 1)
        rows.pop()
        rows.insert(7, verifier)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(AssertionError, "out of order"):
            module.validate_trace(rows, "rid", 64, 1)

    def test_manifest_requires_each_provenance_field_and_consistent_nodes(self):
        module, manifest = harness_module(), manifest_fixture()
        module.validate_manifest(manifest, "a" * 40, "instance")
        for path in (
            ("checkpoint_sha256",),
            ("revisions", "source_base"),
            ("revisions", "transport_reference"),
            ("revisions", "kvcm_adapter_reference"),
            ("revisions", "serving"),
        ):
            broken = copy.deepcopy(manifest)
            item = broken["evidence"]
            for field in path[:-1]:
                item = item[field]
            del item[path[-1]]
            with self.assertRaises(AssertionError):
                module.validate_manifest(broken, "a" * 40, "instance")
        for field in manifest["evidence"]["engines"][0]:
            broken = copy.deepcopy(manifest)
            del broken["evidence"]["engines"][0][field]
            with self.assertRaises(AssertionError):
                module.validate_manifest(broken, "a" * 40, "instance")
        for field in ("pod_uid", "physical_host", "gpu_uuid"):
            broken = copy.deepcopy(manifest)
            broken["evidence"]["engines"][1][field] = broken["evidence"]["engines"][0][
                field
            ]
            with self.assertRaises(AssertionError):
                module.validate_manifest(broken, "a" * 40, "instance")
        for field, value in (
            ("fabric_state", "In Progress"),
            ("fabric_status", "Failed"),
            ("serving_sha", "c" * 40),
            ("checkpoint_sha256", "c" * 64),
        ):
            broken = copy.deepcopy(manifest)
            broken["evidence"]["engines"][0][field] = value
            with self.assertRaises(AssertionError):
                module.validate_manifest(broken, "a" * 40, "instance")

    def test_trace_engine_binding_rejects_wrong_snapshot_and_event(self):
        module = harness_module()
        evidence = module.Evidence(manifest_fixture(), "test-token")
        snapshot = {
            "engine": "wrong",
            "events": [],
            "seq": 0,
            "written": 0,
            "dropped": 0,
            "errors": 0,
        }
        with (
            patch.object(module, "http", return_value=snapshot),
            self.assertRaisesRegex(AssertionError, "identity mismatch"),
        ):
            evidence.collect(initial=True)
        snapshot.update(engine="a", events=[{"engine": "b", "seq": 1}])
        with (
            patch.object(module, "http", return_value=snapshot),
            self.assertRaisesRegex(AssertionError, "identity mismatch"),
        ):
            evidence.collect(initial=True)

    def test_trace_collector_rejects_internal_sequence_gap_and_reset(self):
        module = harness_module()
        evidence = module.Evidence(manifest_fixture(), "test-token")
        snapshot = {
            "engine": "a",
            "seq": 3,
            "written": 3,
            "dropped": 0,
            "errors": 0,
            "events": [{"engine": "a", "seq": 1}, {"engine": "a", "seq": 3}],
        }
        with (
            patch.object(module, "http", return_value=snapshot),
            self.assertRaisesRegex(AssertionError, "sequence gap"),
        ):
            evidence.collect()
        evidence.cursors[0] = 4
        with (
            patch.object(module, "http", return_value=snapshot),
            self.assertRaisesRegex(AssertionError, "reset"),
        ):
            evidence.collect()

    def test_source_trace_checks_release_order_attempt_and_boundaries(self):
        module = harness_module()
        source = [
            {
                "rid": "rid",
                "attempt": 2,
                "phase": "source_pin",
                "hit_tokens": 64,
                "local_tokens": 0,
                "pinned_tokens": 64,
            },
            {"rid": "rid", "attempt": 2, "phase": "source_release", "pinned_tokens": 0},
        ]
        target = [
            {"attempt": 2, "phase": "lookup", "hit_tokens": 64},
            {"attempt": 2, "phase": "reservation", "hit_tokens": 64, "local_tokens": 0},
        ]
        module.validate_source_trace(source, "rid", target)
        for field, value in (("attempt", 3), ("hit_tokens", 128), ("local_tokens", 32)):
            broken = copy.deepcopy(source)
            broken[0][field] = value
            with self.assertRaises(AssertionError):
                module.validate_source_trace(broken, "rid", target)
        with self.assertRaises(AssertionError):
            module.validate_source_trace(source[::-1], "rid", target)


class StreamProtocolTest(CustomTestCase):
    @staticmethod
    def frame(tokens, *, rid="rid", count=None, finish=None, **extra):
        return {
            "output_ids": tokens,
            "meta_info": {
                "id": rid,
                "completion_tokens": len(tokens) if count is None else count,
                "finish_reason": finish,
            },
            **extra,
        }

    @staticmethod
    def lines(*frames, done=True):
        data = b"".join(
            b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames
        )
        if done:
            data += b"data: [DONE]\n\n"
        return data.splitlines(keepends=True)

    def parse(self, frames, count=2, *, done=True):
        evidence = {}
        result = harness_module().consume_generate_stream(
            self.lines(*frames, done=done), "rid", count, 0, evidence
        )
        return result, evidence

    def test_complete_cumulative_stream_compares_entire_sequence(self):
        finish = {"type": "length", "length": 2}
        left, evidence = self.parse(
            [self.frame([11]), self.frame([11, 99], finish=finish)]
        )
        right, _ = self.parse([self.frame([22]), self.frame([22, 99], finish=finish)])
        self.assertEqual(left["output_ids"], [11, 99])
        self.assertNotEqual(left["output_ids"], right["output_ids"])
        self.assertEqual(left["completion_tokens"], 2)
        self.assertTrue(left["done"])
        self.assertEqual(len(evidence["frames"]), 2)

    def test_declared_incremental_mode_rejected_before_network(self):
        module = harness_module()
        with (
            patch.object(module.urllib.request, "urlopen") as request,
            self.assertRaisesRegex(AssertionError, "incremental_streaming_output"),
        ):
            module.generate(
                "http://unused",
                [1],
                "rid",
                2,
                incremental_streaming_output=True,
                evidence={},
            )
        request.assert_not_called()
        with self.assertRaises(AssertionError):
            module.require_cumulative_mode({})
        module.require_cumulative_mode({"incremental_streaming_output": False})

    def test_delta_chunks_cannot_silently_compare_only_last_token(self):
        finish = {"type": "length", "length": 2}
        for first in (11, 22):
            with self.assertRaisesRegex(AssertionError, "token count"):
                self.parse(
                    [self.frame([first]), self.frame([99], count=2, finish=finish)]
                )

    def test_truncated_stream_preserves_received_evidence(self):
        module, evidence = harness_module(), {}
        frame = self.frame([99])
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            module.consume_generate_stream(
                self.lines(frame, done=False), "rid", 8, 0, evidence
            )
        self.assertEqual(evidence["frames"], [frame])
        self.assertFalse(evidence["done"])
        final = self.frame([11, 99], finish={"type": "length", "length": 2})
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            self.parse([final], done=False)
        with self.assertRaisesRegex(AssertionError, "DONE without"):
            self.parse([self.frame([11, 99])])

    def test_wrong_rid_count_prefix_and_terminal_reason_fail(self):
        finish = {"type": "length", "length": 2}
        for frames in (
            [self.frame([11, 99], rid="other", finish=finish)],
            [self.frame([11, 99], count=1, finish=finish)],
            [self.frame([11]), self.frame([22, 99], finish=finish)],
            [self.frame([11, 99], finish={"type": "stop"})],
            [self.frame([99], finish={"type": "length", "length": 1})],
            [self.frame([11, 99], finish={"type": "length", "length": 1})],
        ):
            with self.subTest(frames=frames), self.assertRaises(AssertionError):
                self.parse(frames)

    def test_error_and_frames_after_termination_fail(self):
        module, evidence = harness_module(), {}
        error = {
            "error": {"type": "invalid_request_error", "message": "synthetic failure"}
        }
        with self.assertRaisesRegex(AssertionError, "SSE error"):
            module.consume_generate_stream(self.lines(error), "rid", 2, 0, evidence)
        self.assertEqual(evidence["frames"], [error])
        final = self.frame([11, 99], finish={"type": "length", "length": 2})
        with self.assertRaisesRegex(AssertionError, "extra result"):
            self.parse([final, final])
        with self.assertRaisesRegex(AssertionError, "after DONE"):
            module.consume_generate_stream(
                self.lines(final) + self.lines(final), "rid", 2, 0, {}
            )
        with self.assertRaisesRegex(AssertionError, "truncated SSE line"):
            module.consume_generate_stream(
                [b'data: {"output_ids": [11]'], "rid", 2, 0, {}
            )


if __name__ == "__main__":
    unittest.main()
