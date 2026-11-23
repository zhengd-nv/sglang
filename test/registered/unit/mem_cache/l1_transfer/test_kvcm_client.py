"""Real local gRPC wire tests for the optional peer-L1 candidate client."""

import concurrent.futures
import os
import signal
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path

import grpc

from sglang.srt.mem_cache.l1_transfer import directory as d
from sglang.srt.mem_cache.l1_transfer import kvcm_client as k
from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer.generated import (
    tair_kvcache_meta_service_pb2_grpc as rpc,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.target import KvcmService
from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[5]
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def identity(name):
    return p.EngineIdentity(
        name,
        name + "-boot",
        "domain",
        "instance",
        "model",
        "layout",
        name + "-tp",
        2,
        0,
        "shard",
    )


class KVCMClientTest(CustomTestCase):
    def setUp(self):
        self.service = KvcmService()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.server = grpc.server(self.executor)
        rpc.add_MetaServiceServicer_to_server(self.service, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.target, self.source = identity("target"), identity("source")
        self.entry = d.EngineDirectoryEntry(
            "exact-reporter", self.source, "http://control-host:8000"
        )
        self.directory = d.EngineDirectory(self.target, (self.entry,))
        self.query = p.LookupQuery(
            self.target,
            p.TransferId(self.target.incarnation, "rid", 3),
            tuple(range(-8, 8)),
            tuple(range(1025)),
            256,
            1024,
            64,
        )
        self.client = k.KVCMClient(self.channel, self.directory, timeout_s=0.5)

    def tearDown(self):
        self.channel.close()
        self.server.stop(None).wait()
        self.executor.shutdown(wait=True)

    def lookup(self, hosts, **changes):
        self.service.response.ClearField("hosts")
        for reporter, local in hosts:
            self.service.response.hosts.add(
                host_ip_port=reporter,
                local=local,
                p2p_1_fetch=999,
                p2p_1_total_match=999,
            )
        return self.client.lookup_with_observation(replace(self.query, **changes))

    def test_real_wire_root_keys_signed64_and_restore_once(self):
        outcome = self.lookup([("exact-reporter", 16)])
        self.assertEqual(outcome.result.restore_tokens, 768)
        self.assertEqual(outcome.result.restore_pages, 12)
        self.assertEqual(outcome.control_endpoint, self.entry.control_endpoint)
        self.assertEqual(outcome.observation.reason, k.LookupReason.HIT)
        request = self.service.requests[-1]
        self.assertEqual(
            list(request.block_cache_keys), list(self.query.event_block_keys)
        )
        self.assertEqual(request.query_type, 2)
        self.assertEqual(request.instance_id, "instance")
        self.assertEqual(request.trace_id, "rid")
        self.assertEqual(list(request.medium), ["hbm"])
        self.assertEqual(request.p2p_host_count, 0)
        self.assertEqual(self.client.lookup(self.query).hit_tokens, 1024)

    def test_status_one_only(self):
        for status in (0, 3, 999):
            self.service.response.header.status.code = status
            outcome = self.lookup([("exact-reporter", 16)])
            self.assertIsNone(outcome.result.source)
            self.assertEqual(outcome.observation.reason, k.LookupReason.STATUS_ERROR)
            self.assertEqual(outcome.observation.status_code, status)
        self.service.response.ClearField("header")
        self.assertEqual(
            self.lookup([]).observation.reason, k.LookupReason.STATUS_ERROR
        )

    def test_local_bounds_and_no_p2p_use(self):
        for local in (-1, 0, 4, 17, (1 << 63) - 1):
            outcome = self.lookup([("exact-reporter", local)])
            self.assertIsNone(outcome.result.source)
            self.assertEqual(outcome.result.hit_tokens, 256)
            if local < 0 or local > 16:
                self.assertEqual(
                    outcome.observation.rejections[0].reason,
                    k.RejectionReason.INVALID_LOCAL,
                )
        outcome = self.lookup([("exact-reporter", 16)], max_restore_tokens=960)
        self.assertEqual(outcome.result.hit_tokens, 960)
        self.assertEqual(outcome.result.restore_tokens, 704)

    def test_unknown_reporter_and_exact_mapping(self):
        outcome = self.lookup(
            [("http://control-host:8000", 16), ("exact-reporter ", 16)]
        )
        self.assertIsNone(outcome.result.source)
        self.assertTrue(
            all(
                r.reason is k.RejectionReason.UNKNOWN_REPORTER
                for r in outcome.observation.rejections
            )
        )

    def test_best_single_source_without_host_aggregation(self):
        second = d.EngineDirectoryEntry(
            "second", identity("source2"), "http://second:8000"
        )
        self.client = k.KVCMClient(
            self.channel,
            d.EngineDirectory(self.target, (self.entry, second)),
            timeout_s=0.5,
        )
        outcome = self.lookup([("exact-reporter", 8), ("second", 12)])
        self.assertEqual(outcome.result.source, second.identity)
        self.assertEqual(outcome.result.hit_tokens, 768)
        self.assertEqual(outcome.control_endpoint, second.control_endpoint)

    def test_oversized_host_list_is_a_miss_without_per_host_work(self):
        """A reply listing more hosts than the bound is invalid, even with a valid
        reporter in it; no rejection is built per host."""
        unknown = [(f"h{i}", 16) for i in range(k._MAX_HOSTS)]
        outcome = self.lookup([("exact-reporter", 16)] + unknown)
        self.assertIsNone(outcome.result.source)
        self.assertEqual(outcome.observation.reason, k.LookupReason.INVALID_RESPONSE)
        self.assertEqual(outcome.observation.rejections, ())

    def test_duplicate_reporter_is_rejected(self):
        self.assertIsNone(
            self.lookup([("exact-reporter", 8), ("exact-reporter", 16)]).result.source
        )

    def test_stale_or_foreign_target_has_no_rpc(self):
        for field in (
            "incarnation",
            "domain_id",
            "instance_id",
            "layout_digest",
            "shard_id",
        ):
            target = replace(self.target, **{field: "changed"})
            outcome = self.client.lookup_with_observation(
                replace(
                    self.query,
                    target=target,
                    transfer_id=replace(
                        self.query.transfer_id, target_incarnation=target.incarnation
                    ),
                )
            )
            self.assertEqual(outcome.observation.reason, k.LookupReason.TARGET_MISMATCH)
        self.assertEqual(self.service.requests, [])

    def test_unsupported_feature_gate(self):
        for kwargs in (
            {"is_eagle": True},
            {"use_lora": True},
            {"is_eagle": 1},
            {"use_lora": "false"},
        ):
            with self.assertRaises(ValueError):
                k.KVCMClient(self.channel, self.directory, timeout_s=0.5, **kwargs)
        for field in ("cache_salt", "extra_key"):
            for value in ("tenant", ""):
                outcome = self.lookup([], **{field: value})
                self.assertEqual(
                    outcome.observation.reason, k.LookupReason.UNSUPPORTED_QUERY
                )
        self.assertEqual(self.service.requests, [])

    def test_deadline_and_rpc_error(self):
        self.service.error = grpc.StatusCode.UNAVAILABLE
        outcome = self.client.lookup_with_observation(self.query)
        self.assertEqual(outcome.observation.reason, k.LookupReason.RPC_ERROR)
        self.assertEqual(outcome.observation.grpc_code, "UNAVAILABLE")
        self.service.error = None
        self.service.wait = True
        client = k.KVCMClient(self.channel, self.directory, timeout_s=0.02)
        outcome = client.lookup_with_observation(self.query)
        self.assertEqual(outcome.observation.reason, k.LookupReason.DEADLINE_EXCEEDED)
        self.assertIsNone(outcome.result.source)

    def test_timeout_is_positive_finite_and_bounded(self):
        for timeout in (0, -1, float("inf"), float("nan"), True, "1", 31, 10**1000):
            with self.assertRaises(ValueError):
                k.KVCMClient(self.channel, self.directory, timeout_s=timeout)

    def test_empty_restore_budget_skips_rpc(self):
        outcome = self.lookup([], max_restore_tokens=256)
        self.assertEqual(outcome.observation.reason, k.LookupReason.NO_CANDIDATE)
        self.assertEqual(self.service.requests, [])

    def test_closed_channel_is_an_observable_safe_miss(self):
        self.channel.close()
        outcome = self.client.lookup_with_observation(self.query)
        self.assertIsNone(outcome.result.source)
        self.assertEqual(outcome.observation.reason, k.LookupReason.RPC_ERROR)
        self.assertEqual(outcome.observation.exception_type, "ValueError")


class DependencyIsolationTest(CustomTestCase):
    """Probe optional imports without entering gRPC's process-global fork hooks.

    Pytest may run this after local gRPC fixtures. Explicit posix_spawn avoids
    their prefork locks, even after a server/channel has been closed.
    """

    def test_optional_import_does_not_require_grpc(self):
        script = f"""
import importlib.util, sys
class RejectGrpc:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('grpc', 'google'):
            raise ImportError('optional dependency forbidden')
sys.meta_path.insert(0, RejectGrpc())
s = importlib.util.spec_from_file_location('test_l1', {str(ROOT / "python/sglang/srt/mem_cache/l1_transfer/__init__.py")!r})
m = importlib.util.module_from_spec(s); sys.modules['test_l1'] = m; s.loader.exec_module(m)
from test_l1 import kvcm_client as k, protocol as p, directory as d
target = p.EngineIdentity('t', 'boot', 'domain', 'instance', 'model', 'layout', 'group', 1, 0, 'shard')
query = p.LookupQuery(target, p.TransferId('boot', 'rid', 0), (1, 2), (1, 2), 0, 1, 1)
outcome = k.KVCMClient(None, d.EngineDirectory(target, ())).lookup_with_observation(query)
assert outcome.observation.reason is k.LookupReason.DEPENDENCY_UNAVAILABLE
assert outcome.result.source is None
"""
        deadline = time.monotonic() + 20
        # Isolated Python plus an empty environment makes the dependency probe
        # independent of ambient PYTHONPATH, user site, and credential variables.
        pid = os.posix_spawn(sys.executable, [sys.executable, "-I", "-c", script], {})
        status = None
        try:
            while time.monotonic() < deadline:
                exited, result = os.waitpid(pid, os.WNOHANG)
                if exited:
                    status = result
                    break
                time.sleep(0.01)
            self.assertIsNotNone(status, "optional import probe exceeded 20s")
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        finally:
            if status is None:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)


if __name__ == "__main__":
    unittest.main()
