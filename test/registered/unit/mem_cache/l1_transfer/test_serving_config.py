import importlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_protocol import PACKAGE, identity, register_cpu_ci

from sglang.test.test_utils import CustomTestCase

c = importlib.import_module(f"{PACKAGE}.config")


register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class ConfigTest(CustomTestCase):
    def setUp(self):
        self.observability = NS(kv_events_config='{"publisher":"zmq"}')
        self.uses_dynamo_format = Mock(return_value=False)
        # Keep these CPU-only contract tests independent of accelerator imports.
        self.runtime_modules = patch.dict(
            sys.modules,
            {
                "sglang.srt.runtime_context": NS(
                    get_observability=lambda: self.observability
                ),
                "sglang.srt.disaggregation.kv_events": NS(
                    uses_dynamo_format=self.uses_dynamo_format
                ),
            },
        )
        self.runtime_modules.start()
        self.addCleanup(self.runtime_modules.stop)

    def context(self):
        params = NS(
            pp_size=1,
            attn_cp_size=1,
            is_eagle=False,
            enable_session_radix_cache=False,
            component_registry_override=None,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            enable_kv_cache_events=True,
            dynamo_kv_event_format=False,
            page_size=64,
            token_to_kv_pool_allocator=NS(page_size=64),
        )
        return NS(
            server_args=NS(),
            params=params,
            tp_size=1,
            tp_rank=0,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            is_dsa=False,
            disable_radix_cache=False,
            enable_hierarchical_cache=False,
        )

    def glm_context(self, rank):
        ctx = self.context()
        ctx.tp_size = 4
        ctx.tp_rank = rank
        ctx.is_dsa = True
        ctx.server_args.disaggregation_mode = "prefill"
        ctx.params.enable_kv_cache_events = rank == 0
        return ctx

    @patch.dict(os.environ, {"SGLANG_GLM_DSA_FIXED_FABRIC_POOL": "1"})
    def test_glm_tp4_publisher_rank(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                ctx = self.glm_context(rank)
                c.validate_serving_context(ctx, NS(_tree_core_backend="python"))
                self.assertEqual(ctx.params.enable_kv_cache_events, rank == 0)
                ctx.params.enable_kv_cache_events = rank != 0
                with self.assertRaisesRegex(ValueError, "KV events on TP0 only"):
                    c.validate_serving_context(ctx, NS(_tree_core_backend="python"))

    @patch.dict(os.environ, {"SGLANG_GLM_DSA_FIXED_FABRIC_POOL": "1"})
    def test_global_native_events_required_on_every_rank(self):
        contexts = [self.context()] + [self.glm_context(rank) for rank in range(4)]
        for ctx in contexts:
            for config, dynamo in (
                (None, False),
                ("", False),
                ('{"format":"dynamo"}', True),
            ):
                with self.subTest(tp_size=ctx.tp_size, rank=ctx.tp_rank, config=config):
                    self.observability.kv_events_config = config
                    self.uses_dynamo_format.return_value = dynamo
                    with self.assertRaisesRegex(ValueError, "native SGLang KV events"):
                        c.validate_serving_context(ctx, NS(_tree_core_backend="python"))
            self.observability.kv_events_config = '{"publisher":"zmq"}'
            self.uses_dynamo_format.return_value = False
            ctx.params.dynamo_kv_event_format = True
            with self.assertRaisesRegex(ValueError, "native SGLang KV events"):
                c.validate_serving_context(ctx, NS(_tree_core_backend="python"))

    def test_tp1_publisher_required(self):
        ctx = self.context()
        ctx.params.enable_kv_cache_events = False
        with self.assertRaisesRegex(ValueError, "KV events on TP0 only"):
            c.validate_serving_context(ctx, NS(_tree_core_backend="python"))

    @patch.dict(os.environ, {"SGLANG_GLM_DSA_FIXED_FABRIC_POOL": "1"})
    def test_glm_topology_and_rank_bounds_fail_closed(self):
        for rank in (-1, 4):
            with (
                self.subTest(rank=rank),
                self.assertRaisesRegex(ValueError, "valid TP rank"),
            ):
                c.validate_serving_context(
                    self.glm_context(rank), NS(_tree_core_backend="python")
                )
        for owner, field, value in (
            ("params", "pp_size", 2),
            ("params", "attn_cp_size", 2),
            ("server_args", "dp_size", 2),
            ("server_args", "dcp_size", 2),
            ("server_args", "enable_dp_attention", True),
            ("server_args", "disaggregation_mode", "decode"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ctx = self.glm_context(1)
                setattr(getattr(ctx, owner), field, value)
                c.validate_serving_context(ctx, NS(_tree_core_backend="python"))

    def test_modes_fail_closed(self):
        cache = NS(_tree_core_backend="python")
        ctx = self.context()
        c.validate_serving_context(ctx, cache)
        for field, value in (
            ("tp_size", 4),
            ("is_hybrid_swa", True),
            ("is_dsa", True),
            ("disable_radix_cache", True),
        ):
            with self.subTest(field=field):
                ctx = self.context()
                setattr(ctx, field, value)
                with self.assertRaises(ValueError):
                    c.validate_serving_context(ctx, cache)
        for field, value in (
            ("speculative_algorithm", "EAGLE"),
            ("dcp_size", 2),
            ("disaggregation_mode", "decode"),
        ):
            ctx = self.context()
            setattr(ctx.server_args, field, value)
            with self.assertRaises(ValueError):
                c.validate_serving_context(ctx, cache)

    def test_credentials_only_from_env_not_repr(self):
        data = manifest()
        data["peer_groups"][0]["reporter_id"] = "exact-reporter"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(data))
            with patch.dict(
                os.environ,
                {
                    "SGLANG_NVLINK_CONFIG": str(path),
                    "SGLANG_NVLINK_CONTROL_TOKEN": "test-only-secret",
                },
            ):
                config = c.ServingConfig.from_env()
                self.assertEqual(
                    config.directory.entries[0].reporter_id, "exact-reporter"
                )
                self.assertNotIn("test-only-secret", repr(config))
            data["bearer_token"] = "must-not-be-accepted"
            path.write_text(json.dumps(data))
            with (
                patch.dict(os.environ, {"SGLANG_NVLINK_CONFIG": str(path)}),
                self.assertRaisesRegex(
                    ValueError, "invalid NVLink serving configuration"
                ),
            ):
                c.ServingConfig.from_env()


class PendingCapacityTest(CustomTestCase):
    def data(self):
        return manifest()

    def load(self, data):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(data))
            with patch.dict(
                os.environ,
                {
                    "SGLANG_NVLINK_CONFIG": str(path),
                    "SGLANG_NVLINK_CONTROL_TOKEN": "test-only-secret",
                },
            ):
                return c.ServingConfig.from_env()

    def test_explicit_capacity_accepts_bounded_integers(self):
        for capacity in (4, 32, 64):
            with self.subTest(capacity=capacity):
                data = self.data()
                data["prefetch_pending_capacity"] = capacity
                self.assertEqual(self.load(data).prefetch_pending_capacity, capacity)

    def test_invalid_capacity_is_rejected_without_echoing_configuration(self):
        for capacity in (True, False, 4.0, 32.5, "32", "test-only-secret", None, 3, 65):
            with self.subTest(capacity=capacity):
                data = self.data()
                data["prefetch_pending_capacity"] = capacity
                with self.assertRaises(ValueError) as raised:
                    self.load(data)
                self.assertEqual(
                    str(raised.exception), "invalid NVLink serving configuration"
                )
                self.assertIsNone(raised.exception.__cause__)
                self.assertTrue(raised.exception.__suppress_context__)

    def test_pipeline_depth_is_strict_and_default_off(self):
        self.assertEqual(self.load(self.data()).prefetch_pipeline_depth, 1)
        for depth in (True, False, 1.0, 2.0, "1", "2", None, 0, 3, -1):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.load(self.data() | {"prefetch_pipeline_depth": depth})
        self.assertEqual(
            self.load(
                self.data() | {"prefetch_pipeline_depth": 1}
            ).prefetch_pipeline_depth,
            1,
        )
        self.assertEqual(
            self.load(
                self.data() | {"prefetch_pipeline_depth": 2}
            ).prefetch_pipeline_depth,
            2,
        )


def manifest():
    def group(name):
        return dict(
            reporter_id=name,
            ranks=[
                dict(
                    identity=asdict(identity(name, tp_size=4, tp_rank=rank)),
                    control_endpoint=f"http://127.0.0.1:{31000 + rank}",
                )
                for rank in range(4)
            ],
        )

    return dict(
        local_group=group("target"),
        peer_groups=[group("source")],
        bind_host="127.0.0.1",
        kvcm_endpoint="127.0.0.1:50051",
        trusted_private_domain=True,
        control_profile="slotguard-hash-v3",
        peer_control_profiles={"source": "slotguard-hash-v3"},
    )


if __name__ == "__main__":
    unittest.main()
