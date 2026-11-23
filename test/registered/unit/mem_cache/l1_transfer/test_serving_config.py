import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.l1_transfer import config as c
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.wire import identity
from sglang.test.test_utils import CustomTestCase

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
        self.enterContext(envs.SGLANG_GLM_DSA_FIXED_FABRIC_POOL.override(True))

    def context(self, rank=0):
        params = NS(
            pp_size=1,
            attn_cp_size=1,
            is_eagle=False,
            enable_session_radix_cache=False,
            component_registry_override=None,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            enable_kv_cache_events=rank == 0,
            dynamo_kv_event_format=False,
            page_size=64,
            token_to_kv_pool_allocator=NS(page_size=64),
        )
        server_args = NS(
            disaggregation_mode="prefill",
            dcp_size=1,
            dp_size=1,
            speculative_algorithm=None,
            dllm_algorithm=None,
            enable_lora=False,
            enable_hisparse=False,
            enable_linker_mla_dedup=False,
            attn_dp_size=1,
            enable_mixed_chunk=False,
            enable_hierarchical_cache=False,
        )
        return NS(
            server_args=server_args,
            params=params,
            tp_size=4,
            tp_rank=rank,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            is_dsa=True,
            disable_radix_cache=False,
            enable_hierarchical_cache=False,
        )

    def test_glm_tp4_publisher_rank(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                ctx = self.context(rank)
                c.validate_serving_context(ctx, NS(_tree_core_backend="python"))
                self.assertEqual(ctx.params.enable_kv_cache_events, rank == 0)
                ctx.params.enable_kv_cache_events = rank != 0
                with self.assertRaisesRegex(ValueError, "KV events on TP0 only"):
                    c.validate_serving_context(ctx, NS(_tree_core_backend="python"))

    def test_global_native_events_required_on_every_rank(self):
        contexts = [self.context(rank) for rank in range(4)]
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

    def test_glm_topology_and_rank_bounds_fail_closed(self):
        for rank in (-1, 4):
            with (
                self.subTest(rank=rank),
                self.assertRaisesRegex(ValueError, "valid TP rank"),
            ):
                c.validate_serving_context(
                    self.context(rank), NS(_tree_core_backend="python")
                )
        for owner, field, value in (
            ("params", "pp_size", 2),
            ("params", "attn_cp_size", 2),
            ("server_args", "dp_size", 2),
            ("server_args", "dcp_size", 2),
            ("server_args", "attn_dp_size", 2),
            ("server_args", "disaggregation_mode", "decode"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ctx = self.context(1)
                setattr(getattr(ctx, owner), field, value)
                c.validate_serving_context(ctx, NS(_tree_core_backend="python"))

    def test_modes_fail_closed(self):
        cache = NS(_tree_core_backend="python")
        ctx = self.context()
        c.validate_serving_context(ctx, cache)
        for field, value in (
            ("tp_size", 1),
            ("is_hybrid_swa", True),
            ("is_dsa", False),
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

    def test_dllm_is_rejected_at_startup(self):
        """A DLLM server must fail SlotGuard startup, not its first extend batch."""
        ctx, cache = self.context(), NS(_tree_core_backend="python")
        configurator = NS(
            server_args=ctx.server_args,
            pp_size=1,
            attn_dp_size=1,
            is_draft_worker=False,
            is_hybrid_swa=False,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            kv_shard_size=1,
        )
        config = NS(
            publication_context_entries=65536,
            publication_context_bytes=64 * 1024 * 1024,
            slotguard_limits=NS(max_pages=65536),
        )
        checks = (
            (
                "factory",
                partial(
                    c.validate_slotguard_factory,
                    configurator,
                    config,
                    NS(tp_size=4, attn_dcp_size=1, attn_cp_size=1),
                    NS(enable_hisparse=False),
                    NS(page_size=64),
                ),
            ),
            ("serving", partial(c.validate_serving_context, ctx, cache)),
        )
        for _, check in checks:
            check()
        ctx.server_args.dllm_algorithm = "LowConfidence"
        for name, check in checks:
            with self.subTest(check=name), self.assertRaises(ValueError):
                check()

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
        """Depth 2 was accepted although the target runs one inbound read at a time."""
        self.assertEqual(self.load(self.data()).prefetch_pipeline_depth, 1)
        for depth in (True, False, 1.0, 2.0, "1", "2", None, 0, 2, 3, -1):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                self.load(self.data() | {"prefetch_pipeline_depth": depth})
        self.assertEqual(
            self.load(
                self.data() | {"prefetch_pipeline_depth": 1}
            ).prefetch_pipeline_depth,
            1,
        )

    def test_kvcm_endpoint_needs_host_and_port(self):
        """An endpoint with an empty host or a missing or bad port was accepted,
        so the failure came only at the first KVCM query."""
        for endpoint in ("127.0.0.1:50051", "kvcm.local:1", "[::1]:50051"):
            with self.subTest(endpoint=endpoint):
                data = self.data() | {"kvcm_endpoint": endpoint}
                self.assertEqual(self.load(data).kvcm_endpoint, endpoint)
        for endpoint in (
            ":",
            ":50051",
            "host:",
            "host:port",
            "host:0",
            "host:65536",
            "user@host:50051",
            "host:50051/path",
            " host:50051",
            "::1:50051",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                self.load(self.data() | {"kvcm_endpoint": endpoint})

    def test_manifest_size_is_bounded_before_parsing(self):
        """A manifest of any size was read and parsed on every rank."""
        text = json.dumps(self.data())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            env = {
                "SGLANG_NVLINK_CONFIG": str(path),
                "SGLANG_NVLINK_CONTROL_TOKEN": "test-only-secret",
            }
            for size, accepted in (
                (c.MAX_MANIFEST_BYTES, True),
                (c.MAX_MANIFEST_BYTES + 1, False),
            ):
                with self.subTest(size=size), patch.dict(os.environ, env):
                    # Trailing whitespace keeps the JSON valid at any size.
                    path.write_text(text.ljust(size))
                    if accepted:
                        c.ServingConfig.from_env()
                    else:
                        with self.assertRaises(ValueError):
                            c.ServingConfig.from_env()


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
