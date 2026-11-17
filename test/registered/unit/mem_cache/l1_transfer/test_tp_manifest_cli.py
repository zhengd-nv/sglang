"""Generate real serving configurations without CUDA, networking, or credentials."""

import importlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from test_protocol import PACKAGE, register_cpu_ci

from sglang.test.test_utils import CustomTestCase

c = importlib.import_module(f"{PACKAGE}.config")
m = importlib.import_module(f"{PACKAGE}.tp_manifest")
register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class ManifestCLITest(CustomTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "tp4.json"

    def argv(self, *extra):
        return [
            "--local-engine",
            "p0",
            "--local-incarnation",
            "p0-boot-1",
            "--local-host",
            "127.0.0.1",
            "--local-reporter",
            "p0-tp0-reporter",
            "--local-base-port",
            "31000",
            "--peer-engine",
            "p1",
            "--peer-incarnation",
            "p1-boot-1",
            "--peer-host",
            "127.0.0.2",
            "--peer-reporter",
            "p1-tp0-reporter",
            "--peer-base-port",
            "32000",
            "--domain",
            "fabric-domain",
            "--instance",
            "isolated-instance",
            "--model",
            "/models/glm",
            "--bind-host",
            "0.0.0.0",
            "--kvcm-endpoint",
            "127.0.0.1:50052",
            "--output",
            str(self.output),
            *extra,
        ]

    @staticmethod
    def group(engine, host, port):
        # Freeze the existing GLM TP4 geometry and identity contract.
        digests = (
            "d78a62557c4f299439e9c3da2d222014755e81aebc4e61ec53ba8afe2667b72f",
            "a8e190ca5fcb5d16e9f203debbab0f2ee9ea2275143b55b65db1e0f5b7942e91",
            "ff066d16ccf76ef1f0b057b8db4eff6a6f0078d13067f4f7f2e50df5df5f22a8",
            "b3939bca4e61d1da32cb6010443131f7b96c28360cb40f3d78d3ceccc86e7fa1",
        )
        return {
            "reporter_id": f"{engine}-tp0-reporter",
            "ranks": [
                {
                    "identity": {
                        "engine_id": engine,
                        "incarnation": f"{engine}-boot-1",
                        "domain_id": "fabric-domain",
                        "instance_id": "isolated-instance",
                        "model_id": "/models/glm",
                        "layout_digest": digests[rank],
                        "tp_group_id": f"{engine}-tp4",
                        "tp_size": 4,
                        "tp_rank": rank,
                        "shard_id": f"tp{rank}",
                        "protocol_version": 1,
                    },
                    "control_endpoint": f"http://{host}:{port + rank}",
                }
                for rank in range(4)
            ],
        }

    def expected(self):
        return {
            "local_group": self.group("p0", "127.0.0.1", 31000),
            "peer_groups": [self.group("p1", "127.0.0.2", 32000)],
            "bind_host": "0.0.0.0",
            "kvcm_endpoint": "127.0.0.1:50052",
            "trusted_private_domain": True,
            "bearer_env": "SGLANG_NVLINK_CONTROL_TOKEN",
            "min_headroom_tokens": 8192,
            "control_profile": "slotguard-hash-v3",
            "peer_control_profiles": {"p1": "slotguard-hash-v3"},
            "prefetch_pending_capacity": 32,
            "prefetch_pipeline_depth": 2,
            "publication_context_entries": 65536,
            "publication_context_bytes": 64 * 1024 * 1024,
            "slotguard_limits": asdict(c.SlotGuardLimits()),
        }

    def generate(self, *extra):
        with patch.dict(os.environ, SGLANG_NVLINK_CONTROL_TOKEN="test-only-secret"):
            m.main(self.argv(*extra))
        text = self.output.read_text()
        self.assertNotIn("test-only-secret", text)
        self.assertNotIn("bearer_token", text)
        self.assertTrue(text.endswith("\n"))
        return json.loads(text)

    def test_default_output_is_the_exact_slotguard_manifest(self):
        self.assertEqual(self.generate(), self.expected())
        with patch.dict(
            os.environ,
            SGLANG_NVLINK_CONFIG=str(self.output),
            SGLANG_NVLINK_CONTROL_TOKEN="test-only-secret",
        ):
            for rank in range(4):
                with self.subTest(rank=rank):
                    config = c.ServingConfig.from_env(tp_rank=rank)
                    self.assertEqual(config.directory.target.tp_rank, rank)
                    self.assertEqual(config.bind_address, ("0.0.0.0", 31000 + rank))
                    self.assertEqual(config.directory.entries[0].identity.tp_rank, rank)
                    self.assertEqual(config.control_profile.value, "slotguard-hash-v3")
                    self.assertNotIn("test-only-secret", repr(config))

    def test_mixed_peer_profiles_fail_closed(self):
        data = self.generate()
        data["peer_control_profiles"]["p1"] = "slotguard-v2"
        self.output.write_text(json.dumps(data))
        with patch.dict(
            os.environ,
            SGLANG_NVLINK_CONFIG=str(self.output),
            SGLANG_NVLINK_CONTROL_TOKEN="test-only-secret",
        ):
            with self.assertRaises(ValueError):
                c.ServingConfig.from_env(tp_rank=0)

    def test_explicit_512k_limits(self):
        value = self.generate(
            "--max-pin-pages", "8192", "--max-remote-pinned-pages", "8192"
        )
        self.assertEqual(value["slotguard_limits"]["max_pin_pages"], 8192)
        self.assertEqual(value["slotguard_limits"]["max_remote_pinned_pages"], 8192)

    def test_invalid_limits_fail_before_writing(self):
        for extra in (
            ("--max-pin-pages", "8192"),
            ("--max-pin-pages", "8193", "--max-remote-pinned-pages", "8193"),
        ):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    m.main(self.argv(*extra))
            self.assertFalse(self.output.exists())

    def test_async_unpin_defaults_on_and_only_sync_flag_disables_it(self):
        env = dict(
            SGLANG_NVLINK_CONFIG=str(self.output),
            SGLANG_NVLINK_CONTROL_TOKEN="test-only-secret",
        )
        for extra, expected in (((), True), (("--sync-unpin",), False)):
            with self.subTest(extra=extra):
                if self.output.exists():
                    self.output.unlink()
                data = self.generate(*extra)
                self.assertEqual("async_unpin" in data, not expected)
                with patch.dict(os.environ, env):
                    self.assertIs(c.ServingConfig.from_env().async_unpin, expected)

    def test_existing_output_is_never_overwritten(self):
        self.output.write_text("keep-existing-file\n")
        with self.assertRaises(FileExistsError):
            m.main(self.argv())
        self.assertEqual(self.output.read_text(), "keep-existing-file\n")

    def test_invalid_rank_port_is_rejected_before_output_creation(self):
        with self.assertRaises(ValueError):
            m.main(self.argv("--local-base-port", "65534"))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
