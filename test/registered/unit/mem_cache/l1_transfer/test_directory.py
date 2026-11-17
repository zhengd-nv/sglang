"""Static KVCM directory tests, with no framework or accelerator dependency."""

import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

PACKAGE = "sglang.srt.mem_cache.l1_transfer"
p = importlib.import_module(f"{PACKAGE}.protocol")
d = importlib.import_module(f"{PACKAGE}.directory")

# Load the actual CI marker for this contract module.
ci_spec = importlib.util.spec_from_file_location(
    "_l1_ci_register",
    Path(__file__).resolve().parents[5] / "python/sglang/test/ci/ci_register.py",
)
ci_register = importlib.util.module_from_spec(ci_spec)
ci_spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def identity(engine_id):
    return p.EngineIdentity(
        engine_id,
        f"{engine_id}-boot",
        "domain",
        "instance",
        "model",
        "layout",
        f"{engine_id}-tp",
        2,
        0,
        "shard-0",
    )


class DirectoryTest(CustomTestCase):
    def setUp(self):
        self.source = identity("source")
        self.target = identity("target")
        self.entry = d.EngineDirectoryEntry(
            "reporter-1", self.source, "http://127.0.0.1:8000"
        )
        self.directory = d.EngineDirectory(self.target, (self.entry,))

    def test_exact_reporter_lookup(self):
        self.assertIs(self.directory.lookup("reporter-1", self.source), self.entry)
        for reporter in ("reporter", "reporter-1 ", "source", "REPORTER-1"):
            with self.subTest(reporter=reporter), self.assertRaises(KeyError):
                self.directory.lookup(reporter, self.source)
        self.assertEqual(d.EngineDirectory(self.target, ()).entries, ())

    def test_stale_incarnation_and_identity(self):
        for field in ("incarnation", "engine_id", "tp_group_id"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.directory.lookup(
                    "reporter-1", replace(self.source, **{field: "old"})
                )

    def test_foreign_identity(self):
        for field, value in (
            ("engine_id", "target"),
            ("domain_id", "foreign"),
            ("instance_id", "foreign"),
            ("model_id", "foreign"),
            ("layout_digest", "foreign"),
            ("tp_size", 4),
            ("tp_rank", 1),
            ("shard_id", "foreign"),
        ):
            foreign = replace(self.source, **{field: value})
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.directory.lookup("reporter-1", foreign)
                with self.assertRaises(ValueError):
                    d.EngineDirectory(
                        self.target, (replace(self.entry, identity=foreign),)
                    )

    def test_duplicate_reporter_and_engine_rank(self):
        for entry in (
            self.entry,
            replace(self.entry, identity=identity("source-2")),
            replace(self.entry, reporter_id="reporter-2"),
            replace(
                self.entry,
                reporter_id="reporter-2",
                identity=replace(self.source, incarnation="new"),
            ),
        ):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                d.EngineDirectory(self.target, (self.entry, entry))
        other = replace(
            self.entry, reporter_id="reporter-2", identity=identity("source-2")
        )
        self.assertEqual(
            len(d.EngineDirectory(self.target, (self.entry, other)).entries), 2
        )

    def test_missing_or_mutable_fields(self):
        for changes in (
            {"reporter_id": ""},
            {"reporter_id": None},
            {"identity": None},
            {"identity": {}},
            {"control_endpoint": ""},
            {"control_endpoint": None},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.entry, **changes)
        for entries in ([self.entry], (None,), ({},)):
            with self.assertRaises(ValueError):
                d.EngineDirectory(self.target, entries)
        with self.assertRaises(TypeError):
            d.EngineDirectoryEntry("reporter-1", self.source)

    def test_control_endpoint_validation(self):
        for endpoint in ("http://host:1234", "https://host:443/", "http://[::1]:1234"):
            self.assertEqual(
                replace(self.entry, control_endpoint=endpoint).control_endpoint,
                endpoint,
            )
        for endpoint in (
            "host:80",
            "http://host",
            "http://host:0",
            "http://host:99999",
            "http://host:abc",
            "ftp://host:80",
            "http://:80",
            "http://[::1",
            "http://user:password@host:80",
            "http://user@host:80",
            "http://host:80/path",
            "http://host:80?x=1",
            "http://host:80#fragment",
            " http://host:80",
            "http://ho st:80",
            "http://host:80\n",
            "\x01http://host:80",
            "http://ho\x00st:80",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                replace(self.entry, control_endpoint=endpoint)


if __name__ == "__main__":
    unittest.main()
