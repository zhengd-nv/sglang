"""Fixed ROOT event hashes, independent oracle, and Linux native parity check."""

import ast
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[5]
FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[5]
        / "sgl-router/sgl-kv-indexer/tests/event_hash_golden.json"
    ).read_text()
)
ci_spec = importlib.util.spec_from_file_location(
    "_l1_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(ci_spec)
ci_spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def fixture_cases():
    yield FIXTURE
    page64 = FIXTURE["page64"]
    yield {
        **page64,
        "token_ids": list(range(page64["token_start"], page64["token_end"])),
    }


def real_int64_helper():
    # Execute the actual standalone helper, avoiding unrelated CUDA imports.
    path = ROOT / "python/sglang/srt/mem_cache/utils.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "hash_str_to_int64"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["hash_str_to_int64"]


class EventHashGoldenTest(CustomTestCase):
    def test_independent_sha256_oracle(self):
        for fixture in fixture_cases():
            with self.subTest(page_size=fixture["page_size"]):
                page_size, tokens = fixture["page_size"], fixture["token_ids"]
                prior = b""
                digests, keys = [], []
                for start in range(0, len(tokens) // page_size * page_size, page_size):
                    page_bytes = b"".join(
                        token.to_bytes(4, "little")
                        for token in tokens[start : start + page_size]
                    )
                    prior = hashlib.sha256(prior + page_bytes).digest()
                    digests.append(prior.hex())
                    keys.append(int.from_bytes(prior[:8], "big", signed=True))
                self.assertEqual(digests, fixture["digests"])
                self.assertEqual(keys, fixture["event_keys"])
                self.assertTrue(any(key < 0 for key in keys))

    def test_actual_python_event_int64_helper(self):
        convert = real_int64_helper()
        for fixture in fixture_cases():
            self.assertEqual(
                [convert(value) for value in fixture["digests"]], fixture["event_keys"]
            )

    def test_truncated_chaining_and_storage_namespace_are_not_event_chain(self):
        first_digest = bytes.fromhex(FIXTURE["digests"][0])
        second_page = b"".join(
            token.to_bytes(4, "little") for token in FIXTURE["token_ids"][4:8]
        )
        wrong = hashlib.sha256(first_digest[:8] + second_page).hexdigest()
        self.assertNotEqual(wrong, FIXTURE["digests"][1])
        storage_seed = hashlib.sha256(
            b"sglang-cache-namespace-v1"
            + b"\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            + b"\x00"
        ).digest()
        first_page = b"".join(
            token.to_bytes(4, "little") for token in FIXTURE["token_ids"][:4]
        )
        self.assertNotEqual(
            hashlib.sha256(storage_seed + first_page).hexdigest(), FIXTURE["digests"][0]
        )

    @unittest.skipUnless(
        sys.platform.startswith("linux") and sys.byteorder == "little",
        "actual native hash requires little-endian Linux",
    )
    def test_actual_linux_native_event_helper(self):
        # No oracle substitution: this imports and executes the serving helper.
        from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

        for fixture in fixture_cases():
            with self.subTest(page_size=fixture["page_size"]):
                page_size = fixture["page_size"]
                tokens = fixture["token_ids"]
                complete = tokens[: len(tokens) // page_size * page_size]
                digests = get_hash_str(complete, page_size=page_size)
                self.assertEqual(digests, fixture["digests"])
                self.assertEqual(
                    [hash_str_to_int64(value) for value in digests],
                    fixture["event_keys"],
                )


if __name__ == "__main__":
    unittest.main()
