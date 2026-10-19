# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of request ownership ordering and the real API forwarding."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
MODULE = 'sglang.srt.mem_cache.l1_transfer.peer.request'
spec = importlib.util.spec_from_file_location(
    MODULE, ROOT / 'python/sglang/srt/mem_cache/l1_transfer/peer/request.py')
request = importlib.util.module_from_spec(spec)
spec.loader.exec_module(request)


class FakeEngine:
    server_args = SimpleNamespace(page_size=16)

    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def l1_peer(self, action, **payload):
        self.calls.append((action, payload))
        if self.failure == action:
            raise RuntimeError(action)
        if action == 'restore':
            return dict(pinned_pages=1, restored_tokens=16, local_tokens=0)
        return dict(target_active=False, pinned_pages=0)

    def generate(self, **arguments):
        self.calls.append(('generate', arguments))
        if self.failure == 'generate':
            raise RuntimeError('generate')
        return {'meta_info': {'cached_tokens': 16}, 'text': 'answer'}

    def shutdown(self):
        self.calls.append(('shutdown', {}))


class RequestRestoreTests(unittest.TestCase):
    def setUp(self):
        self.arguments = dict(input_ids=list(range(23)), sampling_params={'n': 1})
        self.restore = dict(source={'host': 'source', 'port': 31000}, prefix_tokens=16)

    def test_success_releases_only_after_generate_and_preserves_result(self):
        engine = FakeEngine()
        result = request.generate_with_restore(engine, self.arguments, self.restore)
        self.assertEqual([x[0] for x in engine.calls], ['restore', 'generate', 'finish'])
        self.assertEqual(engine.calls[0][1]['tokens'], list(range(16)))
        self.assertEqual(engine.calls[1][1]['input_ids'], list(range(23)))
        self.assertTrue(result['meta_info']['l1_restore']['consumer_released'])
        self.assertEqual(result['text'], 'answer')

    def test_failure_never_blindly_finishes_consumption(self):
        for failure, expected in [('restore', ['restore']),
                                  ('generate', ['restore', 'generate', 'shutdown']),
                                  ('finish', ['restore', 'generate', 'finish', 'shutdown'])]:
            with self.subTest(failure=failure):
                engine = FakeEngine(failure)
                with self.assertRaises(RuntimeError):
                    request.generate_with_restore(engine, self.arguments, self.restore)
                self.assertEqual([x[0] for x in engine.calls], expected)

    def test_invalid_requests_do_not_acquire(self):
        cases = [({'stream': True}, self.restore),
                 ({'input_ids': [[1, 2]]}, self.restore),
                 ({'input_ids': [True] * 23}, self.restore),
                 ({'sampling_params': {'n': 2}}, self.restore),
                 ({'sampling_params': [{}]}, self.restore),
                 ({'cache_salt': 'other'}, self.restore),
                 ({'session_id': 'session'}, self.restore),
                 ({'image_data': 'image'}, self.restore),
                 ({}, dict(self.restore, prefix_tokens=15)),
                 ({}, dict(self.restore, prefix_tokens=32)),
                 ({}, dict(self.restore, prefix_tokens=True)),
                 ({}, dict(self.restore, source={'host': 'a', 'port': 0})),
                 ({}, dict(self.restore, fallback=True))]
        for overrides, restore in cases:
            with self.subTest(overrides=overrides, restore=restore):
                engine = FakeEngine()
                with self.assertRaises(ValueError):
                    request.generate_with_restore(engine, dict(self.arguments, **overrides), restore)
                self.assertEqual(engine.calls, [])

    def test_actual_generate_entry_forwards_arguments_without_recursive_restore(self):
        # Compile the real method without importing the GPU framework stack.
        tree = ast.parse((ROOT / 'python/sglang/srt/entrypoints/engine.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        namespace = {}
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                           names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<Engine.generate>', 'exec'), namespace)
        engine = FakeEngine()
        with patch.dict(sys.modules, {MODULE: request}):
            result = namespace['generate'](engine, input_ids=list(range(23)),
                sampling_params={'temperature': 0}, rid='request-1', l1_restore=self.restore)
        self.assertEqual(engine.calls[1][1]['rid'], 'request-1')
        self.assertNotIn('l1_restore', engine.calls[1][1])
        self.assertEqual(result['text'], 'answer')


if __name__ == '__main__':
    unittest.main()
