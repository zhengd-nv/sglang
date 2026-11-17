"""Real CPU guard/Tree/service composition; transports are hermetic CPU fakes.

This exercises B's injected composition seam, not production activation or CUDA.
"""

import json
import os
import tempfile
import time
import unittest
from array import array
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_glm_fabric_adapter import glm_pool
from test_nvlink_backend import pool
from test_publication_capture import reference_native_hash
from test_sglang_adapter import BorrowDriver

from sglang.srt.mem_cache.allocator.slotguard_paged import (
    SlotGuardPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TreeSlot
from sglang.srt.mem_cache.l1_transfer import config, serving_runtime
from sglang.srt.mem_cache.l1_transfer.control_profile import ControlProfile
from sglang.srt.mem_cache.l1_transfer.metadata import event_keys_and_hashes
from sglang.srt.mem_cache.l1_transfer.protocol import (
    EngineIdentity,
    LookupQuery,
    LookupResult,
    TransferId,
)
from sglang.srt.mem_cache.l1_transfer.slot_guard import SlotGuard
from sglang.srt.mem_cache.l1_transfer.slotguard_protocol import PinStatus
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.l1_transfer_cpu import NvlinkEngine
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def identity(name, rank=0):
    return EngineIdentity(
        name,
        name + "-boot",
        "fabric",
        "instance",
        "model",
        "layout",
        name + "-tp",
        4,
        rank,
        f"shard-{rank}",
    )


def cache_fixture(name="source"):
    guard = SlotGuard(
        1,
        identity(name).incarnation,
        max_remote_pinned_pages=32,
        max_active_leases=32,
        max_pending_reclaim_pages=32,
        max_pages=32,
        shared_headroom_credit=True,
    )
    allocator = SlotGuardPagedTokenToKVPoolAllocator(
        2048, 64, torch.float16, "cpu", None, False, slot_guard=guard
    )
    rows = ReqToTokenPool(4, 4096, "cpu", False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            False,
            rows,
            allocator,
            64,
            tree_slots=(TreeSlot(component="FULL", family=TREE_TOKEN),),
            tree_core_backend="python",
        )
    )
    return cache, allocator


class ServingCompositionTest(CustomTestCase):
    def setUp(self):
        self.hash_patch = patch(
            "sglang.srt.mem_cache.utils.get_native_hash", reference_native_hash
        )
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.cache, self.allocator = cache_fixture()
        self.owner = object.__new__(serving_runtime.SlotGuardServingRuntime)
        self.owner.slotguard_mode = True
        self.owner.closed = False
        self.owner.engine = NvlinkEngine(0, _driver=BorrowDriver())
        self.owner.adapter = SGLangPoolAdapter(pool(0), self.owner.engine)
        self.ctx = NS(
            params=NS(token_to_kv_pool_allocator=self.allocator, page_size=64),
            tp_size=4,
            is_dsa=True,
            tp_worker=NS(model_runner=NS(forward_stream=object())),
        )
        from sglang.srt.mem_cache.l1_transfer.directory import (
            EngineDirectory,
            EngineDirectoryEntry,
        )

        self.config = config.ServingConfig(
            EngineDirectory(
                identity("source"),
                (
                    EngineDirectoryEntry(
                        "target", identity("target"), "http://127.0.0.1:1234"
                    ),
                ),
            ),
            ("127.0.0.1", 0),
            "127.0.0.1:50051",
            "test-only-credential",
            control_profile=ControlProfile.SLOTGUARD_HASH_V3,
        )
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for name in (
            "gateway",
            "backend",
            "client",
            "source_service",
            "publication_hook",
            "publication_worker",
            "arbiter",
            "admission",
            "adapter",
            "engine",
            "channel",
        ):
            resource = getattr(self.owner, name, None)
            if resource is not None:
                if name == "admission":
                    resource = resource.worker
                    if not hasattr(resource, "close"):
                        continue
                deadline = time.monotonic() + 2
                while True:
                    try:
                        resource.close()
                        break
                    except RuntimeError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.001)

    def test_slotguard_device_resolution_rejects_every_wrong_backing_or_context(self):
        for fault in (
            "pool",
            "allocator",
            "current",
            "stream",
            "stream_generic",
            "owner",
            "owner_bool",
            "tensor_cpu",
            "tensor_wrong",
            "tensor_generic",
            "missing_tensor",
        ):
            with self.subTest(fault=fault):
                pool = glm_pool(size=2048, coalesced=True)
                pool.device = "cuda"
                allocator = NS(device="cuda", get_kvcache=lambda pool=pool: pool)
                stream = NS(device="cuda:0")
                ctx = NS(tp_worker=NS(model_runner=NS(forward_stream=stream)))
                current = 0
                if fault == "pool":
                    pool.device = "cuda:1"
                elif fault == "allocator":
                    allocator.device = "cuda:1"
                elif fault == "current":
                    current = 1
                elif fault == "stream":
                    stream.device = "cuda:1"
                elif fault == "stream_generic":
                    stream.device = "cuda"
                elif fault == "owner":
                    pool.index_key_cache._fixed_fabric_owner.device_id = 1
                elif fault == "owner_bool":
                    pool._fixed_fabric_owner.device_id = False
                elif fault == "tensor_cpu":
                    pool.kv_buffer[77].device = "cpu"
                elif fault == "tensor_wrong":
                    pool.index_k_with_scale_buffer[77].device = "cuda:1"
                elif fault == "tensor_generic":
                    pool.kv_buffer[77].device = "cuda"
                elif fault == "missing_tensor":
                    pool.kv_buffer.pop()
                with (
                    patch.object(torch.cuda, "current_device", return_value=current),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    serving_runtime.SlotGuardServingRuntime._slotguard_pool_device(
                        ctx, allocator
                    )

    def test_slotguard_resolves_all_ordinals_and_checks_all_156_tensor_devices(self):
        pool = glm_pool(size=2048, coalesced=True)
        pool.device = "cuda"
        allocator = NS(device="cuda", get_kvcache=lambda: pool)
        stream = NS(device="cuda:0")
        ctx = NS(tp_worker=NS(model_runner=NS(forward_stream=stream)))
        for ordinal in range(4):
            device = torch.device("cuda", ordinal)
            stream.device = device
            for owner in (
                pool._fixed_fabric_owner,
                pool.index_key_cache._fixed_fabric_owner,
            ):
                owner.device_id = ordinal
                for tensor in owner.tensors:
                    tensor.device = device
            with patch.object(torch.cuda, "current_device", return_value=ordinal):
                for pool_declared in ("cuda", device):
                    for alloc_declared in ("cuda", device):
                        pool.device, allocator.device = pool_declared, alloc_declared
                        self.assertEqual(
                            serving_runtime.SlotGuardServingRuntime._slotguard_pool_device(
                                ctx, allocator
                            ),
                            device,
                        )
                for index, tensor in enumerate(
                    pool.kv_buffer + pool.index_k_with_scale_buffer
                ):
                    tensor.device = torch.device("cuda", (ordinal + 1) % 4)
                    with (
                        self.subTest(ordinal=ordinal, buffer=index),
                        self.assertRaises(ValueError),
                    ):
                        serving_runtime.SlotGuardServingRuntime._slotguard_pool_device(
                            ctx, allocator
                        )
                    tensor.device = device

    def test_safe_startup_diagnostic_does_not_prevent_common_vote_or_cleanup(self):
        secret = "SECRET_TOKEN_AND_ADDRESS_10.0.0.1"

        class DiagnosticFailure(BaseException):
            pass

        def initialize(owner, *args, **kwargs):
            owner.slotguard_mode = True
            owner._startup_stage = 3
            raise ValueError(secret)

        def cleanup(owner):
            owner.cleanup_done.set()

        for failure_type in (
            None,
            RuntimeError,
            KeyboardInterrupt,
            SystemExit,
            DiagnosticFailure,
        ):
            with self.subTest(failure=failure_type):
                collective = NS(gather=Mock(return_value=[False] * 4))
                groups = NS(collective=Mock(return_value=collective))
                owner = object.__new__(serving_runtime.SlotGuardServingRuntime)
                caught = None
                with (
                    patch.object(
                        serving_runtime.SlotGuardServingRuntime,
                        "_initialize",
                        initialize,
                    ),
                    patch.object(
                        serving_runtime.SlotGuardServingRuntime, "_cleanup", cleanup
                    ),
                    patch.object(
                        serving_runtime.logger,
                        "error",
                        side_effect=(failure_type(secret) if failure_type else None),
                    ) as log,
                ):
                    try:
                        owner.__init__(NS(tp_rank=0), NS(), _startup_groups=groups)
                    except BaseException as exc:  # noqa: BLE001 -- injected interrupts
                        # Catch test injections so pytest itself is not interrupted.
                        caught = exc
                self.assertIs(type(caught), RuntimeError)
                self.assertEqual(str(caught), "SlotGuard startup composition failed")
                collective.gather.assert_called_once_with(False)
                self.assertTrue(owner.closed)
                self.assertTrue(owner.cleanup_done.wait(2))
                self.assertEqual(log.call_count, 1)
                self.assertEqual(
                    log.call_args.args,
                    ("SlotGuard startup failure stage=%d kind=%s", 3, "ValueError"),
                )
                self.assertNotIn(secret, repr(log.call_args))

    def test_diagnostic_isolation_does_not_swallow_original_main_path_interrupt(self):
        class OriginalFailure(BaseException):
            pass

        for failure_type in (KeyboardInterrupt, SystemExit, OriginalFailure):
            with self.subTest(failure=failure_type):
                original = failure_type("PRIVATE_MAIN_PATH_ERROR")
                owner = object.__new__(serving_runtime.SlotGuardServingRuntime)

                def initialize(owner, *args, _original=original, **kwargs):
                    owner.slotguard_mode = True
                    owner._startup_stage = 2
                    raise _original

                with (
                    patch.object(
                        serving_runtime.SlotGuardServingRuntime,
                        "_initialize",
                        initialize,
                    ),
                    patch.object(
                        serving_runtime.SlotGuardServingRuntime,
                        "_cleanup",
                        lambda owner: owner.cleanup_done.set(),
                    ),
                    patch.object(
                        serving_runtime.logger,
                        "error",
                        side_effect=SystemExit("PRIVATE_LOG_ERROR"),
                    ),
                    self.assertRaises(failure_type) as caught,
                ):
                    owner.__init__(NS(tp_rank=0), NS())
                self.assertIs(caught.exception, original)
                self.assertTrue(owner.closed)
                self.assertTrue(owner.cleanup_done.wait(2))

    def test_diagnostic_classification_never_invokes_custom_metaclass_equality(self):
        for mode in ("spoof_builtin", "raise"):
            with self.subTest(mode=mode):
                comparisons = []

                class CustomExceptionMeta(type):
                    def __eq__(cls, other, _mode=mode, _comparisons=comparisons):
                        _comparisons.append(other)
                        if _mode == "raise":
                            raise RuntimeError("PRIVATE_METACLASS_ERROR")
                        return True

                failure_type = CustomExceptionMeta(
                    "CUSTOM_SECRET_CLASS", (Exception,), {}
                )
                original = failure_type("PRIVATE_EXCEPTION_MESSAGE")

                def initialize(owner, *args, _original=original, **kwargs):
                    owner.slotguard_mode = True
                    owner._startup_stage = 3
                    raise _original

                collective = NS(gather=Mock(return_value=[False] * 4))
                groups = NS(collective=Mock(return_value=collective))
                owner = object.__new__(serving_runtime.SlotGuardServingRuntime)
                with (
                    patch.object(
                        serving_runtime.SlotGuardServingRuntime,
                        "_initialize",
                        initialize,
                    ),
                    patch.object(
                        serving_runtime.SlotGuardServingRuntime,
                        "_cleanup",
                        lambda owner: owner.cleanup_done.set(),
                    ),
                    patch.object(serving_runtime.logger, "error") as log,
                    self.assertRaisesRegex(
                        RuntimeError, "SlotGuard startup composition failed"
                    ),
                ):
                    owner.__init__(NS(tp_rank=0), NS(), _startup_groups=groups)
                collective.gather.assert_called_once_with(False)
                self.assertTrue(owner.closed)
                self.assertTrue(owner.cleanup_done.wait(2))
                self.assertEqual(
                    log.call_args.args,
                    ("SlotGuard startup failure stage=%d kind=%s", 3, "Exception"),
                )
                self.assertEqual(log.call_count, 1)
                self.assertEqual(comparisons, [])
                for private in (
                    "CUSTOM_SECRET_CLASS",
                    "PRIVATE_EXCEPTION_MESSAGE",
                    "PRIVATE_METACLASS_ERROR",
                ):
                    self.assertNotIn(private, repr(log.call_args))

    def attach_source(self):
        self.owner._initialize_slotguard_source(
            self.ctx, self.cache, self.config, NS(device="cpu")
        )

    def test_actual_publication_to_independent_service_zero_credit_is_busy(self):
        self.attach_source()
        self.assertIs(self.owner.source_service.guard, self.allocator.slot_guard)
        self.assertIs(self.owner.source_service.arbiter, self.owner.arbiter)
        self.assertIs(self.owner.catalog.guard, self.allocator.slot_guard)
        self.assertIs(self.cache.publication_hook, self.owner.publication_hook)
        self.assertEqual(self.owner.publication_worker.max_context_entries, 65536)
        self.assertEqual(
            self.owner.publication_worker.max_context_bytes, 64 * 1024 * 1024
        )
        values = self.allocator.alloc(192)
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(192))),
                value=values,
                page_span=self.allocator.last_allocation_spans[0],
            )
        )
        deadline = time.monotonic() + 2
        while len(self.allocator.slot_guard.snapshot().published_pages) != 3:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        target = identity("target")
        ticket = TransferId(target.incarnation, "request", 0)
        keys, hashes = event_keys_and_hashes(tuple(range(193)), 64)
        query = LookupQuery(
            target, ticket, keys, tuple(range(193)), 64, 192, 64, prefix_hashes=hashes
        )
        result = LookupResult(query, identity("source"), 192, keys)
        request = ControlProfile.SLOTGUARD_HASH_V3.make_pin(result)
        with patch.object(
            self.cache,
            "match_prefix",
            side_effect=AssertionError("source tree traversal"),
        ):
            response = self.owner.source_service.submit(request).result(2)
        self.assertIs(response.status, PinStatus.BUSY)
        self.assertFalse(self.owner.engine._exports)
        self.assertEqual(self.owner.arbiter.owner_snapshot().owners, 0)
        self.assertFalse(hasattr(self.owner, "dispatcher"))
        self.assertFalse(hasattr(self.owner, "source"))

    def test_exact_capability_validation_and_lifecycle_refusal(self):
        # CPU guard composition is useful for contracts, but cannot activate
        # production CUDA serving or substitute for an explicit device ordinal.
        with self.assertRaisesRegex(ValueError, "CUDA pool ordinal"):
            serving_runtime.SlotGuardServingRuntime._validate_slotguard(
                self.ctx, self.cache, self.config
            )
        for field, value in (("tp_size", 1), ("is_dsa", False)):
            old = getattr(self.ctx, field)
            setattr(self.ctx, field, value)
            with self.assertRaises(ValueError):
                serving_runtime.SlotGuardServingRuntime._validate_slotguard(
                    self.ctx, self.cache, self.config
                )
            setattr(self.ctx, field, old)
        self.ctx.params.token_to_kv_pool_allocator = NS(
            cpu_page_identity=lambda: self.allocator
        )
        with self.assertRaises(ValueError):
            serving_runtime.SlotGuardServingRuntime._validate_slotguard(
                self.ctx, self.cache, self.config
            )
        self.ctx.params.token_to_kv_pool_allocator = self.allocator


class SlotGuardConfigTest(CustomTestCase):
    def data(self):
        def group(name, port):
            return {
                "reporter_id": name,
                "ranks": [
                    {
                        "identity": asdict(identity(name, rank)),
                        "control_endpoint": f"http://127.0.0.1:{port + rank}",
                    }
                    for rank in range(4)
                ],
            }

        return {
            "local_group": group("source", 31000),
            "peer_groups": [group("target", 32000)],
            "bind_host": "127.0.0.1",
            "kvcm_endpoint": "127.0.0.1:50051",
            "trusted_private_domain": True,
            "control_profile": "slotguard-hash-v3",
            "peer_control_profiles": {"target": "slotguard-hash-v3"},
        }

    def load(self, data, rank=0):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data))
            with patch.dict(
                os.environ,
                {
                    "SGLANG_NVLINK_CONFIG": str(path),
                    "SGLANG_NVLINK_CONTROL_TOKEN": "test-only-credential",
                },
            ):
                return config.ServingConfig.from_env(rank)

    def test_explicit_full_manifest_profile_and_bounded_context(self):
        for rank in range(4):
            value = self.load(self.data(), rank)
            self.assertIs(value.control_profile, ControlProfile.SLOTGUARD_HASH_V3)
            self.assertEqual(value.publication_context_entries, 65536)
            self.assertEqual(value.publication_context_bytes, 64 * 1024 * 1024)
        for field, values in (
            ("mode", ("slotguard",)),
            ("control_profile", (None, "v1", "slotguard-v2")),
            (
                "peer_control_profiles",
                (
                    {},
                    {"target": "slotguard-v2"},
                    {"target": "slotguard-hash-v3", "extra": "slotguard-hash-v3"},
                ),
            ),
            ("publication_context_entries", (True, 8192, 39999, 262145)),
            ("publication_context_bytes", (True, 8 * 1024 * 1024, 257 * 1024 * 1024)),
        ):
            for bad in values:
                with self.subTest(field=field, bad=bad):
                    data = self.data()
                    data[field] = bad
                    with self.assertRaises(ValueError):
                        self.load(data)


if __name__ == "__main__":
    unittest.main()
