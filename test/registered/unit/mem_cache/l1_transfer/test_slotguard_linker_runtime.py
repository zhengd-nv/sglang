"""Real guarded cache/source lifecycle through the production flush method."""

import logging
import subprocess
import sys
import time
import unittest
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_slotguard_serving as fixture
import torch
from test_slotguard_external_prefetch import insert_computed

from sglang.srt.mem_cache.allocator.page_identity import PageSpan
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.l1_transfer.role_gate import RoleGate
from sglang.srt.mem_cache.l1_transfer.slotguard_group import SlotGuardAllocationGroup
from sglang.srt.mem_cache.l1_transfer.slotguard_linker import SlotGuardLinker, _Runtime
from sglang.srt.mem_cache.l1_transfer.tp_group import SchedulerCollective
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.l1_transfer_scheduler import serving_method
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


class RuntimeTest(CustomTestCase):
    def setUp(self):
        self.fixture = fixture.ServingCompositionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.attach_source()
        self.runtime = self.fixture.owner
        self.runtime.__class__ = _Runtime
        self.cache, self.alloc = self.fixture.cache, self.fixture.allocator
        self.runtime.cache = self.cache
        self.runtime.min_headroom_tokens = 0
        self.runtime.gate = RoleGate()
        self.runtime.backend = NS(
            _directory=self.fixture.config.directory,
            _gate=self.runtime.gate,
            transport_arbiter=self.runtime.arbiter,
            owner_snapshot=lambda: NS(
                accepted=0, queued=0, working=0, imports=0, batches=0, unknown=0
            ),
            close=lambda: None,
        )
        collective = SchedulerCollective(lambda value: [value] * 4, 0)
        self.group = SlotGuardAllocationGroup(
            collective,
            self.alloc,
            self.runtime.source_service,
            evictable_tokens=self.cache.evictable_size,
            scope="linker-runtime-test",
        )
        self.runtime.allocation_group = self.group
        self.runtime._initialize_target(
            self.cache, self.fixture.config, True, collective
        )
        self.addCleanup(
            lambda: self.runtime.transport._lookup_worker.shutdown(wait=True)
        )
        self.addCleanup(
            lambda: self.runtime.transport._prepare_worker.shutdown(wait=True)
        )
        self.linker = SlotGuardLinker(
            NS(),
            self.fixture.ctx.params,
            components=set(self.cache.components),
            extra_config={},
        )
        self.linker.load_mode = "prefetch"
        self.linker.runtime = self.runtime
        self.cache.linker = UnifiedCacheLinkerWrapper(self.cache, self.linker)
        self.group.maintenance("startup")
        self.scheduler = NS(
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.alloc,
            req_to_token_pool=self.cache.req_to_token_pool,
            pause_state=NS(paused=False),
            gracefully_exit=False,
            running_batch=NS(reqs=[]),
            last_batch=None,
            result_queue=(),
            chunked_req=None,
            cache_scheduler_hooks=self.linker.scheduler_hooks(),
            is_fully_idle=lambda **kwargs: True,
            uocr_runtime=None,
            grammar_manager=NS(clear=lambda: None),
            draft_worker=None,
            metrics_reporter=NS(
                reset_metrics=lambda: None, is_stats_logging_rank=False
            ),
            invariant_checker=NS(run=lambda _: None),
        )
        for name in ("flush_cache", "_flush_cache_pools"):
            method = serving_method("python/sglang/srt/managers/scheduler.py", name)
            method.__globals__.update(
                CheckPhase=NS(FLUSH="flush"), logger=logging.getLogger(__name__)
            )
            setattr(self.scheduler, name, MethodType(method, self.scheduler))

    def tick(self):
        self.scheduler.cache_scheduler_hooks.on_requests_received(
            paused=self.scheduler.pause_state.paused,
            batches=(self.scheduler.running_batch,),
            chunked_req=None,
        )

    def settle(self):
        deadline = time.monotonic() + 3
        while True:
            try:
                self.runtime.assert_quiescent()
                return
            except RuntimeError:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.001)

    def test_production_flush_reclaims_pages_and_rebuilds_publication_epoch(self):
        insert_computed(self.cache, self.group, 128)
        self.settle()
        old_epoch = self.alloc.slot_guard.pool_epoch
        old_hook = self.cache.publication_hook
        self.assertTrue(self.scheduler.flush_cache(empty_cache=False))
        self.assertEqual(self.alloc.slot_guard.pool_epoch, old_epoch + 1)
        self.assertIsNot(self.cache.publication_hook, old_hook)
        self.assertEqual(self.cache.publication_hook.scope.pool_epoch, old_epoch + 1)
        self.assertEqual(self.alloc.verify_byte_accounting(), [])
        self.assertFalse(self.runtime.resetting)
        insert_computed(self.cache, self.group, 64)
        self.settle()
        self.assertTrue(self.scheduler.flush_cache(empty_cache=False))

    def test_busy_flush_keeps_epoch_and_capture_usable(self):
        epoch = self.alloc.slot_guard.pool_epoch
        with patch.object(
            self.runtime.transport,
            "assert_quiescent",
            side_effect=RuntimeError("read live"),
        ):
            self.assertFalse(self.scheduler.flush_cache(empty_cache=False))
        self.assertEqual(self.alloc.slot_guard.pool_epoch, epoch)
        self.assertFalse(self.runtime.source_service.external_snapshot().paused)
        self.assertFalse(self.runtime.resetting)
        self.assertTrue(self.scheduler.flush_cache(empty_cache=False))

    def test_flush_during_pause_does_not_reopen_source_admission(self):
        self.scheduler.pause_state.paused = True
        self.tick()
        with patch.object(
            self.runtime.transport,
            "assert_quiescent",
            side_effect=RuntimeError("read live"),
        ):
            self.assertFalse(self.scheduler.flush_cache(empty_cache=False))
        self.assertTrue(self.runtime.source_service.external_snapshot().paused)
        self.assertTrue(self.scheduler.flush_cache(empty_cache=False))
        self.assertTrue(self.runtime.source_service.external_snapshot().paused)

    def test_direct_cache_reset_cannot_bypass_production_transaction(self):
        epoch = self.alloc.slot_guard.pool_epoch
        with self.assertRaisesRegex(RuntimeError, "Scheduler.flush_cache"):
            self.cache.reset()
        self.assertEqual(self.alloc.slot_guard.pool_epoch, epoch)

    def test_scheduler_pause_keeps_progress_and_source_admission_fenced(self):
        self.scheduler.pause_state.paused = True
        self.tick()
        self.assertTrue(self.runtime.source_service.external_snapshot().paused)
        self.scheduler.pause_state.paused = False
        self.tick()
        self.assertFalse(self.runtime.source_service.external_snapshot().paused)
        self.assertFalse(hasattr(self.runtime, "admission"))
        self.assertFalse(hasattr(self.runtime, "coordinator"))

    def test_idle_hook_selects_new_runtime_without_legacy_wrapper(self):
        self.scheduler.session_controller = NS(maybe_reap=lambda now: None)
        self.scheduler.flush_wrapper = NS(check_pending=lambda: None)
        self.scheduler.external_corpus_manager = None
        method = serving_method(
            "python/sglang/srt/managers/scheduler.py", "process_input_requests"
        )
        method(self.scheduler, [])
        self.assertEqual(self.runtime._ticks, 1)
        self.assertFalse(hasattr(self.scheduler, "linker_prefetch_queue"))


class PrefetchRequestTest(CustomTestCase):
    def test_accepts_the_kv_transfer_the_full_component_builds(self):
        """A real KV PoolTransfer, whose name is a plain string, is submitted."""
        submitted = []
        linker = SlotGuardLinker.__new__(SlotGuardLinker)
        linker.runtime = NS(
            _paused=False,
            transport=NS(prefetch=lambda request, t: submitted.append(t) or "future"),
        )
        span = PageSpan(page_ids=(3, 4), page_size=64, token_offset=0, token_length=128)
        transfer = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(128),
            keys=["a", "b"],
            page_span=span,
        )
        self.assertEqual(linker.prefetch_request("req", [transfer]), "future")
        self.assertEqual(submitted, [transfer])


class LauncherTest(CustomTestCase):
    def test_dynamic_backend_import_does_not_require_legacy_target_modules(self):
        # Fresh interpreter: prior fixture imports must not hide a dependency.
        code = """
import sys
prefix = "sglang.srt.mem_cache.l1_transfer."
for name in ("serving", "prefetch_wrapper", "tp_admission", "test_admin", "slotguard_admin"):
    sys.modules[prefix + name] = None
from sglang.srt.mem_cache.l1_transfer.slotguard_linker import SlotGuardLinker
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
