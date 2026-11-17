"""Task4b1 owner startup and narrow allocation integration, CPU only."""

import os
import tempfile
import time
import unittest
from array import array
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_slotguard_serving as composition
import torch

from sglang.srt.mem_cache.allocator.slotguard_paged import (
    SlotGuardPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TreeSlot
from sglang.srt.mem_cache.l1_transfer.config import SlotGuardLimits
from sglang.srt.mem_cache.l1_transfer.slotguard_runtime import SlotGuardSchedulerHooks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=75, suite="base-a-test-cpu")


class RuntimeFactoryTest(CustomTestCase):
    def test_real_empty_urc_rebinds_same_allocator_pool_and_row_owner(self):
        from test_slotguard_allocator import allocator

        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        alloc = allocator(pages=16, page_size=64, active=False)
        backing = NS(buffer=torch.empty((17, 64)))
        alloc._kvcache = backing
        pointer = backing.buffer.data_ptr()
        warmup = alloc.alloc(128)
        alloc.free(warmup)
        alloc.resize(NS(max_total_num_tokens=512))
        rows = ReqToTokenPool(4, 1024, "cpu", False)
        cache = UnifiedRadixCache(
            CacheInitParams(
                False,
                rows,
                alloc,
                64,
                tree_slots=(TreeSlot(component="FULL", family=TREE_TOKEN),),
                tree_core_backend="python",
            )
        )
        self.assertIsNone(rows._cpu_page_rows)
        self.assertIsNone(alloc.page_rows)
        guard = alloc.slot_guard
        alloc.activate(rows, cache)
        self.assertIs(cache.token_to_kv_pool_allocator, alloc)
        self.assertIs(alloc.get_kvcache(), backing)
        self.assertEqual(backing.buffer.data_ptr(), pointer)
        self.assertIs(alloc.slot_guard, guard)
        self.assertIs(rows._cpu_page_rows, alloc.page_rows)

    def test_dynamic_linker_factory_uses_the_same_guarded_allocator(self):
        self._check_frozen_factory("dynamic")

    def _check_frozen_factory(self, backend):
        from sglang.srt.mem_cache import kv_cache_configurator as m

        config_test = composition.SlotGuardConfigTest()
        config = config_test.load(config_test.data())
        owner = object.__new__(m.KVCacheConfigurator)
        for name, value in dict(
            _nvlink_config_loaded=False,
            _nvlink_serving_config=None,
            is_hybrid_swa=False,
            is_draft_worker=False,
            pp_size=1,
            attn_dp_size=1,
            mambaish_config=None,
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            kv_shard_size=1,
            kv_cache_dtype=torch.float16,
            device="cpu",
            model_config=NS(hf_config=object()),
            server_args=NS(disaggregation_mode="prefill"),
        ).items():
            setattr(owner, name, value)
        memory = NS(
            enable_unified_cache_external_linker=True,
            unified_cache_external_linker_backend=backend,
            unified_cache_external_linker_config={
                "linker_module_path": "sglang.srt.mem_cache.l1_transfer.slotguard_linker",
                "linker": "SlotGuardLinker",
                "load_mode": "prefetch",
            },
            enable_hisparse=False,
        )
        pool = object()
        with (
            patch.object(m, "_is_npu", False),
            patch.object(m, "get_memory", return_value=memory),
            patch.object(
                m, "get_disagg", return_value=NS(disaggregation_mode="prefill")
            ),
            patch.object(
                m,
                "get_parallel",
                return_value=NS(tp_size=4, tp_rank=0, attn_dcp_size=1),
            ),
            patch.object(m, "get_schedule", return_value=NS(page_size=64)),
            patch.object(m, "is_deepseek_dsa", return_value=True),
            patch.object(
                m.current_platform, "get_paged_allocator_cls", return_value=None
            ),
            patch(
                "sglang.srt.mem_cache.l1_transfer.config.ServingConfig.from_env",
                return_value=config,
            ) as load,
            patch.dict(os.environ, {"SGLANG_GLM_DSA_FIXED_FABRIC_POOL": "1"}),
        ):
            alloc = owner._build_token_to_kv_pool_allocator(
                sizes=NS(max_total_num_tokens=256),
                token_to_kv_pool=pool,
                is_compressed_kv_model=False,
                req_to_token_pool=NS(),
                token_to_kv_pool_allocator=None,
            )
            self.assertIs(type(alloc), SlotGuardPagedTokenToKVPoolAllocator)
            self.assertIsNone(alloc.cpu_page_identity())
            self.assertIs(alloc._frozen_serving_config, config)
            self.assertIs(alloc.get_kvcache(), pool)
            self.assertEqual(alloc.slot_guard.owner_counts().credit_remaining, 0)
            load.assert_called_once()
            # Another factory construction cannot reread changing env/files.
            owner._build_token_to_kv_pool_allocator(
                sizes=NS(max_total_num_tokens=256),
                token_to_kv_pool=pool,
                is_compressed_kv_model=False,
                req_to_token_pool=NS(),
                token_to_kv_pool_allocator=None,
            )
            load.assert_called_once()
            self.assertNotIn("bearer", repr(owner.server_args))

    def test_hard_limits_validate_types_and_consistency(self):
        for values in (
            {"max_pages": True},
            {"max_pages": 262145},
            {"max_active_leases": 65537},
            {"max_tombstones": 10},
            {"max_remote_pinned_pages": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SlotGuardLimits(**values)


class SchedulerBoundaryTest(CustomTestCase):
    def test_actual_normal_and_pd_entry_finish_before_metadata_and_waits(self):
        from sglang.test.kits.l1_transfer_scheduler import serving_method

        raw = serving_method(
            "python/sglang/srt/managers/scheduler.py", "_get_new_batch_prefill_raw"
        )
        pd = serving_method(
            "python/sglang/srt/disaggregation/prefill.py",
            "get_next_disagg_prefill_batch_to_run",
        )
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                events, capacities = [], []
                group = NS(
                    runtime_collective=NS(gather=lambda value: [value] * 4),
                    assert_ready_for_forward=lambda: events.append("ready"),
                    allocator=NS(set_common_capacity_tokens=capacities.append),
                )

                def execute(stage, body):
                    events.append("begin")
                    value = body(NS(capacity_tokens=192))
                    events.append("finish")
                    return NS(value=value)

                group.execute = execute
                batch, allocation = NS(), object()
                running = NS(batch_is_full=False)

                def allocate(running, prepared, delayer, *, allocation_only):
                    assert allocation_only and prepared == (0, 256, 64)
                    events.append("allocation-row-commit")
                    return batch, object(), allocation

                def finish(new, adder, running, *, allocation):
                    events.append(("metadata", allocation))
                    return new, running

                owner = NS(
                    enable_overlap=overlap,
                    cache_scheduler_hooks=SlotGuardSchedulerHooks(
                        NS(allocation_group=group)
                    ),
                    _prepare_new_batch_prefill=lambda running: (0, 256, 64),
                    _allocate_new_batch_prefill=allocate,
                    _finish_new_batch_prefill=finish,
                )
                owner._get_new_batch_prefill_raw = MethodType(raw, owner)
                owner.get_new_batch_prefill = lambda running: NS(
                    **dict(
                        zip(
                            ("batch_to_run", "running_batch"),
                            owner._get_new_batch_prefill_raw(None, running),
                        )
                    )
                )
                owner.process_pending_chunked_abort = lambda: events.append("abort")
                owner._process_hicache_events = lambda **kwargs: None
                owner.resolve_waiting_queue_bootstrap = lambda: events.append(
                    "bootstrap"
                )
                owner.process_prefill_chunk = lambda **kwargs: events.append(
                    "pd-copy-wait"
                )
                owner.dp_attn_adapter = NS(
                    maybe_prepare_mlp_sync_batch=lambda batch: batch
                )
                result = pd(owner, running, None)
                self.assertIs(result.batch_to_run, batch)
                self.assertLess(events.index("pd-copy-wait"), events.index("begin"))
                self.assertLess(
                    events.index("allocation-row-commit"), events.index("finish")
                )
                self.assertLess(
                    events.index("finish"), events.index(("metadata", allocation))
                )
                self.assertEqual(capacities, [192, None])

    def test_actual_extend_allocation_commits_cpu_row_before_metadata(self):
        from sglang.srt.managers import schedule_batch as m
        from sglang.srt.sampling.sampling_params import SamplingParams

        cache, alloc = composition.cache_fixture()
        req = m.Req(
            "row", "text", array("q", range(64)), SamplingParams(max_new_tokens=1)
        )
        req.full_untruncated_fill_ids = array("q", range(64))
        req.extend_end = 64
        req.last_node = cache.root_node_handle()
        batch = m.ScheduleBatch(
            reqs=[req],
            device="cpu",
            tree_cache=cache,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=alloc,
            hisparse_coordinator=None,
            model_config=NS(is_encoder_decoder=False, vocab_size=1024),
        )
        batch.maybe_evict_swa = lambda: None
        batch.is_dllm = lambda: False
        snapshot = alloc.begin_allocation_epoch(1)
        alloc.install_allocation_epoch(1, snapshot.held_pages)
        runtime = NS(
            features=NS(enable_encoder_swa_bounded_replay=False),
            mamba=NS(enable_mamba_extra_buffer=False),
        )
        with (
            patch.object(m, "get_exec", return_value=runtime),
            patch(
                "sglang.srt.mem_cache.allocation.attention_backends",
                return_value=("torch_native", "torch_native"),
            ),
            patch.object(
                m,
                "flatten_arrays_to_pinned_cpu",
                side_effect=AssertionError("metadata in epoch"),
            ),
        ):
            allocation = batch.prepare_for_extend_allocation()
        row = req.kv.req_pool_idx
        self.assertEqual(alloc.page_rows.get(row, 0, 64).page_ids, (1,))
        self.assertEqual(
            cache.req_to_token_pool.req_to_token[row, :64].tolist(),
            list(range(64, 128)),
        )
        alloc.finish_allocation_epoch(1, verified_headroom_pages=31)
        with (
            patch.object(m, "get_exec", return_value=runtime),
            patch.object(
                m.SamplingBatchInfo, "from_schedule_batch", return_value=object()
            ),
        ):
            batch.finish_prepare_for_extend(allocation)
        self.assertEqual(batch.prefill_input_ids_cpu.tolist(), list(range(64)))

    def test_real_four_process_runtime_ordering(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="slotguard-runtime-") as directory:
            context = mp.spawn(
                _runtime_worker,
                args=(str(Path(directory) / "rendezvous"),),
                nprocs=4,
                join=False,
            )
            deadline = time.monotonic() + 100
            try:
                while not context.join(timeout=0.2):
                    if time.monotonic() > deadline:
                        self.fail("runtime TP4 workers exceeded bounded deadline")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3)


def _runtime_worker(rank, rendezvous):
    import importlib

    import test_protocol as contract
    import torch.distributed as dist

    contract.PACKAGE = "sglang.srt.mem_cache.l1_transfer"
    contract.p = importlib.import_module(f"{contract.PACKAGE}.protocol")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    model = dist.new_group(
        ranks=[0, 1, 2, 3], backend="gloo", timeout=timedelta(seconds=15)
    )
    from sglang.srt.mem_cache.l1_transfer.directory import (
        EngineDirectoryEntry,
        EngineTPGroup,
    )
    from sglang.srt.mem_cache.l1_transfer.slotguard_runtime import (
        compose_slotguard_runtime,
        create_startup_groups,
    )

    entries = tuple(
        EngineDirectoryEntry(
            "reporter",
            composition.identity("source", r),
            f"http://127.0.0.1:{19000 + r}",
        )
        for r in range(4)
    )
    config = NS(
        directory=NS(target=entries[rank].identity),
        local_group=EngineTPGroup("reporter", entries),
        group_manifest_digest="cpu-startup-manifest",
        min_headroom_tokens=0,
        publication_context_entries=65536,
        publication_context_bytes=64 * 1024 * 1024,
        slotguard_limits=SlotGuardLimits(),
        control_profile=composition.ControlProfile.SLOTGUARD_HASH_V3,
        prefetch_pending_capacity=None,
        prefetch_pipeline_depth=1,
    )
    ctx = NS(
        tp_size=4,
        tp_rank=rank,
        tp_group=model,
        params=NS(tp_cache_group=model, pp_size=1),
        server_args=NS(model_path="wrong" if rank == 3 else "model", dp_size=1),
    )
    try:
        # One bad rank must make all ranks reject before either new_group.
        with patch.object(dist, "new_group", wraps=dist.new_group) as create:
            try:
                create_startup_groups(ctx, config)
                raise AssertionError("rank-local startup error was accepted")
            except ValueError:
                pass
            assert create.call_count == 0
            ctx.server_args.model_path = "model"
            ctx.params.token_to_kv_pool_allocator = NS(
                _frozen_serving_config=None if rank == 3 else config
            )
            try:
                compose_slotguard_runtime(ctx, NS())
                raise AssertionError("missing rank-local frozen snapshot was accepted")
            except ValueError:
                pass
            assert create.call_count == 0
            ctx.params.token_to_kv_pool_allocator._frozen_serving_config = config
            if rank == 3:
                config.prefetch_pipeline_depth = 2
            try:
                compose_slotguard_runtime(ctx, NS())
                raise AssertionError("rank-local configuration mismatch was accepted")
            except ValueError:
                pass
            assert create.call_count == 0
            config.prefetch_pipeline_depth = 1
            groups = create_startup_groups(ctx, config, timeout=timedelta(seconds=15))
            assert create.call_count == 2
        assert groups.target is not groups.allocation
        assert groups.target is not model and groups.allocation is not model
        assert (
            groups.target is not dist.group.WORLD
            and groups.allocation is not dist.group.WORLD
        )
        assert tuple(dist.get_process_group_ranks(groups.target)) == (0, 1, 2, 3)
        assert tuple(dist.get_process_group_ranks(groups.allocation)) == (0, 1, 2, 3)
        _runtime_gateway_failure(rank, groups)
        _runtime_gateway_failure(rank, groups, uncertain_rank=0)
        _runtime_gateway_failure(rank, groups, fail_rank=None, fail_unfence_rank=3)
        _runtime_gateway_failure(rank, groups, fail_rank=None)
        _runtime_entry_failures(rank, groups)
        dist.destroy_process_group(groups.allocation)
        dist.destroy_process_group(groups.target)
    finally:
        dist.destroy_process_group(model)
        dist.destroy_process_group()


def _runtime_gateway_failure(
    rank, groups, *, fail_rank=3, uncertain_rank=None, fail_unfence_rank=None
):
    """Actual serving constructor, real gateways/Gloo, explicit CPU transport."""
    import test_slotguard_group as fixture

    from sglang.srt.mem_cache.l1_transfer.control_http import HTTPControlGateway
    from sglang.srt.mem_cache.l1_transfer.serving_runtime import SlotGuardServingRuntime

    state = fixture._fixture(groups.allocation, pages=8)
    if rank == uncertain_rank:
        # Explicit CPU failure seam: an uncertain native owner must retain all
        # registration backing, even during a failed startup cleanup.
        state.service._uncertain = 1
    native_before = tuple(state.engine._memories)
    owner = object.__new__(SlotGuardServingRuntime)
    gateway = HTTPControlGateway(
        state.service.local_identity,
        tuple(state.service.exact_peers),
        bind_address=("127.0.0.1", 0),
        bearer_token="startup-cpu-test-only",
        trusted_private_domain=True,
        control_profile=composition.ControlProfile.SLOTGUARD_HASH_V3,
        slotguard_service=state.service,
        workers=2,
    )
    starting_fenced = []
    start = gateway.start

    def start_observed():
        starting_fenced.append(state.service.owner_snapshot().fenced)
        assert state.guard.owner_counts().frozen_epoch is None
        state.controller.assert_ready_for_forward()
        return start()

    def initialize(self, ctx, cache, **kwargs):
        self.closed, self.slotguard_mode = False, True
        self.source_service, self.allocation_group = state.service, state.controller
        self.publication_hook, self.backend = NS(freeze=Mock()), NS(close=Mock())
        self.gateway = gateway
        self.adapter, self.engine = state.adapter, state.engine

    failure = (
        patch.object(
            gateway._listener,
            "start",
            side_effect=RuntimeError("listener-start-failure"),
        )
        if rank == fail_rank
        else nullcontext()
    )
    unfence, unfence_calls = state.service.unfence_pins, 0

    def commit(generation):
        nonlocal unfence_calls
        unfence_calls += 1
        if rank == fail_unfence_rank and unfence_calls == 2:
            raise RuntimeError("startup-commit-failure")
        return unfence(generation)

    result = "returned-success"
    try:
        with (
            patch.object(SlotGuardServingRuntime, "_initialize", initialize),
            patch.object(gateway, "start", side_effect=start_observed),
            patch.object(state.service, "unfence_pins", side_effect=commit),
            failure,
        ):
            try:
                owner.__init__(NS(tp_rank=rank), NS(), _startup_groups=groups)
            except RuntimeError:
                result = "failed"
        outcomes = groups.collective(rank).gather(
            (
                result,
                tuple(starting_fenced),
                state.controller.faulted,
                state.service.owner_snapshot().fenced,
            )
        )
        # Ensure the old counterexample also terminates after recording RED.
        if result == "returned-success":
            gateway.close()
            state.service.close()
        else:
            assert owner.cleanup_done.wait(2)
            assert owner.cleanup_error == (
                "ServiceBusy" if rank == uncertain_rank else None
            )
        expected = (
            "failed"
            if fail_rank is not None or fail_unfence_rank is not None
            else "returned-success"
        )
        assert all(row[0] == expected for row in outcomes), outcomes
        assert all(row[1] == (True,) for row in outcomes), outcomes
        assert (
            all(row[2] and row[3] for row in outcomes)
            if expected == "failed"
            else all(not row[2] and not row[3] for row in outcomes)
        ), outcomes
        if rank == uncertain_rank:
            assert tuple(state.engine._memories) == native_before
            assert state.adapter.memories
        assert not any(thread.is_alive() for thread in gateway._threads())
    finally:
        # Resolve only the explicit synthetic test owner before test teardown.
        state.service._uncertain = 0
        fixture._close(state)


def _runtime_entry_failures(rank, groups):
    import test_slotguard_group as fixture

    from sglang.srt.mem_cache.l1_transfer.slotguard_runtime import common_prefill_entry

    for case in ("entry", "chunk-policy", "body", "private-flag"):
        state = fixture._fixture(groups.allocation, pages=4)
        fixture._owned(state, 1)
        state.controller.runtime_collective = groups.collective(rank)
        assert not common_prefill_entry(state.controller, None, None)
        assert state.controller.status().epoch == 0
        if case == "entry":
            operation = lambda: common_prefill_entry(
                state.controller, None if rank == 3 else (0, 64, 64), None
            )
        elif case == "chunk-policy":
            operation = lambda: common_prefill_entry(
                state.controller, (0, 128 if rank == 3 else 64, 64), None
            )
        elif case == "private-flag":
            operation = lambda: state.controller.execute(
                "private-flag",
                lambda _: None,
                request_key="same",
                demand_tokens=64,
                private_free_only=rank != 3,
            )
        else:

            def body(_decision):
                if rank == 3:
                    raise ValueError("rank-local CPU body failure")

            operation = lambda: state.controller.execute("body-failure", body)
        fixture._expect_fault(state, operation)
        fixture._close(state)


if __name__ == "__main__":
    unittest.main()
