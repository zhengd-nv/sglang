"""Real guarded CPU allocation/tree publication through the upstream wrapper.

Transport is a controllable running Future. A separate four-process Gloo test
checks that completion, common refusal and allocation failures agree across TP.
"""

import tempfile
import time
import unittest
from array import array
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.l1_transfer.slotguard_group import (
    AllocationEpochError,
    SlotGuardAllocationGroup,
)
from sglang.srt.mem_cache.l1_transfer.slotguard_prefetch import (
    SlotGuardPrefetchOwner,
)
from sglang.srt.mem_cache.l1_transfer.tp_group import SchedulerCollective
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.scheduler import ServingFixture, cache_fixture
from sglang.test.kits.slotguard.source import close_group_fixture, group_fixture
from sglang.test.kits.slotguard.wire import reference_native_hash
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=75, suite="base-a-test-cpu")


class Backend:
    load_mode = "prefetch"
    request_prefetch = False
    pools = {PoolName.KV: object()}

    def __init__(self):
        self.restorable = [2]
        self.futures = []
        self.owner = None

    def prefetch_allocation_owner(self):
        return self.owner

    def lookup(self, rid, transfers):
        return self.restorable

    def submit_lookup(self, rid, transfers):
        return UnifiedCacheLinker.submit_lookup(self, rid, transfers)

    def prefetch(self, transfers):
        future = Future()
        # Cancelling a live request must not mark a device writer completed.
        future.set_running_or_notify_cancel()
        self.futures.append(future)
        return future

    def offload(self, transfers, session_id=None):
        return False

    def reset(self):
        if any(not future.done() for future in self.futures):
            raise RuntimeError("writer still running")


class SourceFence:
    """Control-plane stub only; allocation and page accounting are real."""

    def __init__(self, guard):
        self.guard, self.generation = guard, 0

    def owner_snapshot(self):
        return NS(generation=self.generation)

    def fence_pins(self, generation):
        self.generation = generation

    def prevalidate_unfence_pins(self, generation):
        assert generation == self.generation

    def unfence_pins(self, generation):
        return generation == self.generation


def setup_cache(cache, group, *, headroom=0, backend=None):
    owner = SlotGuardPrefetchOwner(cache, group, headroom_tokens=lambda: headroom)
    backend = Backend() if backend is None else backend
    backend.owner = owner
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    cache.linker = wrapper
    return owner, backend, wrapper


def start_prefetch(wrapper, req):
    """Submit the lookup, then take the cache-event tick that allocates and loads."""
    wrapper.prefetch(req)
    wrapper.check_prefetch_events()


def request(cache, *, tokens=128, attempt=0):
    req = NS(
        rid="restore",
        cache_request_handle=CacheRequestHandle("restore", attempt),
        extra_key=None,
        cache_salt=None,
        positional_embed_overrides=None,
        full_untruncated_fill_ids=array("q", range(tokens)),
        lock=None,
        # Fields the wrapper reads to report request features to the backend.
        session=None,
        multimodal_inputs=None,
        input_embeds=None,
        is_dllm=lambda: False,
        beam_group=None,
        return_logprob=False,
        kv=NS(holds_kv=False),
    )

    def match(*args, **kwargs):
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.full_untruncated_fill_ids))
        )
        req.last_node = result.last_device_node
        req.prefix_len = result.device_prefix_len

    req.init_next_round_input = match
    req._compute_max_prefix_len = lambda length: length
    match()
    return req


def insert_computed(cache, group, tokens):
    def body(_decision):
        alloc = cache.token_to_kv_pool_allocator
        indices = alloc.alloc(tokens)
        span = alloc.last_allocation_spans[0]
        return cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(tokens))),
                value=indices,
                page_span=span,
                inserted_len=tokens,
            )
        )

    return group.execute("test-computed-prefix", body).value


class ExternalPrefetchTest(CustomTestCase):
    def setUp(self):
        self.hash_patch = patch(
            "sglang.srt.mem_cache.utils.get_native_hash", reference_native_hash
        )
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.cache, self.alloc = cache_fixture()
        self.cache._all_reduce_attn_groups = lambda tensor, op: None
        self.group = SlotGuardAllocationGroup(
            SchedulerCollective(lambda value: [value] * 4, 0),
            self.alloc,
            SourceFence(self.alloc.slot_guard),
            evictable_tokens=self.cache.evictable_size,
            scope="prefetch-test",
        )
        self.owner, self.backend, self.wrapper = setup_cache(self.cache, self.group)

    def test_publish_preserves_local_prefix_and_tail_page_identity(self):
        insert_computed(self.cache, self.group, 64)
        original = self.cache.path_cpu_page_span(request(self.cache).last_node)
        req = request(self.cache, tokens=192)
        start_prefetch(self.wrapper, req)
        state = self.wrapper.prefetches[req.cache_request_handle]
        tail = state.component_transfers[0][1].page_span
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        req.init_next_round_input()
        actual = self.cache.path_cpu_page_span(req.last_node)
        self.assertEqual(actual.page_ids, original.page_ids + tail.page_ids)
        self.assertEqual(actual.token_length, 192)
        self.assertEqual(self.alloc.verify_byte_accounting(), [])

    def test_publication_hold_survives_rejected_attempt_until_real_lock(self):
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        self.cache.pop_prefetch_loaded_span(req.cache_request_handle)
        self.cache.evict(EvictParams(num_tokens=128))
        req.init_next_round_input()
        self.assertEqual(req.prefix_len, 128)
        with self.assertRaises(RuntimeError):
            self.wrapper.adopt_prefetch(req)
        req.lock = self.cache.lock(req.last_node)
        self.wrapper.adopt_prefetch(req)
        self.assertFalse(self.wrapper.prefetch_holds)
        self.cache.unlock(req.lock)
        self.group.execute(
            "test-evict", lambda _: self.cache.evict(EvictParams(num_tokens=128))
        )
        self.assertEqual(self.alloc.conservation_counts(), (0, 32, 0, 32))

    def test_cancelled_running_read_keeps_slots_until_drained(self):
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        self.wrapper.release_request(req.cache_request_handle)
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.alloc.available_size(), 30 * 64)
        self.assertIn(req.cache_request_handle, self.wrapper.prefetches)
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        self.assertFalse(self.wrapper.prefetches)
        self.assertFalse(self.wrapper.prefetch_holds)
        self.assertEqual(self.alloc.conservation_counts(), (0, 32, 0, 32))

    def test_replaced_attempt_releases_its_old_publication_hold(self):
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        self.assertIn(req.cache_request_handle, self.wrapper.prefetch_holds)
        replacement = request(self.cache, attempt=1)
        start_prefetch(self.wrapper, replacement)
        self.assertNotIn(req.cache_request_handle, self.wrapper.prefetch_holds)
        self.assertEqual(self.cache.protected_size(), 0)

    def test_common_headroom_refusal_does_not_launch_or_evict(self):
        self.owner.headroom_tokens = lambda: 31 * 64
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        self.assertFalse(self.backend.futures)
        # The common refusal discards the entry; the next tick finishes it.
        self.wrapper.check_prefetch_events()
        self.assertFalse(self.wrapper.prefetches)
        self.assertEqual(self.alloc.conservation_counts(), (0, 32, 0, 32))
        self.assertFalse(self.group.faulted)

    def test_duplicate_tree_insert_reclaims_only_the_fetched_tail(self):
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        insert_computed(self.cache, self.group, 128)
        canonical = self.cache.path_cpu_page_span(request(self.cache).last_node)
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.alloc.conservation_counts(), (2, 30, 0, 32))
        self.assertEqual(
            self.cache.path_cpu_page_span(request(self.cache).last_node), canonical
        )
        self.assertEqual(self.alloc.verify_byte_accounting(), [])

    def test_publication_inside_batch_epoch_does_not_start_nested_epoch(self):
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        self.backend.futures[-1].set_result(True)
        epoch = self.group.status().epoch
        self.group.execute(
            "test-batch-selection", lambda _: self.wrapper.check_prefetch_events()
        )
        self.assertEqual(self.group.status().epoch, epoch + 1)
        self.assertEqual(request(self.cache).prefix_len, 128)

    def test_reset_cannot_lose_the_journal_of_an_undrained_read(self):
        req = request(self.cache)
        start_prefetch(self.wrapper, req)
        with self.assertRaises(RuntimeError):
            self.wrapper.reset()
        self.assertIn(req.cache_request_handle, self.wrapper.prefetches)
        self.assertEqual(self.alloc.available_size(), 30 * 64)
        self.backend.futures[-1].set_result(False)
        self.wrapper.check_prefetch_events()
        self.wrapper.reset()
        self.assertEqual(self.alloc.conservation_counts(), (0, 32, 0, 32))

    def test_guarded_pool_cannot_bypass_the_allocation_owner(self):
        self.cache.linker = wrapper = UnifiedCacheLinkerWrapper(self.cache, Backend())
        with self.assertRaisesRegex(RuntimeError, "no allocation owner"):
            start_prefetch(wrapper, request(self.cache))


class SourceRuntimeTest(CustomTestCase):
    def setUp(self):
        self.fixture = ServingFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.close)

    def test_source_can_be_constructed_without_a_target_backend_or_admission(self):
        fixture = self.fixture
        fixture.owner._initialize_slotguard_source(
            fixture.ctx, fixture.cache, fixture.config, NS(device="cpu")
        )
        runtime = fixture.owner.source_runtime
        self.assertIs(runtime.slot_guard, fixture.allocator.slot_guard)
        self.assertIs(runtime.source_service.guard, fixture.allocator.slot_guard)
        self.assertFalse(hasattr(runtime, "admission"))
        self.assertFalse(hasattr(runtime, "backend"))
        self.assertEqual(runtime.slot_guard.owner_counts().credit_remaining, 0)

    def test_partial_source_construction_remains_visible_to_startup_cleanup(self):
        fixture = self.fixture
        with (
            patch(
                "sglang.srt.mem_cache.l1_transfer.slotguard_service.SlotGuardService",
                side_effect=RuntimeError("source startup rejected"),
            ),
            self.assertRaises(RuntimeError),
        ):
            fixture.owner._initialize_slotguard_source(
                fixture.ctx, fixture.cache, fixture.config, NS(device="cpu")
            )
        self.assertIs(
            fixture.owner.publication_worker,
            fixture.owner.source_runtime.publication_worker,
        )
        self.assertIs(fixture.owner.arbiter, fixture.owner.source_runtime.arbiter)


class CommonPrefetchTest(CustomTestCase):
    def test_four_rank_completion_refusal_and_allocation_failure(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="slotguard-prefetch-") as directory:
            context = mp.spawn(
                _worker, args=(str(Path(directory) / "rdzv"),), nprocs=4, join=False
            )
            deadline = time.monotonic() + 100
            try:
                while not context.join(timeout=0.2):
                    if time.monotonic() > deadline:
                        self.fail("prefetch TP4 workers exceeded deadline")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3)


def _worker(rank, rendezvous):
    import torch.distributed as dist

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TreeSlot
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    group = dist.new_group([0, 1, 2, 3], backend="gloo")
    try:
        state = group_fixture(group, pages=16)
        cache = UnifiedRadixCache(
            CacheInitParams(
                False,
                ReqToTokenPool(4, 4096, "cpu", False),
                state.alloc,
                64,
                tree_slots=(TreeSlot(component="FULL", family=TREE_TOKEN),),
                tree_core_backend="python",
            )
        )
        cache._all_reduce_attn_groups = lambda tensor, op: dist.all_reduce(
            tensor, op, group=group
        )
        state.controller._evictable_tokens = cache.evictable_size
        owner, backend, wrapper = setup_cache(cache, state.controller)
        with patch("sglang.srt.mem_cache.utils.get_native_hash", reference_native_hash):
            req = request(cache)
            start_prefetch(wrapper, req)
            if rank == 0:
                backend.futures[-1].set_result(True)
            wrapper.check_prefetch_events()
            assert (
                len(wrapper.prefetches) == 1 and state.alloc.available_size() == 14 * 64
            )
            if rank != 0:
                backend.futures[-1].set_result(True)
            state.controller.execute(
                "test-complete-in-batch", lambda _: wrapper.check_prefetch_events()
            )
            assert not wrapper.prefetches
            wrapper.release_request(req.cache_request_handle)

            owner.headroom_tokens = lambda: 16 * 64
            denied = request(cache, tokens=256, attempt=1)
            before = len(backend.futures)
            start_prefetch(wrapper, denied)
            wrapper.check_prefetch_events()
            assert len(backend.futures) == before
            assert not wrapper.prefetches and not state.controller.faulted

            owner.headroom_tokens = lambda: 0
            failed = request(cache, tokens=256, attempt=2)
            try:
                with (
                    patch.object(state.alloc, "alloc", return_value=None)
                    if rank == 0
                    else nullcontext()
                ):
                    start_prefetch(wrapper, failed)
                raise AssertionError("rank-local allocation failure was accepted")
            except AllocationEpochError:
                assert state.controller.faulted
        close_group_fixture(state)
    finally:
        dist.destroy_process_group()


class AsyncBackend(Backend):
    request_prefetch = True

    def __init__(self):
        super().__init__()
        self.lookups = {}
        self.cancelled = []
        self.released = []
        self.loads = []

    def submit_prefetch_lookup(self, context, transfers):
        future = Future()
        future.set_running_or_notify_cancel()
        self.lookups[context.request] = (context, future)
        return future

    def prefetch_request(self, request, transfers):
        self.loads.append(request)
        return self.prefetch(transfers)

    def cancel_prefetch_request(self, request):
        self.cancelled.append(request)

    def release_prefetch_request(self, request):
        self.released.append(request)
        self.lookups.pop(request)

    def poll_prefetch(self):
        pass

    def reset(self):
        super().reset()
        if any(not f.done() for _, f in self.lookups.values()):
            raise RuntimeError("lookup running")


class AsyncPrefetchTest(CustomTestCase):
    def setUp(self):
        ExternalPrefetchTest.setUp(self)
        self.backend = AsyncBackend()
        self.backend.owner = self.owner
        # request_prefetch is read when the wrapper is built.
        self.wrapper = UnifiedCacheLinkerWrapper(self.cache, self.backend)
        self.cache.linker = self.wrapper

    def test_lookup_is_nonblocking_and_allocates_only_after_result(self):
        req = request(self.cache)
        before = self.alloc.available_size()
        self.wrapper.prefetch(req)
        self.assertEqual(self.alloc.available_size(), before)
        self.wrapper.check_prefetch_events()
        self.assertFalse(self.backend.loads)
        ctx, future = self.backend.lookups[req.cache_request_handle]
        self.assertEqual(ctx.token_ids, req.full_untruncated_fill_ids.tobytes())
        future.set_result([1, 2])
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.backend.loads, [req.cache_request_handle])
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        req.init_next_round_input()
        self.assertEqual(req.prefix_len, 128)
        self.assertEqual(self.backend.released, [req.cache_request_handle])
        self.assertEqual(self.alloc.verify_byte_accounting(), [])

    def test_lookup_completion_inside_batch_epoch_defers_allocation(self):
        req = request(self.cache)
        self.wrapper.prefetch(req)
        self.backend.lookups[req.cache_request_handle][1].set_result([2])
        self.group.execute(
            "batch-selection", lambda _: self.wrapper.check_prefetch_events()
        )
        self.assertFalse(self.backend.loads)
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.backend.loads, [req.cache_request_handle])
        self.backend.futures[-1].set_result(False)
        self.wrapper.check_prefetch_events()

    def test_abort_running_lookup_releases_without_allocation(self):
        req = request(self.cache)
        before = self.alloc.available_size()
        self.wrapper.prefetch(req)
        future = self.backend.lookups[req.cache_request_handle][1]
        self.wrapper.release_request(req.cache_request_handle)
        self.wrapper.check_prefetch_events()
        self.assertIn(req.cache_request_handle, self.wrapper.prefetches)
        future.set_result([2])
        self.wrapper.check_prefetch_events()
        self.assertFalse(self.wrapper.prefetches)
        self.assertFalse(self.backend.loads)
        self.assertEqual(self.alloc.available_size(), before)
        self.assertEqual(self.backend.released, [req.cache_request_handle])

    def test_replacement_does_not_pair_old_lookup_with_new_read(self):
        old, new = request(self.cache), request(self.cache, attempt=1)
        self.wrapper.prefetch(old)
        old_future = self.backend.lookups[old.cache_request_handle][1]
        self.wrapper.prefetch(new)
        old_future.set_result([2])
        self.backend.lookups[new.cache_request_handle][1].set_result([1])
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.backend.loads, [new.cache_request_handle])
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        new.init_next_round_input()
        self.assertEqual(new.prefix_len, 64)
        self.assertEqual(
            set(self.backend.released),
            {old.cache_request_handle, new.cache_request_handle},
        )

    def test_load_abort_retains_pages_until_writer_drained(self):
        req = request(self.cache)
        before = self.alloc.available_size()
        self.wrapper.prefetch(req)
        self.backend.lookups[req.cache_request_handle][1].set_result([2])
        self.wrapper.check_prefetch_events()
        self.wrapper.release_request(req.cache_request_handle)
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.alloc.available_size(), before - 128)
        self.backend.futures[-1].set_result(True)
        self.wrapper.check_prefetch_events()
        self.assertEqual(self.alloc.available_size(), before)
        req.init_next_round_input()
        self.assertEqual(req.prefix_len, 0)

    def test_reset_does_not_erase_running_lookup(self):
        req = request(self.cache)
        self.wrapper.prefetch(req)
        with self.assertRaises(RuntimeError):
            self.wrapper.reset()
        self.assertIn(req.cache_request_handle, self.wrapper.prefetches)
        self.backend.lookups[req.cache_request_handle][1].set_result([])
        self.wrapper.reset()
        self.assertFalse(self.wrapper.prefetches)
        self.assertEqual(self.backend.released, [req.cache_request_handle])

    def test_lookup_error_and_miss_release_exact_handle(self):
        for failed in (False, True):
            req = request(self.cache, attempt=int(failed))
            self.wrapper.prefetch(req)
            future = self.backend.lookups[req.cache_request_handle][1]
            if failed:
                future.set_exception(TimeoutError())
            else:
                future.set_result([])
            self.wrapper.check_prefetch_events()
            self.assertFalse(self.wrapper.prefetches)
            self.assertIn(req.cache_request_handle, self.backend.released)
        self.assertFalse(self.backend.loads)


if __name__ == "__main__":
    unittest.main()
