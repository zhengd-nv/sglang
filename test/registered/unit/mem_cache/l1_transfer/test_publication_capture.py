"""Canonical CPU capture worker and real Python Tree publication boundaries."""

import hashlib
import struct
import threading
import time
import unittest
from array import array
from dataclasses import replace
from unittest.mock import patch

import test_slotguard_allocator as allocator_tests
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.l1_transfer.hash_pin import HashLookupResult
from sglang.srt.mem_cache.l1_transfer.metadata import event_keys_and_hashes
from sglang.srt.mem_cache.l1_transfer.protocol import (
    EngineIdentity,
    LookupQuery,
    LookupResult,
    TransferId,
)
from sglang.srt.mem_cache.l1_transfer.publication_capture import (
    CanonicalPublicationHook,
    ProducerEvents,
    PublicationCaptureWorker,
)
from sglang.srt.mem_cache.l1_transfer.slotguard_catalog import (
    PublicationCatalog,
    PublicationScope,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


def reference_native_hash(tokens, prior_digest=None, page_size=None):
    """hash_binding.cpp: SHA256(full parent digest + little-endian uint32 page)."""
    raw = tuple(tokens)
    hashes = []
    pages = (
        (raw,)
        if page_size is None
        else tuple(
            raw[start : start + page_size] for start in range(0, len(raw), page_size)
        )
    )
    for page in pages:
        # RadixKey bigram iteration yields the exact interleaved uint32 pairs.
        page = tuple(
            value
            for token in page
            for value in (token if isinstance(token, tuple) else (token,))
        )
        prior_digest = hashlib.sha256(
            (prior_digest or b"") + struct.pack(f"<{len(page)}I", *page)
        ).digest()
        hashes.append(prior_digest.hex())
    return hashes if page_size else hashes[-1]


class Ready:
    value = True

    def ready(self):
        return self.value


class CaptureTreeTest(CustomTestCase):
    def setUp(self):
        # The native extension deliberately rejects macOS; patch only its hash
        # implementation with the source-equivalent CPU reference in this test.
        patcher = patch(
            "sglang.srt.mem_cache.utils.get_native_hash", reference_native_hash
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self, **limits):
        cache, alloc, rows = allocator_tests.runtime_fixture(pages=64)
        allocator_tests.epoch(alloc)
        catalog = PublicationCatalog(alloc.slot_guard)
        worker = PublicationCaptureWorker(catalog, autostart=False, **limits)
        ready = Ready()
        hook = CanonicalPublicationHook(
            worker,
            alloc,
            PublicationScope("model", "layout", 4, 1),
            device="cpu",
            readiness_factory=lambda: ready,
        )
        hook.attach(cache)
        return cache, alloc, rows, catalog, worker, hook, ready

    def insert(self, cache, alloc, tokens):
        values = alloc.alloc(len(tokens))
        span = alloc.last_allocation_spans[0]
        result = cache.insert(
            InsertParams(key=RadixKey(array("q", tokens)), value=values, page_span=span)
        )
        return result, span

    def lookup(self, catalog, tokens):
        fields = {
            "domain_id": "fabric",
            "instance_id": "instance",
            "model_id": "model",
            "layout_digest": "layout",
            "tp_size": 4,
            "tp_rank": 0,
            "shard_id": "rank0",
        }
        target = EngineIdentity(
            "target", "target", tp_group_id="target-group", **fields
        )
        source = EngineIdentity(
            "source", "source", tp_group_id="source-group", **fields
        )
        tokens = tuple(tokens)
        keys, hashes = event_keys_and_hashes(tokens, 4)
        query = LookupQuery(
            target,
            TransferId("target", "request", 0),
            keys,
            tokens + (999,),
            0,
            len(tokens),
            4,
            prefix_hashes=hashes,
        )
        result = LookupResult(query, source, len(tokens), query.event_block_keys)
        return catalog.lookup(HashLookupResult.from_lookup(result))

    def node(self, cache, tokens):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return cache.tree_core.node_by_id(match.last_device_node)

    def test_root_is_captured_after_insert_and_prepared_only_on_worker(self):
        cache, alloc, _, catalog, worker, _, _ = self.fixture()
        _, span = self.insert(cache, alloc, range(12))
        self.assertEqual(catalog.snapshot().records, 0)
        self.assertEqual(worker.owner_snapshot().queued, 1)
        worker.progress_once()
        self.assertEqual(catalog.snapshot().records, 3)
        self.assertEqual(
            alloc.slot_guard.snapshot().published_pages, alloc.page_refs(span)
        )

    def test_split_partial_overlap_and_child_continuation_use_full_parent_sha(self):
        cache, alloc, _, catalog, worker, _, _ = self.fixture()
        _, original = self.insert(cache, alloc, range(12))
        root_content = (
            self.node(cache, range(12)).component_data[ComponentType.FULL].content_span
        )
        worker.progress_once()
        prefix = self.node(cache, range(8))
        prefix_content = prefix.component_data[ComponentType.FULL].content_span
        suffix_content = (
            next(iter(prefix.children.values()))
            .component_data[ComponentType.FULL]
            .content_span
        )
        self.assertIs(prefix_content.token_bytes, root_content.token_bytes)
        self.assertIs(suffix_content.token_bytes, root_content.token_bytes)
        self.assertEqual(suffix_content.parent_cursor, prefix_content.cursor)
        branch_tokens = (*range(8), 101, 102, 103, 104)
        _, branch = self.insert(cache, alloc, branch_tokens)
        worker.progress_once()
        self.assertEqual(
            self.lookup(catalog, branch_tokens),
            alloc.page_refs(original.slice(0, 8))
            + alloc.page_refs(branch.slice(8, 12)),
        )
        _, extended = self.insert(cache, alloc, range(16))
        # The old ROOT keys are truncated int64; continuation needs full SHA256.
        with patch(
            "sglang.srt.mem_cache.utils.get_hash_str",
            wraps=__import__(
                "sglang.srt.mem_cache.utils", fromlist=["get_hash_str"]
            ).get_hash_str,
        ) as hashes:
            worker.progress_once()
        self.assertEqual(
            hashes.call_args.args[1],
            reference_native_hash(tuple(range(12)), page_size=4)[-1],
        )
        self.assertEqual(
            self.lookup(catalog, range(16)),
            alloc.page_refs(original) + alloc.page_refs(extended.slice(12, 16)),
        )

    def test_dedup_retries_survivors_and_never_republishes_private_duplicate(self):
        cache, alloc, _, catalog, worker, _, _ = self.fixture()
        _, canonical = self.insert(cache, alloc, range(12))
        worker.progress_once()
        records = catalog.snapshot().records
        _, duplicate = self.insert(cache, alloc, range(12))
        self.assertNotEqual(duplicate, canonical)
        worker.progress_once()
        self.assertEqual(catalog.snapshot().records, records)
        self.assertEqual(self.lookup(catalog, range(12)), alloc.page_refs(canonical))
        self.assertTrue(
            set(duplicate.page_ids).isdisjoint(
                ref.page_id for ref in alloc.slot_guard.snapshot().published_pages
            )
        )

    def test_four_ranks_with_recorders_disabled_all_capture_canonical_pages(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                cache, alloc, _, catalog, worker, _, _ = self.fixture()
                cache.tree_core.kv_events.enabled = False
                with patch.object(
                    cache.tree_core.kv_events,
                    "record_store",
                    wraps=cache.tree_core.kv_events.record_store,
                ) as recorder:
                    _, span = self.insert(cache, alloc, range(8))
                # Even an explicitly disabled/no-op recorder cannot suppress capture.
                self.assertTrue(recorder.called)
                worker.progress_once()
                self.assertEqual(self.lookup(catalog, range(8)), alloc.page_refs(span))

    def test_owner_does_no_publication_construction_hash_or_device_readback(self):
        cache, alloc, _, _, worker, _, _ = self.fixture()
        with (
            patch(
                "sglang.srt.mem_cache.l1_transfer.publication_capture.Publication",
                side_effect=AssertionError("owner Publication"),
            ),
            patch(
                "sglang.srt.mem_cache.utils.get_hash_str",
                side_effect=AssertionError("owner hash"),
            ),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("owner D2H")),
        ):
            self.insert(cache, alloc, range(12))
        self.assertEqual(worker.owner_snapshot().queued_entries, 1)
        worker.progress_once()

    def test_prefetch_private_ready_is_inert_then_checkpoint_publishes_survivors(self):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        cache, alloc, rows, catalog, worker, _, _ = self.fixture()
        canonical = None
        for attempt in range(2):
            req = Req(
                str(attempt),
                "x",
                array("q", range(8)),
                SamplingParams(max_new_tokens=4),
            )
            req.full_untruncated_fill_ids = array("q", range(12))
            req.last_node = cache.root_node_handle()
            req.kv.req_pool_idx = rows.alloc_rows(1)[0]
            req.kv.kv_allocated_len = req.kv.kv_committed_len = 12
            values = alloc.alloc(12)
            span = alloc.last_allocation_spans[0]
            rows.write(
                (req.kv.req_pool_idx, slice(0, 12)),
                values,
                cpu_page_writes=((req.kv.req_pool_idx, 0, span),),
            )
            # Allocated/adopted/READY private KV has not entered canonical Tree.
            self.assertEqual(worker.owner_snapshot().queued, 0)
            before = catalog.snapshot().records
            with patch.object(cache, "insert", wraps=cache.insert) as inserts:
                cache.checkpoint_into_tree(req, up_to=12)
            self.assertEqual(inserts.call_count, 2)
            worker.progress_once()
            adopted = cache.path_cpu_page_span(req.last_node)
            if canonical is None:
                canonical = adopted
            self.assertEqual(adopted, canonical)
            self.assertEqual(
                self.lookup(catalog, range(12)), alloc.page_refs(canonical)
            )
            if attempt:
                self.assertEqual(catalog.snapshot().records, before)
            cache.unlock(req.lock)
            rows.free_rows([req.kv.req_pool_idx])

    def test_insert_action_failure_aborts_all_capture_and_retry_can_recover(self):
        cache, alloc, _, catalog, worker, hook, _ = self.fixture()
        original_apply = cache._apply_cache_actions
        with (
            patch.object(
                cache, "_apply_cache_actions", side_effect=RuntimeError("action failed")
            ),
            self.assertRaises(RuntimeError),
        ):
            self.insert(cache, alloc, range(8))
        self.assertIsNone(hook._staged)
        self.assertEqual(worker.owner_snapshot().queued, 0)
        self.assertEqual(catalog.snapshot().records, 0)
        self.assertEqual(cache._apply_cache_actions, original_apply)
        self.insert(cache, alloc, range(8))
        worker.progress_once()
        self.assertIsNotNone(self.lookup(catalog, range(8)))

    def test_queue_full_safe_miss_requires_owner_reencounter(self):
        cache, alloc, _, catalog, worker, _, _ = self.fixture(capacity=1)
        self.insert(cache, alloc, range(8))
        self.insert(cache, alloc, range(12))
        self.assertEqual(worker.owner_snapshot().status, "QUEUE_FULL")
        self.assertEqual(worker.owner_snapshot().queued, 1)
        worker.progress_once()
        self.assertIsNone(self.lookup(catalog, range(12)))
        # No worker Tree read/retry: only a later canonical owner insert retries.
        worker.progress_once()
        self.assertIsNone(self.lookup(catalog, range(12)))
        self.insert(cache, alloc, range(12))
        worker.progress_once()
        self.assertIsNotNone(self.lookup(catalog, range(12)))

    def test_pending_ready_free_and_aba_never_revive_old_generation(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                cache, alloc, _, catalog, worker, _, ready = self.fixture()
                ready.value = False
                _, old = self.insert(cache, alloc, range(8))
                old_refs = alloc.page_refs(old)
                worker.progress_once()
                self.assertEqual(worker.owner_snapshot().pending, 1)
                self.assertIsNone(self.lookup(catalog, range(8)))
                with self.assertRaises(RuntimeError):
                    worker.close()
                cache.evict(EvictParams(num_tokens=8))
                if reuse:
                    # Reuse exactly these physical IDs in a later common epoch.
                    alloc.finish_allocation_epoch(1)
                    allocator_tests.epoch(alloc, 2)
                    self.assertIsNotNone(alloc.alloc(64 * 4))
                    newer = alloc.page_refs(alloc.last_allocation_spans[0])
                    self.assertTrue(
                        set(old.page_ids).issubset(ref.page_id for ref in newer)
                    )
                    self.assertTrue(set(old_refs).isdisjoint(newer))
                ready.value = True
                worker.progress_once()
                self.assertEqual(worker.owner_snapshot().pending, 0)
                self.assertIsNone(self.lookup(catalog, range(8)))
                worker.close()

    def test_query_error_is_uncertain_owner_and_refuses_close(self):
        class Uncertain:
            def ready(self):
                raise RuntimeError("query failed")

        cache, alloc, _, catalog, worker, hook, _ = self.fixture()
        hook._readiness_factory = Uncertain
        self.insert(cache, alloc, range(8))
        worker.progress_once()
        owners = worker.owner_snapshot()
        self.assertEqual((owners.pending, owners.uncertain), (1, 1))
        self.assertEqual(owners.pool_epoch, 1)
        self.assertIsNone(self.lookup(catalog, range(8)))
        with self.assertRaises(RuntimeError):
            worker.close()
        self.assertFalse(worker.owner_snapshot().closed)

    def test_missing_parent_and_bounded_context_receipts_are_safe_misses(self):
        charge = PublicationCaptureWorker.RECEIPT_BYTES
        cache, alloc, _, catalog, worker, _, _ = self.fixture(
            max_context_entries=1, max_context_bytes=charge
        )
        self.insert(cache, alloc, range(12))
        worker.progress_once()
        # Splitting inside an old chunk needs its explicit internal-page receipt.
        self.node(cache, range(4))
        branch = (*range(4), 201, 202, 203, 204)
        self.insert(cache, alloc, branch)
        worker.progress_once()
        owners = worker.owner_snapshot()
        self.assertLessEqual(owners.context_entries, 1)
        self.assertLessEqual(owners.context_bytes, charge)
        # A root-prefix retry is allowed to recover its receipt, then the child
        # can continue; if the tiny map loses it first, the outcome stays MISS.
        found = self.lookup(catalog, branch)
        self.assertTrue(found is None or len(found) == 2)

    def test_namespace_and_incomplete_page_never_publish(self):
        cache, alloc, _, catalog, worker, _, _ = self.fixture()
        values = alloc.alloc(8)
        span = alloc.last_allocation_spans[0]
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(8)), cache_salt="salt"),
                value=values,
                page_span=span,
            )
        )
        worker.progress_once()
        self.assertEqual(catalog.snapshot().records, 0)
        self.assertEqual(worker.owner_snapshot().status, "UNSUPPORTED_CONTENT")
        values = alloc.alloc(8)
        partial = alloc.last_allocation_spans[0].slice(0, 7)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(7))), value=values[:7], page_span=partial
            )
        )
        worker.progress_once()
        self.assertIsNotNone(self.lookup(catalog, range(4)))
        self.assertEqual(catalog.snapshot().records, 1)

    def test_resurrection_rebinds_same_content_cursor_to_current_generation(self):
        cache, alloc, _, catalog, worker, _, _ = self.fixture()
        _, old_span = self.insert(cache, alloc, range(12))
        worker.progress_once()
        node = self.node(cache, range(12))
        cd = node.component_data[ComponentType.FULL]
        original_content = cd.content_span
        old_refs = alloc.page_refs(old_span)
        # A synthetic host backing permits the real Tree demotion/resurrection
        # path on CPU. This is not enabling HiCache support for this hook.
        cd.host_value = torch.arange(12)
        cache.evict(EvictParams(num_tokens=12))
        self.assertIsNone(cd.value)
        alloc.finish_allocation_epoch(1)
        allocator_tests.epoch(alloc, 2)
        all_values = alloc.alloc(64 * 4)
        all_span = alloc.last_allocation_spans[0]
        start = all_span.page_ids.index(old_span.page_ids[0]) * 4
        reused = all_span.slice(start, start + 12)
        self.assertEqual(reused.page_ids, old_span.page_ids)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(12))),
                value=all_values[start : start + 12],
                page_span=reused,
            )
        )
        self.assertIs(cd.content_span.context, original_content.context)
        self.assertIs(cd.content_span.token_bytes, original_content.token_bytes)
        current = alloc.page_refs(reused)
        self.assertNotEqual(current, old_refs)
        worker.progress_once()
        self.assertEqual(self.lookup(catalog, range(12)), current)

    def test_missing_internal_parent_has_explicit_miss_without_tree_reads(self):
        cache, alloc, _, catalog, worker, hook, ready = self.fixture(
            max_context_entries=1
        )
        self.insert(cache, alloc, range(12))
        worker.progress_once()
        prefix = self.node(cache, range(4))
        suffix = (
            next(iter(prefix.children.values()))
            .component_data[ComponentType.FULL]
            .content_span
        )
        worker.enqueue(hook.scope, suffix, alloc.page_refs(suffix.page_span), ready)
        worker.progress_once()
        self.assertEqual(worker.owner_snapshot().status, "MISSING_PARENT")
        self.assertEqual(worker.owner_snapshot().context_entries, 1)
        self.assertEqual(catalog.snapshot().records, 3)

    def test_capture_backing_bytes_and_staging_entries_are_hard_bounded(self):
        cache, alloc, _, _, worker, hook, _ = self.fixture(
            max_capture_bytes=120, max_chunk_bytes=120
        )
        self.insert(cache, alloc, range(8))
        self.assertEqual(worker.owner_snapshot().queued_bytes, 112)
        self.insert(cache, alloc, range(12))
        self.assertGreater(worker.owner_snapshot().dropped, 0)
        self.assertLessEqual(worker.owner_snapshot().queued_bytes, 120)
        worker.progress_once()
        # A suffix view retains the full original chunk backing for accounting.
        parent = self.node(cache, range(4))
        suffix = (
            next(iter(parent.children.values()))
            .component_data[ComponentType.FULL]
            .content_span
        )
        self.assertEqual(len(suffix.token_bytes), 64)
        self.assertEqual(suffix.length, 4)
        self.assertFalse(
            worker.enqueue(
                hook.scope,
                replace(suffix, token_bytes=b"x" * 128),
                alloc.page_refs(suffix.page_span),
                Ready(),
            )
        )
        self.assertEqual(worker.owner_snapshot().queued_bytes, 0)

    def test_default_tree_is_inert_and_attach_rejects_nonempty_or_wrong_guard(self):
        cache, alloc, _ = allocator_tests.runtime_fixture()
        allocator_tests.epoch(alloc)
        self.insert(cache, alloc, range(8))
        self.assertFalse(hasattr(cache, "publication_hook"))
        node = self.node(cache, range(8))
        self.assertIsNone(node.component_data[ComponentType.FULL].content_span)
        catalog = PublicationCatalog(alloc.slot_guard)
        worker = PublicationCaptureWorker(catalog, autostart=False)
        hook = CanonicalPublicationHook(
            worker, alloc, PublicationScope("model", "layout", 4, 1), device="cpu"
        )
        with self.assertRaises(ValueError):
            hook.attach(cache)
        other = allocator_tests.allocator()
        with self.assertRaises(ValueError):
            CanonicalPublicationHook(worker, other, hook.scope, device="cpu")
        with self.assertRaises(ValueError):
            CanonicalPublicationHook(worker, alloc, hook.scope, device="cuda:0")
        with self.assertRaises(ValueError):
            CanonicalPublicationHook(
                worker, alloc, replace(hook.scope, page_size=64), device="cpu"
            )

    def test_cuda_cannot_use_cpu_readiness_factory_to_bypass_real_certificate(self):
        for forward, factory in (
            (None, ProducerEvents),
            (object(), ProducerEvents),
            (None, None),
        ):
            with self.subTest(
                forward_present=forward is not None, custom_factory=factory is not None
            ):
                cache, alloc, _ = allocator_tests.runtime_fixture()
                allocator_tests.epoch(alloc)
                alloc.alloc(8)  # Allocate on the real CPU pool before declaration.
                before = alloc.slot_guard.owner_counts()
                alloc.device = "cuda:0"  # Hardwareless reproduction of CUDA branch.
                catalog = PublicationCatalog(alloc.slot_guard)
                worker = PublicationCaptureWorker(catalog, autostart=False)
                scope = PublicationScope("model", "layout", 4, 1)
                with (
                    patch.object(
                        ProducerEvents,
                        "record",
                        side_effect=AssertionError("real CUDA certificate required"),
                    ) as record,
                    self.assertRaises(ValueError),
                ):
                    CanonicalPublicationHook(
                        worker,
                        alloc,
                        scope,
                        device="cuda:0",
                        forward_stream=forward,
                        readiness_factory=factory,
                    ).attach(cache)
                worker.progress_once()
                record.assert_not_called()
                self.assertEqual(alloc.slot_guard.owner_counts(), before)
                self.assertEqual(alloc.slot_guard.snapshot().published_pages, ())
                self.assertEqual(catalog.snapshot().records, 0)
                self.assertEqual(worker.owner_snapshot().queued, 0)
                self.assertFalse(cache.tree_core.root_node.children)
                worker.close()

    def test_cuda_tree_commit_records_actual_streams_and_retains_unknown_query(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                cache, alloc, _ = allocator_tests.runtime_fixture()
                allocator_tests.epoch(alloc)
                values = alloc.alloc(8)
                span = alloc.last_allocation_spans[0]
                alloc.device = "cuda:0"  # No CUDA allocation or tensor in this test.
                catalog = PublicationCatalog(alloc.slot_guard)
                worker = PublicationCaptureWorker(catalog, autostart=False)
                scope = PublicationScope("model", "layout", 4, 1)
                forward, current = object(), object()
                events = []

                class Event:
                    def __init__(self, output=events):
                        self.stream, self.done, self.unknown = None, False, False
                        self.queries = 0
                        output.append(self)

                    def record(self, stream):
                        self.stream = stream

                    def query(self):
                        self.queries += 1
                        if self.unknown:
                            raise RuntimeError("unknown CUDA event")
                        return self.done

                    def synchronize(self):
                        raise AssertionError("must only query CUDA events")

                hook = CanonicalPublicationHook(
                    worker, alloc, scope, device="cuda:0", forward_stream=forward
                )
                hook.attach(cache)
                with (
                    patch.object(torch.cuda, "Event", Event),
                    patch.object(torch.cuda, "current_stream", return_value=current),
                    patch.object(
                        ProducerEvents, "record", wraps=ProducerEvents.record
                    ) as record,
                    patch.object(
                        hook,
                        "_readiness_factory",
                        side_effect=AssertionError("CUDA cannot consume CPU factory"),
                    ) as cpu_factory,
                ):
                    cache.insert(
                        InsertParams(
                            key=RadixKey(array("q", range(8))),
                            value=values,
                            page_span=span,
                        )
                    )
                record.assert_called_once_with(forward, "cuda:0")
                cpu_factory.assert_not_called()
                self.assertEqual(
                    tuple(event.stream for event in events), (forward, current)
                )
                worker.progress_once()
                self.assertEqual(
                    (catalog.snapshot().pending, catalog.snapshot().uncertain), (1, 0)
                )
                self.assertEqual(tuple(event.queries for event in events), (1, 1))
                self.assertEqual(alloc.slot_guard.snapshot().published_pages, ())
                for event in events:
                    event.done = True
                events[-1].unknown = unknown
                worker.progress_once()
                if unknown:
                    self.assertEqual(
                        (catalog.snapshot().pending, catalog.snapshot().uncertain),
                        (1, 1),
                    )
                    self.assertEqual(alloc.slot_guard.snapshot().published_pages, ())
                    with self.assertRaises(RuntimeError):
                        worker.close()
                else:
                    self.assertEqual(
                        self.lookup(catalog, range(8)), alloc.page_refs(span)
                    )
                    worker.close()

    def test_epoch_change_requires_sealed_quiescent_rebind_before_next_insert(self):
        cache, alloc, _, _, worker, hook, _ = self.fixture()
        alloc.finish_allocation_epoch(1)
        alloc.clear()
        self.assertEqual(alloc.slot_guard.pool_epoch, 2)
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            self.insert(cache, alloc, range(8))
        self.assertIsNone(hook._staged)
        self.assertEqual(worker.owner_snapshot().queued, 0)

    def test_abort_if_final_action_barrier_fails_after_result(self):
        cache, alloc, _, catalog, worker, hook, _ = self.fixture()
        original = cache._apply_cache_actions
        calls = 0

        def apply(actions):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("final barrier failed")
            original(actions)

        with (
            patch.object(cache, "_apply_cache_actions", apply),
            self.assertRaises(RuntimeError),
        ):
            self.insert(cache, alloc, range(8))
        self.assertEqual(calls, 2)
        self.assertIsNone(hook._staged)
        self.assertEqual(
            (worker.owner_snapshot().queued, catalog.snapshot().records), (0, 0)
        )

    def test_close_refuses_queued_preparing_and_active_owner_transaction(self):
        cache, alloc, _, _, worker, hook, _ = self.fixture()
        hook.begin()
        self.assertTrue(hook.owner_snapshot().active)
        self.assertEqual(hook.owner_snapshot().staged_entries, 0)
        with self.assertRaises(RuntimeError):
            hook.close()
        hook.abort()
        self.insert(cache, alloc, range(8))
        with self.assertRaises(RuntimeError):
            worker.close()
        entered, release = threading.Event(), threading.Event()
        original = worker._prepare

        def prepare(capture):
            entered.set()
            release.wait(2)
            original(capture)

        with patch.object(worker, "_prepare", prepare):
            thread = threading.Thread(target=worker.progress_once)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                owners = worker.owner_snapshot()
                self.assertEqual((owners.preparing, owners.preparing_entries), (1, 1))
                self.assertGreater(owners.preparing_bytes, 0)
                with self.assertRaises(RuntimeError):
                    worker.close()
            finally:
                release.set()
                thread.join(2)
        worker.close()
        self.assertTrue(worker.owner_snapshot().closed)

    def test_worker_progresses_pending_producer_without_scheduler_ticks(self):
        cache, alloc, _, catalog, unused, hook, ready = self.fixture()
        worker = PublicationCaptureWorker(catalog, poll_interval_s=0.002)
        hook.worker = worker
        ready.value = False
        try:
            _, span = self.insert(cache, alloc, range(8))
            deadline = time.monotonic() + 2
            while worker.owner_snapshot().pending != 1 and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertEqual(worker.owner_snapshot().pending, 1)
            ready.value = True
            deadline = time.monotonic() + 2
            while (
                self.lookup(catalog, range(8)) is None and time.monotonic() < deadline
            ):
                time.sleep(0.002)
            self.assertEqual(self.lookup(catalog, range(8)), alloc.page_refs(span))
            deadline = time.monotonic() + 2
            while worker.owner_snapshot().progressing and time.monotonic() < deadline:
                time.sleep(0.002)
            worker.close()
        finally:
            unused.close()


class ProducerEventTest(CustomTestCase):
    def test_nonboolean_or_failed_event_queries_are_uncertain(self):
        class Query:
            def query(self):
                return object()

        with self.assertRaises(RuntimeError):
            ProducerEvents((Query(),)).ready()

    def test_actual_forward_and_current_streams_are_recorded_and_only_queried(self):
        events = []

        class Event:
            def __init__(self):
                self.stream = None
                self.done = False
                events.append(self)

            def record(self, stream):
                self.stream = stream

            def query(self):
                return self.done

            def synchronize(self):
                raise AssertionError("producer readiness must never synchronize")

        forward, current = object(), object()
        with (
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "current_stream", return_value=current),
        ):
            certificate = ProducerEvents.record(forward, "cuda:0")
        self.assertEqual(tuple(event.stream for event in events), (forward, current))
        self.assertFalse(certificate.ready())
        for event in events:
            event.done = True
        self.assertTrue(certificate.ready())

    def test_partial_event_record_failure_is_not_ready(self):
        with (
            patch.object(torch.cuda, "Event", side_effect=RuntimeError("driver")),
            patch.object(torch.cuda, "current_stream", return_value=object()),
        ):
            certificate = ProducerEvents.record(object(), "cuda:0")
        self.assertTrue(certificate.uncertain)
        with self.assertRaises(RuntimeError):
            certificate.ready()


if __name__ == "__main__":
    unittest.main()
