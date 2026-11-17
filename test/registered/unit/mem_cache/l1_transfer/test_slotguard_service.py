"""Independent workers with the real guard, wire codec and fake CPU driver."""

import gc
import importlib
import threading
import unittest
import weakref
from concurrent.futures import TimeoutError
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from test_nvlink_backend import pool
from test_protocol import PACKAGE, identity, p, register_cpu_ci
from test_sglang_adapter import BorrowDriver, SGLangPoolAdapter
from test_slotguard_catalog import (
    Ready,
    c,
    guard,
    hashes,
    hp,
    keys,
    lease,
    s,
    seed_credit,
)

from sglang.test.kits.l1_transfer_cpu import NvlinkEngine
from sglang.test.test_utils import CustomTestCase

v = importlib.import_module(f"{PACKAGE}.slotguard_protocol")
a = importlib.import_module(f"{PACKAGE}.transport_role_arbiter")
svc = importlib.import_module(f"{PACKAGE}.slotguard_service")
codec = importlib.import_module(f"{PACKAGE}.wire_codec")
w = importlib.import_module(f"{PACKAGE}.slotguard_control_wire")
layout = importlib.import_module(f"{PACKAGE}.transport.layout")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class ServiceTest(CustomTestCase):
    def setUp(self):
        self.g = guard(shared_headroom_credit=True)
        seed_credit(self.g, 32)
        self.refs = self.g.register_allocation((1, 2, 3))
        self.catalog = c.PublicationCatalog(self.g, event_keys=keys)
        ready = Ready()
        ready.value = True
        self.catalog.submit(
            c.Publication(
                c.PublicationScope("model-1", "layout-1", 64, 1),
                None,
                tuple(tuple(range(start, start + 64)) for start in (0, 64, 128)),
                keys(tuple(range(192)), 64),
                self.refs,
                ready,
            )
        )
        self.catalog.progress()
        self.engine = NvlinkEngine(0, _driver=BorrowDriver())
        self.adapter = SGLangPoolAdapter(pool(0), self.engine)
        self.arbiter = a.TransportRoleArbiter()
        self.service = svc.SlotGuardService(
            guard=self.g,
            catalog=self.catalog,
            local_identity=identity("source"),
            exact_peers=(identity(),),
            engine=self.engine,
            adapter=self.adapter,
            arbiter=self.arbiter,
            max_requests=2,
            max_records=8,
        )

    def tearDown(self):
        self.service.assert_quiescent()
        self.service.close()
        self.adapter.close()
        self.engine.close()

    def lookup(self, attempt=0):
        """Target discovery result; Pin carries only its compact hash proof."""
        target = identity()
        tokens = tuple(range(193))
        query = p.LookupQuery(
            target,
            p.TransferId(target.incarnation, "rid", attempt),
            keys(tokens, 64),
            tokens,
            64,
            192,
            64,
            prefix_hashes=hashes(tokens, 64),
        )
        return p.LookupResult(query, identity("source"), 192, query.event_block_keys)

    def pin(self, attempt=0):
        result = self.lookup(attempt)
        identity_ = s.LeaseIdentity(
            identity("source").incarnation,
            result.query.target.incarnation,
            result.transfer_id,
            f"nonce-{attempt}",
        )
        return v.PinReq(hp.HashLookupResult.from_lookup(result), identity_)

    def unpin(self, response):
        return v.UnpinReq(response.request.reference, response.lease_id, True, True)

    def test_all_buffers_exact_interval_replay_and_ack_loss(self):
        req = self.pin()
        response = self.service.submit(req).result(2)
        self.assertEqual(response.status, v.PinStatus.PREPARED)
        self.assertEqual(response.pages, self.refs[1:])
        self.assertEqual(
            {x.buffer_name for x in response.buffer_bindings},
            set(self.adapter.bindings),
        )
        codec.decode_export(response, self.adapter.layout, "target-owner")
        self.assertEqual(
            w.decode(w.encode(response), v.PinPreparedResp, request_context=req),
            response,
        )
        self.assertEqual(self.service.submit(req).result(2), response)
        self.assertEqual(self.service.owner_snapshot().leases, 1)
        self.assertEqual(
            self.service.submit(self.unpin(response)).result(2).status,
            v.UnpinStatus.RELEASED,
        )
        self.assertEqual(
            self.service.submit(self.unpin(response)).result(2).status,
            v.UnpinStatus.ALREADY_RELEASED,
        )
        self.assertEqual(self.service.submit(req).result(2).status, v.PinStatus.MISS)
        self.assertFalse(self.engine._exports)

    def test_partial_export_cleanup_failure_quarantines_then_exact_unpin_retries(self):
        original_export = self.engine.export_memory
        original_release = self.engine.release_export
        calls = []

        def export(memory):
            if calls:
                raise RuntimeError("export failure")
            descriptor = original_export(memory)
            calls.append(descriptor)
            return descriptor

        self.engine.export_memory = export
        self.engine.release_export = lambda _: (_ for _ in ()).throw(
            RuntimeError("unknown cleanup")
        )
        req = self.pin()
        response = self.service.submit(req).result(2)
        self.assertEqual(response.status, v.PinStatus.BUSY)
        self.assertEqual(self.g.owner_counts().active_leases, 1)
        self.assertEqual(self.service.owner_snapshot().uncertain, 1)
        self.assertEqual(self.g.owner_counts().credit_remaining, 30)
        self.assertFalse(self.arbiter.acquire(a.TransportRole.INBOUND, lease(1)))
        self.engine.export_memory = original_export
        self.engine.release_export = original_release
        # Authoritative query recovers the opaque lease identity, never exports.
        state = self.service.query(req.identity)
        ack = self.service.submit(
            v.UnpinReq(req.reference, state.lease_id, True, True)
        ).result(2)
        self.assertEqual(ack.status, v.UnpinStatus.RELEASED)
        self.assertFalse(self.engine._exports)
        self.assertEqual(self.g.owner_counts().credit_remaining, 30)

    def test_partial_export_proved_cleanup_releases_guard_and_role(self):
        original = self.engine.export_memory
        count = []

        def export(memory):
            if count:
                raise RuntimeError("export failure")
            count.append(1)
            return original(memory)

        self.engine.export_memory = export
        self.assertEqual(
            self.service.submit(self.pin()).result(2).status, v.PinStatus.BUSY
        )
        self.assertEqual(self.g.owner_counts().active_leases, 0)
        self.assertEqual(self.arbiter.owner_snapshot().owners, 0)

    def test_cancel_timeout_pending_ownership_fence_and_seal(self):
        entered, proceed = threading.Event(), threading.Event()
        original = self.engine.export_memory

        def export(memory):
            entered.set()
            proceed.wait(2)
            return original(memory)

        self.engine.export_memory = export
        req = self.pin()
        future = self.service.submit(req)
        self.assertTrue(entered.wait(2))
        with self.assertRaises(TimeoutError):
            future.result(0.001)
        self.assertTrue(future.cancel())
        duplicate = self.service.submit(req)
        self.service.fence_pins(1)
        self.assertFalse(self.service.seal_if_empty(1))
        self.assertEqual(
            self.service.submit(self.pin(1)).result(2).status, v.PinStatus.BUSY
        )
        proceed.set()
        duplicate.result(2)
        state = self.service.query(req.identity)
        self.assertEqual(state.state, "PREPARED")
        self.assertEqual(self.g.owner_counts().active_leases, 1)
        self.service.submit(
            v.UnpinReq(req.reference, state.lease_id, True, True)
        ).result(2)
        self.assertTrue(self.service.seal_if_empty(1))
        self.assertEqual(
            self.service.submit(v.UnpinReq(req.reference, state.lease_id, True, True))
            .result(2)
            .status,
            v.UnpinStatus.ALREADY_RELEASED,
        )

    def test_unpin_cannot_overtake_accepted_pin_and_revive_exports(self):
        accepted, proceed = threading.Event(), threading.Event()
        original = self.service._record

        def record(*args):
            value = original(*args)
            accepted.set()
            proceed.wait(2)
            return value

        self.service._record = record
        req = self.pin()
        future = self.service.submit(req)
        self.assertTrue(accepted.wait(2))
        state = self.service.query(req.identity)
        cleanup_req = v.UnpinReq(req.reference, state.lease_id, True, True)
        cleanup = self.service.submit(cleanup_req)
        try:
            self.assertEqual(cleanup.result(0.5).status, v.UnpinStatus.BUSY)
            self.assertEqual(self.service.query(req.identity).state, "PREPARING")
            self.assertEqual(self.service.owner_snapshot().leases, 1)
        finally:
            proceed.set()
            future.result(2)
            retry = self.service.submit(cleanup_req).result(2)
        self.assertEqual(retry.status, v.UnpinStatus.RELEASED)
        self.assertFalse(self.engine._exports)
        self.assertEqual(self.service.submit(req).result(2).status, v.PinStatus.MISS)

    def test_pending_pin_cannot_cross_fence_after_local_epoch_finish(self):
        entered, proceed = threading.Event(), threading.Event()
        original = self.catalog.lookup

        def lookup(result):
            entered.set()
            proceed.wait(5)
            return original(result)

        self.catalog.lookup = lookup
        request = self.pin()
        future = self.service.submit(request)
        try:
            self.assertTrue(entered.wait(2))
            self.service.fence_pins(1)
            self.assertFalse(self.service.unfence_pins(0))
            self.assertFalse(self.service.seal_if_empty(1))
            with self.assertRaises(svc.ServiceBusy):
                self.service.close()
            self.g.begin_epoch(1)
            self.g.install_epoch(1, ())
            self.g.finish_epoch(1, verified_headroom_pages=32)
        finally:
            proceed.set()
        response = future.result(2)
        try:
            self.assertEqual(response.status, v.PinStatus.BUSY)
            self.assertFalse(self.engine._exports)
            self.assertEqual(self.g.owner_counts().active_leases, 0)
            self.assertEqual(self.arbiter.owner_snapshot().owners, 0)
            self.assertEqual(self.service.owner_snapshot().owners, 0)
        finally:
            if response.status is v.PinStatus.PREPARED:
                self.service.submit(self.unpin(response)).result(2)
        self.assertTrue(self.service.seal_if_empty(1))
        self.assertFalse(self.service.unfence_pins(1))
        self.service.close()
        self.assertFalse(self.service.unfence_pins(1))
        self.assertEqual(
            self.service.submit(self.pin(1)).result(2).status, v.PinStatus.BUSY
        )

    def test_unpin_preparing_b_never_blocks_prepared_a_on_the_sole_worker(self):
        prepared_a = self.service.submit(self.pin()).result(2)
        entered, proceed = threading.Event(), threading.Event()
        original = self.catalog.lookup

        def lookup(result):
            entered.set()
            proceed.wait(5)
            return original(result)

        self.catalog.lookup = lookup
        request_b = self.pin(1)
        pin_b = self.service.submit(request_b)
        self.assertTrue(entered.wait(2))
        state_b = self.service.query(request_b.identity)
        unpin_b_req = v.UnpinReq(request_b.reference, state_b.lease_id, True, True)
        unpin_b = self.service.submit(unpin_b_req)
        unpin_a = self.service.submit(self.unpin(prepared_a))
        try:
            self.assertEqual(self.service._unpins._max_workers, 1)
            self.assertEqual(unpin_b.result(0.5).status, v.UnpinStatus.BUSY)
            self.assertEqual(unpin_a.result(0.5).status, v.UnpinStatus.RELEASED)
            self.assertEqual(self.service.query(request_b.identity).state, "PREPARING")
            self.assertEqual(self.service.owner_snapshot().leases, 1)
        finally:
            proceed.set()
            pin_b.result(2)
            self.service.submit(unpin_b_req).result(2)
            self.service.submit(self.unpin(prepared_a)).result(2)
        self.assertFalse(self.engine._exports)
        self.assertEqual(self.g.owner_counts().active_leases, 0)

    def test_identity_conflict_stale_peer_and_bounded_history(self):
        req = self.pin()
        response = self.service.submit(req).result(2)
        changed = replace(req, result=replace(req.result, prefix_hash="0" * 64))
        self.assertEqual(
            self.service.submit(changed).result(2).status, v.PinStatus.INCOMPATIBLE
        )
        bad = replace(
            req,
            result=replace(
                req.result, source=replace(req.result.source, incarnation="stale")
            ),
            identity=replace(req.identity, source_incarnation="stale"),
        )
        self.assertEqual(
            self.service.submit(bad).result(2).status, v.PinStatus.STALE_ENGINE
        )
        with self.assertRaises(svc.ServiceBusy):
            self.service.close()
        self.service.submit(self.unpin(response)).result(2)
        for attempt in range(1, 10):
            request = self.pin(attempt)
            done = self.service.submit(request).result(2)
            if done.status is v.PinStatus.PREPARED:
                self.service.submit(self.unpin(done)).result(2)
        self.assertLessEqual(self.service.owner_snapshot().records, 8)

    def test_all_156_buffers_survive_real_export_codec_without_dma(self):
        self.service.close()
        buffers = tuple(
            layout.BufferLayout(f"layer{i}.kv", 64, "full-mla-dsa") for i in range(156)
        )
        capacity, extent = 33, 33 * 64
        memory = self.engine.allocate_memory(len(buffers) * extent)
        adapter = SimpleNamespace(
            layout=layout.PoolLayout("uint8", 64, buffers),
            capacity=capacity,
            memories=(memory,),
            bindings={
                buffer.name: (layout.RegionSpan(0, extent, memory, i * extent),)
                for i, buffer in enumerate(buffers)
            },
        )
        self.service = svc.SlotGuardService(
            guard=self.g,
            catalog=self.catalog,
            local_identity=identity("source"),
            exact_peers=(identity(),),
            engine=self.engine,
            adapter=adapter,
            arbiter=self.arbiter,
        )
        response = self.service.submit(self.pin()).result(2)
        self.assertEqual(response.status, v.PinStatus.PREPARED)
        body = w.encode(response)
        self.assertNotIn(b"prefix_hash", body)
        self.assertLess(len(body), 160 * 1024)
        decoded = codec.decode_export(
            w.decode(
                w.encode(response), v.PinPreparedResp, request_context=response.request
            ),
            adapter.layout,
            "target-owner",
        )
        self.assertEqual(len(decoded.bindings), 156)
        self.assertEqual(self.engine._driver.submits, 0)
        self.service.submit(self.unpin(response)).result(2)
        self.engine.unregister_memory(memory)

    def test_queue_cancel_keeps_capacity_and_unpin_has_independent_worker(self):
        response = self.service.submit(self.pin()).result(2)
        entered, proceed = threading.Event(), threading.Event()
        original = self.catalog.lookup

        def lookup(result):
            entered.set()
            proceed.wait(2)
            return original(result)

        self.catalog.lookup = lookup
        first = self.service.submit(self.pin(1))
        self.assertTrue(entered.wait(2))
        second = self.service.submit(self.pin(2))
        self.assertTrue(second.cancel())
        self.assertEqual(
            self.service.submit(self.pin(3)).result(2).status, v.PinStatus.BUSY
        )
        self.assertEqual(
            self.service.submit(self.unpin(response)).result(2).status,
            v.UnpinStatus.RELEASED,
        )
        self.assertEqual(self.service.owner_snapshot().requests, 2)
        proceed.set()
        first_response = first.result(2)
        # The cancelled request still acquires and owns its source grant.
        second_response = self.service.submit(self.pin(2)).result(2)
        for done in (first_response, second_response):
            self.assertEqual(done.status, v.PinStatus.PREPARED)
            self.service.submit(self.unpin(done)).result(2)

    def test_cheap_owner_snapshot_includes_inbound_and_no_full_guard_census(self):
        with patch.object(
            self.g, "snapshot", side_effect=AssertionError("full census")
        ):
            before = self.service.owner_snapshot()
            self.assertEqual(before.owners, 0)
            self.assertTrue(self.arbiter.acquire(a.TransportRole.INBOUND, lease(9)))
            held = self.service.owner_snapshot()
            self.assertGreater(held.owners, 0)
            self.assertNotEqual(before.revision, held.revision)
            self.service.fence_pins(1)
            self.assertFalse(self.service.seal_if_empty(1))
            self.arbiter.release(a.TransportRole.INBOUND, lease(9))
            self.assertTrue(self.service.seal_if_empty(1))

    def test_partial_unpin_failure_keeps_journal_then_retries_exact_tail(self):
        done = self.service.submit(self.pin()).result(2)
        original = self.engine.release_export
        released = []

        def release(lease_id):
            if released:
                raise RuntimeError("remaining export unknown")
            original(lease_id)
            released.append(lease_id)

        self.engine.release_export = release
        req = self.unpin(done)
        self.assertEqual(self.service.submit(req).result(2).status, v.UnpinStatus.BUSY)
        self.assertEqual(self.g.owner_counts().active_leases, 1)
        self.assertEqual(len(self.engine._exports), len(done.regions) - 1)
        self.engine.release_export = original
        self.assertEqual(
            self.service.submit(req).result(2).status, v.UnpinStatus.RELEASED
        )
        self.assertFalse(self.engine._exports)

    def test_fence_unfence_requires_exact_generation_and_cannot_reopen_seal(self):
        self.service.fence_pins(2)
        self.assertFalse(self.service.unfence_pins(1))
        self.assertEqual(
            self.service.submit(self.pin()).result(2).status, v.PinStatus.BUSY
        )
        self.assertTrue(self.service.unfence_pins(2))
        done = self.service.submit(self.pin()).result(2)
        self.assertEqual(done.status, v.PinStatus.PREPARED)
        self.service.fence_pins(3)
        self.assertFalse(self.service.seal_if_empty(3))
        self.service.submit(self.unpin(done)).result(2)
        self.assertTrue(self.service.seal_if_empty(3))
        self.assertFalse(self.service.unfence_pins(3))

    def test_completion_byte_bound_cleans_partial_exports_and_retains_tombstone(self):
        request = self.pin()
        binding_bytes = len(w.reference_bytes(request.reference))
        self.service.max_retained_bytes = 2 * len(w.encode(request)) + binding_bytes + 1
        self.assertEqual(
            self.service.submit(request).result(2).status, v.PinStatus.BUSY
        )
        self.assertFalse(self.engine._exports)
        snapshot = self.service.owner_snapshot()
        self.assertEqual(snapshot.owners, 0)
        self.assertEqual(snapshot.retained_bytes, binding_bytes)
        self.assertEqual(snapshot.records, 1)
        self.assertEqual(
            self.service.submit(request).result(2).status, v.PinStatus.MISS
        )

    def test_terminal_reference_bytes_remain_bounded_and_counted(self):
        request = self.pin()
        weak_request = weakref.ref(request)
        binding_bytes = len(w.reference_bytes(request.reference))
        response = self.service.submit(request).result(2)
        record = self.service._records[request.identity.transfer_id]
        self.assertEqual(
            record.request_bytes, len(w.encode(request)) + len(request._wire_body)
        )
        self.service.submit(self.unpin(response)).result(2)
        del request, response
        gc.collect()
        self.assertIsNone(weak_request(), "terminal history retained full Pin/body")
        self.assertEqual(self.service.owner_snapshot().retained_bytes, binding_bytes)
        self.service.max_retained_bytes = binding_bytes
        self.assertEqual(
            self.service.submit(self.pin(1)).result(2).status, v.PinStatus.BUSY
        )
        self.assertEqual(self.service.owner_snapshot().records, 1)

    def test_forged_reference_cannot_grant_or_release_and_never_rehashes_pin(self):
        request = self.pin()
        with patch.object(w, "pin_binding", side_effect=AssertionError("rehashed Pin")):
            response = self.service.submit(request).result(2)
            self.assertEqual(response.status, v.PinStatus.PREPARED)
            self.assertEqual(self.service.submit(request).result(2), response)
            proof = self.unpin(response)
            forged = replace(proof.reference, content_digest="0" * 64)
            with self.assertRaises(ValueError):
                self.service.submit(forged)
            for reference, status in (
                (forged, v.UnpinStatus.INCOMPATIBLE),
                (
                    replace(
                        proof.reference,
                        identity=replace(
                            proof.reference.identity, nonce="forged-nonce"
                        ),
                    ),
                    v.UnpinStatus.INCOMPATIBLE,
                ),
                (
                    replace(
                        proof.reference,
                        source=replace(
                            proof.reference.source, engine_id="forged-source"
                        ),
                    ),
                    v.UnpinStatus.STALE_ENGINE,
                ),
            ):
                self.assertEqual(
                    self.service.submit(replace(proof, reference=reference))
                    .result(2)
                    .status,
                    status,
                )
                self.assertEqual(self.g.owner_counts().active_leases, 1)
            self.assertEqual(
                self.service.submit(proof).result(2).status, v.UnpinStatus.RELEASED
            )

    def test_stale_or_rebound_unpin_never_releases_original_lease(self):
        done = self.service.submit(self.pin()).result(2)
        original = self.unpin(done)
        self.assertEqual(
            self.service.submit(replace(original, lease_id="other")).result(2).status,
            v.UnpinStatus.INCOMPATIBLE,
        )
        changed = replace(
            done.request, result=replace(done.request.result, prefix_hash="0" * 64)
        )
        self.assertEqual(
            self.service.submit(replace(original, reference=changed.reference))
            .result(2)
            .status,
            v.UnpinStatus.INCOMPATIBLE,
        )
        self.assertEqual(self.g.owner_counts().active_leases, 1)
        self.service.submit(original).result(2)

    def test_readiness_progress_never_blocks_pin_or_unpin_worker(self):
        entered, proceed = threading.Event(), threading.Event()

        class Certificate:
            def ready(self):
                entered.set()
                proceed.wait(2)
                return True

        self.catalog.submit(
            c.Publication(
                c.PublicationScope("model-1", "layout-1", 64, 1),
                None,
                (tuple(range(64)),),
                (sum(range(64)),),
                self.refs[:1],
                Certificate(),
            )
        )
        progress = self.service.publication_progress()
        self.assertTrue(entered.wait(2))
        done = self.service.submit(self.pin()).result(2)
        self.assertEqual(done.status, v.PinStatus.PREPARED)
        self.assertEqual(
            self.service.submit(self.unpin(done)).result(2).status,
            v.UnpinStatus.RELEASED,
        )
        proceed.set()
        self.assertEqual(progress.result(2), 1)


if __name__ == "__main__":
    unittest.main()
