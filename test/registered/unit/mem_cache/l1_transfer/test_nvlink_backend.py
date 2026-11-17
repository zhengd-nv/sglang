"""Target lifecycle against a real SlotGuard source service and fake CUDA bytes."""

import importlib
import json
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_protocol import PACKAGE, identity, p, register_cpu_ci
from test_sglang_adapter import BorrowDriver, SGLangPoolAdapter
from test_slotguard_catalog import Ready, c, guard, hashes, keys, s, seed_credit

from sglang.test.kits.l1_transfer_cpu import ExportResponse, NvlinkEngine
from sglang.test.test_utils import CustomTestCase

b = importlib.import_module(f"{PACKAGE}.nvlink_backend")
codec = importlib.import_module(f"{PACKAGE}.wire_codec")
l = importlib.import_module(f"{PACKAGE}.role_gate")
d = importlib.import_module(f"{PACKAGE}.directory")
k = importlib.import_module(f"{PACKAGE}.kvcm_client")
a = importlib.import_module(f"{PACKAGE}.transport_role_arbiter")
v = importlib.import_module(f"{PACKAGE}.slotguard_protocol")
svc = importlib.import_module(f"{PACKAGE}.slotguard_service")
V3 = importlib.import_module(
    f"{PACKAGE}.control_profile"
).ControlProfile.SLOTGUARD_HASH_V3
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def pool(device):
    result = type("MHATokenToKVPool", (), {})()
    result.use_hnd, result.kv_cache_layout = False, "nhd"
    result.size, result.page_size = 2048, 64
    result.store_dtype, result.start_layer = NS(itemsize=1), 0
    names = ("k0", "v0", "k1", "v1")
    extent, stride, base = 2112, 4096, 65536
    result._kv_buffer_descs = [
        NS(name=name, shape=(2112, 1, 1), row_bytes=1, tokens_per_row=1)
        for name in names
    ]
    result.get_contiguous_buf_infos = lambda: (
        [base + i * stride for i in range(4)],
        [extent] * 4,
        [64] * 4,
    )
    result._post_capture_owner = NS(
        _final_num_tokens=2048,
        page_size=64,
        store_dtype=result.store_dtype,
        device_id=device,
        _specs=[
            NS(desc=desc, offset=i * stride)
            for i, desc in enumerate(result._kv_buffer_descs)
        ],
        _arena=NS(
            base=base,
            reserved=16384,
            granularity=64,
            _closed=False,
            _allocation=NS(
                _closed=False,
                _mappings=[
                    (base + i * stride + offset, length, 0)
                    for i in range(4)
                    for offset, length in ((0, 704), (704, 1408))
                ],
            ),
        ),
    )
    return result


class SourceClient:
    """Real SlotGuardService behind the control client seam, with fault knobs."""

    def __init__(self, service):
        self.service = service
        self.calls = []
        self.lose_ack = 0
        self.pin_filter = lambda response: response
        self.ack_filter = lambda response: response
        self.hold_pin = False
        self.held = None
        self.hold_ack = False
        self.held_ack = None

    def pin(self, endpoint, req):
        self.calls.append(("pin", req, threading.get_ident()))
        out = Future()

        def done(f):
            value = self.pin_filter(f.result())
            if self.hold_pin:
                self.held = (out, value)
            else:
                out.set_result(value)

        self.service.submit(req).add_done_callback(done)
        return out

    def unpin(self, endpoint, req):
        self.calls.append(("unpin", req, threading.get_ident()))
        out = Future()

        def done(f):
            value = f.result()
            if self.lose_ack:
                self.lose_ack -= 1
                out.set_exception(TimeoutError())
            elif self.hold_ack:
                self.held_ack = (out, value)
            else:
                out.set_result(self.ack_filter(value))

        self.service.submit(req).add_done_callback(done)
        return out


class BackendTest(CustomTestCase):
    def setUp(self):
        self.driver = BorrowDriver()
        self.driver.state = 1
        self.source_engine = NvlinkEngine(0, _driver=self.driver)
        self.target_engine = NvlinkEngine(1, _driver=self.driver)
        self.source_pool = SGLangPoolAdapter(pool(0), self.source_engine)
        self.target_pool = SGLangPoolAdapter(pool(1), self.target_engine)
        self.source = identity("source", tp_size=1)
        self.target = identity("target", tp_size=1)
        tokens = tuple(range(1025))
        query = p.LookupQuery(
            self.target,
            p.TransferId(self.target.incarnation, "rid-1", 0),
            keys(tokens, 64),
            tokens,
            256,
            1024,
            64,
            prefix_hashes=hashes(tokens, 64),
        )
        self.result = p.LookupResult(query, self.source, 1024, query.event_block_keys)
        self.pages = tuple(range(17, 29))[::-1]
        self.slots = p.PrivateSlotView(
            self.result.transfer_id,
            256,
            1024,
            64,
            tuple(x for page in self.pages for x in range(page * 64, (page + 1) * 64)),
            self.pages,
        )
        # [0, 256) is the target's local prefix; the source publishes the whole
        # chain, and only its [256, 1024) pages are exported.
        self.source_pages = tuple(range(3, 27, 2))
        self.guard = guard(shared_headroom_credit=True)
        seed_credit(self.guard, 32)
        refs = self.guard.register_allocation((2, 4, 6, 8) + self.source_pages)
        self.catalog = c.PublicationCatalog(self.guard, event_keys=keys)
        ready = Ready()
        ready.value = True
        self.catalog.submit(
            c.Publication(
                c.PublicationScope(
                    self.source.model_id,
                    self.source.layout_digest,
                    64,
                    self.guard.pool_epoch,
                ),
                None,
                tuple(tokens[i : i + 64] for i in range(0, 1024, 64)),
                keys(tokens, 64),
                refs,
                ready,
            )
        )
        self.catalog.progress()
        self.arbiter = a.TransportRoleArbiter()
        self.service = svc.SlotGuardService(
            guard=self.guard,
            catalog=self.catalog,
            local_identity=self.source,
            exact_peers=(self.target,),
            engine=self.source_engine,
            adapter=self.source_pool,
            arbiter=self.arbiter,
        )
        self.client = SourceClient(self.service)
        self.directory = d.EngineDirectory(
            self.target,
            (d.EngineDirectoryEntry("reporter", self.source, "http://source:1234"),),
        )
        self.gate = l.RoleGate()
        self.gate.acquire(l.EngineRole.INBOUND, self.result.transfer_id)
        self.backend = b.NVLinkTransferBackend(
            self.target_engine,
            self.target_pool,
            self.directory,
            k.KVCMClient(None, self.directory),
            self.client,
            self.gate,
            control_profile=V3,
            rpc_timeout_s=0.2,
            copy_timeout_s=1,
            poll_interval_s=0.001,
        )
        for index, spans in enumerate(self.source_pool.bindings.values()):
            for span in spans:
                data = bytes(
                    (index * 47 + i + span.buffer_offset) % 251
                    for i in range(span.length)
                )
                self.source_engine.upload(span.memory, data, span.region_offset)
        for memory in self.target_pool.memories:
            self.target_engine.upload(memory, b"\xa5" * memory.size)

    def source_locked(self):
        return self.guard.owner_counts().active_leases > 0

    def occupy_source_role(self):
        """Another inbound transfer owns the source's single transport role."""
        other = p.TransferId("peer-boot-1", "busy", 0)
        owner = s.LeaseIdentity(self.source.incarnation, "peer-boot-1", other, "busy")
        self.assertTrue(self.arbiter.acquire(a.TransportRole.INBOUND, owner))
        return owner

    def advance(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.001)
        self.fail("lifecycle did not reach expected state")

    def status(self):
        return self.backend.poll_prefetch(self.result.transfer_id)

    def start(self):
        return self.backend.prefetch(self.result, self.slots)

    def finish(self, expected=p.PrefetchStatus.READY):
        self.advance(lambda: self.status() == expected)

    def tearDown(self):
        if self.backend._live is not None and self.status() in (
            p.PrefetchStatus.READY,
            p.PrefetchStatus.FAILED_SAFE,
        ):
            self.backend.retire(self.result.transfer_id)
        if self.backend._live is None:
            self.backend.close()
        # Quarantined targets keep their source lease; reconcile it directly
        # with the authoritative source query before closing the fixture.
        for kind, req, _ in self.client.calls:
            if kind == "pin":
                state = self.service.query(req.identity)
                if state is not None and state.state == "PREPARED":
                    self.service.submit(
                        v.UnpinReq(req.reference, state.lease_id, True, True)
                    ).result(2)
        deadline = time.monotonic() + 2
        while True:
            try:
                self.service.assert_quiescent()
                break
            except svc.ServiceBusy:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.001)
        self.service.close()

    def test_restore_768_slots_all_buffers_mappings_and_sentinels(self):
        self.start()
        self.finish()
        self.assertEqual(self.driver.submits, 1)
        self.assertFalse(self.source_locked())
        self.assertFalse(self.source_engine._exports)
        for index, spans in enumerate(self.target_pool.bindings.values()):
            actual = b"".join(
                self.target_engine.download(
                    span.memory, span.length, span.region_offset
                )
                for span in spans
            )
            expected = bytearray(b"\xa5" * 2112)
            for source_page, target_page in zip(self.source_pages, self.pages):
                expected[target_page * 64 : (target_page + 1) * 64] = bytes(
                    (index * 47 + i) % 251
                    for i in range(source_page * 64, (source_page + 1) * 64)
                )
            self.assertEqual(actual, expected)
        self.assertEqual(self.gate.role, l.EngineRole.INBOUND)
        self.assertTrue(
            all(call[2] != threading.get_ident() for call in self.client.calls)
        )
        self.backend.retire(self.result.transfer_id)
        self.backend.close()

    def test_explicit_pin_miss(self):
        with patch.object(self.catalog, "lookup", return_value=None):
            self.start()
            self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual([x[0] for x in self.client.calls], ["pin"])

    def test_ready_trace_is_enqueued_before_scheduler_can_observe_ready(self):
        observations = []

        def observe(phase, ticket=None, **fields):
            if phase == "backend_state" and fields["status"] == "READY":
                acquired = self.backend._lock.acquire(blocking=False)
                if acquired:
                    self.backend._lock.release()
                observations.append((acquired, self.backend._records[ticket].status))

        self.backend.observer = NS(emit=observe, emit_slotguard=lambda *a, **k: None)
        self.start()
        self.finish()
        self.assertEqual(observations, [(False, p.PrefetchStatus.RELEASING)])

    def test_explicit_pin_busy(self):
        owner = self.occupy_source_role()
        self.start()
        self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertEqual(self.driver.submits, 0)
        self.assertFalse(self.source_locked())
        self.arbiter.release(a.TransportRole.INBOUND, owner)

    def test_late_pin_after_cancel_cleans_exact_lease_without_dma(self):
        self.client.hold_pin = True
        self.start()
        self.advance(lambda: self.client.held is not None)
        self.backend.cancel_prefetch(self.result.transfer_id)
        future, response = self.client.held
        future.set_result(response)
        self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertEqual(self.driver.submits, 0)
        self.assertFalse(self.source_locked())
        self.assertEqual(self.client.calls[-1][1].lease_id, response.lease_id)

    def test_lost_pin_reply_quarantines_even_if_later_received(self):
        self.client.hold_pin = True
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        future, response = self.client.held
        future.set_result(response)
        self.assertTrue(self.source_locked())
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual(len(self.client.calls), 1)
        with self.assertRaises(b.BusyError):
            self.backend.retire(self.result.transfer_id)
        with self.assertRaises(b.BusyError):
            self.backend.close()

    def test_wrong_pin_request_never_imports_or_unpins_other_lease(self):
        query = replace(
            self.result.query, transfer_id=replace(self.result.transfer_id, attempt=8)
        )
        other = V3.make_pin(replace(self.result, query=query))
        self.client.pin_filter = lambda response: replace(response, request=other)
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertTrue(self.source_locked())
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual(len(self.client.calls), 1)

    def test_malformed_metadata_is_released_before_any_import(self):
        before = self.driver.next
        self.client.pin_filter = lambda response: replace(
            response,
            regions=(replace(response.regions[0], descriptor=b'{"version":2}'),)
            + response.regions[1:],
        )
        self.start()
        self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertFalse(self.source_locked())
        self.assertEqual(self.driver.next, before)
        self.assertEqual(self.driver.submits, 0)

    def test_cancel_copying_waits_for_drain_and_unpin_ack(self):
        self.driver.state = 0
        self.start()
        self.advance(lambda: self.driver.submits == 1)
        self.assertEqual(
            self.backend.cancel_prefetch(self.result.transfer_id),
            p.PrefetchStatus.COPYING,
        )
        self.assertTrue(self.source_locked())
        self.assertTrue(self.target_engine._batches)
        self.driver.state = 1
        self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertFalse(self.source_locked())
        self.assertFalse(self.target_engine._batches)

    def test_copy_failed_drained_releases_safely(self):
        self.driver.state = 2
        self.start()
        self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertFalse(self.source_locked())
        self.assertFalse(self.target_engine._batches)
        self.assertFalse(self.source_engine._exports)

    def test_copy_uncertain_keeps_all_resources(self):
        self.driver.state = 3
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertTrue(self.source_locked())
        record = self.backend._records[self.result.transfer_id]
        self.assertEqual(len(record.imports), 8)
        self.assertTrue(self.target_engine._batches)
        self.assertEqual(len(self.client.calls), 1)

    def test_submit_baseexception_is_quarantined_not_pending(self):
        def fail(ranges):
            raise KeyboardInterrupt()

        self.driver.submit = fail
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertTrue(self.target_engine._batches)
        self.assertTrue(self.source_locked())
        self.assertEqual(len(self.client.calls), 1)

    def test_partial_import_and_failed_rollback_keeps_source_locked(self):
        original_info, original_close = self.driver.info, self.driver.close
        initial = self.driver.next

        def info(handle):
            if handle == initial + 2:
                raise RuntimeError("import tracking failed")
            return original_info(handle)

        def close(handle):
            if handle == initial + 2:
                raise RuntimeError("unmap failed")
            return original_close(handle)

        self.driver.info, self.driver.close = info, close
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertTrue(self.target_engine._pending_cleanup)
        self.assertEqual(len(self.backend._records[self.result.transfer_id].imports), 1)
        self.assertTrue(self.source_locked())
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual(len(self.client.calls), 1)

    def test_first_import_exception_without_public_handle_is_not_unmap_proof(self):
        def fail(device, descriptor):
            raise RuntimeError("unknown import side effects")

        self.driver.import_region = fail
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertFalse(self.backend._records[self.result.transfer_id].imports)
        self.assertTrue(self.source_locked())
        self.assertEqual(len(self.client.calls), 1)

    def test_pure_binding_failure_releases_known_imports_without_dma(self):
        with patch.object(
            self.target_engine, "bind_pages", side_effect=ValueError("bad binding")
        ):
            self.start()
            self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertFalse(self.backend._records[self.result.transfer_id].imports)
        self.assertFalse(self.source_locked())
        self.assertEqual(self.driver.submits, 0)

    def test_failed_batch_release_cannot_unmap_or_unpin(self):
        def fail(handle):
            raise RuntimeError("release failed")

        self.driver.release_batch = fail
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertEqual(len(self.backend._records[self.result.transfer_id].imports), 8)
        self.assertTrue(self.target_engine._batches)
        self.assertTrue(self.source_locked())
        self.assertEqual(len(self.client.calls), 1)

    def test_failed_unmap_cannot_unpin(self):
        def fail(handle):
            raise RuntimeError("unmap failed")

        self.driver.close = fail
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertFalse(self.target_engine._batches)
        self.assertEqual(len(self.backend._records[self.result.transfer_id].imports), 8)
        self.assertTrue(self.source_locked())
        self.assertEqual(len(self.client.calls), 1)

    def test_copy_timeout_never_proves_drain(self):
        self.driver.state = 0
        self.backend._copy_timeout = 0.01
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertTrue(self.target_engine._batches)
        self.assertTrue(self.source_locked())

    def test_stale_ack_does_not_grant_ready(self):
        self.client.ack_filter = lambda response: replace(
            response, status=v.UnpinStatus.STALE_ENGINE
        )
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertEqual(len(self.client.calls), 4)

    def test_target_does_not_allocate_or_release_scheduler_role(self):
        with (
            patch.object(
                self.target_engine, "allocate_memory", side_effect=AssertionError()
            ),
            patch.object(self.gate, "acquire", side_effect=AssertionError()),
            patch.object(self.gate, "release", side_effect=AssertionError()),
        ):
            self.start()
            self.finish()
            self.backend.retire(self.result.transfer_id)
            self.backend.close()
        self.assertEqual(self.gate.role, l.EngineRole.INBOUND)

    def test_retire_required_before_new_attempt_and_old_attempt_rejected(self):
        old_result, old_slots = self.result, self.slots
        self.start()
        self.finish()
        self.gate.release(l.EngineRole.INBOUND, old_result.transfer_id)
        new_id = replace(old_result.transfer_id, attempt=1)
        self.gate.acquire(l.EngineRole.INBOUND, new_id)
        self.result = replace(
            old_result, query=replace(old_result.query, transfer_id=new_id)
        )
        self.slots = replace(old_slots, transfer_id=new_id)
        with self.assertRaises(b.BusyError):
            self.start()
        self.backend.retire(old_result.transfer_id)
        self.start()
        self.finish()
        self.gate.release(l.EngineRole.INBOUND, new_id)
        self.gate.acquire(l.EngineRole.INBOUND, old_result.transfer_id)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.backend.prefetch(old_result, old_slots)
        self.assertEqual(self.driver.submits, 2)

    def test_record_capacity_never_evicts_attempt_fence(self):
        self.backend._max_records = 1
        self.start()
        self.finish()
        self.backend.retire(self.result.transfer_id)
        new_id = replace(self.result.transfer_id, attempt=1)
        self.gate.release(l.EngineRole.INBOUND, self.result.transfer_id)
        self.gate.acquire(l.EngineRole.INBOUND, new_id)
        result = replace(
            self.result, query=replace(self.result.query, transfer_id=new_id)
        )
        with self.assertRaises(b.BusyError):
            self.backend.prefetch(result, replace(self.slots, transfer_id=new_id))
        self.assertEqual(len(self.backend._records), 1)

    def test_closed_backend_and_tp4_are_rejected(self):
        self.backend.close()
        with self.assertRaises(b.BusyError):
            self.start()
        directory = d.EngineDirectory(replace(self.target, tp_size=4), ())
        with self.assertRaisesRegex(ValueError, "TP=1"):
            b.NVLinkTransferBackend(
                self.target_engine,
                self.target_pool,
                directory,
                self.backend._kvcm,
                self.client,
                self.gate,
                control_profile=V3,
            )

    def test_exact_directory_target_and_source_required(self):
        result = replace(self.result, source=replace(self.source, incarnation="stale"))
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.backend.prefetch(result, self.slots)
        target = replace(self.target, tp_group_id="other")
        result = replace(self.result, query=replace(self.result.query, target=target))
        with self.assertRaisesRegex(ValueError, "target identity"):
            self.backend.prefetch(result, self.slots)

    def test_real_kvcm_lookup_on_background_caller(self):
        import concurrent.futures

        import grpc
        import test_kvcm_client as wire_test

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        service = wire_test.Service()
        service.response.hosts.add(host_ip_port="reporter", local=16)
        server = grpc.server(executor)
        wire_test.rpc.add_MetaServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        try:
            self.backend._kvcm = k.KVCMClient(channel, self.directory)
            result = executor.submit(self.backend.lookup, self.result.query).result(
                timeout=2
            )
            self.assertEqual(result, self.result)
            self.assertEqual(len(service.requests), 1)
        finally:
            channel.close()
            server.stop(None).wait()
            executor.shutdown(wait=True)

    def test_ack_loss_retries_same_lease_without_recopied_bytes(self):
        self.client.lose_ack = 1
        self.start()
        self.finish()
        unpins = [x[1] for x in self.client.calls if x[0] == "unpin"]
        self.assertEqual(len(unpins), 2)
        self.assertEqual(unpins[0], unpins[1])
        self.assertEqual(self.driver.submits, 1)
        self.assertEqual(self.guard.owner_counts().active_leases, 0)

    def test_no_terminal_status_before_release_ack_and_cancel_stays_intent(self):
        self.client.hold_ack = True
        self.start()
        self.advance(lambda: self.client.held_ack is not None)
        self.assertFalse(self.source_locked())
        self.assertFalse(self.target_engine._batches)
        self.assertEqual(self.status(), p.PrefetchStatus.RELEASING)
        start = time.monotonic()
        for _ in range(100):
            self.status()
            self.backend.cancel_prefetch(self.result.transfer_id)
        self.assertLess(time.monotonic() - start, 0.1)
        with self.assertRaises(b.BusyError):
            self.backend.retire(self.result.transfer_id)
        self.client.held_ack[0].set_result(self.client.held_ack[1])
        self.finish(p.PrefetchStatus.FAILED_SAFE)

    def test_blocked_pin_control_call_does_not_block_enqueue_or_cancel(self):
        entered, proceed = threading.Event(), threading.Event()
        original = self.client.pin

        def blocked(*args):
            self.assertNotEqual(threading.get_ident(), self.backend._owner)
            entered.set()
            if not proceed.wait(2):
                raise RuntimeError("test Pin not released")
            return original(*args)

        self.client.pin = blocked
        self.start()
        self.advance(entered.is_set)
        start = time.monotonic()
        self.start()
        self.status()
        self.backend.cancel_prefetch(self.result.transfer_id)
        self.assertLess(time.monotonic() - start, 0.1)
        proceed.set()
        self.finish(p.PrefetchStatus.FAILED_SAFE)
        self.assertEqual(self.driver.submits, 0)
        self.assertFalse(self.source_locked())

    def test_wrong_ack_cannot_make_ready(self):
        self.client.ack_filter = lambda response: replace(
            response, request=replace(response.request, lease_id="different-lease")
        )
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertEqual(self.driver.submits, 1)
        self.assertFalse(self.target_engine._batches)
        self.assertEqual(len(self.client.calls), 4)

    def test_ack_retry_exhaustion_is_bounded_and_quarantined(self):
        self.client.lose_ack = 99
        self.start()
        self.finish(p.PrefetchStatus.QUARANTINED)
        self.assertEqual(len(self.client.calls), 4)
        self.assertEqual(self.driver.submits, 1)
        self.assertFalse(self.backend._records[self.result.transfer_id].imports)

    def _use_async_unpin(self):
        self.backend.close()
        self.backend = b.NVLinkTransferBackend(
            self.target_engine,
            self.target_pool,
            self.directory,
            k.KVCMClient(None, self.directory),
            self.client,
            self.gate,
            control_profile=V3,
            rpc_timeout_s=0.2,
            copy_timeout_s=1,
            poll_interval_s=0.001,
            async_unpin=True,
        )

    def test_async_unpin_is_ready_after_unmap_before_release_ack(self):
        """READY needs drained DMA and unmapped imports only; the ACK follows."""
        self._use_async_unpin()
        self.client.hold_ack = True
        self.start()
        self.finish()
        self.advance(lambda: self.client.held_ack is not None)
        self.assertFalse(self.target_engine._batches)
        self.assertFalse(self.backend._records[self.result.transfer_id].imports)
        self.client.held_ack[0].set_result(self.client.held_ack[1])
        self.advance(lambda: not self.backend.owner_snapshot().working)
        self.assertEqual(self.backend.owner_snapshot().unknown, 0)

    def test_async_unpin_failure_stays_visible_after_ready(self):
        """A lost release ACK after READY must not vanish with the retired record."""
        self._use_async_unpin()
        self.client.lose_ack = 99
        self.start()
        self.finish()
        self.advance(lambda: self.backend.owner_snapshot().unknown == 1)
        self.assertEqual(len([x for x in self.client.calls if x[0] == "unpin"]), 3)

    def test_duplicate_object_gate_and_attempt_fences(self):
        self.client.hold_pin = True
        self.start()
        self.assertEqual(self.start(), self.result.transfer_id)
        pages = tuple(reversed(self.pages))
        different = replace(
            self.slots,
            page_ids=pages,
            slot_indices=tuple(
                x for page in pages for x in range(page * 64, (page + 1) * 64)
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.backend.prefetch(self.result, different)
        with self.assertRaisesRegex(ValueError, "reconstructed"):
            self.backend.prefetch(replace(self.result), self.slots)
        self.gate.release(l.EngineRole.INBOUND, self.result.transfer_id)
        with self.assertRaises(l.LeaseRejected):
            self.start()
        self.gate.acquire(l.EngineRole.OUTBOUND, self.result.transfer_id)
        with self.assertRaises(l.LeaseRejected):
            self.start()
        self.gate.release(l.EngineRole.OUTBOUND, self.result.transfer_id)
        self.gate.acquire(l.EngineRole.INBOUND, self.result.transfer_id)
        self.advance(lambda: self.client.held is not None)
        self.client.held[0].set_result(self.client.held[1])
        self.finish()
        self.backend.retire(self.result.transfer_id)
        self.assertIsInstance(
            self.backend._records[self.result.transfer_id], b._Retired
        )
        self.assertFalse(
            hasattr(self.backend._records[self.result.transfer_id], "result")
        )
        with self.assertRaisesRegex(ValueError, "retired"):
            self.start()
        self.assertEqual(self.driver.submits, 1)

    def test_enqueue_never_serializes_or_hashes_prompt_on_scheduler(self):
        import dataclasses
        import hashlib

        owner = threading.get_ident()

        def guarded(fn):
            def call(*args, **kwargs):
                self.assertNotEqual(threading.get_ident(), owner)
                return fn(*args, **kwargs)

            return call

        # Patch only while enqueueing: the source Scheduler's existing RPC
        # fingerprint path is intentionally outside this target-backend test.
        with (
            patch.object(dataclasses, "asdict", guarded(dataclasses.asdict)),
            patch.object(json, "dumps", guarded(json.dumps)),
            patch.object(hashlib, "sha256", guarded(hashlib.sha256)),
        ):
            self.start()
            self.start()
            self.status()
        self.finish()

    def test_all_scheduler_apis_reject_wrong_thread(self):
        errors = []

        def other():
            for fn in (
                self.start,
                self.status,
                lambda: self.backend.cancel_prefetch(self.result.transfer_id),
                lambda: self.backend.retire(self.result.transfer_id),
                self.backend.close,
            ):
                try:
                    fn()
                except l.SchedulerThreadViolation:
                    errors.append(True)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join(1)
        self.assertEqual(len(errors), 5)
        with self.assertRaises(l.SchedulerThreadViolation):
            self.backend.lookup(self.result.query)

    def test_blocked_cuda_stages_do_not_block_scheduler(self):
        # Each stage executes on the single worker while Scheduler snapshots and
        # idempotent enqueue remain local. Cancellation is tested on the last stage.
        for name in ("import_region", "submit", "poll", "release_batch", "close"):
            original = getattr(self.driver, name)
            entered, proceed = threading.Event(), threading.Event()

            def blocked(*args, original=original, entered=entered, proceed=proceed):
                self.assertNotEqual(threading.get_ident(), self.backend._owner)
                entered.set()
                if not proceed.wait(2):
                    raise RuntimeError("test stage was not released")
                return original(*args)

            setattr(self.driver, name, blocked)
            setattr(self, "stage_" + name, (entered, proceed))
        self.start()
        for name in ("import_region", "submit", "poll", "release_batch", "close"):
            entered, proceed = getattr(self, "stage_" + name)
            self.advance(entered.is_set)
            start = time.monotonic()
            for _ in range(100):
                self.status()
            self.start()  # same attempt is local/idempotent, even while CUDA blocks
            if name == "close":
                self.backend.cancel_prefetch(self.result.transfer_id)
            self.assertLess(time.monotonic() - start, 0.1)
            self.assertTrue(self.source_locked())
            proceed.set()
        self.finish(p.PrefetchStatus.FAILED_SAFE)


class CodecTest(CustomTestCase):
    def setUp(self):
        self.driver = BorrowDriver()
        self.engine = NvlinkEngine(0, _driver=self.driver)
        self.adapter = SGLangPoolAdapter(pool(0), self.engine)
        self.descriptors = {
            m.region_id: self.engine.export_memory(m) for m in self.adapter.memories
        }
        regions, bindings = codec.encode_export(
            identity("source"), "lease", self.adapter, self.descriptors
        )
        self.response = ExportResponse(
            NS(result=NS(source=identity("source"))),
            "lease",
            regions,
            bindings,
            tuple(range(4, 16)),
        )

    def decode(self, response):
        return codec.decode_export(response, self.adapter.layout, "0" * 32)

    def test_full_multi_mapping_roundtrip(self):
        decoded = self.decode(self.response)
        self.assertEqual(decoded.layout, self.adapter.layout)
        self.assertEqual(decoded.capacity, 33)
        self.assertEqual(len(decoded.descriptors), 8)
        self.assertEqual([len(spans) for _, spans in decoded.bindings], [2] * 4)

    def test_binding_validation_before_cuda(self):
        original = self.response.buffer_bindings[0]
        for change in (
            lambda x: x.update(version=True),
            lambda x: x.update(capacity=True),
            lambda x: x.update(capacity=34),
            lambda x: x.update(dtype="wrong"),
            lambda x: x.update(page_size=32),
            lambda x: x["buffer"].update(layout="wrong"),
            lambda x: x["buffer"].update(page_bytes=32),
            lambda x: x["spans"][1].update(region_id="unknown"),
            lambda x: x["spans"][1].update(buffer_offset=703),
            lambda x: x["spans"][1].update(length=1409),
            lambda x: x["spans"][1].update(region_offset=1),
            lambda x: x.update(address=12345),
        ):
            value = json.loads(original.descriptor)
            change(value)
            changed = replace(original, descriptor=json.dumps(value).encode())
            response = replace(
                self.response,
                buffer_bindings=(changed,) + self.response.buffer_bindings[1:],
            )
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.decode(response)

    def test_region_validation_before_cuda(self):
        original = self.response.regions[0]
        for change in (
            lambda x: x.update(version=2),
            lambda x: x.update(lease_id="wrong"),
            lambda x: x["source"].update(incarnation="wrong"),
            lambda x: x["transport"].update(owner_id="0" * 32),
            lambda x: x["transport"].update(access="write"),
            lambda x: x["transport"].update(transport="other"),
            lambda x: x["transport"].update(transport_handle="AAA="),
            lambda x: x["transport"].update(size=True),
            lambda x: x["transport"].update(address=12345),
        ):
            value = json.loads(original.descriptor)
            change(value)
            changed = replace(original, descriptor=json.dumps(value).encode())
            response = replace(
                self.response, regions=(changed,) + self.response.regions[1:]
            )
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.decode(response)
        response = replace(self.response, source_page_ids=tuple(range(30, 42)))
        with self.assertRaises(ValueError):
            self.decode(response)

    def test_duplicate_json_fields_and_aliased_native_handles(self):
        first, second = self.response.regions[:2]
        changed = replace(
            first,
            descriptor=first.descriptor.replace(
                b'"version":1', b'"version":1,"version":1'
            ),
        )
        with self.assertRaises(ValueError):
            self.decode(
                replace(self.response, regions=(changed,) + self.response.regions[1:])
            )
        obj = json.loads(second.descriptor)
        obj["transport"]["transport_handle"] = json.loads(first.descriptor)[
            "transport"
        ]["transport_handle"]
        changed = replace(second, descriptor=json.dumps(obj).encode())
        with self.assertRaises(ValueError):
            self.decode(
                replace(
                    self.response, regions=(first, changed) + self.response.regions[2:]
                )
            )


if __name__ == "__main__":
    unittest.main()
