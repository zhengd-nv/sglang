"""Bounded v2 512K contracts using the real CPU-only package loader."""

import base64
import hashlib
import importlib
import json
import struct
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_protocol import PACKAGE, p, register_cpu_ci
from test_slotguard_catalog import Ready, c, hashes, keys, lease, result, s, seed_credit

from sglang.test.test_utils import CustomTestCase

v = importlib.import_module(f"{PACKAGE}.slotguard_protocol")
w = importlib.import_module(f"{PACKAGE}.slotguard_control_wire")
limits = importlib.import_module(f"{PACKAGE}.slotguard_limits")
config = importlib.import_module(f"{PACKAGE}.config")
receipts = importlib.import_module(f"{PACKAGE}.slotguard_receipts")
register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def request(hit=460800, page_size=64, prefix_hash=None):
    """Compact hash Pin restoring [0, hit); its size does not grow with hit."""
    if prefix_hash is None:
        prefix_hash = hashlib.sha256(b"512k-prefix").hexdigest()
    return v.PinReq(
        replace(
            result(),
            local_hit_tokens=0,
            hit_tokens=hit,
            page_size=page_size,
            prefix_hash=prefix_hash,
        ),
        lease(),
    )


def prepared(req):
    return v.PinPreparedResp(
        req,
        v.PinStatus.PREPARED,
        "lease",
        1000,
        (p.RegionDescriptor("region", b"descriptor"),),
        (p.BufferBinding("kv", "region", b"binding"),),
        tuple(s.PageRef(1, i + 1, i + 2) for i in range(req.result.restore_pages)),
    )


class Bounded512KTest(CustomTestCase):
    def test_complete_512000_pin_7200_pages_prepared_and_unpin_roundtrips(self):
        req = request()
        self.assertEqual(req.result.restore_pages, 7200)
        body = w.encode(req)
        self.assertLess(len(body), 4096)
        decoded = w.decode(body, v.PinReq)
        self.assertEqual(decoded, req)
        self.assertEqual(decoded.reference, req.reference)
        self.assertEqual(w.encode(decoded), body)
        response = prepared(req)
        response_body = w.encode(response)
        self.assertLessEqual(len(response_body), w.MAX_BODY_BYTES)
        self.assertEqual(
            w.decode(response_body, v.PinPreparedResp, request_context=decoded),
            response,
        )
        unpin = v.UnpinReq(req.reference, "lease", True, True)
        for message in (unpin, v.UnpinResp(unpin, v.UnpinStatus.RELEASED)):
            self.assertEqual(w.decode(w.encode(message), type(message)), message)
        with self.assertRaises(w.WireError):
            w.decode(body, v.PinReq, max_body_bytes=len(body) - 1)
        with (
            patch.object(w.json, "loads", side_effect=AssertionError("parsed")),
            self.assertRaises(w.WireError),
        ):
            w.decode(b" " * (w.MAX_BODY_BYTES + 1), v.PinReq)

    def test_exact_prefix_page_and_packed_bounds(self):
        req = request(hit=limits.MAX_PREFIX_TOKENS)
        self.assertEqual(w.decode(w.encode(req), v.PinReq), req)
        with self.assertRaises(ValueError):
            request(hit=limits.MAX_PREFIX_TOKENS + 64)
        req = request(hit=8192 * 32, page_size=32)
        response = prepared(req)
        self.assertEqual(len(response.pages), 8192)
        self.assertEqual(
            w.decode(w.encode(response), v.PinPreparedResp, request_context=req),
            response,
        )
        with self.assertRaises(ValueError):
            request(hit=8193 * 32, page_size=32)
        with self.assertRaises(ValueError):
            replace(response, pages=response.pages + (s.PageRef(1, 9000, 1),))
        vector = w._pack_ints((0,) * limits.MAX_PACKED_ITEMS)
        self.assertEqual(len(vector["data"]), limits.MAX_PACKED_BASE64_BYTES)
        self.assertEqual(len(w._unpack_ints(vector, 0)), limits.MAX_PACKED_ITEMS)
        with self.assertRaises(w.WireError):
            w._pack_ints((0,) * (limits.MAX_PACKED_ITEMS + 1))
        for value in (True, -(1 << 63) - 1, 1 << 63):
            with self.subTest(value=value), self.assertRaises(w.WireError):
                w._pack_ints((value,))
        for change in (
            {"count": True},
            {"count": limits.MAX_PACKED_ITEMS + 1},
            {"data": vector["data"] + "="},
            {"data": "!" * len(vector["data"])},
            {"count": limits.MAX_PACKED_ITEMS - 1},
        ):
            # The private helper preserves base64's ValueError; decode wraps it
            # into WireError at the public boundary.
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                w._unpack_ints({**vector, **change}, 0)

    def test_page_table_count_identity_and_generation_validation(self):
        response = prepared(request())
        raw = json.loads(w.encode(response))
        table = raw["payload"]["pages"]
        variants = [{**table, "count": value} for value in (True, 8193, 7199)]
        for name in ("page_ids", "generations"):
            variants.extend({**table, name: value} for value in (True, "!", ""))
            for value in (0, -1):
                encoded = base64.b64encode(struct.pack("<7200q", *([value] * 7200)))
                variants.append({**table, name: encoded.decode("ascii")})
        ids = [page.page_id for page in response.pages]
        ids[-1] = ids[0]
        variants.append({**table, "page_ids": w._pack_ints(tuple(ids))["data"]})
        for table in variants:
            changed = {**raw, "payload": {**raw["payload"], "pages": table}}
            with self.subTest(fields=table.keys()), self.assertRaises(w.WireError):
                w.decode(
                    json.dumps(changed).encode(),
                    v.PinPreparedResp,
                    request_context=response.request,
                )
        changed = replace(
            response,
            pages=response.pages[:-1] + (replace(response.pages[-1], generation=9000),),
        )
        self.assertNotEqual(
            receipts.ordered_page_digest(response.pages),
            receipts.ordered_page_digest(changed.pages),
        )

    def test_generic_bounds_namespace_metadata_and_small_bytes_unchanged(self):
        codec = importlib.import_module(f"{PACKAGE}.wire_codec")
        self.assertEqual((w.MAX_ITEMS, w.MAX_STRING), (300000, 4 * 1024 * 1024))
        self.assertEqual(w.MAX_BODY_BYTES, 16 * 1024 * 1024)
        self.assertEqual(codec.MAX_METADATA_BYTES, 4 * 1024 * 1024)
        with self.assertRaises(w.WireError):
            w._typed([0] * (w.MAX_ITEMS + 1), tuple[str, ...])
        with self.assertRaises(w.WireError):
            w._typed("x" * (w.MAX_STRING + 1), str)
        req = v.PinReq(result(), lease())
        self.assertEqual(
            hashlib.sha256(w.encode(req)).hexdigest(),
            "f536c798c1f6d099b079188ff9ae44578d6414bb58cdf3b3a115351ad57e9f4e",
        )
        with self.assertRaises(ValueError):
            replace(req.result, cache_salt="x" * 4097)
        response = prepared(req)
        with self.assertRaises(ValueError):
            replace(
                response,
                regions=(
                    p.RegionDescriptor("region", b"x" * codec.MAX_METADATA_BYTES),
                ),
            )


class LimitsWiringTest(CustomTestCase):
    def test_large_pin_quota_and_deferred_reclaim(self):
        req = request()
        guard = config.SlotGuardLimits(
            max_pin_pages=8192,
            max_remote_pinned_pages=8192,
            max_pending_reclaim_pages=8192,
        ).make_guard(req.result.source.incarnation)
        seed_credit(guard, 8192)
        refs = guard.register_allocation(tuple(range(1, 8194)))
        first, remaining = refs[:7200], refs[7200:]
        guard.publish(first)
        guard.publish(remaining)
        guard.try_pin(req.identity, first)
        second = replace(
            req.identity,
            transfer_id=replace(req.identity.transfer_id, attempt=1),
            nonce="second",
        )
        with self.assertRaises(s.QuotaExceeded):
            guard.try_pin(second, remaining)
        self.assertEqual(guard.snapshot().held_pages, first)
        self.assertEqual(guard.on_free(first).deferred, first)
        guard.release(req.identity, first)
        guard.begin_epoch(1)
        guard.install_epoch(1, ())
        self.assertEqual(guard.drain_reclaim(1), first)
        guard.finish_epoch(1, verified_headroom_pages=8192)
        self.assertEqual(guard.snapshot().held_pages, ())

    def test_defaults_caps_and_explicit_serving_catalog_bound(self):
        defaults = config.SlotGuardLimits()
        self.assertEqual(
            (defaults.max_pin_pages, defaults.max_remote_pinned_pages), (4096, 4096)
        )
        for value in (True, 8193):
            with self.assertRaises(ValueError):
                config.SlotGuardLimits(
                    max_pin_pages=value, max_remote_pinned_pages=9000
                )
        with self.assertRaises(ValueError):
            config.SlotGuardLimits(max_pin_pages=8192)
        configured = config.SlotGuardLimits(
            max_pin_pages=8192, max_remote_pinned_pages=8192
        )
        guard = configured.make_guard("source")
        self.assertEqual(c.PublicationCatalog(guard).max_publication_pages, 4096)
        serving = importlib.import_module(f"{PACKAGE}.serving_runtime")
        capture = importlib.import_module(f"{PACKAGE}.publication_capture")
        service = importlib.import_module(f"{PACKAGE}.slotguard_service")
        owner = NS(engine=object(), adapter=object())
        ctx = NS(
            params=NS(token_to_kv_pool_allocator=NS(slot_guard=guard)),
            tp_worker=NS(model_runner=NS(forward_stream=object())),
        )
        profile = importlib.import_module(f"{PACKAGE}.control_profile")
        cfg = NS(
            slotguard_limits=configured,
            control_profile=profile.ControlProfile.SLOTGUARD_HASH_V3,
            publication_context_entries=65536,
            publication_context_bytes=64 * 1024 * 1024,
            directory=NS(target=result().source, entries=()),
        )
        with (
            patch.object(capture, "PublicationCaptureWorker"),
            patch.object(capture, "CanonicalPublicationHook"),
            patch.object(service, "SlotGuardService"),
        ):
            serving.SlotGuardServingRuntime._initialize_slotguard_source(
                owner, ctx, object(), cfg, NS(device="cpu")
            )
        self.assertEqual(owner.catalog.max_publication_pages, 8192)

    def test_7200_page_publication_lookup_and_receipts(self):
        tokens = tuple(range(460800))
        req = request(prefix_hash=hashes(tokens, 64)[-1])
        configured = config.SlotGuardLimits(
            max_pin_pages=8192, max_remote_pinned_pages=8192
        )
        guard = configured.make_guard(req.result.source.incarnation)
        refs = guard.register_allocation(tuple(range(1, 7201)))
        ready = Ready()
        ready.value = True
        publication = c.Publication(
            c.PublicationScope("model-1", "layout-1", 64, 1),
            None,
            tuple(tokens[start : start + 64] for start in range(0, 460800, 64)),
            keys(tokens, 64),
            refs,
            ready,
        )
        catalog = c.PublicationCatalog(
            guard, max_publication_pages=8192, event_keys=keys
        )
        catalog.submit(publication)
        self.assertEqual(catalog.progress(), 1)
        self.assertEqual(catalog.lookup(req.result), refs)
        fields = receipts.publication_receipt(publication)
        self.assertTrue(receipts.valid_receipt("publication_ready", fields, None))
        fields = receipts.lease_receipt(req.reference, "lease", refs)
        self.assertTrue(
            receipts.valid_receipt(
                "target_pin_validated", fields, req.result.transfer_id
            )
        )
        for value in (8192, 8193, True):
            self.assertEqual(
                receipts.valid_receipt(
                    "target_pin_validated",
                    {**fields, "page_count": value},
                    req.result.transfer_id,
                ),
                value == 8192,
            )


if __name__ == "__main__":
    unittest.main()
