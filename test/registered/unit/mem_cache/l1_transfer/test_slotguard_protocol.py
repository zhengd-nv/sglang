import base64
import hashlib
import importlib
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_protocol import PACKAGE, p, register_cpu_ci
from test_slotguard_catalog import lease, lookup_result, result

from sglang.test.test_utils import CustomTestCase

v = importlib.import_module(f"{PACKAGE}.slotguard_protocol")
w = importlib.import_module(f"{PACKAGE}.slotguard_control_wire")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class ProtocolTest(CustomTestCase):
    def test_compact_messages_never_reprocess_original_pin(self):
        with patch.object(w, "pin_binding", wraps=w.pin_binding) as binding:
            req = v.PinReq(result(), lease())
            self.assertEqual(binding.call_count, 1)
            for _ in range(3):
                response = v.PinPreparedResp(req, v.PinStatus.MISS)
                unpin = v.UnpinReq(req.reference, "lease", True, True)
                for message in (
                    response,
                    unpin,
                    v.UnpinResp(unpin, v.UnpinStatus.RELEASED),
                ):
                    body = w.encode(message)
                    self.assertLess(len(body), 8192)
                    self.assertNotIn(b"prefix_hash", body)
            self.assertEqual(binding.call_count, 1)
        body = w.encode(req)
        self.assertIn(b"prefix_hash", body)
        self.assertEqual(w.decode(body, v.PinReq).reference, req.reference)

    def test_every_original_pin_field_binds_reference(self):
        req = v.PinReq(result(), lease())
        r = req.result
        variants = [replace(req, identity=replace(req.identity, nonce="new-nonce"))]
        for name, value in (
            ("local_hit_tokens", 0),
            ("hit_tokens", 4),
            ("prefix_hash", "0" * 64),
        ):
            variants.append(replace(req, result=replace(r, **{name: value})))
        changed_source = replace(r.source, incarnation="new-source")
        variants.append(
            replace(
                req,
                result=replace(r, source=changed_source),
                identity=replace(
                    req.identity, source_incarnation=changed_source.incarnation
                ),
            )
        )
        changed_transfer = replace(r.transfer_id, attempt=1)
        variants.append(
            replace(
                req,
                result=replace(r, transfer_id=changed_transfer),
                identity=replace(req.identity, transfer_id=changed_transfer),
            )
        )
        body = w.encode(v.PinPreparedResp(req, v.PinStatus.MISS))
        for changed in variants:
            with self.subTest(reference=changed.reference):
                self.assertNotEqual(changed.reference, req.reference)
                with self.assertRaises(w.WireError):
                    w.decode(body, v.PinPreparedResp, request_context=changed)

    def test_strict_result_ranges_depth_and_reference_schema(self):
        req = v.PinReq(result(), lease())
        for name, invalid in (
            ("local_hit_tokens", True),
            ("hit_tokens", -1),
            ("page_size", 0),
            ("prefix_hash", "A" * 64),
            ("prefix_hash", "0" * 63),
        ):
            with (
                self.subTest(name=name, invalid=invalid),
                self.assertRaises(ValueError),
            ):
                replace(req.result, **{name: invalid})
        with self.assertRaises(w.WireError):
            w.decode(b"[" * 18 + b"0" + b"]" * 18, v.PinReq)
        unpin = v.UnpinReq(req.reference, "lease", True, True)
        raw = w.encode(unpin)
        for body in (
            raw.replace(b'"content_digest":', b'"unknown":'),
            raw.replace(b'"protocol_version":3', b'"protocol_version":true'),
            raw.replace(
                b'"capability":"slotguard-hash-prepared-v3"', b'"capability":"v1"'
            ),
        ):
            self.assertNotEqual(body, raw)
            with self.assertRaises(w.WireError):
                w.decode(body, v.UnpinReq)

    def test_compact_reply_requires_exact_local_context(self):
        req = v.PinReq(result(), lease())
        reply = v.PinPreparedResp(req, v.PinStatus.MISS)
        body = w.encode(reply)
        self.assertNotIn(b"prefix_hash", body)
        self.assertLess(len(body), 8192)
        self.assertEqual(w.decode(body, v.PinPreparedResp, request_context=req), reply)
        with self.assertRaises(w.WireError):
            w.decode(body, v.PinPreparedResp)
        changed = v.PinReq(replace(req.result, prefix_hash="0" * 64), req.identity)
        with self.assertRaises(w.WireError):
            w.decode(body, v.PinPreparedResp, request_context=changed)

    def test_reply_capability_is_rejected_before_export_descriptors(self):
        req = v.PinReq(result(), lease())
        raw = json.loads(w.encode(v.PinPreparedResp(req, v.PinStatus.MISS)))
        raw["payload"]["capability"] = "wrong-capability"
        raw["payload"]["regions"] = [{}]
        original = w._typed
        seen = []

        def tracked(value, expected, depth=0, request_context=None):
            seen.append(expected)
            return original(value, expected, depth, request_context)

        with patch.object(w, "_typed", tracked), self.assertRaises(w.WireError):
            w.decode(json.dumps(raw).encode(), v.PinPreparedResp, request_context=req)
        self.assertNotIn(p.RegionDescriptor, seen)

    def test_compact_unpin_and_computed_binding(self):
        req = v.PinReq(result(), lease())
        ref = req.reference
        self.assertEqual(ref, w.decode(w.encode(req), v.PinReq).reference)
        unpin = v.UnpinReq(ref, "lease", True, True)
        body = w.encode(unpin)
        self.assertNotIn(b"prefix_hash", body)
        self.assertLess(len(body), 8192)
        self.assertEqual(w.decode(body, v.UnpinReq), unpin)
        obj = json.loads(w.encode(req))
        obj["payload"]["_reference"] = json.loads(body)["payload"]["reference"]
        with self.assertRaises(w.WireError):
            w.decode(json.dumps(obj).encode(), v.PinReq)

    def test_nominal_roundtrip_and_version_rejection(self):
        request = v.PinReq(result(), lease())
        self.assertEqual(w.decode(w.encode(request), v.PinReq), request)
        self.assertEqual(p.PROTOCOL_VERSION, 1)
        for changes in (
            {"protocol_version": 1},
            {"capability": "other"},
            {"identity": lease(1)},
        ):
            with self.assertRaises(ValueError):
                replace(request, **changes)

    def test_strict_fields_types_duplicate_and_bounds(self):
        req = v.PinReq(result(), lease())
        encoded = w.encode(req)
        variants = [
            encoded.replace(b'"version":3', b'"version":true'),
            encoded.replace(b'"version":3', b'"version":3,"version":3'),
            encoded.replace(b'"page_size":2', b'"page_size":true'),
        ]
        obj = json.loads(encoded)
        obj["payload"]["unknown"] = 0
        variants.append(json.dumps(obj).encode())
        for data in variants:
            with self.assertRaises(w.WireError):
                w.decode(data, v.PinReq)
        with self.assertRaises(w.WireError):
            w.decode(encoded, v.PinReq, max_body_bytes=10)
        with self.assertRaises(ValueError):
            v.UnpinReq(req, "lease", False, True)
        with self.assertRaises(ValueError):
            v.PinPreparedResp(req, v.PinStatus.PREPARED)


c = importlib.import_module(f"{PACKAGE}.control_profile")
V3 = c.ControlProfile.SLOTGUARD_HASH_V3


class ProfileTest(CustomTestCase):
    def test_identity_can_be_reserved_before_worker_pin_construction(self):
        identity = lease()
        request = V3.make_pin(lookup_result(), identity=identity)
        self.assertIs(request.identity, identity)
        with self.assertRaises(ValueError):
            V3.make_pin(lookup_result(), identity=lease(1))
        self.assertEqual(
            V3.paths, ("/v3/slotguard-hash/pin", "/v3/slotguard-hash/unpin")
        )
        response = v.PinPreparedResp(request, v.PinStatus.MISS)
        self.assertTrue(V3.pin_matches(response, request))
        self.assertFalse(V3.prepared(response))

    def test_profile_never_guesses_from_foreign_messages_or_strings(self):
        for value in (None, "v1", "slotguard-v2", "slotguard-hash-v3", 2):
            with self.assertRaises(ValueError):
                c.require_profile(value)
        req = V3.make_pin(lookup_result())
        with self.assertRaises(ValueError):
            V3.endpoints(v.PinPreparedResp(req, v.PinStatus.MISS))
        changed = replace(req, identity=replace(req.identity, nonce="another"))
        self.assertFalse(
            V3.pin_matches(v.PinPreparedResp(changed, v.PinStatus.MISS), req)
        )
        self.assertEqual(p.PROTOCOL_VERSION, 1)


class PackedWireTest(CustomTestCase):
    def test_cached_encoder_never_reserializes_or_rehashes(self):
        req = v.PinReq(result(), lease())
        body = w.encode(req)
        self.assertEqual(hashlib.sha256(body).hexdigest(), req.reference.content_digest)
        self.assertIs(body, req._wire_body)
        with (
            patch.object(w, "_body", side_effect=AssertionError("serialized again")),
            patch.object(w, "decode", side_effect=AssertionError("decoded again")),
        ):
            self.assertIs(w.encode(req), body)
            with self.assertRaises(w.WireError):
                w.encode(req, max_body_bytes=len(body) - 1)
        decoded = w.decode(body, v.PinReq)
        self.assertEqual(decoded, req)
        self.assertEqual(decoded.reference, req.reference)
        self.assertEqual(decoded._wire_body, body)

    def test_internal_fields_cannot_be_forged(self):
        raw = json.loads(w.encode(v.PinReq(result(), lease())))
        for field in ("_wire_body", "_wire_size", "_reference"):
            changed = json.loads(json.dumps(raw))
            changed["payload"][field] = "forged"
            with self.subTest(field=field), self.assertRaises(w.WireError):
                w.decode(json.dumps(changed).encode(), v.PinReq)

    def test_fixed_little_endian_signed_int64_bytes(self):
        values = (0, 1, -1, -(1 << 63), (1 << 63) - 1)
        encoded = w._pack_ints(values)
        expected = (
            b"\0" * 8
            + b"\1"
            + b"\0" * 7
            + b"\xff" * 8
            + b"\0" * 7
            + b"\x80"
            + b"\xff" * 7
            + b"\x7f"
        )
        self.assertEqual(base64.b64decode(encoded["data"]), expected)
        self.assertEqual(w._unpack_ints(encoded, 0), values)

    def test_pack_ints_rejects_non_int64_values(self):
        for invalid in (True, 1.0, -(1 << 63) - 1, 1 << 63):
            with self.subTest(invalid=invalid), self.assertRaises(w.WireError):
                w._pack_ints((invalid,))

    def test_canonical_binding_ignores_json_field_order_and_whitespace(self):
        req = v.PinReq(result(), lease())
        raw = json.loads(w.encode(req))
        raw["payload"] = dict(reversed(list(raw["payload"].items())))
        raw["payload"]["result"] = dict(
            reversed(list(raw["payload"]["result"].items()))
        )
        decoded = w.decode(json.dumps(raw, indent=2).encode(), v.PinReq)
        self.assertEqual(decoded.reference, req.reference)
        self.assertEqual(w.encode(decoded), w.encode(req))


if __name__ == "__main__":
    unittest.main()
