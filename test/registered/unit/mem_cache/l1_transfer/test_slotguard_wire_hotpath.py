"""Preparse guard, complete PageRef table and validated reply cache."""

import base64
import itertools
import json
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import bounded_json as bj
from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer import slot_guard as s
from sglang.srt.mem_cache.l1_transfer import slotguard_control_wire as w
from sglang.srt.mem_cache.l1_transfer import slotguard_protocol as v
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.source import ServiceFixture
from sglang.test.test_utils import CustomTestCase

v, w = v, w
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class WireHotpathTest(CustomTestCase):
    def setUp(self):
        self.base = ServiceFixture()
        self.base.setUp()
        self.response = self.base.service.submit(self.base.pin()).result(2)

    def tearDown(self):
        self.base.service.submit(self.base.unpin(self.response)).result(2)
        self.base.tearDown()

    def test_table_preserves_every_ordered_page_reference(self):
        response = replace(self.response, pages=tuple(reversed(self.response.pages)))
        body = w.encode(response)
        table = json.loads(body)["payload"]["pages"]
        self.assertEqual(
            table.keys(), {"codec", "pool_epoch", "count", "page_ids", "generations"}
        )
        self.assertEqual(table["codec"], "pageref-i64le-v1")
        self.assertEqual(table["count"], len(response.pages))
        self.assertEqual(table["pool_epoch"], response.pages[0].pool_epoch)
        self.assertEqual(
            struct.unpack(f"<{table['count']}q", base64.b64decode(table["page_ids"])),
            response.source_page_ids,
        )
        self.assertEqual(
            w.decode(body, v.PinPreparedResp, request_context=response.request),
            response,
        )
        largest = (1 << 63) - 1
        pages = tuple(
            s.PageRef(largest, largest - i, largest) for i in range(len(response.pages))
        )
        extreme = replace(response, pages=pages)
        self.assertEqual(
            w.decode(
                w.encode(extreme), v.PinPreparedResp, request_context=extreme.request
            ),
            extreme,
        )

    def test_strict_page_table_rejects_ambiguous_shapes_and_invalid_values(self):
        raw = json.loads(w.encode(self.response))
        table = raw["payload"]["pages"]
        variants = [[], {}, {**table, "codec": "pages-only"}, {**table, "extra": 1}]
        for name in ("count", "pool_epoch"):
            for value in (True, -1, 1 << 63, "1"):
                variants.append({**table, name: value})
        variants += [{**table, "count": 4097}, {**table, "count": table["count"] + 1}]
        for name in ("page_ids", "generations"):
            for value in (True, [], "!", table[name] + "=", ""):
                variants.append({**table, name: value})
            for value in (0, -1):
                data = struct.pack(f"<{table['count']}q", *([value] * table["count"]))
                variants.append({**table, name: base64.b64encode(data).decode()})
        variants.append(
            {**table, "page_ids": base64.b64encode(struct.pack("<2q", 1, 1)).decode()}
        )
        # Two int64 entries have two padding characters, with unused pad bits.
        encoded = table["generations"]
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        index = alphabet.index(encoded[-3])
        variants.append(
            {**table, "generations": encoded[:-3] + alphabet[index | 1] + "=="}
        )
        for table in variants:
            changed = {**raw, "payload": {**raw["payload"], "pages": table}}
            with self.subTest(table=table), self.assertRaises(w.WireError):
                w.decode(
                    json.dumps(changed).encode(),
                    v.PinPreparedResp,
                    request_context=self.response.request,
                )

    def test_wrong_context_rejected_before_table_or_exports(self):
        raw = json.loads(w.encode(self.response))
        raw["payload"]["pages"] = {"codec": "unsolicited"}
        raw["payload"]["regions"] = [{}]
        typed = w._typed

        def guarded(value, expected, depth=0, request_context=None):
            if expected is p.RegionDescriptor:
                raise AssertionError("unsolicited export visited")
            return typed(value, expected, depth, request_context)

        with (
            patch.object(w, "_typed", guarded),
            patch.object(
                w, "_unpack_pages", side_effect=AssertionError("table visited")
            ),
            self.assertRaises(w.WireError),
        ):
            w.decode(json.dumps(raw).encode(), v.PinPreparedResp)

    def test_cold_validated_cache_warm_bound_decode_and_parallel_encoding(self):
        response = replace(self.response)
        self.assertIsNone(response._wire_body)
        with patch.object(w, "decode", wraps=w.decode) as decode:
            body = w.encode(response)
            self.assertEqual(decode.call_count, 1)
        self.assertIs(response._wire_body, body)
        with (
            patch.object(w, "_body", side_effect=AssertionError("serialized twice")),
            patch.object(w, "decode", side_effect=AssertionError("validated twice")),
        ):
            self.assertIs(w.encode(response), body)
            with self.assertRaises(w.WireError):
                w.encode(response, max_body_bytes=len(body) - 1)
        decoded = w.decode(body, v.PinPreparedResp, request_context=response.request)
        self.assertEqual(w.encode(decoded), body)
        self.assertIs(w.encode(decoded), decoded._wire_body)
        cold = replace(response)
        with ThreadPoolExecutor(max_workers=8) as pool:
            bodies = tuple(pool.map(lambda _: w.encode(cold), range(32)))
        self.assertTrue(all(item is cold._wire_body for item in bodies))
        with self.assertRaises(TypeError):
            v.PinPreparedResp(response.request, response.status, _wire_body=body)
        raw = json.loads(body)
        raw["payload"]["_wire_body"] = "forged"
        with self.assertRaises(w.WireError):
            w.decode(
                json.dumps(raw).encode(),
                v.PinPreparedResp,
                request_context=response.request,
            )

    def test_cold_cache_never_skips_nested_outgoing_validation(self):
        region = replace(self.response.regions[0], descriptor=b"a" * 20)
        response = replace(self.response, regions=(region,) + self.response.regions[1:])
        with patch.object(w, "MAX_STRING", 16), self.assertRaises(w.WireError):
            w.encode(response)
        self.assertIsNone(response._wire_body)


# The pre-parse shape check exactly as decode() applies it.
_shape_check = partial(
    bj.checked_json_text,
    max_depth=w.MAX_DEPTH,
    max_items=w.MAX_ITEMS,
    error=w.WireError,
)


class V2DepthTest(CustomTestCase):
    def test_all_json_byte_encodings_reject_depth_before_loads(self):
        for encoding in (
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
            "utf-32",
            "utf-32-le",
            "utf-32-be",
        ):
            body = ("[" * (w.MAX_DEPTH + 1) + "0" + "]" * (w.MAX_DEPTH + 1)).encode(
                encoding
            )
            with (
                self.subTest(encoding=encoding),
                patch.object(
                    w.json, "loads", side_effect=AssertionError("parser reached")
                ),
                self.assertRaises(w.WireError),
            ):
                w.decode(body, v.PinReq)

    def test_structural_regex_and_c_string_skip_handle_escapes_unicode(self):
        request = ServiceFixture().pin()
        for text in ("[]{}" * 20000, '\\"' * 10000, "雪😀\ud800", "a" * 200000):
            # The lease nonce is the request's free-form bounded text field.
            request = replace(
                request, identity=replace(request.identity, nonce=text[:4096])
            )
            body = w.encode(request)
            for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-32"):
                self.assertEqual(
                    w.decode(body.decode().encode(encoding), v.PinReq), request
                )
        body = json.dumps({"x": '[]{}\\"' * 20000}).encode()
        with patch.object(bj, "_scanstring", wraps=bj._scanstring) as scan:
            _shape_check(body)
            self.assertEqual(scan.call_count, 2)

    def test_malformed_strings_and_mismatched_closers_fail_before_loads(self):
        bodies = [
            b'"unterminated',
            b'"\\',
            b'"bad\\x"',
            b'"bad\x00"',
            b"][",
            b"{]",
            b'\\"unterminated',
        ]
        for body in bodies:
            with (
                self.subTest(body=body),
                patch.object(
                    w.json, "loads", side_effect=AssertionError("parser reached")
                ),
                self.assertRaises(w.WireError),
            ):
                w.decode(body, v.PinReq)

    def test_depth_boundary_escaped_quote_runs_and_literal_surrogates(self):
        for characters in itertools.product('[]{}\\"a', repeat=3):
            quoted = json.dumps("".join(characters))
            for prefix, suffix in (("[", "]"), ('{"x":', "}")):
                allowed = prefix * w.MAX_DEPTH + quoted + suffix * w.MAX_DEPTH
                self.assertEqual(_shape_check(allowed.encode()), allowed)
                too_deep = prefix + allowed + suffix
                with (
                    patch.object(
                        w.json, "loads", side_effect=AssertionError("parser reached")
                    ),
                    self.assertRaises(w.WireError),
                ):
                    w.decode(too_deep.encode(), v.PinReq)
        request = ServiceFixture().pin()
        for namespace in ("雪😀", "\ud800", "\udfff"):
            raw = json.loads(w.encode(request))
            raw["payload"]["identity"]["nonce"] = namespace
            text = json.dumps(raw, ensure_ascii=False)
            for encoding in (
                "utf-8",
                "utf-16-le",
                "utf-16-be",
                "utf-32-le",
                "utf-32-be",
            ):
                decoded = w.decode(text.encode(encoding, "surrogatepass"), v.PinReq)
                self.assertEqual(decoded.identity.nonce, namespace)


if __name__ == "__main__":
    unittest.main()
