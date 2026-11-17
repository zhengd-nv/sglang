"""Hash-only Pin contracts; CPU only, no inference performance claims."""

import hashlib
import importlib
import struct
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_protocol import PACKAGE, identity, p, register_cpu_ci
from test_slotguard_catalog import Ready, c, guard

from sglang.test.test_utils import CustomTestCase

v = importlib.import_module(f"{PACKAGE}.slotguard_protocol")
w = importlib.import_module(f"{PACKAGE}.slotguard_control_wire")
profiles = importlib.import_module(f"{PACKAGE}.control_profile")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def hashes(tokens, page_size, prior=None):
    output = []
    for start in range(0, len(tokens), page_size):
        digest = hashlib.sha256()
        if prior is not None:
            digest.update(bytes.fromhex(prior))
        page = tokens[start : start + page_size]
        digest.update(struct.pack(f"<{len(page)}I", *page))
        prior = digest.hexdigest()
        output.append(prior)
    return tuple(output)


def keys(tokens, page_size):
    return tuple(
        int.from_bytes(bytes.fromhex(h)[:8], "big", signed=True)
        for h in hashes(tokens, page_size)
    )


def hit(length=193, hit_tokens=192, local=64, page_size=64):
    tokens = tuple(i % 100003 for i in range(length))
    complete = tokens[: length // page_size * page_size]
    full = hashes(complete, page_size)
    q = p.LookupQuery(
        identity(),
        p.TransferId(identity().incarnation, "rid", 0),
        keys(complete, page_size),
        tokens,
        local,
        (length - 1) // page_size * page_size,
        page_size,
        prefix_hashes=full,
    )
    return p.LookupResult(
        q, identity("source"), hit_tokens, q.event_block_keys[: hit_tokens // page_size]
    )


class HashPinTest(CustomTestCase):
    def test_dirty_suffix_proof_and_evicted_ancestor_are_fail_closed(self):
        g = guard()
        refs = g.register_allocation((1, 2, 3))
        catalog = c.PublicationCatalog(g, event_keys=keys, max_records=3)
        scope = c.PublicationScope("model-1", "layout-1", 64, 1)
        ready = Ready()
        ready.value = True
        tokens = tuple(range(192))
        parent = catalog.submit(
            c.Publication(
                scope, None, (tokens[:64],), keys(tokens[:64], 64), refs[:1], ready
            )
        )
        catalog.submit(
            c.Publication(
                scope,
                parent,
                (tokens[64:128], tokens[128:]),
                keys(tokens, 64)[1:],
                refs[1:],
                ready,
            )
        )
        catalog.progress()
        compact = profiles.ControlProfile.SLOTGUARD_HASH_V3.make_pin(hit()).result
        self.assertEqual(catalog.lookup(compact), refs[1:])
        # Evict only the root while retaining the terminal strong-index entry.
        catalog.submit(
            c.Publication(
                scope, None, (tokens[:64],), keys(tokens[:64], 64), refs[:1], ready
            )
        )
        self.assertIsNone(catalog.lookup(compact))

    def test_forged_publication_key_never_builds_strong_index(self):
        g = guard()
        refs = g.register_allocation((1, 2, 3))
        catalog = c.PublicationCatalog(g, event_keys=keys)
        ready = Ready()
        ready.value = True
        tokens = tuple(range(192))
        expected = keys(tokens, 64)
        catalog.submit(
            c.Publication(
                c.PublicationScope("model-1", "layout-1", 64, 1),
                None,
                tuple(tokens[i : i + 64] for i in range(0, 192, 64)),
                (expected[0] + 1,) + expected[1:],
                refs,
                ready,
            )
        )
        catalog.progress()
        compact = profiles.ControlProfile.SLOTGUARD_HASH_V3.make_pin(hit()).result
        self.assertIsNone(catalog.lookup(compact))

    def test_hash_service_replay_generation_and_exact_unpin(self):
        from test_slotguard_service import ServiceTest

        fixture = ServiceTest()
        fixture.setUp()
        try:
            tokens = tuple(range(192))
            ready = Ready()
            ready.value = True
            fixture.catalog.submit(
                c.Publication(
                    c.PublicationScope("model-1", "layout-1", 64, 1),
                    None,
                    tuple(tokens[i : i + 64] for i in range(0, 192, 64)),
                    keys(tokens, 64),
                    fixture.refs,
                    ready,
                )
            )
            fixture.catalog.progress()
            profile = profiles.ControlProfile.SLOTGUARD_HASH_V3
            request = profile.make_pin(hit())
            incoming = profile.decode(
                w.encode(request), v.PinReq, max_body_bytes=1 << 24
            )
            # Hash lookup must not call the old ROOT-key/token verifier.
            with patch.object(
                fixture.catalog,
                "event_keys",
                side_effect=AssertionError("Pin hashed tokens"),
            ):
                response = fixture.service.submit(incoming).result(2)
            self.assertEqual(response.status, v.PinStatus.PREPARED)
            self.assertEqual(response.pages, fixture.refs[1:])
            decoded = profile.decode(
                w.encode(response),
                v.PinPreparedResp,
                max_body_bytes=1 << 24,
                request_context=request,
            )
            self.assertTrue(profile.pin_matches(decoded, request))
            self.assertEqual(fixture.service.submit(incoming).result(2), response)
            conflict = replace(
                request, result=replace(request.result, prefix_hash="0" * 64)
            )
            self.assertEqual(
                fixture.service.submit(conflict).result(2).status,
                v.PinStatus.INCOMPATIBLE,
            )
            unpin = profile.make_unpin(decoded)
            ack = fixture.service.submit(unpin).result(2)
            self.assertTrue(profile.released(ack, unpin))
            self.assertEqual(ack.protocol_version, 3)
            self.assertEqual(
                fixture.service.submit(unpin).result(2).status,
                v.UnpinStatus.ALREADY_RELEASED,
            )
        finally:
            fixture.tearDown()

    def test_compact_bounds_namespace_and_capability(self):
        request = profiles.ControlProfile.SLOTGUARD_HASH_V3.make_pin(hit())
        for kwargs in (
            {"prefix_hash": "0" * 16},
            {"cache_salt": "salt"},
            {"extra_key": "adapter"},
            {"local_hit_tokens": 65},
            {"hit_tokens": 0},
            {"page_size": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(request.result, **kwargs)
        with self.assertRaises(ValueError):
            replace(request, capability="slotguard-local-prepared-v2")

    def test_fixed_wire_size_and_no_token_vectors(self):
        sizes = []
        for n in (128000, 256000, 512000):
            result = hit(n, n // 10 * 9, 0)
            request = profiles.ControlProfile.SLOTGUARD_HASH_V3.make_pin(result)
            body = w.encode(request)
            sizes.append(len(body))
            self.assertNotIn(b"prefix_token_ids", body)
            self.assertNotIn(b"event_block_keys", body)
            self.assertEqual(w.decode(body, v.PinReq), request)
            self.assertLess(len(body), 3000)
        self.assertLess(max(sizes) - min(sizes), 10)

    def test_pin_never_sends_the_local_full_hash_list(self):
        request = profiles.ControlProfile.SLOTGUARD_HASH_V3.make_pin(hit())
        body = w.encode(request)
        self.assertNotIn(b"prefix_hashes", body)
        self.assertEqual(w.decode(body, v.PinReq), request)

    def test_strong_index_exact_slice_and_stale_pages(self):
        g = guard()
        refs = g.register_allocation((1, 2, 3))
        catalog = c.PublicationCatalog(g, event_keys=keys)
        ready = Ready()
        tokens = tuple(range(192))
        catalog.submit(
            c.Publication(
                c.PublicationScope("model-1", "layout-1", 64, 1),
                None,
                tuple(tokens[i : i + 64] for i in range(0, 192, 64)),
                keys(tokens, 64),
                refs,
                ready,
            )
        )
        compact = profiles.ControlProfile.SLOTGUARD_HASH_V3.make_pin(hit()).result
        self.assertIsNone(catalog.lookup(compact))
        ready.value = True
        catalog.progress()
        self.assertEqual(catalog.lookup(compact), refs[1:])
        self.assertIsNone(catalog.lookup(replace(compact, prefix_hash="0" * 64)))
        g.on_free(refs)
        self.assertIsNone(catalog.lookup(compact))


if __name__ == "__main__":
    unittest.main()
