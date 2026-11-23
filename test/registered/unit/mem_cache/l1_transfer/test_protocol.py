"""Contract tests using the production L1 package identity."""

import unittest
from dataclasses import replace

from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.wire import identity
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def query(**changes):
    target = identity()
    return replace(
        p.LookupQuery(
            target=target,
            transfer_id=p.TransferId(target.incarnation, "rid-1", 0),
            event_block_keys=tuple(range(-8, 8)),
            prefix_token_ids=tuple(range(1025)),
            local_hit_tokens=256,
            max_restore_tokens=1024,
            page_size=64,
        ),
        **changes,
    )


def hit(**changes):
    q = query()
    return replace(
        p.LookupResult(q, identity("source"), 1024, q.event_block_keys), **changes
    )


def slots(**changes):
    result = hit()
    pages = tuple(range(4, 16))
    return replace(
        p.PrivateSlotView(
            transfer_id=result.transfer_id,
            local_hit_tokens=256,
            hit_tokens=1024,
            page_size=64,
            slot_indices=tuple(range(256, 1024)),
            page_ids=pages,
        ),
        **changes,
    )


class ProtocolTest(CustomTestCase):
    def test_unknown_version(self):
        for obj in (identity(), p.TransferId("boot", "r", 0), query(), hit(), slots()):
            with self.subTest(type=type(obj)), self.assertRaises(ValueError):
                replace(obj, protocol_version=2)

    def test_rank_must_be_below_tp_size(self):
        with self.assertRaises(ValueError):
            identity(tp_rank=2)

    def test_source_scope(self):
        for field, value in (
            ("engine_id", "target"),
            ("domain_id", "other"),
            ("instance_id", "other"),
            ("model_id", "other"),
            ("layout_digest", "other"),
            ("tp_size", 4),
            ("tp_rank", 1),
            ("shard_id", "other"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                hit(source=replace(identity("source"), **{field: value}))

    def test_stale_target_incarnation(self):
        with self.assertRaises(ValueError):
            query(transfer_id=p.TransferId("old-boot", "rid-1", 0))

    def test_query_reserves_last_prompt_token(self):
        # Even a page-aligned full prompt must reserve its final token.
        with self.assertRaises(ValueError):
            query(prefix_token_ids=tuple(range(1024)))
        self.assertEqual(
            query(
                prefix_token_ids=tuple(range(1024)), max_restore_tokens=960
            ).max_restore_tokens,
            960,
        )

    def test_root_event_keys_int64_and_full_prefix_coverage(self):
        keys = (-(1 << 63), (1 << 63) - 1) + query().event_block_keys[2:]
        self.assertEqual(query(event_block_keys=keys).event_block_keys, keys)
        for key in (-(1 << 63) - 1, 1 << 63):
            with self.subTest(key=key), self.assertRaises(ValueError):
                query(event_block_keys=(key,) + keys[1:])
        with self.assertRaises(ValueError):
            query(prefix_token_ids=(-1,) + tuple(range(1024)))
        for invalid_keys in (
            query().event_block_keys[:-1],
            query().event_block_keys + (5,),
        ):
            with self.assertRaises(ValueError):
                query(event_block_keys=invalid_keys)

    def test_result_range_and_keys(self):
        result = hit()
        self.assertEqual((result.restore_tokens, result.restore_pages), (768, 12))
        # TP group IDs may differ between peers.
        self.assertNotEqual(result.source.tp_group_id, result.query.target.tp_group_id)
        slots().validate_for(result)
        q = query()
        miss = p.LookupResult(q, None, 256, q.event_block_keys[:4])
        self.assertEqual(miss.restore_tokens, 0)
        for changes in (
            {"source": None},
            {"hit_tokens": 256, "event_block_keys": q.event_block_keys[:4]},
            {"hit_tokens": 192},
            {"hit_tokens": 1088},
            {"hit_tokens": 1023},
            {"event_block_keys": q.event_block_keys[4:]},
            {"event_block_keys": (99,) + q.event_block_keys[1:]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hit(**changes)

    def test_private_slots_mapping(self):
        pages = tuple(reversed(range(4, 16)))
        indices = tuple(
            slot for page in pages for slot in range(page * 64, (page + 1) * 64)
        )
        slots(page_ids=pages, slot_indices=indices).validate_for(hit())
        for changes in (
            {"slot_indices": tuple(range(255, 1023))},
            {"slot_indices": tuple(range(256, 1023))},
            {"slot_indices": tuple(range(256, 1025))},
            {"slot_indices": (257, 256) + tuple(range(258, 1024))},
            {"slot_indices": tuple(range(768)), "page_ids": tuple(range(12))},
            {"page_ids": (4,) * 12},
            {"page_ids": tuple(range(4, 15))},
            {"hit_tokens": 256},
            {"local_hit_tokens": 255},
            {"page_size": 0},
        ):
            with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                slots(**changes)

    def test_slots_ownership(self):
        for transfer in (
            p.TransferId("old-boot", "rid-1", 0),
            p.TransferId("target-boot-1", "other-rid", 0),
            p.TransferId("target-boot-1", "rid-1", 1),
        ):
            with self.subTest(transfer=transfer), self.assertRaises(ValueError):
                slots(transfer_id=transfer).validate_for(hit())
        for q in (
            query(local_hit_tokens=192),
            query(page_size=32, event_block_keys=tuple(range(32))),
        ):
            result = p.LookupResult(q, identity("source"), 1024, q.event_block_keys)
            with self.assertRaises(ValueError):
                slots().validate_for(result)

    def test_prefetch_status_literals(self):
        self.assertEqual(
            {status.value for status in p.PrefetchStatus},
            {
                "PIN_PENDING",
                "COPYING",
                "RELEASING",
                "ACKED",
                "READY",
                "FAILED_SAFE",
                "QUARANTINED",
            },
        )


if __name__ == "__main__":
    unittest.main()
