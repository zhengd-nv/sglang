"""Synthetic receipt consistency checks, never native transport evidence."""

import copy
import hashlib
import sys
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slotguard_receipt_audit import validate_slotguard_receipts  # noqa: E402


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(*, worker_progress=False):
    sources, targets, expected = [], [], {"source": [], "target": []}
    for rank in range(4):
        source = dict(engine="p0", tp_group_id="p0-tp4", tp_rank=rank)
        target = dict(engine="p1", tp_group_id="p1-tp4", tp_rank=rank)
        expected["source"].append(source)
        expected["target"].append(target)
        base = dict(
            profile="slotguard-v2", control_version=2, data_abi_version=1, pool_epoch=7
        )
        binding = dict(
            base,
            page_count=1800,
            source_rank=rank,
            target_rank=rank,
            **{
                field: sha(f"rank-{rank}-{field}")
                for field in (
                    "page_set_digest",
                    "content_digest",
                    "identity_digest",
                    "lease_digest",
                )
            },
        )

        def event(phase, seq, context, fields, ticket=True):
            return dict(
                context,
                seq=seq,
                monotonic_ns=seq * 100,
                phase="slotguard_" + phase,
                **({"rid": "native-rid", "attempt": 3} if ticket else {}),
                **fields,
            )

        publication = dict(
            base,
            page_count=1800,
            page_set_digest=binding["page_set_digest"],
            scope_digest=sha("scope"),
            publication_digest=sha(f"publication-{rank}"),
            capture_count=1,
            ready_proof=True,
        )
        sources.append(
            [
                event("publication_ready", 1, source, publication, False),
                event("source_prepared", 2, source, dict(binding, pin_generation=9)),
                event("source_unpin_released", 3, source, binding),
            ]
        )
        phases = (
            "target_pin_validated",
            "group_copy",
            "target_dma_drained",
            "target_imports_unmapped",
            "target_unpin_ack",
            "group_ready",
        )
        targets.append(
            [
                event(
                    phase,
                    index + 1,
                    target,
                    dict(
                        binding,
                        **(
                            {"ranks": 4, "decision": "COPY" if index == 1 else "READY"}
                            if phase in ("group_copy", "group_ready")
                            else {}
                        ),
                    ),
                )
                for index, phase in enumerate(phases)
            ]
        )
        if worker_progress:
            targets[-1][1]["phase"] = "slotguard_rank_copy"
            targets[-1][1].pop("decision")
            targets[-1][1].pop("ranks")
            targets[-1][1].update(progress_mode="worker", initial_grant="PIN")
            for row in targets[-1]:
                row["seq"] += 2
                row["monotonic_ns"] = row["seq"] * 100
            targets[-1][:0] = [
                dict(
                    target,
                    phase="group_permit",
                    seq=1,
                    monotonic_ns=100,
                    rid="native-rid",
                    attempt=3,
                    status="PIN",
                ),
                dict(
                    target,
                    phase="pin_start",
                    seq=2,
                    monotonic_ns=200,
                    rid="native-rid",
                    attempt=3,
                ),
            ]
    return sources, targets, expected


def audit(sources, targets, expected, **overrides):
    options = dict(rid="native-rid", attempt=3, source_pool_epoch=7, page_count=1800)
    options.update(overrides)
    return validate_slotguard_receipts(sources, targets, expected=expected, **options)


class ReceiptAuditTests(CustomTestCase):
    def test_explicit_hash_v3_chain_rejects_v2_and_mixed_publication(self):
        sources, targets, expected = fixture(worker_progress=True)
        for trace in sources + targets:
            for row in trace:
                if row["phase"].startswith("slotguard_"):
                    row.update(profile="slotguard-hash-v3", control_version=3)
        self.assertEqual(
            audit(sources, targets, expected, profile="slotguard-hash-v3")[
                "progress_mode"
            ],
            "worker",
        )
        with self.assertRaises(AssertionError):
            audit(sources, targets, expected)
        sources[0][0].update(profile="slotguard-v2", control_version=2)
        with self.assertRaises(AssertionError):
            audit(sources, targets, expected, profile="slotguard-hash-v3")

    def test_worker_mode_requires_full_same_mode_chain_and_initial_pin_grant(self):
        sources, targets, expected = fixture(worker_progress=True)
        self.assertEqual(audit(sources, targets, expected)["progress_mode"], "worker")
        for mutation in (
            "missing_grant",
            "late_grant",
            "missing_ack",
            "mixed_mode",
            "forged_copy",
            "wrong_ticket",
            "duplicate_grant",
        ):
            sources, targets, expected = fixture(worker_progress=True)
            rows = targets[3]
            if mutation == "missing_grant":
                rows.pop(0)
            elif mutation == "late_grant":
                rows[0]["phase"], rows[1]["phase"] = rows[1]["phase"], rows[0]["phase"]
                rows[1]["status"] = "PIN"
            elif mutation == "missing_ack":
                rows.pop(-2)
            elif mutation == "wrong_ticket":
                rows[0]["rid"] = "another"
            elif mutation == "duplicate_grant":
                rows[1].update(phase="group_permit", status="PIN")
            else:
                rows[3].update(phase="slotguard_group_copy", decision="COPY", ranks=4)
                rows[3].pop("progress_mode")
                rows[3].pop("initial_grant")
                if mutation == "forged_copy":
                    rows[4].update(
                        phase="slotguard_rank_copy",
                        progress_mode="worker",
                        initial_grant="PIN",
                    )
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                audit(sources, targets, expected)

    def test_same_rank_complete_binding_not_cross_rank_physical_equality(self):
        sources, targets, expected = fixture()
        result = audit(sources, targets, expected)
        self.assertEqual(result["kind"], "RECEIPT_CHAIN_VALID_ONLY")
        self.assertEqual(len(result["ranks"]), 4)
        self.assertEqual(len({row["page_set_digest"] for row in result["ranks"]}), 4)

    def test_missing_duplicate_out_of_order_phase_and_source_release_fail(self):
        for which, rank, operation in (
            ("target", 3, "missing"),
            ("target", 0, "duplicate"),
            ("target", 2, "order"),
            ("source", 1, "missing"),
            ("source", 0, "duplicate"),
        ):
            sources, targets, expected = fixture()
            rows = targets[rank] if which == "target" else sources[rank]
            if operation == "missing":
                rows.pop()
            elif operation == "duplicate":
                rows.append(copy.deepcopy(rows[-1]))
            else:
                rows[1], rows[2] = rows[2], rows[1]
            with self.subTest(which=which, rank=rank, operation=operation):
                with self.assertRaises(AssertionError):
                    audit(sources, targets, expected)

    def test_tampered_binding_schema_versions_identity_count_and_generation_fail(self):
        mutations = {
            "page_set_digest": sha("changed"),
            "content_digest": sha("changed"),
            "identity_digest": sha("changed"),
            "lease_digest": sha("changed"),
            "pool_epoch": 8,
            "page_count": 1799,
            "source_rank": 0,
            "target_rank": 0,
            "engine": "other",
            "tp_group_id": "other",
            "control_version": True,
            "data_abi_version": True,
            "profile": "v1",
            "seq": True,
            "monotonic_ns": 0,
            "extra": "not allowed",
        }
        for field, value in mutations.items():
            sources, targets, expected = fixture()
            targets[3][2][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit(sources, targets, expected)
        sources, targets, expected = fixture()
        sources[2][1]["pin_generation"] = True
        with self.assertRaises(AssertionError):
            audit(sources, targets, expected)

    def test_unready_publication_or_wrong_permit_cannot_pass(self):
        for field, value in (("ready_proof", False), ("pool_epoch", 6)):
            sources, targets, expected = fixture()
            sources[0][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit(sources, targets, expected)
        for field, value in (("decision", "READY"), ("ranks", 3)):
            sources, targets, expected = fixture()
            targets[0][1][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                audit(sources, targets, expected)

    def test_caller_context_and_bounds_cannot_select_a_successful_subset(self):
        sources, targets, expected = fixture()
        for overrides in (
            {"rid": "other"},
            {"attempt": True},
            {"source_pool_epoch": 8},
            {"page_count": True},
            {"page_count": 0},
            {"page_count": 4097},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(AssertionError):
                audit(sources, targets, expected, **overrides)
        with self.assertRaises(AssertionError):
            audit(sources, targets[:3], expected)
        expected["target"][3] = expected["target"][2]
        with self.assertRaises(AssertionError):
            audit(sources, targets, expected)

    def test_clocks_are_only_ordered_locally_not_compared_between_hosts(self):
        sources, targets, expected = fixture()
        for rows in sources:
            for row in rows:
                row["monotonic_ns"] += 10**12
        self.assertEqual(
            audit(sources, targets, expected)["kind"], "RECEIPT_CHAIN_VALID_ONLY"
        )

    def test_unknown_phase_and_late_publication_fail(self):
        sources, targets, expected = fixture()
        targets[0][0]["phase"] = "slotguard_invented_ok"
        with self.assertRaises(AssertionError):
            audit(sources, targets, expected)
        sources, targets, expected = fixture()
        publication = sources[0].pop(0)
        publication.update(seq=4, monotonic_ns=400)
        sources[0].append(publication)
        with self.assertRaises(AssertionError):
            audit(sources, targets, expected)

    def test_malformed_expected_group_ids_are_rejected_before_set_construction(self):
        for value in (None, True, {}, []):
            sources, targets, expected = fixture()
            expected["source"][0]["tp_group_id"] = value
            with self.subTest(value=value), self.assertRaises(AssertionError):
                audit(sources, targets, expected)


if __name__ == "__main__":
    unittest.main()
