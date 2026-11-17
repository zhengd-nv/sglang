"""Bounded same-rank SlotGuard receipt consistency, not native acceptance.

Caller must independently validate complete trace windows, current engine/process
provenance, restore coverage, byte correctness and true OFF. This module cannot
derive physical pages from digests or prove that a publication is the requested
prefix. The source PREPARED authority remains the reviewed catalog/guard code.
Clocks are not compared across machines; physical PageRef digests are compared
between source and target for each rank, never between different ranks.
"""

import re

SLOTGUARD_PROFILES = ("slotguard-v2", "slotguard-hash-v3")

_META = {"seq", "monotonic_ns", "engine", "phase", "tp_rank", "tp_group_id"}
_BASE = {"profile", "control_version", "data_abi_version", "pool_epoch"}
_LEASE = _BASE | {
    "page_count",
    "page_set_digest",
    "content_digest",
    "identity_digest",
    "lease_digest",
    "source_rank",
    "target_rank",
}
_PUBLICATION = _BASE | {
    "page_count",
    "page_set_digest",
    "scope_digest",
    "publication_digest",
    "capture_count",
    "ready_proof",
}
_TARGET = (
    "target_pin_validated",
    "group_copy",
    "target_dma_drained",
    "target_imports_unmapped",
    "target_unpin_ack",
    "group_ready",
)
_WORKER_TARGET = tuple(
    "rank_copy" if phase == "group_copy" else phase for phase in _TARGET
)
_PHASES = {
    "publication_ready": _PUBLICATION,
    "source_prepared": _LEASE | {"pin_generation"},
    "source_unpin_released": _LEASE,
    **{phase: _LEASE for phase in _TARGET},
    "group_copy": _LEASE | {"decision", "ranks"},
    "rank_copy": _LEASE | {"progress_mode", "initial_grant"},
    "group_ready": _LEASE | {"decision", "ranks"},
}
_BINDING = _LEASE | {"rid", "attempt"}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value < 2**63


def _text(value, maximum=128):
    return type(value) is str and 0 < len(value) <= maximum


def _trace(
    rows, context, rank, *, runtime_source_identity=None, profile="slotguard-v2"
):
    from glm_long_context_runtime import receipt_page_limit

    max_pages = receipt_page_limit(runtime_source_identity)
    _require(
        type(context) is dict
        and set(context) == {"engine", "tp_group_id", "tp_rank"}
        and _text(context["engine"])
        and _text(context["tp_group_id"])
        and type(context["tp_rank"]) is int
        and context["tp_rank"] == rank,
        "receipt context must contain the exact ordered TP4 rank",
    )
    _require(type(rows) is list and len(rows) <= 131072, "bounded trace required")
    selected, previous_seq, previous_time = [], 0, 0
    for row in rows:
        _require(type(row) is dict, "invalid trace row")
        _require(
            all(row.get(key) == value for key, value in context.items())
            and type(row.get("tp_rank")) is int
            and _integer(row.get("seq"), 1)
            and row["seq"] > previous_seq
            and _integer(row.get("monotonic_ns"), 1)
            and row["monotonic_ns"] >= previous_time
            and _text(row.get("phase")),
            "trace identity or local ordering changed",
        )
        previous_seq, previous_time = row["seq"], row["monotonic_ns"]
        if not row["phase"].startswith("slotguard_"):
            continue
        phase = row["phase"][len("slotguard_") :]
        _require(phase in _PHASES, "unknown SlotGuard receipt phase")
        ticket = set() if phase == "publication_ready" else {"rid", "attempt"}
        _require(set(row) == _META | _PHASES[phase] | ticket, "receipt schema changed")
        _require(
            row["profile"] == profile
            and type(row["control_version"]) is int
            and row["control_version"] == (3 if profile == "slotguard-hash-v3" else 2)
            and type(row["data_abi_version"]) is int
            and row["data_abi_version"] == 1,
            "receipt control/data version mismatch",
        )
        for key in _PHASES[phase]:
            value = row[key]
            if key.endswith("_digest"):
                valid = type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
            elif key in (
                "profile",
                "decision",
                "ready_proof",
                "progress_mode",
                "initial_grant",
            ):
                continue
            else:
                valid = _integer(value)
            _require(valid, "invalid bounded receipt scalar")
        _require(1 <= row["page_count"] <= max_pages, "invalid receipt page count")
        if phase == "publication_ready":
            _require(
                row["ready_proof"] is True and row["capture_count"] == 1,
                "publication is not producer-ready",
            )
        else:
            _require(
                _text(row["rid"], 256)
                and _integer(row["attempt"])
                and row["source_rank"] == row["target_rank"] == rank,
                "receipt transfer/rank binding changed",
            )
        if phase in ("group_copy", "group_ready"):
            _require(
                row["ranks"] == 4
                and row["decision"] == ("COPY" if phase == "group_copy" else "READY"),
                "receipt is not an all-four permit",
            )
        if phase == "rank_copy":
            _require(
                row["progress_mode"] == "worker" and row["initial_grant"] == "PIN",
                "invalid rank-local transaction authority",
            )
        selected.append((phase, row))
    return selected


def validate_slotguard_receipts(
    sources,
    targets,
    *,
    expected,
    rid,
    attempt,
    source_pool_epoch,
    page_count,
    runtime_source_identity=None,
    profile="slotguard-v2",
):
    """Require all four complete same-rank chains for one already-audited RID.

    READY alone, an arbitrary 'OK' rename, or three successful ranks is never
    sufficient. Caller supplies attested identities and exact restored page
    count; expected is not inferred from whichever events happened to succeed.
    """
    from glm_long_context_runtime import receipt_page_limit

    max_pages = receipt_page_limit(runtime_source_identity)
    _require(
        profile in SLOTGUARD_PROFILES, "explicit SlotGuard receipt profile required"
    )
    _require(
        type(sources) is list
        and type(targets) is list
        and len(sources) == len(targets) == 4
        and type(expected) is dict
        and set(expected) == {"source", "target"}
        and all(type(rows) is list and len(rows) == 4 for rows in expected.values()),
        "all four source and target ranks required",
    )
    _require(
        _text(rid, 256)
        and _integer(attempt)
        and _integer(source_pool_epoch)
        and _integer(page_count, 1)
        and page_count <= max_pages,
        "invalid transfer bounds",
    )
    for side in ("source", "target"):
        _require(
            all(
                type(context) is dict and _text(context.get("tp_group_id"))
                for context in expected[side]
            )
            and len({context.get("tp_group_id") for context in expected[side]}) == 1,
            "each side must name one complete TP4 group",
        )
    _require(
        expected["source"][0].get("tp_group_id")
        != expected["target"][0].get("tp_group_id"),
        "source and target groups must differ",
    )
    result, modes = [], set()
    for rank in range(4):
        source = _trace(
            sources[rank],
            expected["source"][rank],
            rank,
            runtime_source_identity=runtime_source_identity,
            profile=profile,
        )
        target = _trace(
            targets[rank],
            expected["target"][rank],
            rank,
            runtime_source_identity=runtime_source_identity,
            profile=profile,
        )

        def related(rows):
            return [
                (phase, event)
                for phase, event in rows
                if event.get("rid") == rid and event.get("attempt") == attempt
            ]

        s, t = related(source), related(target)
        worker = any(phase == "rank_copy" for phase, _ in t)
        modes.add("worker" if worker else "scheduler")
        _require(
            [phase for phase, _ in s] == ["source_prepared", "source_unpin_released"]
            and [phase for phase, _ in t]
            == list(_WORKER_TARGET if worker else _TARGET),
            "missing, duplicate, or reordered receipt chain",
        )
        if worker:
            # Ordinary admission evidence is the initial common transaction
            # authority. Bind the same rank/RID/attempt before actual Pin RPC;
            # a constant string inside rank_copy alone proves no common grant.
            ordinary = [
                row
                for row in targets[rank]
                if row.get("rid") == rid and row.get("attempt") == attempt
            ]
            grants = [row for row in ordinary if row["phase"] == "group_permit"]
            starts = [row for row in ordinary if row["phase"] == "pin_start"]
            _require(
                len(grants) == len(starts) == 1
                and grants[0].get("status") == "PIN"
                and grants[0]["seq"] < starts[0]["seq"] < t[0][1]["seq"],
                "missing or reordered initial common PIN grant",
            )
        prepared = s[0][1]
        _require(
            prepared["pool_epoch"] == source_pool_epoch
            and prepared["page_count"] == page_count,
            "restored page coverage or source epoch differs",
        )
        binding = {key: prepared[key] for key in _BINDING}
        _require(
            all(
                {key: event[key] for key in _BINDING} == binding
                for _, event in [*s, *t]
            ),
            "same-rank lease/page/content identity differs",
        )
        _require(
            any(
                phase == "publication_ready"
                and event["pool_epoch"] == source_pool_epoch
                and event["seq"] < prepared["seq"]
                for phase, event in source
            ),
            "source has no preceding producer-ready publication in this epoch",
        )
        result.append(
            dict(
                rank=rank,
                pin_generation=prepared["pin_generation"],
                **{key: prepared[key] for key in _LEASE},
            )
        )
    _require(len(modes) == 1, "mixed rank progress modes")
    proof = dict(kind="RECEIPT_CHAIN_VALID_ONLY", ranks=result)
    if modes == {"worker"}:
        proof["progress_mode"] = "worker"
    return proof
