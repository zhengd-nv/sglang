"""Finite scalar receipt vocabulary; digest construction is worker-only.

An ordered PageRef digest covers pool epoch, physical ID and every generation.
Rank-specific digests are deliberately *not* a TP consensus value. The existing
common TransferId/H/layout-group decision remains the all-rank authority. Default
mode records COPY; worker progress records a rank-local exact Pin proof under
the initial common PIN transaction grant. Receipts never authorize progress.
"""

import hashlib
import json
import struct
from types import MappingProxyType

from .slotguard_control_wire import reference_bytes
from .slotguard_limits import MAX_PIN_PAGES

_BASE = frozenset({"profile", "control_version", "data_abi_version", "pool_epoch"})
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
PHASE_FIELDS = {
    "publication_ready": _PUBLICATION,
    "source_prepared": _LEASE | {"pin_generation"},
    "source_unpin_released": _LEASE,
    "target_pin_validated": _LEASE,
    "group_copy": _LEASE | {"decision", "ranks"},
    "rank_copy": _LEASE | {"progress_mode", "initial_grant"},
    "target_dma_drained": _LEASE,
    "target_imports_unmapped": _LEASE,
    "target_unpin_ack": _LEASE,
    "group_ready": _LEASE | {"decision", "ranks"},
}
_DIGESTS = frozenset(
    {
        "page_set_digest",
        "content_digest",
        "identity_digest",
        "lease_digest",
        "scope_digest",
        "publication_digest",
    }
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def ordered_page_digest(pages):
    """Complete, ordered physical binding; never invoke from Scheduler."""
    digest = hashlib.sha256()
    for page in pages:
        digest.update(
            struct.pack("!QQQ", page.pool_epoch, page.page_id, page.generation)
        )
    return digest.hexdigest()


def _base(epoch):
    return {
        "profile": "slotguard-hash-v3",
        "control_version": 3,
        "data_abi_version": 1,
        "pool_epoch": epoch,
    }


def lease_receipt(reference, lease_id, pages):
    """Precompute once off owner, then pass only small immutable CPU scalars.

    PinReference.content_digest already hashes the complete immutable Pin body.
    The identity digest includes exact source/target identities and nonce via
    reference_bytes, and the lease digest binds the exact opaque release ID.
    """
    return MappingProxyType(
        dict(
            **_base(pages[0].pool_epoch),
            page_count=len(pages),
            page_set_digest=ordered_page_digest(pages),
            content_digest=reference.content_digest,
            identity_digest=hashlib.sha256(reference_bytes(reference)).hexdigest(),
            lease_digest=_digest(lease_id),
            source_rank=reference.source.tp_rank,
            target_rank=reference.target.tp_rank,
        )
    )


def publication_receipt(publication):
    scope = publication.scope
    scope_digest = _digest(
        (
            scope.model_id,
            scope.layout_digest,
            scope.page_size,
            scope.pool_epoch,
            scope.cache_salt,
            scope.extra_key,
        )
    )
    pages = ordered_page_digest(publication.pages)
    return MappingProxyType(
        dict(
            **_base(scope.pool_epoch),
            page_count=len(publication.pages),
            page_set_digest=pages,
            scope_digest=scope_digest,
            capture_count=1,
            publication_digest=_digest(
                (scope_digest, publication.parent_cid, publication.page_tokens, pages)
            ),
            ready_proof=True,
        )
    )


def valid_receipt(phase, fields, ticket):
    if type(phase) is not str or len(phase) > 32 or phase not in PHASE_FIELDS:
        return False
    if fields.keys() != PHASE_FIELDS[phase]:
        return False
    if any(type(value) not in (str, int, bool) for value in fields.values()):
        return False
    if phase == "publication_ready":
        if ticket is not None or fields["ready_proof"] is not True:
            return False
    else:
        from .protocol import TransferId

        if type(ticket) is not TransferId or len(ticket.rid) > 256:
            return False
    if (
        type(fields["profile"]) is not str
        or (fields["profile"], fields["control_version"]) != ("slotguard-hash-v3", 3)
        or fields["data_abi_version"] != 1
    ):
        return False
    for key, value in fields.items():
        if key in _DIGESTS:
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                return False
        elif key in (
            "profile",
            "decision",
            "ready_proof",
            "progress_mode",
            "initial_grant",
        ):
            continue
        elif type(value) is not int or not 0 <= value < 2**63:
            return False
    if not 1 <= fields["page_count"] <= MAX_PIN_PAGES:
        return False
    if phase == "publication_ready":
        return fields["capture_count"] == 1
    if not (0 <= fields["source_rank"] < 4 and 0 <= fields["target_rank"] < 4):
        return False
    if phase == "rank_copy":
        return fields["progress_mode"] == "worker" and fields["initial_grant"] == "PIN"
    return phase not in ("group_copy", "group_ready") or (
        type(fields["decision"]) is str
        and fields["ranks"] == 4
        and fields["decision"] == ("COPY" if phase == "group_copy" else "READY")
    )


def emit_receipt(observer, phase, ticket, receipt, **fields):
    if observer is not None and receipt is not None:
        observer.emit_slotguard(phase, ticket, **receipt, **fields)
