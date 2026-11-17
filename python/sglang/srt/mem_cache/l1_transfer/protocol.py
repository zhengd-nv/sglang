"""Immutable peer-L1 contracts; no network, allocator, or transport implementation.

Event keys are opaque signed int64 ROOT event hashes supplied by the caller, never
storage hashes. The source must authoritatively match tokens AND event keys before
pinning. A lease timeout never permits unlocking memory. READY is legal only after
DMA has drained, imports have been unmapped, and an unpin ACK has arrived. Unknown
DMA completion requires QUARANTINED, retaining allocations and pins for recovery.
"""

from dataclasses import dataclass, field
from enum import Enum

from .int_vectors import is_int64_tuple

PROTOCOL_VERSION = 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= INT64_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {INT64_MAX}]")


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _version(value: int) -> None:
    _integer("protocol_version", value)
    if value != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")


def _typed(name: str, value: object, cls: type) -> None:
    if not isinstance(value, cls):
        raise ValueError(f"{name} must be {cls.__name__}")  # noqa: TRY004 -- stable wire validation contract


def _integers(name: str, values: tuple[int, ...], minimum: int = 0) -> None:
    if type(values) is not tuple:
        raise ValueError(f"{name} must be an immutable tuple")
    if not is_int64_tuple(values, minimum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {INT64_MAX}]")


def _interval(local_hit_tokens: int, hit_tokens: int, page_size: int) -> None:
    _integer("page_size", page_size, 1)
    _integer("local_hit_tokens", local_hit_tokens)
    _integer("hit_tokens", hit_tokens)
    if local_hit_tokens % page_size or hit_tokens % page_size:
        raise ValueError("prefix boundaries must be page aligned")
    if hit_tokens < local_hit_tokens:
        raise ValueError("hit_tokens cannot precede local_hit_tokens")


@dataclass(frozen=True)
class EngineIdentity:
    engine_id: str
    incarnation: str
    domain_id: str
    instance_id: str
    model_id: str
    layout_digest: str
    tp_group_id: str
    tp_size: int
    tp_rank: int
    shard_id: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        for name in (
            "engine_id",
            "incarnation",
            "domain_id",
            "instance_id",
            "model_id",
            "layout_digest",
            "tp_group_id",
            "shard_id",
        ):
            _text(name, getattr(self, name))
        _integer("tp_size", self.tp_size, 1)
        _integer("tp_rank", self.tp_rank)
        if self.tp_rank >= self.tp_size:
            raise ValueError("tp_rank must be smaller than tp_size")


def validate_peer(source: EngineIdentity, target: EngineIdentity) -> None:
    """Require one distinct compatible engine; TP group IDs may differ."""
    _typed("source", source, EngineIdentity)
    _typed("target", target, EngineIdentity)
    if source.engine_id == target.engine_id:
        raise ValueError("source and target must be distinct engines")
    for name in (
        "domain_id",
        "instance_id",
        "model_id",
        "layout_digest",
        "tp_size",
        "tp_rank",
        "shard_id",
    ):
        if getattr(source, name) != getattr(target, name):
            raise ValueError(f"incompatible peer {name}")


def group_identity(identity: EngineIdentity) -> tuple:
    """Rank-independent group binding; rank layout/shard validation stays exact."""
    return tuple(
        getattr(identity, name)
        for name in (
            "engine_id",
            "incarnation",
            "domain_id",
            "instance_id",
            "model_id",
            "tp_group_id",
            "tp_size",
            "protocol_version",
        )
    )


@dataclass(frozen=True)
class TransferId:
    target_incarnation: str
    rid: str
    attempt: int
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _text("target_incarnation", self.target_incarnation)
        _text("rid", self.rid)
        _integer("attempt", self.attempt)


@dataclass(frozen=True)
class LookupQuery:
    target: EngineIdentity
    transfer_id: TransferId
    event_block_keys: tuple[int, ...]
    prefix_token_ids: tuple[int, ...]
    local_hit_tokens: int
    max_restore_tokens: int
    page_size: int
    cache_salt: str | None = None
    extra_key: str | None = None
    protocol_version: int = PROTOCOL_VERSION
    # Discovery-only native full hashes. Never included in the V1/V2 wire ABI.
    prefix_hashes: tuple[str, ...] = field(
        default=(), repr=False, compare=False, kw_only=True
    )

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _typed("target", self.target, EngineIdentity)
        _typed("transfer_id", self.transfer_id, TransferId)
        if self.target.incarnation != self.transfer_id.target_incarnation:
            raise ValueError("transfer_id has a stale target incarnation")
        _integers("event_block_keys", self.event_block_keys, INT64_MIN)
        _integers("prefix_token_ids", self.prefix_token_ids)
        _interval(self.local_hit_tokens, self.max_restore_tokens, self.page_size)
        if self.max_restore_tokens >= len(self.prefix_token_ids):
            raise ValueError("max_restore_tokens must reserve the last prompt token")
        if len(self.event_block_keys) != len(self.prefix_token_ids) // self.page_size:
            raise ValueError("event_block_keys must cover all complete ROOT pages")
        if type(self.prefix_hashes) is not tuple or (
            self.prefix_hashes
            and (
                len(self.prefix_hashes) != len(self.event_block_keys)
                or any(type(h) is not str or len(h) != 64 for h in self.prefix_hashes)
            )
        ):
            raise ValueError("invalid local full-prefix hash metadata")
        for name in ("cache_salt", "extra_key"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")


@dataclass(frozen=True)
class LookupResult:
    """One source and an exact [L, H) interval, or source=None and H=L."""

    query: LookupQuery
    source: EngineIdentity | None
    hit_tokens: int
    event_block_keys: tuple[int, ...]
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _typed("query", self.query, LookupQuery)
        _interval(self.query.local_hit_tokens, self.hit_tokens, self.query.page_size)
        _integers("event_block_keys", self.event_block_keys, INT64_MIN)
        if self.hit_tokens > self.query.max_restore_tokens:
            raise ValueError("hit exceeds caller's maximum restorable prefix")
        expected = self.query.event_block_keys[
            : self.hit_tokens // self.query.page_size
        ]
        if self.event_block_keys != expected:
            raise ValueError("result must contain the exact ROOT keys through H")
        if self.source is None:
            if self.restore_tokens:
                raise ValueError("miss must have H=L")
        else:
            validate_peer(self.source, self.query.target)
            if not self.restore_tokens:
                raise ValueError("hit must have H>L")

    @property
    def transfer_id(self) -> TransferId:
        return self.query.transfer_id

    @property
    def restore_tokens(self) -> int:
        return max(0, self.hit_tokens - self.query.local_hit_tokens)

    @property
    def restore_pages(self) -> int:
        return self.restore_tokens // self.query.page_size


@dataclass(frozen=True)
class PrivateSlotView:
    """Materialized Scheduler-owned destination map, never an allocator handle.

    The backend cannot allocate, free, or evict these slots. Each page must be
    complete and contiguous internally; different pages need not be adjacent.
    """

    transfer_id: TransferId
    local_hit_tokens: int
    hit_tokens: int
    page_size: int
    slot_indices: tuple[int, ...]
    page_ids: tuple[int, ...]
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _typed("transfer_id", self.transfer_id, TransferId)
        _interval(self.local_hit_tokens, self.hit_tokens, self.page_size)
        _integers("slot_indices", self.slot_indices, 1)
        _integers("page_ids", self.page_ids, 1)
        count = self.hit_tokens - self.local_hit_tokens
        if count == 0 or len(self.slot_indices) != count:
            raise ValueError("private slots must cover exactly nonempty [L,H)")
        if len(self.page_ids) != count // self.page_size:
            raise ValueError("private page count does not match [L,H)")
        if len(set(self.page_ids)) != len(self.page_ids):
            raise ValueError("private pages must not alias")
        for i, page in enumerate(self.page_ids):
            actual = self.slot_indices[i * self.page_size : (i + 1) * self.page_size]
            if actual != tuple(
                range(page * self.page_size, (page + 1) * self.page_size)
            ):
                raise ValueError("slots must match complete, contiguous physical pages")

    def validate_for(self, result: LookupResult) -> None:
        _typed("result", result, LookupResult)
        if result.source is None or (
            self.transfer_id != result.transfer_id
            or self.local_hit_tokens != result.query.local_hit_tokens
            or self.hit_tokens != result.hit_tokens
            or self.page_size != result.query.page_size
        ):
            raise ValueError("private slots do not belong to this transfer interval")


@dataclass(frozen=True)
class RegionDescriptor:
    region_id: str
    descriptor: bytes

    def __post_init__(self) -> None:
        _text("region_id", self.region_id)
        if type(self.descriptor) is not bytes or not self.descriptor:
            raise ValueError("region descriptor must be nonempty immutable bytes")


@dataclass(frozen=True)
class BufferBinding:
    buffer_name: str
    region_id: str
    descriptor: bytes

    def __post_init__(self) -> None:
        _text("buffer_name", self.buffer_name)
        _text("region_id", self.region_id)
        if type(self.descriptor) is not bytes or not self.descriptor:
            raise ValueError("buffer binding must be nonempty immutable bytes")


class PrefetchStatus(str, Enum):
    PIN_PENDING = "PIN_PENDING"
    COPYING = "COPYING"
    RELEASING = "RELEASING"
    ACKED = "ACKED"  # TP-local drain/unmap/Unpin ACK; not group-ready or adoptable
    READY = "READY"
    FAILED_SAFE = "FAILED_SAFE"
    QUARANTINED = "QUARANTINED"
