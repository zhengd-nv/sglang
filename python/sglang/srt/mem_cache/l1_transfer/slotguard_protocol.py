"""Hash-only SlotGuard preparation over data ABI v1.

PREPARED never authorizes DMA by itself. The target must first gather every
required source TP shard, then use its existing drain/unmap/ACK gate.
"""

from dataclasses import dataclass, field
from enum import Enum

from .hash_pin import HASH_CAPABILITY, HASH_VERSION, HashLookupResult
from .protocol import (
    BufferBinding,
    EngineIdentity,
    RegionDescriptor,
    _integer,
    validate_peer,
)
from .slot_guard import LeaseIdentity, PageRef, _text
from .slotguard_limits import MAX_PIN_PAGES
from .wire_codec import MAX_METADATA_BYTES, MAX_REGIONS, MAX_SPANS

CONTROL_VERSION = HASH_VERSION
SLOTGUARD_CAPABILITY = HASH_CAPABILITY


def _version(obj):
    if (
        type(obj.protocol_version) is not int
        or obj.protocol_version != CONTROL_VERSION
        or type(obj.capability) is not str
        or obj.capability != SLOTGUARD_CAPABILITY
    ):
        raise ValueError("unsupported SlotGuard control version/capability")


class PinStatus(str, Enum):
    PREPARED = "PREPARED"
    MISS = "MISS"
    BUSY = "BUSY"
    INCOMPATIBLE = "INCOMPATIBLE"
    STALE_ENGINE = "STALE_ENGINE"


@dataclass(frozen=True)
class PinReference:
    """Exact correlation/release binding, never evidence of source cache coverage."""

    source: EngineIdentity
    target: EngineIdentity
    identity: LeaseIdentity
    content_digest: str
    protocol_version: int = CONTROL_VERSION
    capability: str = SLOTGUARD_CAPABILITY

    def __post_init__(self):
        _version(self)
        if (
            type(self.source) is not EngineIdentity
            or type(self.target) is not EngineIdentity
        ):
            raise ValueError("exact source/target required")
        validate_peer(self.source, self.target)
        if type(self.identity) is not LeaseIdentity or (
            self.identity.source_incarnation != self.source.incarnation
            or self.identity.target_incarnation != self.target.incarnation
        ):
            raise ValueError("reference identity mismatch")
        if (
            type(self.content_digest) is not str
            or len(self.content_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.content_digest)
        ):
            raise ValueError("canonical SHA256 digest required")
        for identity in (self.source, self.target):
            for value in vars(identity).values():
                if type(value) is str:
                    _text("identity", value)


@dataclass(frozen=True)
class PinReq:
    result: HashLookupResult
    identity: LeaseIdentity
    protocol_version: int = CONTROL_VERSION
    capability: str = SLOTGUARD_CAPABILITY
    _reference: PinReference = field(init=False, repr=False, compare=False)
    _wire_size: int = field(init=False, repr=False, compare=False)
    _wire_body: bytes = field(init=False, repr=False, compare=False)

    @property
    def reference(self):
        return self._reference

    def __post_init__(self):
        _version(self)
        if type(self.result) is not HashLookupResult or self.result.source is None:
            raise ValueError("Pin requires exact lookup hit")
        if type(self.identity) is not LeaseIdentity or (
            self.identity.source_incarnation != self.result.source.incarnation
            or self.identity.target_incarnation != self.result.query.target.incarnation
            or self.identity.transfer_id != self.result.transfer_id
        ):
            raise ValueError("Pin lease/query identity mismatch")
        q = self.result.query
        if self.result.restore_pages > MAX_PIN_PAGES:
            raise ValueError("Pin content exceeds bounded control limits")
        for identity in (self.result.source, q.target):
            for value in vars(identity).values():
                if type(value) is str:
                    _text("identity", value)
        # The cached wire body is never re-decoded on encode, so every string
        # it carries must already be within the decoder's bounds.
        _text("rid", self.result.transfer_id.rid)
        for value in (q.cache_salt, q.extra_key):
            if value is not None and (type(value) is not str or len(value) > 4096):
                raise ValueError("namespace exceeds control limit")
        # Compute once from complete validated content, independently on each
        # endpoint. These derived fields are absent from the strict wire schema.
        from .slotguard_control_wire import pin_binding

        digest, body = pin_binding(self)
        object.__setattr__(
            self,
            "_reference",
            PinReference(
                self.result.source,
                q.target,
                self.identity,
                digest,
                self.protocol_version,
                self.capability,
            ),
        )
        object.__setattr__(self, "_wire_size", len(body))
        object.__setattr__(self, "_wire_body", body)


@dataclass(frozen=True)
class PinPreparedResp:
    request: PinReq
    status: PinStatus
    lease_id: str | None = None
    lease_duration_ms: int | None = None
    regions: tuple[RegionDescriptor, ...] = ()
    buffer_bindings: tuple[BufferBinding, ...] = ()
    pages: tuple[PageRef, ...] = ()
    protocol_version: int = CONTROL_VERSION
    capability: str = SLOTGUARD_CAPABILITY
    # Lazy canonical body, populated only by strict outgoing wire validation.
    # Service retention charges this in addition to semantic response metadata.
    _wire_body: bytes | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def source_page_ids(self):
        return tuple(page.page_id for page in self.pages)

    def __post_init__(self):
        _version(self)
        if type(self.request) is not PinReq or type(self.status) is not PinStatus:
            raise ValueError("nominal v2 request/status required")
        if (self.protocol_version, self.capability) != (
            self.request.protocol_version,
            self.request.capability,
        ):
            raise ValueError("response control profile mismatch")
        for name, cls, bound in (
            ("regions", RegionDescriptor, MAX_REGIONS),
            ("buffer_bindings", BufferBinding, MAX_SPANS),
            ("pages", PageRef, MAX_PIN_PAGES),
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(values) > bound
                or any(type(x) is not cls for x in values)
            ):
                raise ValueError("invalid bounded immutable preparation data")
        if self.status is not PinStatus.PREPARED:
            if (
                self.lease_id is not None
                or self.lease_duration_ms is not None
                or self.regions
                or self.buffer_bindings
                or self.pages
            ):
                raise ValueError("unsuccessful preparation cannot expose exports")
            return
        _text("lease_id", self.lease_id)
        _integer("lease_duration_ms", self.lease_duration_ms, 1)
        if (
            len(self.pages) != self.request.result.restore_pages
            or len(set(self.pages)) != len(self.pages)
            or len({page.pool_epoch for page in self.pages}) != 1
            or len(set(self.source_page_ids)) != len(self.pages)
            or any(page.page_id == 0 for page in self.pages)
        ):
            raise ValueError("preparation requires exact distinct physical pages")
        region_ids = {r.region_id for r in self.regions}
        names = {b.buffer_name for b in self.buffer_bindings}
        if (
            not region_ids
            or len(region_ids) != len(self.regions)
            or not names
            or len(names) != len(self.buffer_bindings)
            or any(b.region_id not in region_ids for b in self.buffer_bindings)
            or sum(len(x.descriptor) for x in (*self.regions, *self.buffer_bindings))
            > MAX_METADATA_BYTES
        ):
            raise ValueError("preparation requires bounded unique export bindings")


@dataclass(frozen=True)
class UnpinReq:
    reference: PinReference
    lease_id: str
    accesses_drained: bool
    imports_unmapped: bool
    protocol_version: int = CONTROL_VERSION
    capability: str = SLOTGUARD_CAPABILITY

    def __post_init__(self):
        _version(self)
        if type(self.reference) is not PinReference:
            raise ValueError("exact original v2 Pin reference required")
        if (self.protocol_version, self.capability) != (
            self.reference.protocol_version,
            self.reference.capability,
        ):
            raise ValueError("Unpin control profile mismatch")
        _text("lease_id", self.lease_id)
        if self.accesses_drained is not True or self.imports_unmapped is not True:
            raise ValueError("Unpin requires drained and unmapped proof")


class UnpinStatus(str, Enum):
    RELEASED = "RELEASED"
    ALREADY_RELEASED = "ALREADY_RELEASED"
    BUSY = "BUSY"
    STALE_ENGINE = "STALE_ENGINE"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass(frozen=True)
class UnpinResp:
    request: UnpinReq
    status: UnpinStatus
    protocol_version: int = CONTROL_VERSION
    capability: str = SLOTGUARD_CAPABILITY

    def __post_init__(self):
        _version(self)
        if type(self.request) is not UnpinReq or type(self.status) is not UnpinStatus:
            raise ValueError("nominal v2 Unpin request/status required")
        if (self.protocol_version, self.capability) != (
            self.request.protocol_version,
            self.request.capability,
        ):
            raise ValueError("Unpin response profile mismatch")


def prepared_response(request, *args):
    return PinPreparedResp(
        request,
        *args,
        protocol_version=request.protocol_version,
        capability=request.capability,
    )


def unpin_response(request, status):
    return UnpinResp(request, status, request.protocol_version, request.capability)
