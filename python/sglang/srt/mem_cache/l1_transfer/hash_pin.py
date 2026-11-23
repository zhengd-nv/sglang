"""Compact native prefix proof; physical ownership is still acquired by SlotGuard."""

import hashlib
import struct
from dataclasses import dataclass

from .protocol import EngineIdentity, TransferId, _integer, validate_peer
from .slotguard_limits import MAX_PIN_PAGES, MAX_PREFIX_TOKENS

HASH_VERSION = 3
HASH_CAPABILITY = "slotguard-hash-prepared-v3"


def validate_hash(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("canonical full SHA256 prefix required")


def page_hash(tokens, prior=None):
    """Native ROOT hash: prior digest followed by little-endian u32 atoms.

    Publication worker only. This verifier never runs in the Pin lookup path.
    """
    digest = hashlib.sha256()
    if prior is not None:
        validate_hash(prior)
        digest.update(bytes.fromhex(prior))
    if any(type(token) is not int or not 0 <= token < 1 << 32 for token in tokens):
        raise ValueError("native ROOT token atom must fit u32")
    digest.update(struct.pack(f"<{len(tokens)}I", *tokens))
    return digest.hexdigest()


@dataclass(frozen=True)
class HashLookupResult:
    source: EngineIdentity
    target: EngineIdentity
    transfer_id: TransferId
    local_hit_tokens: int
    hit_tokens: int
    page_size: int
    prefix_hash: str
    cache_salt: str | None = None
    extra_key: str | None = None

    def __post_init__(self):
        if (
            type(self.source) is not EngineIdentity
            or type(self.target) is not EngineIdentity
        ):
            raise ValueError("exact hash Pin peers required")
        validate_peer(self.source, self.target)
        if (
            type(self.transfer_id) is not TransferId
            or self.transfer_id.target_incarnation != self.target.incarnation
        ):
            raise ValueError("exact target transfer required")
        for name, minimum in (
            ("local_hit_tokens", 0),
            ("hit_tokens", 1),
            ("page_size", 1),
        ):
            _integer(name, getattr(self, name), minimum)
        if (
            not self.local_hit_tokens < self.hit_tokens <= MAX_PREFIX_TOKENS
            or self.local_hit_tokens % self.page_size
            or self.hit_tokens % self.page_size
            or self.restore_pages > MAX_PIN_PAGES
        ):
            raise ValueError("bounded aligned hash Pin interval required")
        if self.cache_salt is not None or self.extra_key is not None:
            raise ValueError("hash Pin V3 supports only unnamespaced native ROOT")
        validate_hash(self.prefix_hash)

    @property
    def query(self):
        # Shared export decoder needs only exact peer, interval and page size.
        return self

    @property
    def restore_tokens(self):
        return self.hit_tokens - self.local_hit_tokens

    @property
    def restore_pages(self):
        return self.restore_tokens // self.page_size

    @classmethod
    def from_lookup(cls, result):
        query = result.query
        if not query.prefix_hashes:
            raise ValueError("hash Pin requires discovery's native full-prefix hashes")
        return cls(
            result.source,
            query.target,
            result.transfer_id,
            query.local_hit_tokens,
            result.hit_tokens,
            query.page_size,
            query.prefix_hashes[result.hit_tokens // query.page_size - 1],
            query.cache_salt,
            query.extra_key,
        )
