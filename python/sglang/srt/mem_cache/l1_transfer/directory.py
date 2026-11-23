"""Static exact KVCM reporter-to-engine directory; no inferred identities."""

from dataclasses import dataclass
from urllib.parse import urlsplit

from .protocol import EngineIdentity, _text, _typed, group_identity, validate_peer


@dataclass(frozen=True)
class EngineTPGroup:
    """One exact TP0 reporter hint and four authoritative rank bindings.

    The reporter does not assert rank readability. Each source Scheduler must
    independently validate its shard before the group grants a Pin response.
    """

    reporter_id: str
    ranks: tuple["EngineDirectoryEntry", ...]

    def __post_init__(self):
        _text("reporter_id", self.reporter_id)
        if len(self.reporter_id) > 4096:
            raise ValueError("group reporter exceeds metadata bound")
        if type(self.ranks) is not tuple or len(self.ranks) != 4:
            raise ValueError("engine group must contain exactly four ordered ranks")
        groups, endpoints = set(), set()
        for rank, entry in enumerate(self.ranks):
            _typed("rank", entry, EngineDirectoryEntry)
            if entry.identity.tp_size != 4 or entry.identity.tp_rank != rank:
                raise ValueError("engine group rank mapping must be exact TP4")
            if entry.reporter_id != self.reporter_id:
                raise ValueError("all ranks must bind the exact TP0 reporter hint")
            if any(
                len(value) > (4096 if name == "model_id" else 256)
                for name, value in vars(entry.identity).items()
                if isinstance(value, str)
            ):
                raise ValueError("group identity exceeds bounded control metadata")
            groups.add(group_identity(entry.identity))
            endpoint = urlsplit(entry.control_endpoint)
            endpoints.add((endpoint.scheme, endpoint.hostname, endpoint.port))
        if len(groups) != 1 or len(endpoints) != 4:
            raise ValueError("mixed group boot identity or duplicate control endpoint")

    def directory(self, peers, rank):
        if type(rank) is not int or not 0 <= rank < 4:
            raise ValueError("directory selection requires exact TP rank in [0,4)")
        target = self.ranks[rank].identity
        for peer in peers:
            _typed("peer", peer, EngineTPGroup)
            for own, other in zip(self.ranks, peer.ranks):
                validate_peer(other.identity, own.identity)
        return EngineDirectory(target, tuple(peer.ranks[rank] for peer in peers))


@dataclass(frozen=True)
class EngineDirectoryEntry:
    """reporter_id is exact KVCM hosts[].host_ip_port / KV_INDEXER_WORKER_ID.

    It is independent of the control endpoint and is never parsed as a URI.
    """

    reporter_id: str
    identity: EngineIdentity
    control_endpoint: str

    def __post_init__(self) -> None:
        _text("reporter_id", self.reporter_id)
        _typed("identity", self.identity, EngineIdentity)
        _text("control_endpoint", self.control_endpoint)
        endpoint = urlsplit(self.control_endpoint)
        if (
            self.control_endpoint != self.control_endpoint.strip()
            or any(
                ch.isspace() or ord(ch) < 32 or ord(ch) == 127
                for ch in self.control_endpoint
            )
            or endpoint.scheme not in ("http", "https")
            or not endpoint.hostname
            or endpoint.port is None
            or endpoint.port == 0
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in ("", "/")
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("control_endpoint must be an http(s) host:port endpoint")


@dataclass(frozen=True)
class EngineDirectory:
    """A target-scoped snapshot mapping exact reporter IDs to single engine ranks.

    Reload the snapshot when engines restart. lookup requires the exact expected
    identity, preventing a stale candidate from silently resolving to a new engine
    incarnation. No host aggregation, fuzzy match, or identity guessing is allowed.
    """

    target: EngineIdentity
    entries: tuple[EngineDirectoryEntry, ...]

    def __post_init__(self) -> None:
        _typed("target", self.target, EngineIdentity)
        if type(self.entries) is not tuple:
            raise ValueError("entries must be an immutable tuple")
        reporters = set()
        engine_ranks = set()
        for entry in self.entries:
            _typed("entry", entry, EngineDirectoryEntry)
            validate_peer(entry.identity, self.target)
            rank = (entry.identity.engine_id, entry.identity.tp_rank)
            if entry.reporter_id in reporters or rank in engine_ranks:
                raise ValueError("duplicate reporter_id or engine-rank entry")
            reporters.add(entry.reporter_id)
            engine_ranks.add(rank)

    def lookup(
        self, reporter_id: str, expected_identity: EngineIdentity
    ) -> EngineDirectoryEntry:
        _text("reporter_id", reporter_id)
        validate_peer(expected_identity, self.target)
        for entry in self.entries:
            if entry.reporter_id == reporter_id:
                if entry.identity != expected_identity:
                    raise ValueError("reporter identity or incarnation mismatch")
                return entry
        raise KeyError(reporter_id)
