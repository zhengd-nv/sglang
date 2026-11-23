"""Bounded KVCM candidate lookup for backend execution threads only.

Never call this synchronous client on the Scheduler thread. The caller owns its
gRPC channel and runs lookup on the transfer backend's worker thread. KVCM local
counts are hints, not proof that all TP ranks or required KV pools are ready.
PinPrefix remains authoritative, including token/hash matching and incarnation.

This POC only accepts non-EAGLE, no-LoRA, no-salt requests. No storage hash,
P2P count, KVCM location URI, or multi-host aggregation is used. Optional gRPC
dependencies are loaded only during lookup, leaving disabled installations alone.
"""

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .directory import EngineDirectory
from .protocol import LookupQuery, LookupResult, _typed

# Arbitrary bound on the per-host work of one lookup; a longer reply is invalid.
_MAX_HOSTS = 1024


class LookupReason(str, Enum):
    HIT = "HIT"
    NO_CANDIDATE = "NO_CANDIDATE"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    STATUS_ERROR = "STATUS_ERROR"
    RPC_ERROR = "RPC_ERROR"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class RejectionReason(str, Enum):
    UNKNOWN_REPORTER = "UNKNOWN_REPORTER"
    INCOMPATIBLE_REPORTER = "INCOMPATIBLE_REPORTER"
    DUPLICATE_REPORTER = "DUPLICATE_REPORTER"
    INVALID_LOCAL = "INVALID_LOCAL"
    NO_GAIN = "NO_GAIN"


@dataclass(frozen=True)
class CandidateRejection:
    reporter_id: str
    reason: RejectionReason


@dataclass(frozen=True)
class LookupObservation:
    reason: LookupReason
    rejections: tuple[CandidateRejection, ...] = ()
    status_code: int | None = None
    grpc_code: str | None = None
    exception_type: str | None = None


@dataclass(frozen=True)
class KVCMLookupOutcome:
    result: LookupResult
    control_endpoint: str | None
    observation: LookupObservation


class KVCMClient:
    """One target's directory and a caller-owned gRPC Channel.

    The directory must be refreshed explicitly on engine restart. KVCM responses
    carry no incarnation, so their reporter ID only selects the pinned directory
    identity; an old source incarnation can only be rejected by the later Pin RPC.
    Query target identity must exactly match this directory's target snapshot.
    """

    def __init__(
        self,
        channel: Any,
        directory: EngineDirectory,
        *,
        timeout_s: float = 0.1,
        is_eagle: bool = False,
        use_lora: bool = False,
    ):
        _typed("directory", directory, EngineDirectory)
        if is_eagle is not False or use_lora is not False:
            raise ValueError("peer-L1 KVCM POC requires non-EAGLE and no LoRA")
        if (
            type(timeout_s) not in (int, float)
            or not 0 < timeout_s <= 30
            or not math.isfinite(timeout_s)
        ):
            raise ValueError("timeout_s must be finite and in (0, 30]")
        self._channel = channel
        self._directory = directory
        self._timeout_s = timeout_s

    def lookup(self, query: LookupQuery) -> LookupResult:
        """Convenience method; use lookup_with_observation for fallback reasons."""
        return self.lookup_with_observation(query).result

    def lookup_with_observation(self, query: LookupQuery) -> KVCMLookupOutcome:
        """Return a safe miss on query/transport failure, with a bounded reason.

        Exception text and remote error messages are intentionally not retained:
        observations carry codes, never prompt tokens or possible credentials.
        """
        _typed("query", query, LookupQuery)

        def miss(reason: LookupReason, **details: Any) -> KVCMLookupOutcome:
            return KVCMLookupOutcome(
                LookupResult(
                    query,
                    None,
                    query.local_hit_tokens,
                    query.event_block_keys[: query.local_hit_tokens // query.page_size],
                ),
                None,
                LookupObservation(reason, **details),
            )

        if query.target != self._directory.target:
            return miss(LookupReason.TARGET_MISMATCH)
        if query.cache_salt is not None or query.extra_key is not None:
            return miss(LookupReason.UNSUPPORTED_QUERY)
        if query.max_restore_tokens == query.local_hit_tokens:
            return miss(LookupReason.NO_CANDIDATE)
        try:
            import grpc

            from .generated import tair_kvcache_meta_service_pb2 as pb
            from .generated import tair_kvcache_meta_service_pb2_grpc as rpc
        except ImportError as exc:
            return miss(
                LookupReason.DEPENDENCY_UNAVAILABLE, exception_type=type(exc).__name__
            )

        try:
            request = pb.GetHostCacheStateRequest(
                trace_id=query.transfer_id.rid,
                instance_id=query.target.instance_id,
                query_type=pb.QT_PREFIX_MATCH,
                block_cache_keys=query.event_block_keys,
                medium=["hbm"],
                p2p_host_count=0,
            )
            response = rpc.MetaServiceStub(self._channel).GetHostCacheState(
                request, timeout=self._timeout_s
            )
        except grpc.RpcError as exc:
            code = exc.code()
            reason = (
                LookupReason.DEADLINE_EXCEEDED
                if code == grpc.StatusCode.DEADLINE_EXCEEDED
                else LookupReason.RPC_ERROR
            )
            return miss(reason, grpc_code=code.name)
        except Exception as exc:
            return miss(LookupReason.RPC_ERROR, exception_type=type(exc).__name__)

        try:
            if response.header.status.code != pb.OK:
                return miss(
                    LookupReason.STATUS_ERROR, status_code=response.header.status.code
                )
            if len(response.hosts) > _MAX_HOSTS:
                return miss(LookupReason.INVALID_RESPONSE)
            entries = {entry.reporter_id: entry for entry in self._directory.entries}
            counts = Counter(host.host_ip_port for host in response.hosts)
            rejected = []
            best = None
            best_hit = query.local_hit_tokens
            for host in response.hosts:
                reporter = host.host_ip_port
                reason = None
                if counts[reporter] != 1:
                    reason = RejectionReason.DUPLICATE_REPORTER
                elif host.local < 0 or host.local > len(query.event_block_keys):
                    reason = RejectionReason.INVALID_LOCAL
                elif reporter not in entries:
                    reason = RejectionReason.UNKNOWN_REPORTER
                else:
                    entry = entries[reporter]
                    try:
                        self._directory.lookup(reporter, entry.identity)
                    except (KeyError, ValueError):
                        reason = RejectionReason.INCOMPATIBLE_REPORTER
                    else:
                        # local is a ROOT prefix boundary, not a tail length.
                        hit = min(
                            host.local * query.page_size, query.max_restore_tokens
                        )
                        if hit <= query.local_hit_tokens:
                            reason = RejectionReason.NO_GAIN
                        elif hit > best_hit:
                            best, best_hit = entry, hit
                if reason is not None:
                    rejected.append(CandidateRejection(reporter, reason))
            if best is None:
                return miss(LookupReason.NO_CANDIDATE, rejections=tuple(rejected))
            result = LookupResult(
                query,
                best.identity,
                best_hit,
                query.event_block_keys[: best_hit // query.page_size],
            )
            return KVCMLookupOutcome(
                result,
                best.control_endpoint,
                LookupObservation(LookupReason.HIT, tuple(rejected)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            return miss(
                LookupReason.INVALID_RESPONSE, exception_type=type(exc).__name__
            )
