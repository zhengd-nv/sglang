"""Explicit private-domain configuration; credentials are read only from env."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sglang.srt.environ import envs

from .control_profile import ControlProfile
from .directory import EngineDirectory, EngineDirectoryEntry, EngineTPGroup
from .protocol import EngineIdentity
from .slotguard_limits import MAX_PIN_PAGES

# Arbitrary bound, read before parsing; a generated manifest takes about 2.5 KiB
# per engine, so this leaves room for every engine of an NVLink domain.
MAX_MANIFEST_BYTES = 1 << 20


def layout_digest(layout):
    return hashlib.sha256(
        json.dumps(asdict(layout), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SlotGuardLimits:
    """Hard RAM bounds, including the finalized post-capture physical domain."""

    max_pages: int = 65536
    max_remote_pinned_pages: int = 4096
    max_active_leases: int = 1024
    max_pending_reclaim_pages: int = 65536
    max_pin_pages: int = 4096
    max_tombstones: int = 65536

    def __post_init__(self):
        for name, ceiling in (
            ("max_pages", 262144),
            ("max_remote_pinned_pages", 262144),
            ("max_active_leases", 65536),
            ("max_pending_reclaim_pages", 262144),
            ("max_pin_pages", MAX_PIN_PAGES),
            ("max_tombstones", 262144),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid bounded SlotGuard limit")
        if (
            self.max_remote_pinned_pages > self.max_pages
            or self.max_pending_reclaim_pages > self.max_pages
            or self.max_pin_pages > self.max_remote_pinned_pages
            or self.max_active_leases > self.max_tombstones
        ):
            raise ValueError("inconsistent SlotGuard limits")

    def make_guard(self, incarnation):
        from .slot_guard import SlotGuard

        return SlotGuard(
            1,
            incarnation,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            shared_headroom_credit=True,
        )


@dataclass(frozen=True)
class ServingConfig:
    directory: EngineDirectory
    bind_address: tuple[str, int]
    kvcm_endpoint: str
    bearer_token: str = field(repr=False)
    min_headroom_tokens: int = 256
    local_group: EngineTPGroup | None = None
    group_manifest_digest: str | None = None
    prefetch_pending_capacity: int | None = None
    prefetch_pipeline_depth: int = 1
    control_profile: ControlProfile = ControlProfile.SLOTGUARD_HASH_V3
    publication_context_entries: int = 65536
    publication_context_bytes: int = 64 * 1024 * 1024
    slotguard_limits: SlotGuardLimits = field(default_factory=SlotGuardLimits)
    # Publish READY once DMA drained and imports unmapped; the Unpin round
    # trip then finishes off the critical path.
    async_unpin: bool = False
    # Lifetime bound, not a window: the source keeps a replay record for every
    # transfer it accepts, so Pin is BUSY once this many exist until restart.
    source_max_records: int = 1024

    def __post_init__(self):
        if type(self.async_unpin) is not bool:
            raise ValueError("async Unpin requires exact bool")

    @classmethod
    def from_env(cls, tp_rank=0):
        path = envs.SGLANG_NVLINK_CONFIG.get()
        if not path:
            raise ValueError("nvlink requires SGLANG_NVLINK_CONFIG")
        try:
            with Path(path).open("rb") as stream:
                text = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(text) > MAX_MANIFEST_BYTES:
                raise ValueError("manifest exceeds its size bound")
            data = json.loads(text)
            if set(data) - {
                "kvcm_endpoint",
                "bearer_env",
                "trusted_private_domain",
                "min_headroom_tokens",
                "local_group",
                "peer_groups",
                "bind_host",
                "prefetch_pending_capacity",
                "prefetch_pipeline_depth",
                "control_profile",
                "peer_control_profiles",
                "publication_context_entries",
                "publication_context_bytes",
                "slotguard_limits",
                "async_unpin",
                "source_max_records",
            }:
                raise ValueError("unknown configuration fields")
            if data.get("trusted_private_domain") is not True:
                raise ValueError("private-domain acknowledgement required")
            from urllib.parse import urlsplit

            def parse_group(value):
                if set(value) != {"reporter_id", "ranks"}:
                    raise ValueError("invalid group fields")
                if any(
                    set(rank) != {"identity", "control_endpoint"}
                    for rank in value["ranks"]
                ):
                    raise ValueError("invalid rank binding fields")
                return EngineTPGroup(
                    value["reporter_id"],
                    tuple(
                        EngineDirectoryEntry(
                            value["reporter_id"],
                            EngineIdentity(**rank["identity"]),
                            rank["control_endpoint"],
                        )
                        for rank in value["ranks"]
                    ),
                )

            local_group = parse_group(data["local_group"])
            peers = tuple(parse_group(peer) for peer in data["peer_groups"])
            directory = local_group.directory(peers, tp_rank)
            manifest_digest = hashlib.sha256(
                json.dumps(
                    (asdict(local_group), [asdict(peer) for peer in peers]),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            bind = (
                data["bind_host"],
                urlsplit(local_group.ranks[tp_rank].control_endpoint).port,
            )
            token = os.environ[data.get("bearer_env", "SGLANG_NVLINK_CONTROL_TOKEN")]
            endpoint = data["kvcm_endpoint"]
            headroom = data.get("min_headroom_tokens", 256)
            pending_capacity = data.get("prefetch_pending_capacity")
            pipeline_depth = data.get("prefetch_pipeline_depth", 1)
            # The target transport runs one inbound read at a time.
            if type(pipeline_depth) is not int or pipeline_depth != 1:
                raise ValueError("pipeline depth must be integer 1")
            if "prefetch_pending_capacity" in data and (
                type(pending_capacity) is not int or not 4 <= pending_capacity <= 64
            ):
                raise ValueError("pending capacity must be an integer in [4,64]")
            if not directory.entries or type(headroom) is not int or headroom < 1:
                raise ValueError("peers and positive headroom required")
            if not isinstance(endpoint, str) or any(ch.isspace() for ch in endpoint):
                raise ValueError("explicit KVCM host:port required")
            kvcm = urlsplit(f"//{endpoint}")
            # .port raises ValueError for a non-numeric or out-of-range port.
            if (
                kvcm.netloc != endpoint
                or not kvcm.hostname
                or kvcm.username is not None
                or not kvcm.port
            ):
                raise ValueError("explicit KVCM host:port required")
            profile = ControlProfile(data["control_profile"])
            context_entries = data.get("publication_context_entries", 65536)
            context_bytes = data.get("publication_context_bytes", 64 * 1024 * 1024)
            source_records = data.get("source_max_records", 1024)
            # Arbitrary ceiling that bounds the host memory of the replay history.
            if type(source_records) is not int or not 1 <= source_records <= 1 << 20:
                raise ValueError("source replay record bound out of range")
            limits = SlotGuardLimits(**data.get("slotguard_limits", {}))
            # Operator-supplied immutable manifest acknowledgement, not a
            # runtime negotiation or authenticated capability handshake.
            expected = {
                entry.identity.engine_id: profile.value
                for peer in peers
                for entry in peer.ranks
            }
            if data.get("peer_control_profiles") != expected:
                raise ValueError("exact TP4 peer control profiles required")
            if (
                type(context_entries) is not int
                or not 40000 <= context_entries <= 262144
                or type(context_bytes) is not int
                or not 40 * 1024 * 1024 <= context_bytes <= 256 * 1024 * 1024
                or context_bytes < context_entries * 1024
            ):
                raise ValueError(
                    "publication context exceeds bounded working-set budget"
                )
            return cls(
                directory,
                bind,
                endpoint,
                token,
                headroom,
                local_group,
                manifest_digest,
                pending_capacity,
                pipeline_depth,
                profile,
                context_entries,
                context_bytes,
                limits,
                # On by default; false keeps Unpin before READY.
                data.get("async_unpin", True),
                source_records,
            )
        except (KeyError, ValueError, TypeError, OSError):
            # Do not echo config, tokens, prompt payloads or credential values.
            raise ValueError("invalid NVLink serving configuration") from None


def validate_slotguard_factory(configurator, config, parallel, memory, schedule):
    """Reject unsupported paths before constructing an inactive factory pool."""
    args = configurator.server_args
    if (
        parallel.tp_size != 4
        or parallel.attn_dcp_size != 1
        or parallel.attn_cp_size != 1
        or configurator.pp_size != 1
        or configurator.attn_dp_size != 1
        or schedule.page_size != 64
        or configurator.is_draft_worker
        or configurator.is_hybrid_swa
        or configurator.mambaish_config is not None
        or configurator.hybrid_gdn_config is not None
        or configurator.hybrid_kda_config is not None
        or configurator.kv_shard_size != 1
        or memory.enable_hisparse
        or args.disaggregation_mode != "prefill"
        or not envs.SGLANG_GLM_DSA_FIXED_FABRIC_POOL.get()
        or envs.SGLANG_DEBUG_MEMORY_POOL.get()
        or args.enable_mixed_chunk
        or args.enable_hierarchical_cache
        or args.enable_lora
        or args.speculative_algorithm is not None
        or args.dllm_algorithm is not None
        or config.publication_context_entries < config.slotguard_limits.max_pages
        or config.publication_context_bytes < config.publication_context_entries * 1024
    ):
        raise ValueError("unsupported SlotGuard factory configuration")


def validate_serving_context(ctx, cache):
    """Reject unsupported serving paths before creating any network/CUDA owner."""
    args, params = ctx.server_args, ctx.params
    if (
        ctx.tp_size != 4
        or not ctx.is_dsa
        or params.pp_size != 1
        or params.attn_cp_size != 1
        or args.dcp_size != 1
        or args.dp_size != 1
    ):
        raise ValueError("NVLink prefetch requires TP4 DSA with PP=CP=DCP=DP=1")
    if (
        not envs.SGLANG_GLM_DSA_FIXED_FABRIC_POOL.get()
        or args.disaggregation_mode != "prefill"
    ):
        raise ValueError("NVLink prefetch requires fixed FABRIC pool and prefill role")
    if (
        ctx.is_hybrid_swa
        or ctx.is_hybrid_ssm
        or params.is_eagle
        or ctx.disable_radix_cache
        or ctx.enable_hierarchical_cache
        or params.enable_session_radix_cache
        or params.component_registry_override
        or params.mtp_draft_device_pools
        or params.mtp_draft_mamba_pools
        or args.speculative_algorithm is not None
        or args.enable_lora
        or args.enable_hisparse
        or args.enable_linker_mla_dedup
        or args.attn_dp_size != 1
        or args.dllm_algorithm is not None
    ):
        raise ValueError("NVLink prefetch requires a Full-only GLM MLA/DSA cache")
    if cache._tree_core_backend != "python":
        raise ValueError(
            "NVLink source snapshots currently require the Python URC core"
        )
    from sglang.srt.disaggregation.kv_events import uses_dynamo_format
    from sglang.srt.runtime_context import get_observability

    # The resolved global setting lives in runtime_context. Scheduler enables
    # local events only on the publisher rank; under the supported PP/CP/DP=1
    # topology that is TP0.
    kv_events_config = get_observability().kv_events_config
    if (
        not kv_events_config
        or uses_dynamo_format(kv_events_config)
        or params.dynamo_kv_event_format
    ):
        raise ValueError(
            "NVLink requires native SGLang KV events and an external KVCM reporter"
        )
    if not 0 <= ctx.tp_rank < ctx.tp_size:
        raise ValueError("NVLink requires a valid TP rank")
    if params.enable_kv_cache_events != (ctx.tp_rank == 0):
        raise ValueError("NVLink requires local KV events on TP0 only")
    if params.token_to_kv_pool_allocator.page_size != params.page_size:
        raise ValueError("NVLink requires identical allocator/tree page size")
