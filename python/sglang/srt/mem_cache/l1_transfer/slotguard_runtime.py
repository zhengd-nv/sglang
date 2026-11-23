"""Startup-only dedicated groups and Scheduler owner hooks for SlotGuard.

No process group is created in a Scheduler tick or a background worker.
"""

import threading
from dataclasses import dataclass
from datetime import timedelta

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.cache_scheduler_hooks import (
    CacheSchedulerHooks,
)

from .tp_group import SchedulerCollective


@dataclass(frozen=True)
class SlotGuardStartupGroups:
    target: object
    allocation: object
    members: tuple[int, ...]
    owner: int

    def check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("SlotGuard startup belongs to the Scheduler owner")

    def collective(self, rank):
        self.check_owner()
        return SchedulerCollective.from_process_group(self.target, rank)


def create_startup_groups(ctx, config, *, timeout=timedelta(seconds=30)):
    """World4 only. All world ranks vote before the same two new_group calls.

    The startup WORLD vote only validates topology/configuration. Neither of
    the returned groups aliases WORLD, the model/cache group, or PD polling.
    Larger model worlds require a separate world-owner group creation design.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise ValueError("SlotGuard startup currently requires model world size four")
    try:
        model = ctx.tp_group.cpu_group
        members = tuple(dist.get_process_group_ranks(model))
        cache_members = tuple(dist.get_process_group_ranks(ctx.params.tp_cache_group))
        identity = config.directory.target
        valid = (
            ctx.tp_size == 4
            and ctx.params.pp_size == 1
            and ctx.server_args.dp_size == 1
            and members == cache_members == (0, 1, 2, 3)
            and dist.get_rank(model)
            == dist.get_rank(ctx.params.tp_cache_group)
            == ctx.tp_rank
            and identity.tp_rank == ctx.tp_rank == dist.get_rank()
            and identity.tp_size == 4
            and config.local_group is not None
            and config.local_group.ranks[ctx.tp_rank].identity == identity
            and identity.model_id == ctx.server_args.model_path
            and type(config.group_manifest_digest) is str
            and 0 < len(config.group_manifest_digest) <= 128
        )
        vote = (
            bool(valid),
            config.group_manifest_digest,
            identity.model_id,
            identity.domain_id,
            config.min_headroom_tokens,
            config.publication_context_entries,
            config.publication_context_bytes,
            config.slotguard_limits,
            config.control_profile.value,
            config.prefetch_pending_capacity,
            config.prefetch_pipeline_depth,
            config.async_unpin,
            envs.SGLANG_NVLINK_VERIFY_BYTES.get(),
        )
    except Exception:  # noqa: BLE001 -- never split startup collective ordering
        vote = (False,)
    votes = [None] * 4
    dist.all_gather_object(votes, vote)
    if not all(row[0] for row in votes) or any(row != votes[0] for row in votes):
        raise ValueError("SlotGuard startup capability/manifest mismatch")
    # Fixed order: target/admin control first, allocation epochs second.
    target = dist.new_group(ranks=[0, 1, 2, 3], backend="gloo", timeout=timeout)
    allocation = dist.new_group(ranks=[0, 1, 2, 3], backend="gloo", timeout=timeout)
    return SlotGuardStartupGroups(
        target, allocation, (0, 1, 2, 3), threading.get_ident()
    )


def compose_slotguard_runtime(ctx, cache):
    """Compose the opt-in public path from the allocator's frozen config."""
    from .slotguard_linker import _Runtime

    # A rank without a frozen config still joins the startup WORLD capability
    # vote; it must not strand peers by returning early.
    config = ctx.params.token_to_kv_pool_allocator._frozen_serving_config
    groups = create_startup_groups(ctx, config)
    return _Runtime(ctx, cache, _config=config, _startup_groups=groups)


def common_prefill_entry(group, prepared, error):
    """All ranks vote after grammar/empty/priority/chunk-size preparation.

    The target control group carries only short flags here. No allocation
    epoch begins for a commonly empty pass. A divergent entry fails closed
    using the allocation controller's regular all-phase failure protocol.
    """
    try:
        # Only five CPU scalars (batch size, chunk size, prefill token limit, mixed
        # decode tokens, tile size) cross this stage; divergent dynamic chunk or
        # budget policy must fault before allocation.
        flags = group.runtime_collective.gather((prepared, error is not None))
    except Exception:  # noqa: BLE001 -- broken control group cannot reopen Pins
        group.fault_control_stage()
    if any(flag[1] for flag in flags) or any(flag != flags[0] for flag in flags):

        def fail(_decision):
            raise RuntimeError("prefill entry mismatch")

        group.execute("prefill-entry-failure", fail)
    return prepared is not None


class SlotGuardSchedulerHooks(CacheSchedulerHooks):
    """Scheduler steps of one SlotGuard runtime, in its common TP4 epochs."""

    def __init__(self, runtime):
        self._runtime = runtime
        self._group = runtime.allocation_group

    def on_requests_received(self, *, paused, batches, chunked_req):
        self._runtime.scheduler_tick(
            paused=paused, batches=batches, chunked_req=chunked_req
        )

    def agree_prefill_entry(self, prepare):
        prepared, error = None, None
        try:
            prepared = prepare()
        except Exception as exc:  # noqa: BLE001 -- every rank must reach the vote
            error = exc
        return prepared if common_prefill_entry(self._group, prepared, error) else None

    def run_prefill_allocation(self, allocate):
        allocator = self._group.allocator

        def body(decision):
            allocator.set_common_capacity_tokens(decision.capacity_tokens)
            try:
                return allocate()
            finally:
                allocator.set_common_capacity_tokens(None)

        value = self._group.execute("normal-prefill", body).value
        self._group.assert_ready_for_forward()
        return value

    def assert_ready_for_forward(self):
        self._group.assert_ready_for_forward()

    def flush(self, *, idle, paused, reset_pools):
        return self._runtime.flush(idle=idle, paused=paused, reset_pools=reset_pools)
