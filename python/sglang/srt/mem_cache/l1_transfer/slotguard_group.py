"""Scheduler-owned common TP4 allocation epochs, on a dedicated CPU group.

The caller puts only budget/select/evict/allocate/CPU-row commits in ``execute``.
Forward, driver/network waits, and target copy work happen after it returns. No
getter or source worker enters a collective. Group creation and runtime hooks
belong to the startup/Scheduler integration, not this controller.
"""

import hashlib
import re
import threading
from dataclasses import dataclass

from sglang.srt.mem_cache.l1_transfer.tp_group import SchedulerCollective


class AllocationEpochError(RuntimeError):
    """Bounded public failure code; never includes a request or local exception."""


@dataclass(frozen=True)
class AllocationDecision:
    epoch: int
    admitted: bool
    capacity_tokens: int
    protected_pages: tuple
    reclaim_pages: tuple


@dataclass(frozen=True)
class AllocationResult:
    decision: AllocationDecision
    value: object = None


@dataclass(frozen=True)
class AllocationStatus:
    epoch: int
    active: bool
    faulted: bool
    maintenance_needed: bool


class SlotGuardAllocationGroup:
    """One owner and one permanently assigned exact-TP4 allocation collective.

    ``evictable_tokens`` is an O(1), owner-local *raw* tree counter. All capacity
    units at this API are tokens; page alignment and minimum headroom are
    checked. Protected local ownership is conservatively deducted in full.
    ``collective`` injection is for an already validated dedicated group; use
    ``from_process_group`` to validate raw runtime ProcessGroups.
    """

    def __init__(
        self,
        collective,
        allocator,
        service,
        *,
        evictable_tokens,
        scope,
        min_headroom_tokens=0,
    ):
        allocator.assert_allocation_owner()
        self._owner = threading.get_ident()
        self.collective, self.allocator, self.service = collective, allocator, service
        self.guard = allocator.slot_guard
        self._evictable_tokens = evictable_tokens
        self.scope = self._label(scope)
        self.min_headroom_tokens = min_headroom_tokens
        self._pages(min_headroom_tokens)
        if not callable(evictable_tokens) or service.guard is not self.guard:
            raise ValueError("allocation controller requires exact owner inputs")
        if not self.guard.shared_headroom_credit_enabled:
            raise ValueError("common allocation epochs require shared credit")
        self._epoch = 0
        self._generation = service.owner_snapshot().generation
        self._active = self.faulted = False
        self._verified_revision = None
        self._maintenance_pending = False

    @classmethod
    def from_process_group(cls, group, allocator, service, *, members, **kwargs):
        import torch.distributed as dist

        if group is None or group is dist.group.WORLD:
            raise ValueError("allocation requires a dedicated CPU/Gloo group")
        actual = tuple(dist.get_process_group_ranks(group))
        if (
            len(actual) != 4
            or actual != tuple(members)
            or len(set(actual)) != 4
            or dist.get_rank() not in actual
        ):
            raise ValueError("allocation group must match exact ordered TP4 members")
        collective = SchedulerCollective.from_process_group(group, dist.get_rank(group))
        return cls(collective, allocator, service, **kwargs)

    @staticmethod
    def _label(value):
        if (
            type(value) is not str
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is None
        ):
            raise ValueError("bounded allocation stage/scope identifier required")
        return value

    def _pages(self, tokens):
        if type(tokens) is not int or tokens < 0 or tokens % self.allocator.page_size:
            raise ValueError("allocation capacity must contain whole pages")
        return tokens // self.allocator.page_size

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise AllocationEpochError("OWNER_THREAD")

    def assert_ready_for_forward(self):
        self._check_owner()
        if self.faulted or self._active:
            raise AllocationEpochError("ALLOCATION_FENCED")

    def fault_control_stage(self):
        """Permanent owner fence after a failed common control-stage vote.

        A broken control group cannot run an allocation reconciliation epoch.
        This performs no collective and never releases retained ownership.
        """
        self._check_owner()
        self._fault("CONTROL_STAGE_FAILURE")

    def maintenance_needed(self):
        """Cheap local hint only; 4b must make the fixed-stage common decision."""
        self._check_owner()
        counts = self.guard.owner_counts()
        return (
            self.faulted
            or self._maintenance_pending
            or counts.revision != self._verified_revision
        )

    def status(self):
        self._check_owner()
        return AllocationStatus(
            self._epoch, self._active, self.faulted, self.maintenance_needed()
        )

    def _fault(self, code):
        # Neither successful allocation nor successful local finish is rolled
        # back. Even a rank which finished its guard remains permanently fenced.
        self.faulted = True
        try:
            self._generation = max(
                self._generation, self.service.owner_snapshot().generation
            )
            self.service.fence_pins(self._generation)
        except Exception:  # noqa: BLE001, S110 -- still fence allocator, expose only code
            pass
        finally:
            self.allocator.fault_allocation_epoch(self._epoch)
        raise AllocationEpochError(code) from None

    def _exchange(self, phase, function):
        try:
            value, code = function(), None
        except Exception:  # noqa: BLE001 -- all owners must enter the failure vote
            value, code = None, "LOCAL_FAILURE"
        try:
            rows = self.collective.gather((self._epoch, phase, code, value))
        except Exception:  # noqa: BLE001 -- a broken group cannot safely resume
            self._fault("COLLECTIVE_FAILURE")
        if any(row[:2] != (self._epoch, phase) for row in rows):
            self._fault("PHASE_DIVERGENCE")
        if any(row[2] is not None for row in rows):
            self._fault(f"{phase}_FAILURE")
        return [row[3] for row in rows]

    def _capacity(self):
        allocator = self.allocator
        if allocator.verify_byte_accounting():
            raise ValueError("physical allocation bookkeeping mismatch")
        evictable = self._evictable_tokens()
        self._pages(evictable)
        owned, free, retired, _capacity = allocator.conservation_counts()
        counts = self.guard.owner_counts()
        if counts.local_owned_pages != owned or counts.retired_pages != retired:
            raise ValueError("guard and physical allocator ownership disagree")
        if evictable > owned * allocator.page_size:
            raise ValueError("raw evictable counter exceeds local ownership")
        capacity = allocator.available_and_evictable_size(evictable)
        if allocator.available_size() != free * allocator.page_size:
            raise ValueError("physical free counter mismatch")
        return max(0, capacity - self.min_headroom_tokens)

    def execute(
        self,
        stage,
        body,
        *,
        request_key=None,
        demand_tokens=None,
        private_free_only=False,
    ):
        """Run one bounded CPU allocation body, or commonly deny private demand.

        A denied demand is a successful no-body epoch and refreshes credit.
        ``request_key`` is hashed before exchange. Allocations/page-span capture
        in ``body`` must finish before return; the returned value may be used for
        subsequent private copy/driver work only after this method succeeds.
        """
        self.assert_ready_for_forward()
        self._active = True
        self._epoch += 1
        frozen_held_count = 0

        def begin():
            nonlocal frozen_held_count
            # Startup/4b2 Admin may independently advance the source fence.
            # Retain this controller and epoch counter across guard.reset.
            self._generation = (
                max(self._generation, self.service.owner_snapshot().generation) + 1
            )
            self.service.fence_pins(self._generation)
            stage_id = self._label(stage)
            if type(private_free_only) is not bool or (
                private_free_only and demand_tokens is None
            ):
                raise ValueError("private free-only demand requires explicit tokens")
            if not callable(body):
                raise TypeError("CPU allocation body required")
            if demand_tokens is not None:
                self._pages(demand_tokens)
                if type(request_key) is not str or not 0 < len(request_key) <= 256:
                    raise ValueError("private demand requires a bounded request key")
            elif request_key is not None:
                raise ValueError("request key requires explicit private demand")
            snapshot = self.allocator.begin_allocation_epoch(self._epoch)
            frozen_held_count = len(snapshot.held_pages)
            held = self.allocator.logical_pages(snapshot.held_pages)
            reclaim = self.allocator.logical_pages(snapshot.reclaim_candidate_pages)
            if len(held) + len(reclaim) > 2 * self.allocator.num_pages:
                raise ValueError("allocation proposal exceeds pool bound")
            request_digest = (
                hashlib.sha256(request_key.encode()).hexdigest()
                if request_key is not None
                else None
            )
            return (
                (
                    stage_id,
                    self.scope,
                    self.allocator.allocation_signature(),
                    self.allocator.num_pages,
                    self.allocator.page_size,
                    self.min_headroom_tokens,
                    self.guard.max_remote_pinned_pages,
                    demand_tokens,
                    request_digest,
                    self._generation,
                    private_free_only,
                ),
                held,
                reclaim,
            )

        proposals = self._exchange("BEGIN", begin)
        if any(row[0] != proposals[0][0] for row in proposals):
            self._fault("PROPOSAL_DIVERGENCE")
        protected = set().union(*(set(row[1]) for row in proposals))
        reclaim = set(proposals[0][2]).intersection(*(set(row[2]) for row in proposals))
        protected, reclaim = (
            tuple(sorted(protected)),
            tuple(sorted(reclaim - protected)),
        )

        def install():
            self.allocator.install_allocation_epoch(
                self._epoch,
                self.allocator.local_page_refs(protected),
                reclaim_refs=self.allocator.local_page_refs(reclaim),
            )
            capacity = self._capacity()
            if private_free_only:
                # HELD private slots cannot evict local cache. All ranks agree
                # this actual FREE bound before any rank allocates.
                capacity = min(
                    capacity,
                    max(0, self.allocator.available_size() - self.min_headroom_tokens),
                )
            return capacity

        capacities = self._exchange("INSTALL", install)
        capacity = min(capacities)
        admitted = demand_tokens is None or demand_tokens <= capacity
        decision = AllocationDecision(
            self._epoch, admitted, capacity, protected, reclaim
        )
        value = None

        def run_body():
            nonlocal value
            if admitted:
                value = body(decision)
            local_capacity = self._capacity()
            local_credit = min(
                self._pages(local_capacity), self.guard.max_remote_pinned_pages
            )
            self.allocator.prevalidate_finish_allocation_epoch(
                self._epoch, verified_headroom_pages=local_credit
            )
            self.service.prevalidate_unfence_pins(self._generation)
            return self.allocator.allocation_signature(), local_credit

        outcomes = self._exchange("BODY", run_body)
        if any(row[0] != outcomes[0][0] for row in outcomes):
            self._fault("BODY_HISTORY_DIVERGENCE")
        credit = min(row[1] for row in outcomes)

        def finish():
            self.allocator.finish_allocation_epoch(
                self._epoch, verified_headroom_pages=credit
            )

        self._exchange("FINISH", finish)
        counts = self.guard.owner_counts()
        self._verified_revision = counts.revision
        self._maintenance_pending = bool(
            counts.held_pages != frozen_held_count
            or counts.pending_reclaim_pages
            or counts.reclaim_overflow_pages
        )

        # No source service can reopen before every guard/allocator finished.
        def unfence():
            if not self.service.unfence_pins(self._generation):
                raise ValueError("source admission failed to reopen")

        self._exchange("UNFENCE", unfence)
        self._active = False
        return AllocationResult(decision, value)

    def maintenance(self, stage="idle-maintenance"):
        """Callable idle epoch; the runtime must call it commonly on all ranks."""
        return self.execute(stage, lambda _decision: None)
