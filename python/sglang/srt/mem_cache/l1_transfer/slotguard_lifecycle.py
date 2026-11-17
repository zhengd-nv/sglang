"""Short owner proofs for the explicitly injected SlotGuard composition.

Capture and target observations belong to their Scheduler owner. Background
workers expose locked counters; this module never reads a Tree, tensor pool or
native engine journal. Cached local pages and bounded replay metadata remain
reported, but are not external ownership obligations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetOwners:
    active: int
    parked: int
    pending: int
    private: int
    receipts: int
    workqueue: int
    completions: int
    working: int
    unknown: int
    records: int
    revision: tuple

    @property
    def owners(self):
        return sum(
            (
                self.active,
                self.parked,
                self.pending,
                self.private,
                self.receipts,
                self.workqueue,
                self.completions,
                self.working,
                self.unknown,
            )
        )


@dataclass(frozen=True)
class BackendOwners:
    accepted: int
    queued: int
    working: int
    imports: int
    batches: int
    unknown: int
    revision: int

    @property
    def owners(self):
        return sum(
            (
                self.accepted,
                self.queued,
                self.working,
                self.imports,
                self.batches,
                self.unknown,
            )
        )


@dataclass(frozen=True)
class SlotGuardOwners:
    mode: str
    profile: str
    pool_epoch: int
    generation: int
    capture: object
    catalog: object
    source: object
    transport: object
    guard: object
    target: TargetOwners
    backend: BackendOwners
    revisions: tuple

    @property
    def quiescent(self):
        capture, guard = self.capture, self.guard
        return not any(
            (
                capture.active,
                capture.staged_entries,
                capture.worker.owners,
                capture.worker.uncertain,
                self.catalog.pending,
                self.catalog.uncertain,
                self.source.owners,
                self.source.uncertain,
                self.transport.owners,
                guard.active_leases,
                guard.held_pages,
                guard.retired_pages,
                guard.pending_reclaim_pages,
                guard.reclaim_overflow_pages,
                guard.frozen_epoch is not None,
                self.target.owners,
                self.backend.owners,
            )
        )


def owner_snapshot(serving):
    """Owner-only bounded observation, with missing visibility failing closed."""
    try:
        capture = serving.publication_hook.owner_snapshot()
        source = serving.source_service.owner_snapshot()
        catalog = serving.catalog.snapshot()
        transport = serving.arbiter.owner_snapshot()
        guard = serving.slot_guard.owner_counts()
        target = serving.admission.owner_snapshot()
        backend = serving.backend.owner_snapshot()
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("SlotGuard owner visibility is incomplete") from exc
    return SlotGuardOwners(
        "slotguard",
        serving.backend._profile.value,
        serving.slot_guard.pool_epoch,
        source.generation,
        capture,
        catalog,
        source,
        transport,
        guard,
        target,
        backend,
        (
            capture.revision,
            capture.worker.revision,
            catalog.revision,
            source.revision,
            transport.revision,
            guard,
            target.revision,
            backend.revision,
        ),
    )
