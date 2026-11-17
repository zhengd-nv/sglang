"""Request-aware bridge to the existing SlotGuard network transaction.

No allocator, tree mutation or worker collective lives here. The upstream
prefetch wrapper owns all destination slots. A running Future is completed only
after a common drain decision; uncertain owners are retained and require engine
restart, never a fabricated failed-copy completion.

The target runs one inbound read at a time. Finished lookups are published in
submission order, and one that found pages stays unpublished until the previous
read retires or its request is released, so a queued hit holds no device slots.
"""

import threading
import uuid
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

from .metadata import DeferredSlots, event_keys_and_hashes
from .nvlink_backend import PrefetchNotAccepted
from .protocol import (
    LookupQuery,
    LookupResult,
    PrefetchStatus,
    PrivateSlotView,
    TransferId,
    group_identity,
)
from .role_gate import EngineRole
from .slot_guard import LeaseIdentity
from .tp_group import worker_decision
from .transport_role_arbiter import TransportRole


class LinkerQuarantined(RuntimeError):
    """Restart required: retained pages/leases must not be reused."""


@dataclass
class _Lookup:
    context: object
    future: Future
    ticket: TransferId
    job: Future | None = None
    result: LookupResult | None = None
    error: Exception | None = None
    cancelled: bool = False
    released: bool = False


@dataclass
class _Read:
    lookup: _Lookup
    future: Future
    identity: LeaseIdentity
    fence: DeferredSlots | None = None
    prepare: Future | None = None
    started: bool = False
    synthetic: tuple = ("ENQUEUED", False)
    synthetic_status: PrefetchStatus | None = None
    cancelled: bool = False
    uncertain: bool = False
    expected_pages: tuple | None = None


class SlotGuardLinkerTransport:
    def __init__(self, backend, collective, *, forward_stream=None, capacity=64):
        self.backend, self.collective = backend, collective
        self.forward_stream = forward_stream
        self.target = backend._directory.target
        self.gate, self.arbiter = backend._gate, backend.transport_arbiter
        self.capacity = capacity
        self._next_transfer_attempt = 0
        self.lookups = {}
        self.retired_lookups = []
        self.active = None
        # Published lookup whose read has not started; it holds the inbound turn.
        self.reserved = None
        self.disabled = False
        self.closed = False
        self._owner = threading.get_ident()
        self._lookup_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="l1-lookup"
        )
        self._prepare_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="l1-prepare"
        )

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("linker progress belongs to Scheduler")

    def submit_lookup(self, context):
        self._check_owner()
        self._reap_lookups()
        if self.disabled or self.closed:
            return None
        if context.request in self.lookups:
            raise ValueError("duplicate prefetch handle")
        if (
            context.extra_key is not None
            or context.cache_salt is not None
            or context.is_bigram
            or getattr(context, "features", ())
        ):
            return None
        if len(self.lookups) + len(self.retired_lookups) >= self.capacity:
            return None
        ticket = TransferId(
            self.target.incarnation, context.request.rid, self._next_transfer_attempt
        )
        self._next_transfer_attempt += 1
        record = _Lookup(context, Future(), ticket)
        self.lookups[context.request] = record
        try:
            record.job = self._lookup_worker.submit(self._lookup, record)
        except Exception:
            del self.lookups[context.request]
            raise
        return record.future

    def _lookup(self, record):
        # The Scheduler publishes the result; this worker only computes it.
        if record.cancelled:
            return
        try:
            ctx = record.context
            tokens = array("q")
            tokens.frombytes(ctx.token_ids)
            keys, hashes = event_keys_and_hashes(tokens, ctx.page_size)
            query = LookupQuery(
                self.target,
                record.ticket,
                keys,
                tuple(tokens),
                ctx.local_hit_tokens,
                ctx.max_restore_tokens,
                ctx.page_size,
                prefix_hashes=hashes,
            )
            record.result = self.backend.lookup(query)
        except Exception as exc:
            record.error = exc

    def _publish_lookups(self):
        waiting = self.active is not None or self.reserved is not None
        for record in self.lookups.values():
            if record.future.done():
                continue
            if not record.job.done():
                return  # The single lookup worker finishes in submission order.
            result = record.result
            needs_read = (
                record.error is None
                and result.source is not None
                and result.restore_pages > 0
            )
            if needs_read and waiting:
                continue  # Misses behind it may pass; later hits keep their order.
            if not record.future.set_running_or_notify_cancel():
                continue
            if record.error is not None:
                record.future.set_exception(record.error)
                continue
            if needs_read:
                self.reserved, waiting = record, True
            record.future.set_result(list(range(1, result.restore_pages + 1)))

    def prefetch(self, request, transfer):
        self._check_owner()
        record = self.lookups[request]
        if self.reserved is record:
            # The read either starts below or is refused; both end the turn.
            self.reserved = None
        if not record.future.done() or record.future.cancelled():
            raise ValueError("prefetch requires completed exact lookup")
        result = record.result
        if result is None or result.source is None:
            return None
        # Reserve CPU roles on every rank before any producer event/copy. Partial
        # reservations are released only after the common rejection decision.
        identity = LeaseIdentity(
            result.source.incarnation,
            self.target.incarnation,
            result.transfer_id,
            uuid.uuid4().hex,
        )
        gate_held = role_held = False
        error = False
        try:
            if not (self.disabled or self.closed or record.cancelled or self.active):
                gate_held = self.gate.acquire(EngineRole.INBOUND, result.transfer_id)
                if gate_held:
                    role_held = self.arbiter.acquire(TransportRole.INBOUND, identity)
        except Exception:
            error = True
        vote = (
            request.rid,
            request.attempt_id,
            group_identity(result.source),
            len(transfer.device_indices),
            gate_held and role_held,
            error,
        )
        try:
            votes = self.collective.gather(vote)
            if any(v[-1] for v in votes) or any(
                (v[:2], v[3]) != (votes[0][:2], votes[0][3]) for v in votes
            ):
                self.disabled = True
                raise LinkerQuarantined("prefetch identity/role vote failed")
            # Independent directory queries may choose different peer hints.
            # No Pin or device read has started: reject together and release
            # only our reservations, just as for a rank with a busy role.
            if any(v[2] != votes[0][2] for v in votes) or not all(v[4] for v in votes):
                if role_held:
                    self.arbiter.release(TransportRole.INBOUND, identity)
                if gate_held:
                    self.gate.release(EngineRole.INBOUND, result.transfer_id)
                return None
        except Exception:
            self.disabled = True
            raise
        read = _Read(record, Future(), identity)
        read.future.set_running_or_notify_cancel()
        self.active = read
        span = getattr(transfer, "page_span", None)
        read.expected_pages = None if span is None else span.page_ids
        # From this point every exception retains a running Future and the roles.
        try:
            read.fence = DeferredSlots.capture(
                transfer.device_indices,
                self.forward_stream,
                retain=lambda fence: setattr(read, "fence", fence),
            )
            read.prepare = self._prepare_worker.submit(self._prepare, read)
        except Exception:
            read.uncertain = True
        return read.future

    @staticmethod
    def _prepare(read):
        try:
            indices = read.fence.materialize()
        except Exception:
            return False, None, None
        try:
            original = read.lookup.result
            query = original.query
            end = query.local_hit_tokens + len(indices)
            result = replace(
                original,
                hit_tokens=end,
                event_block_keys=query.event_block_keys[: end // query.page_size],
            )
            slots = PrivateSlotView(
                result.transfer_id,
                query.local_hit_tokens,
                end,
                query.page_size,
                indices,
                tuple(i // query.page_size for i in indices[:: query.page_size]),
            )
            slots.validate_for(result)
            if (
                read.expected_pages is not None
                and slots.page_ids != read.expected_pages
            ):
                raise ValueError("device indices differ from guarded CPU page identity")
            return True, result, slots
        except Exception:
            return True, None, None

    def cancel(self, request):
        self._check_owner()
        record = self.lookups.get(request)
        if record is not None:
            record.cancelled = True
            record.future.cancel()
        if self.active is not None and self.active.lookup.context.request == request:
            self.active.cancelled = True

    def release(self, request):
        self._check_owner()
        record = self.lookups.get(request)
        if record is None:
            return
        if self.active is not None and self.active.lookup is record:
            raise RuntimeError("cannot release lookup while its read is active")
        if self.reserved is record:
            self.reserved = None
        del self.lookups[request]
        if record.job is not None and not record.job.done():
            self.retired_lookups.append(record)
        self._reap_lookups()

    def _reap_lookups(self):
        self.retired_lookups[:] = [r for r in self.retired_lookups if not r.job.done()]

    def _snapshot(self):
        read = self.active
        if read is None:
            return None
        request = read.lookup.context.request
        prepared = read.prepare is not None and read.prepare.done()
        drained = valid = False
        if prepared:
            drained, result, _ = read.prepare.result()
            valid = result is not None
        ticket = read.lookup.result.transfer_id
        status = (
            self.backend.poll_prefetch(ticket)
            if read.started
            else read.synthetic_status
        )
        phase = self.backend.group_snapshot(ticket) if read.started else read.synthetic
        return (
            request.rid,
            request.attempt_id,
            prepared,
            drained,
            valid,
            read.cancelled,
            read.started,
            status,
            phase,
            read.uncertain,
        )

    def poll(self):
        self._check_owner()
        self._poll_read()
        self._publish_lookups()

    def _poll_read(self):
        self._reap_lookups()
        local = None
        try:
            if self.backend.owner_snapshot().unknown:
                self.disabled = True
            local = self._snapshot()
        except Exception:
            self.disabled = True
        try:
            votes = self.collective.gather((self.disabled, local))
            if any(v[0] for v in votes):
                raise LinkerQuarantined(
                    "linker disabled; retained ownership requires restart"
                )
            states = [v[1] for v in votes]
            if all(s is None for s in states):
                return
            if any(s is None for s in states) or any(
                s[:2] != states[0][:2] for s in states
            ):
                raise LinkerQuarantined("active handle mismatch")
            if any(
                s[9]
                or (s[2] and not s[3])
                or s[7] is PrefetchStatus.QUARANTINED
                or s[8][0] == "UNKNOWN"
                for s in states
            ):
                raise LinkerQuarantined("device drain uncertain")
            outcome = None
            error = False
            try:
                outcome = self._advance(states)
            except Exception:
                error = True
            if any(self.collective.gather(error)):
                raise LinkerQuarantined("partial progress/cleanup failed")
            if outcome is not None:
                read, self.active = self.active, None
                read.future.set_result(outcome)
        except Exception:
            self.disabled = True
            raise

    def _advance(self, states):
        read = self.active
        ticket = read.lookup.result.transfer_id
        cancelled = any(s[5] for s in states)
        read.cancelled |= cancelled
        if not all(s[2] for s in states):
            return None
        if not any(s[6] or s[7] is not None for s in states):
            if cancelled or not all(s[4] for s in states):
                self._retire(read)
                return False
            # Each worker initially waits at ENQUEUED. A local typed refusal
            # joins that rendezvous synthetically; no peer can Pin before DROP.
            _, result, slots = read.prepare.result()
            try:
                self.backend.prefetch(result, slots, identity=read.identity)
                read.started = True
            except PrefetchNotAccepted:
                read.synthetic_status = PrefetchStatus.FAILED_SAFE
            return None
        failed = cancelled or any(s[7] is PrefetchStatus.FAILED_SAFE for s in states)
        if failed and read.started:
            self.backend.cancel_prefetch(ticket)
        decision = worker_decision([s[8] for s in states], cancelled=failed)
        if decision == "QUARANTINE":
            raise LinkerQuarantined("worker rendezvous uncertain")
        if decision is not None:
            if read.started:
                self.backend.group_permit(ticket, decision)
            elif decision == "DROP":
                read.synthetic = ("RELEASE", False)
            elif decision == "ABORT":
                read.synthetic = ("DRAINED", False)
            elif decision == "RELEASE":
                read.synthetic = ("RELEASE", False)
        safe = (PrefetchStatus.ACKED, PrefetchStatus.READY, PrefetchStatus.FAILED_SAFE)
        if all(s[7] in safe for s in states):
            if (
                read.started
                and self.backend.poll_prefetch(ticket) is PrefetchStatus.ACKED
            ):
                if failed:
                    self.backend.cancel_prefetch(ticket)
                else:
                    self.backend.group_commit_ready(ticket)
            self._retire(read)
            return not failed
        return None

    def _retire(self, read):
        if read.started:
            self.backend.retire(read.lookup.result.transfer_id)
        self.gate.release(EngineRole.INBOUND, read.lookup.result.transfer_id)
        self.arbiter.release(TransportRole.INBOUND, read.identity)

    def assert_quiescent(self):
        self._check_owner()
        self._reap_lookups()
        owners = self.backend.owner_snapshot()
        if (
            self.disabled
            or self.active is not None
            or self.retired_lookups
            or any(
                r.job is not None and not r.job.done() for r in self.lookups.values()
            )
            or owners.accepted
            or owners.queued
            or owners.unknown
            or owners.working
            or owners.imports
            or owners.batches
        ):
            raise RuntimeError("linker owners prevent reset/close")

    def reset(self):
        self.assert_quiescent()
        # Wrapper releases each handle after its journal/anchor has been cleared.

    def close(self):
        self.assert_quiescent()
        self.closed = True
        self.backend.close()
        self._lookup_worker.shutdown(wait=False)
        self._prepare_worker.shutdown(wait=False)
