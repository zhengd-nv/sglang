# ruff: noqa: BLE001
# Driver failures retain bounded journals and require authoritative cleanup.
"""Independent CPU source Pin/Unpin workers with immutable physical grants.

No Scheduler inbox, tree adapter, allocator, TP collective or network belongs
here. This is a local-shard preparation service, not a deployable TP protocol.
Every accepted request, journal, response and replay binding is bounded. A
cancelled reply, timeout or expired lease never releases source ownership.
"""

import threading
import time
import uuid
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass, field

from . import slotguard_protocol as p
from .observation import emit
from .protocol import EngineIdentity, _integer, validate_peer
from .slot_guard import (
    LeaseIdentity,
    OwnerCounts,
    PageRef,
    PageUnavailable,
    ReplayConflict,
)
from .slotguard_control_wire import encode, reference_bytes
from .slotguard_receipts import emit_receipt, lease_receipt
from .transport_role_arbiter import TransportRole
from .wire_codec import decode_export, encode_export


class ServiceBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class ExternalSourceSnapshot:
    """External obligations only; valid while the allocator ledger is inactive."""

    requests: int
    leases: int
    uncertain: int
    publications: int
    exports: int
    transport_owners: int
    revision: tuple
    generation: int
    paused: bool

    @property
    def owners(self):
        return sum(
            (
                self.requests,
                self.leases,
                self.uncertain,
                self.publications,
                self.exports,
                self.transport_owners,
            )
        )


@dataclass(frozen=True)
class OwnerSnapshot:
    requests: int
    leases: int
    uncertain: int
    publications: int
    records: int
    retained_bytes: int
    exports: int
    transport_owners: int
    guard_counts: OwnerCounts
    revision: tuple
    generation: int
    fenced: bool
    sealed: bool

    @property
    def owners(self):
        """Proof obligations, with overlapping lease/transport categories."""
        return (
            self.requests
            + self.leases
            + self.publications
            + self.exports
            + self.transport_owners
            + self.guard_counts.active_leases
        )


@dataclass(frozen=True)
class LeaseSnapshot:
    identity: LeaseIdentity
    lease_id: str
    state: str


@dataclass
class _PinCharge:
    """One Pin's wire-equivalent/cache/response accounting, not Python RSS."""

    size: int
    holders: int = 1
    response_bytes: int = 0


@dataclass
class _Submission:
    request: p.PinReq | p.UnpinReq | None
    future: Future | None
    charge: _PinCharge | None = None
    replay_charge: _PinCharge | None = None
    active: bool = True
    admitted_ns: int = 0


@dataclass
class _Record:
    identity: LeaseIdentity
    reference: p.PinReference
    lease_id: str
    request_bytes: int
    binding_bytes: int
    lock: threading.Condition = field(default_factory=threading.Condition)
    state: str = "PREPARING"
    pages: tuple[PageRef, ...] = ()
    journal: list = field(default_factory=list)
    response: p.PinPreparedResp | None = None
    response_bytes: int = 0
    request_charge: _PinCharge | None = None
    pinned: bool = False
    role: bool = False
    pin_generation: int | None = None


class SlotGuardService:
    def __init__(
        self,
        *,
        guard,
        catalog,
        local_identity,
        exact_peers,
        engine,
        adapter,
        arbiter,
        max_requests=128,
        max_records=1024,
        max_retained_bytes=64 * 1024 * 1024,
        pin_workers=2,
        lease_duration_ms=30000,
        observer=None,
    ):
        for value in (
            max_requests,
            max_records,
            max_retained_bytes,
            pin_workers,
            lease_duration_ms,
        ):
            _integer("service limit", value, 1)
        self.observer = observer
        if (
            type(local_identity) is not EngineIdentity
            or type(exact_peers) is not tuple
            or not exact_peers
            or any(type(peer) is not EngineIdentity for peer in exact_peers)
            or len(set(exact_peers)) != len(exact_peers)
        ):
            raise ValueError("explicit exact local and peer identities required")
        for peer in exact_peers:
            validate_peer(local_identity, peer)
        if (
            guard.source_incarnation != local_identity.incarnation
            or catalog.guard is not guard
        ):
            raise ValueError("guard/catalog/local identity mismatch")
        if not guard.shared_headroom_credit_enabled:
            raise ValueError("independent service requires cumulative owner credit")
        adapter.layout.validate()
        self.required_buffer_names = tuple(
            buffer.name for buffer in adapter.layout.buffers
        )
        if set(self.required_buffer_names) != set(adapter.bindings):
            raise ValueError("all required adapter buffer bindings must exist")
        self.guard, self.catalog = guard, catalog
        self.local_identity, self.exact_peers = local_identity, frozenset(exact_peers)
        self.engine, self.adapter, self.arbiter = engine, adapter, arbiter
        self.max_requests, self.max_records = max_requests, max_records
        self.max_retained_bytes, self.lease_duration_ms = (
            max_retained_bytes,
            lease_duration_ms,
        )
        self._admission = threading.RLock()
        self._records = {}
        self._nonces = {}
        self._requests = {p.PinReq: 0, p.UnpinReq: 0}
        self._owners = self._uncertain = self._retained_bytes = self._revision = 0
        self._exports = 0
        self._generation = 0
        self._fenced = self._sealed = self._closed = self._pub_running = False
        self._close_admission = False
        self._maintenance_paused = False
        self._pins = ThreadPoolExecutor(
            max_workers=pin_workers, thread_name_prefix="slotguard-pin"
        )
        self._unpins = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slotguard-unpin"
        )
        self._publications = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="slotguard-publish"
        )

    @staticmethod
    def _busy(request):
        return (
            p.prepared_response(request, p.PinStatus.BUSY)
            if type(request) is p.PinReq
            else p.unpin_response(request, p.UnpinStatus.BUSY)
        )

    def submit(self, request) -> Future:
        if type(request) not in (p.PinReq, p.UnpinReq):
            raise ValueError("nominal v2 request required")
        result = Future()
        family = type(request)
        size = self._fingerprint(request)[1] if family is p.PinReq else 0
        with self._admission:
            if (
                self._closed
                or self._close_admission
                or self._requests[family] >= self.max_requests
                or (
                    family is p.PinReq
                    and (self._fenced or self._sealed or self._maintenance_paused)
                )
                or (
                    family is p.PinReq
                    and self._retained_bytes + size > self.max_retained_bytes
                )
            ):
                result.set_result(self._busy(request))
                return result
            self._requests[family] += 1
            charge = _PinCharge(size) if family is p.PinReq else None
            self._retained_bytes += size
            submission = _Submission(request, result, charge)
            if self.observer is not None:
                submission.admitted_ns = time.monotonic_ns()
            self._revision += 1
            executor = self._pins if family is p.PinReq else self._unpins
            try:
                # The executor retains only this clearable envelope, not an
                # unaccounted full request/Future in its positional arguments.
                executor.submit(self._run, submission)
            except BaseException:
                # submit can raise after putting its work item on the executor
                # queue (e.g. thread startup failure). _run first takes this
                # admission lock, so it cannot have started source work yet.
                submission.active = False
                submission.request = submission.future = None
                if charge is not None:
                    self._drop_charge(charge)
                self._requests[family] -= 1
                self._revision += 1
                raise
        return result

    def _hold_charge(self, charge):
        # All charge operations run under _admission. They retain no payload.
        if charge is None or charge.holders < 1:
            raise RuntimeError("Pin charge has no service holder")
        charge.holders += 1

    def _drop_charge(self, charge):
        if charge.holders < 1:
            raise RuntimeError("Pin charge already released")
        charge.holders -= 1
        if charge.holders == 0:
            self._retained_bytes -= charge.size + charge.response_bytes

    def _run(self, submission):
        with self._admission:
            if not submission.active:
                return  # Rolled back submit; queued envelope owns no payload.
            request, future = submission.request, submission.future
        family = type(request)
        value = error = None
        try:
            try:
                value = (
                    self._pin(request, submission)
                    if family is p.PinReq
                    else self._unpin(request)
                )
            except Exception as exc:
                error = exc
            try:
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(value)
            except InvalidStateError:
                # Cancellation affects delivery only. Accepted work still ran.
                pass
        finally:
            # set_result/set_exception invoke callbacks synchronously. Keep all
            # holds through callbacks, then explicitly clear every worker-owned
            # full payload before refunding its last charge (never rely on GC).
            request = future = value = error = None
            submission.request = submission.future = None
            with self._admission:
                submission.active = False
                for charge in (submission.replay_charge, submission.charge):
                    if charge is not None:
                        self._drop_charge(charge)
                self._requests[family] -= 1
                self._revision += 1

    def _peer_status(self, request):
        status = self._reference_peer_status(request.reference)
        if status is not None:
            return status
        if request.result.page_size != self.adapter.layout.page_size:
            return p.PinStatus.INCOMPATIBLE
        return None

    def _reference_peer_status(self, reference):
        source, target = reference.source, reference.target
        if source != self.local_identity:
            return (
                p.PinStatus.STALE_ENGINE
                if source.engine_id != self.local_identity.engine_id
                or source.incarnation != self.local_identity.incarnation
                else p.PinStatus.INCOMPATIBLE
            )
        if target not in self.exact_peers:
            return (
                p.PinStatus.STALE_ENGINE
                if any(
                    peer.engine_id == target.engine_id
                    and peer.incarnation != target.incarnation
                    for peer in self.exact_peers
                )
                else p.PinStatus.INCOMPATIBLE
            )
        return None

    @staticmethod
    def _fingerprint(request):
        # Count full semantic request wire-equivalent storage and its immutable
        # canonical body cache separately. Terminal records retain neither.
        return request.reference, request._wire_size + len(request._wire_body)

    def _record(self, request, reference):
        identity = request.identity
        binding_size = len(reference_bytes(reference))
        with self._admission:
            record = self._records.get(identity.transfer_id)
            if record is not None:
                if record.identity != identity or record.reference != reference:
                    raise ReplayConflict("transfer content/identity conflict")
                return record, False
            if identity.nonce in self._nonces:
                raise ReplayConflict("nonce is already bound")
            if (
                len(self._records) >= self.max_records
                or self._retained_bytes + binding_size > self.max_retained_bytes
            ):
                raise ServiceBusy("bounded lease/replay history exhausted")
            record = _Record(identity, reference, uuid.uuid4().hex, 0, binding_size)
            self._records[identity.transfer_id] = record
            self._nonces[identity.nonce] = identity
            self._retained_bytes += binding_size
            self._owners += 1
            self._revision += 1
            return record, True

    def _state(self, record, state):
        with self._admission:
            self._uncertain -= record.state == "UNCERTAIN"
            record.state = state
            self._uncertain += state == "UNCERTAIN"
            self._revision += 1
            record.lock.notify_all()

    def _terminal(self, record, state):
        with self._admission:
            self._state(record, state)
            self._owners -= 1
            # Exact compact replay/release history remains admitted and counted;
            # zero active ownership never means replay tombstones can be erased.
            record.request_bytes = record.response_bytes = 0
            record.pages = ()
            record.response = None
            if record.request_charge is not None:
                self._drop_charge(record.request_charge)
                record.request_charge = None

    def _release(self, record):
        # Journal pops only after explicit engine success; retries retain tail.
        while record.journal:
            self.engine.release_export(record.journal[-1].lease_id)
            record.journal.pop()
            with self._admission:
                self._exports -= 1
                self._revision += 1
        with self._admission:
            if record.pinned:
                self.guard.release(record.identity, record.pages)
                record.pinned = False
            if record.role:
                self.arbiter.release(TransportRole.OUTBOUND, record.identity)
                record.role = False

    def _pin(self, request, submission):
        if self.observer is not None:
            emit(
                self.observer,
                "source_pin_worker_start",
                request.identity.transfer_id,
                duration_ns=time.monotonic_ns() - submission.admitted_ns,
            )
        status = self._peer_status(request)
        if status is not None:
            return p.prepared_response(request, status)
        try:
            record, new = self._record(request, request.reference)
        except ReplayConflict:
            return p.prepared_response(request, p.PinStatus.INCOMPATIBLE)
        except ServiceBusy:
            return self._busy(request)
        with record.lock:
            if not new:
                while record.state == "PREPARING":
                    record.lock.wait()
                if record.state == "PREPARED":
                    with self._admission:
                        # A replay returns the ORIGINAL request object, while
                        # its independently decoded input has its own charge.
                        # Unpin may drop the record during callback delivery.
                        submission.replay_charge = record.request_charge
                        self._hold_charge(submission.replay_charge)
                    return record.response
                return p.prepared_response(
                    request,
                    (
                        p.PinStatus.BUSY
                        if record.state == "UNCERTAIN"
                        else p.PinStatus.MISS
                    ),
                )
            try:
                pin_cpu_start = time.thread_time_ns()
                stage_start = time.monotonic_ns() if self.observer is not None else 0
                pages = self.catalog.lookup(request.result)
                if self.observer is not None:
                    emit(
                        self.observer,
                        "source_pin_catalog",
                        request.identity.transfer_id,
                        duration_ns=time.monotonic_ns() - stage_start,
                    )
                if pages is None:
                    self._terminal(record, "FAILED")
                    return p.prepared_response(request, p.PinStatus.MISS)
                with self._admission:
                    # Accepted work may outlive a lookup or local epoch finish.
                    # Fence admission and acquisition in the same transaction.
                    if (
                        self._fenced
                        or self._sealed
                        or self._closed
                        or self._maintenance_paused
                    ):
                        raise ServiceBusy("Pin acquisition is fenced")
                    if not self.arbiter.acquire(
                        TransportRole.OUTBOUND, record.identity
                    ):
                        raise ServiceBusy("transport role busy")
                    record.role = True
                    record.pages = pages
                    stage_start = (
                        time.monotonic_ns() if self.observer is not None else 0
                    )
                    self.guard.try_pin(record.identity, pages)
                    if self.observer is not None:
                        emit(
                            self.observer,
                            "source_pin_guard",
                            request.identity.transfer_id,
                            duration_ns=time.monotonic_ns() - stage_start,
                        )
                    record.pinned = True
                    # Admission generation at exact guard acquisition, not a
                    # later fence counter read while exports are completing.
                    record.pin_generation = self._generation
                stage_start = time.monotonic_ns() if self.observer is not None else 0
                if set(self.required_buffer_names) != set(self.adapter.bindings):
                    raise ValueError("source full buffer set changed")
                for memory in self.adapter.memories:
                    record.journal.append(self.engine.export_memory(memory))
                    with self._admission:
                        self._exports += 1
                        self._revision += 1
                descriptors = {
                    descriptor.region_id: descriptor for descriptor in record.journal
                }
                if len(descriptors) != len(record.journal):
                    raise ValueError("aliased export regions")
                regions, bindings = encode_export(
                    self.local_identity, record.lease_id, self.adapter, descriptors
                )
                if self.observer is not None:
                    emit(
                        self.observer,
                        "source_pin_export",
                        request.identity.transfer_id,
                        duration_ns=time.monotonic_ns() - stage_start,
                    )
                stage_start = time.monotonic_ns() if self.observer is not None else 0
                response = p.prepared_response(
                    request,
                    p.PinStatus.PREPARED,
                    record.lease_id,
                    self.lease_duration_ms,
                    regions,
                    bindings,
                    pages,
                )
                built = time.monotonic_ns()
                # Validate the same complete v1 metadata that a target must check.
                decode_export(
                    response, self.adapter.layout, "validation-" + record.lease_id
                )
                checked = time.monotonic_ns()
                # The wire-equivalent semantic metadata and retained canonical
                # response body are separate fees, as for the initial full Pin.
                # The same body is reused by HTTP encoding; this is not RSS.
                response_size = len(encode(response)) + len(response._wire_body)
                if self.observer is not None:
                    now = time.monotonic_ns()
                    emit(
                        self.observer,
                        "source_pin_response",
                        request.identity.transfer_id,
                        duration_ns=now - stage_start,
                        build_ns=built - stage_start,
                        selfcheck_ns=checked - built,
                        size_ns=now - checked,
                        # Pin thread CPU from catalog lookup to here; the wall
                        # sum of the stages minus this is time it did not run.
                        pin_cpu_ns=time.thread_time_ns() - pin_cpu_start,
                    )
                with self._admission:
                    if self._retained_bytes + response_size > self.max_retained_bytes:
                        raise ServiceBusy("bounded completion storage exhausted")
                    self._retained_bytes += response_size
                    submission.charge.response_bytes = response_size
                    self._hold_charge(submission.charge)
                    record.request_charge = submission.charge
                    record.request_bytes = submission.charge.size
                    record.response_bytes = response_size
                    record.response = response
                    self._state(record, "PREPARED")
                if self.observer is not None:
                    emit_receipt(
                        self.observer,
                        "source_prepared",
                        record.identity.transfer_id,
                        lease_receipt(record.reference, record.lease_id, pages),
                        pin_generation=record.pin_generation,
                    )
                return response
            except Exception as error:
                try:
                    self._release(record)
                except Exception:
                    self._state(record, "UNCERTAIN")
                else:
                    self._terminal(record, "FAILED")
                status = (
                    p.PinStatus.MISS
                    if isinstance(error, PageUnavailable)
                    else (
                        p.PinStatus.INCOMPATIBLE
                        if isinstance(error, ReplayConflict)
                        else p.PinStatus.BUSY
                    )
                )
                return p.prepared_response(request, status)

    def _unpin(self, request):
        reference = request.reference
        status = self._reference_peer_status(reference)
        if status is not None:
            return p.unpin_response(request, p.UnpinStatus(status.value))
        with self._admission:
            record = self._records.get(reference.identity.transfer_id)
            if (
                record is None
                or record.reference != reference
                or record.lease_id != request.lease_id
            ):
                return p.unpin_response(request, p.UnpinStatus.INCOMPATIBLE)
        # The sole Unpin worker must never wait for another lease's prepare.
        # Even the creator-not-yet-locked window remains PREPARING and BUSY.
        if not record.lock.acquire(False):
            return p.unpin_response(request, p.UnpinStatus.BUSY)
        try:
            if record.state == "PREPARING":
                return p.unpin_response(request, p.UnpinStatus.BUSY)
            if record.state == "RELEASED":
                return p.unpin_response(request, p.UnpinStatus.ALREADY_RELEASED)
            if record.state == "FAILED":
                return p.unpin_response(request, p.UnpinStatus.INCOMPATIBLE)
            try:
                receipt = (
                    lease_receipt(record.reference, record.lease_id, record.pages)
                    if self.observer is not None and record.pages
                    else None
                )
                self._release(record)
            except Exception:
                self._state(record, "UNCERTAIN")
                return p.unpin_response(request, p.UnpinStatus.BUSY)
            self._terminal(record, "RELEASED")
            emit_receipt(
                self.observer,
                "source_unpin_released",
                record.identity.transfer_id,
                receipt,
            )
            return p.unpin_response(request, p.UnpinStatus.RELEASED)
        finally:
            record.lock.release()

    def query(self, identity):
        """Local authoritative reconciliation metadata; no exported handles."""
        if type(identity) is not LeaseIdentity:
            raise TypeError("exact lease identity required")
        with self._admission:
            record = self._records.get(identity.transfer_id)
            if record is None or record.identity != identity:
                return None
            return LeaseSnapshot(record.identity, record.lease_id, record.state)

    def submit_publication(self, publication):
        with self._admission:
            if self._closed or self._fenced or self._sealed or self._maintenance_paused:
                raise ServiceBusy("publications are fenced")
            cid = self.catalog.submit(publication)
            self._revision += 1
        self.publication_progress()
        return cid

    def publication_progress(self):
        """At most one bounded readiness sweep; event queries must not wait."""
        with self._admission:
            if self._closed or self._sealed or self._pub_running:
                return None
            self._pub_running = True
            self._revision += 1
            return self._publications.submit(self._progress)

    def _progress(self):
        try:
            return self.catalog.progress()
        finally:
            with self._admission:
                self._pub_running = False
                self._revision += 1

    def fence_pins(self, generation):
        _integer("generation", generation)
        with self._admission:
            if generation < self._generation:
                raise ValueError("stale fence generation")
            self._generation = generation
            self._fenced = True
            self._revision += 1

    def pause_for_maintenance(self, generation):
        """Reversible Pin/publication fence; accepted exact Unpin keeps draining."""
        _integer("generation", generation)
        with self._admission:
            if self._closed or self._sealed or generation < self._generation:
                raise ServiceBusy("maintenance cannot reopen a final seal")
            self._generation = generation
            self._fenced = self._maintenance_paused = True
            self._revision += 1

    def resume_after_maintenance(self, generation):
        _integer("generation", generation)
        with self._admission:
            if self._closed or self._sealed or generation != self._generation:
                raise ServiceBusy("maintenance generation mismatch or final seal")
            self._maintenance_paused = self._fenced = False
            self._revision += 1

    def external_snapshot(self):
        """Never consult SlotGuard/CPU rows; replay records are not live owners."""
        with self._admission:
            catalog, roles = self.catalog.snapshot(), self.arbiter.owner_snapshot()
            return ExternalSourceSnapshot(
                sum(self._requests.values()) + self._pub_running,
                self._owners,
                self._uncertain,
                catalog.pending,
                self._exports,
                roles.owners,
                (self._revision, catalog.revision, roles.revision),
                self._generation,
                self._maintenance_paused,
            )

    def unfence_pins(self, generation):
        """Owner calls only after its all-rank epoch/credit finish gate."""
        _integer("generation", generation)
        with self._admission:
            if generation != self._generation or self._sealed or self._closed:
                return False
            self._fenced = False
            self._revision += 1
            return True

    def prevalidate_unfence_pins(self, generation):
        """Read-only owner vote; maintenance fencing must not permanently seal."""
        _integer("generation", generation)
        with self._admission:
            if (
                generation != self._generation
                or not self._fenced
                or self._sealed
                or self._closed
            ):
                raise ServiceBusy("source admission cannot finish this epoch")

    def owner_snapshot(self):
        with self._admission:
            catalog, roles = self.catalog.snapshot(), self.arbiter.owner_snapshot()
            counts = self.guard.owner_counts()
            return OwnerSnapshot(
                sum(self._requests.values()) + self._pub_running,
                self._owners,
                self._uncertain,
                catalog.pending,
                len(self._records),
                self._retained_bytes,
                self._exports,
                roles.owners,
                counts,
                (self._revision, catalog.revision, roles.revision, counts),
                self._generation,
                self._fenced,
                self._sealed,
            )

    def _empty_locked(self):
        return (
            not any(self._requests.values())
            and not self._pub_running
            and not self._owners
            and not self._exports
            and not self._uncertain
            and self.catalog.snapshot().pending == 0
            and self.arbiter.owner_snapshot().owners == 0
            and self.guard.owner_counts().active_leases == 0
        )

    def seal_if_empty(self, generation, *, close_admission=False):
        _integer("generation", generation)
        if type(close_admission) is not bool:
            raise ValueError("close admission requires explicit bool")
        with self._admission:
            if generation != self._generation or not self._fenced:
                return False
            if not self._empty_locked():
                return False
            self._sealed = True
            self._close_admission |= close_admission
            self._revision += 1
            return True

    def assert_quiescent(self):
        with self._admission:
            if not self._empty_locked():
                raise ServiceBusy(
                    "accepted work, uncertain publication or lease remains"
                )

    def close(self):
        with self._admission:
            # Pin acceptance and shutdown share this lock. On rejection remain
            # fenced, while exact Unpin and known cleanup are still admitted.
            self._fenced = True
            self._revision += 1
            if not self._empty_locked():
                raise ServiceBusy("close requires proven quiescence")
            self._sealed = True
            self._close_admission = True
            self._closed = True
        for executor in (self._pins, self._unpins, self._publications):
            executor.shutdown(wait=True)
