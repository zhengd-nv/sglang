"""Target prefetch lifecycle over registered TP1 MHA or coordinated TP4 GLM pools.

Scheduler calls enqueue/snapshot/cancel only. A single bounded daemon worker owns
control Futures and all CUDA operations; synchronous KVCM lookup is worker-only
and is dispatched by the serving admission worker. The serving factory installs
the explicit private Pin network service. Caller must keep its adapter, pool, engine and
private slots alive and unchanged until terminal adoption/free is acknowledged
with retire(). QUARANTINED retains ownership indefinitely for reconciliation.
"""

import math
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Protocol

from .control_profile import (
    ControlProfile,
    PinRequest,
    PinResponse,
    UnpinRequest,
    UnpinResponse,
    require_profile,
)
from .directory import EngineDirectory
from .kvcm_client import KVCMClient
from .observation import emit
from .protocol import (
    LookupQuery,
    LookupResult,
    PrefetchStatus,
    PrivateSlotView,
    TransferId,
    _integer,
    _typed,
)
from .role_gate import EngineRole, RoleGate, SchedulerThreadViolation
from .slotguard_receipts import emit_receipt, lease_receipt
from .transport.api import BusyError, TransferState
from .transport.kv_peer_engine import KvPeerEngine
from .transport.sglang_adapter import SGLangPoolAdapter
from .wire_codec import decode_export


class PrefixControlClient(Protocol):
    """Injected transport; methods run on the backend worker, never Scheduler.

    Futures must preserve exact typed request/response identity. A lost Pin reply
    may hide a live source lease; the backend must retain the attempt and cannot
    manufacture an Unpin. Network authentication/serving integration is external.
    """

    def pin(self, endpoint: str, request: PinRequest) -> Future[PinResponse]: ...
    def unpin(self, endpoint: str, request: UnpinRequest) -> Future[UnpinResponse]: ...


class PrefetchNotAccepted(BusyError):
    """Definite pre-enqueue refusal; no worker has received this attempt's slots.

    Only this typed failure authorizes admission to release a drained metadata
    reservation. An arbitrary BusyError/exception does not prove non-submission.
    """


@dataclass
class _Record:
    result: LookupResult
    slots: PrivateSlotView
    endpoint: str
    identity: object = None
    cancel: threading.Event = field(default_factory=threading.Event)
    status: PrefetchStatus = PrefetchStatus.PIN_PENDING
    done: bool = False
    # Worker-only resource journal. Never read/mutated by Scheduler APIs.
    pin_future: Future | None = None
    response: PinResponse | None = None
    imports: dict = field(default_factory=dict)
    batch: object = None
    unpin_futures: list = field(default_factory=list)
    failure: str | None = None
    receipt: object = None


@dataclass(frozen=True)
class _Retired:
    status: PrefetchStatus


class NVLinkTransferBackend:
    """One accepted, unretired attempt; at most 4096 records per incarnation.

    No timeout unlocks the source. Pin loss, uncertain DMA, ANY import exception,
    or failed unmap quarantines the retained journal. Even an import exception
    before returning a handle may leave engine-owned native cleanup outstanding.
    Unpin retry resends the identical lease proof, never reimports or recopies.
    """

    def __init__(
        self,
        engine: KvPeerEngine,
        adapter: SGLangPoolAdapter,
        directory: EngineDirectory,
        kvcm: KVCMClient,
        control: PrefixControlClient,
        role_gate: RoleGate,
        *,
        rpc_timeout_s: float = 1.0,
        copy_timeout_s: float = 30.0,
        poll_interval_s: float = 0.001,
        unpin_attempts: int = 3,
        max_records: int = 4096,
        observer=None,
        verify_bytes: bool = False,
        group_coordinated: bool = False,
        control_profile: ControlProfile,
        transport_arbiter=None,
        async_unpin: bool = False,
    ):
        _typed("engine", engine, KvPeerEngine)
        if type(async_unpin) is not bool:
            raise ValueError("async_unpin must be an exact bool")
        self._async_unpin = async_unpin
        self._profile = require_profile(control_profile)
        self.transport_arbiter = transport_arbiter
        if transport_arbiter is not None:
            from .transport_role_arbiter import TransportRoleArbiter

            if (
                type(transport_arbiter) is not TransportRoleArbiter
                or directory.target.tp_size != 4
                or not group_coordinated
            ):
                raise ValueError(
                    "exact transport arbiter requires coordinated TP4 SlotGuard"
                )
        if hasattr(control, "_profile") and control._profile is not self._profile:
            raise ValueError("backend/client control profiles must match")
        _typed("adapter", adapter, SGLangPoolAdapter)
        _typed("directory", directory, EngineDirectory)
        _typed("kvcm", kvcm, KVCMClient)
        _typed("role_gate", role_gate, RoleGate)
        role_gate.check_thread()
        if adapter.engine is not engine or not adapter.memories:
            raise ValueError("adapter must already be registered with this engine")
        if directory.target.tp_size != 1 and not (
            directory.target.tp_size == 4 and group_coordinated
        ):
            raise ValueError(
                "target backend requires TP=1 or the TP4 group coordinator"
            )
        for value in (rpc_timeout_s, copy_timeout_s, poll_interval_s):
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not 0 < value <= 60
            ):
                raise ValueError("worker timeout/interval must be finite in (0,60]")
        _integer("unpin_attempts", unpin_attempts, 1)
        _integer("max_records", max_records, 1)
        if unpin_attempts > 16 or max_records > 4096:
            raise ValueError("retry/record bound exceeded")
        self._engine, self._adapter, self._directory = engine, adapter, directory
        self.group_coordinated = group_coordinated
        self.observer, self._verify_bytes = observer, verify_bytes
        self._kvcm, self._control, self._gate = kvcm, control, role_gate
        self._owner = threading.get_ident()
        self._rpc_timeout, self._copy_timeout = rpc_timeout_s, copy_timeout_s
        self._interval, self._attempts, self._max_records = (
            poll_interval_s,
            unpin_attempts,
            max_records,
        )
        self._lock = threading.Lock()
        self._records = {}
        self._latest = {}
        self._live = None
        self._owner_revision = self._import_owners = self._batch_owners = 0
        self._worker_active = False
        self._unknown_owners = 0
        self._closed = False
        self._queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(
            target=self._work, name="l1-nvlink-target", daemon=True
        )
        self._worker.start()

    def _check_thread(self):
        if threading.get_ident() != self._owner:
            raise SchedulerThreadViolation("target backend belongs to Scheduler thread")

    def lookup(self, query: LookupQuery) -> LookupResult:
        """Blocking KVCM read: call from an external background lookup worker."""
        if threading.get_ident() == self._owner:
            raise SchedulerThreadViolation("KVCM lookup must run off Scheduler")
        if query.target != self._directory.target:
            raise ValueError("lookup target identity mismatch")
        start = time.monotonic_ns()
        result = self._kvcm.lookup(query)
        emit(
            self.observer,
            "lookup",
            query.transfer_id,
            local_tokens=query.local_hit_tokens,
            hit_tokens=result.hit_tokens,
            duration_ns=time.monotonic_ns() - start,
        )
        return result

    def prefetch(
        self,
        result: LookupResult,
        slots: PrivateSlotView,
        *,
        identity=None,
    ) -> TransferId:
        self._check_thread()
        _typed("result", result, LookupResult)
        _typed("slots", slots, PrivateSlotView)
        slots.validate_for(result)
        transfer_id = result.transfer_id
        if self.transport_arbiter is not None:
            from .slot_guard import LeaseIdentity
            from .transport_role_arbiter import TransportRole

            if (
                type(identity) is not LeaseIdentity
                or identity.transfer_id != transfer_id
                or identity.source_incarnation != result.source.incarnation
                or identity.target_incarnation != result.query.target.incarnation
                or not self.transport_arbiter.owns(TransportRole.INBOUND, identity)
            ):
                raise ValueError(
                    "exact preallocated INBOUND lease reservation required"
                )
        self._gate.require_owner(EngineRole.INBOUND, transfer_id)
        if self._closed:
            raise PrefetchNotAccepted("backend is closed")
        if result.query.target != self._directory.target:
            raise ValueError("target identity/directory mismatch")
        entries = [
            entry
            for entry in self._directory.entries
            if entry.identity == result.source
        ]
        if len(entries) != 1:
            raise ValueError("exact source identity absent from directory")
        if slots.page_size != self._adapter.layout.page_size or any(
            page >= self._adapter.capacity for page in slots.page_ids
        ):
            raise ValueError("private slots exceed target layout/capacity")
        key = (transfer_id.target_incarnation, transfer_id.rid)
        if transfer_id.attempt < self._latest.get(key, -1):
            raise ValueError("stale transfer attempt")
        previous = self._records.get(transfer_id)
        if previous is not None:
            # This is a LOCAL enqueue API, not an RPC. Exact object identity is
            # a constant-time fence; do not hash/serialize an entire prompt on
            # Scheduler. Reconstructed equal objects are intentionally rejected.
            if isinstance(previous, _Retired):
                raise ValueError("retired transfer cannot be enqueued again")
            if (
                previous.result is not result
                or previous.slots is not slots
                or previous.identity != identity
            ):
                raise ValueError("conflicting or reconstructed duplicate transfer")
            return transfer_id
        if self._live is not None or len(self._records) >= self._max_records:
            raise PrefetchNotAccepted("target attempt/record capacity exhausted")
        record = _Record(result, slots, entries[0].control_endpoint, identity)
        if self.group_coordinated:
            from .tp_group import WorkerRendezvous

            record.group = WorkerRendezvous()
        self._records[transfer_id] = record
        self._latest[key] = transfer_id.attempt
        self._live = transfer_id
        self._owner_revision += 1
        self._queue.put_nowait(record)
        return transfer_id

    def poll_prefetch(self, transfer_id: TransferId) -> PrefetchStatus:
        self._check_thread()
        with self._lock:
            return self._records[transfer_id].status

    def group_snapshot(self, transfer_id):
        self._check_thread()
        return self._records[transfer_id].group.snapshot()

    def group_permit(self, transfer_id, decision):
        self._check_thread()
        record = self._records[transfer_id]
        before_release = None
        if decision == "COPY" and record.receipt is not None:
            before_release = lambda: emit_receipt(
                self.observer,
                "group_copy",
                transfer_id,
                record.receipt,
                decision="COPY",
                ranks=4,
            )
        record.group.permit(decision, before_release=before_release)

    def group_commit_ready(self, transfer_id):
        self._check_thread()
        record = self._records[transfer_id]
        if (
            not self.group_coordinated
            or record.status is not PrefetchStatus.ACKED
            or not record.done
        ):
            raise RuntimeError("group READY requires local completed Unpin ACK")
        emit_receipt(
            self.observer,
            "group_ready",
            transfer_id,
            record.receipt,
            decision="READY",
            ranks=4,
        )
        self._publish(record, PrefetchStatus.READY, done=True)

    def cancel_prefetch(self, transfer_id: TransferId) -> PrefetchStatus:
        self._check_thread()
        with self._lock:
            record = self._records[transfer_id]
            if isinstance(record, _Record):
                record.cancel.set()
                if record.status in (PrefetchStatus.READY, PrefetchStatus.ACKED):
                    record.status = PrefetchStatus.FAILED_SAFE
            return record.status

    def retire(self, transfer_id: TransferId):
        """Relinquish a drained transport; backend frees no allocator/tree state.

        TP4 READY parking may retire before normal row adoption: private slots
        and the local anchor receipt remain owned by the Scheduler reservation.
        """
        self._check_thread()
        with self._lock:
            record = self._records[transfer_id]
            if isinstance(record, _Retired):
                return
            if not record.done or record.status not in (
                PrefetchStatus.READY,
                PrefetchStatus.FAILED_SAFE,
            ):
                raise BusyError("cannot retire live or quarantined resources")
            self._records[transfer_id] = _Retired(record.status)
            self._live = None
            self._owner_revision += 1

    def owner_snapshot(self):
        """Short counters, without acquiring the driver lock or its journals.

        The accepted attempt remains charged from enqueue through retirement,
        including native calls which have not yet returned a journal handle.
        """
        from .slotguard_lifecycle import BackendOwners

        self._check_thread()
        with self._lock:
            return BackendOwners(
                int(self._live is not None),
                self._queue.qsize(),
                int(self._worker_active),
                self._import_owners,
                self._batch_owners,
                self._unknown_owners,
                self._owner_revision,
            )

    def _resource_owners(self, *, imports=0, batches=0, working=None, unknown=0):
        with self._lock:
            self._import_owners += imports
            self._batch_owners += batches
            self._unknown_owners += unknown
            if working is not None:
                self._worker_active = working
            self._owner_revision += 1

    def close(self):
        """Nonblocking worker shutdown; does not close borrowed engine/pool."""
        self._check_thread()
        if self._live is not None:
            raise BusyError(
                "retire all safe attempts before close; quarantine needs reconciliation"
            )
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    def _publish(self, record, status, *, done=False):
        with self._lock:
            if (
                status in (PrefetchStatus.READY, PrefetchStatus.ACKED)
                and record.cancel.is_set()
            ):
                status = PrefetchStatus.FAILED_SAFE
            # Commit the trace transition before exposing the state to Scheduler
            # pollers. emit only enqueues bounded scalars; it performs no I/O.
            emit(
                self.observer,
                "backend_state",
                record.result.transfer_id,
                status=status.value,
            )
            record.status, record.done = status, done

    def _work(self):
        while True:
            record = self._queue.get()
            if record is None:
                return
            self._resource_owners(working=True)
            try:
                self._execute(record)
            except BaseException as exc:  # noqa: BLE001 - any lost worker must quarantine
                # No traceback retention (may capture prompt/transport objects).
                record.failure = type(exc).__name__
                self._resource_owners(unknown=1, working=False)
                if self.group_coordinated:
                    record.group.unknown()
                self._publish(record, PrefetchStatus.QUARANTINED, done=True)
                return  # No new work is legal until explicit reconciliation.
            self._resource_owners(working=False)
            # Drop the worker's full prompt after the Scheduler retires its record.
            record = None

    def _execute(self, record):
        if self.group_coordinated:
            decision = record.group.arrive("ENQUEUED", True)
            if decision == "DROP":
                self._publish(record, PrefetchStatus.FAILED_SAFE, done=True)
                return
            if decision != "PIN":
                raise RuntimeError("group did not authorize Pin submission")
        if record.cancel.is_set() and not self.group_coordinated:
            self._publish(record, PrefetchStatus.FAILED_SAFE, done=True)
            return
        request = self._profile.make_pin(record.result, identity=record.identity)
        ticket = record.result.transfer_id
        start = time.monotonic_ns()
        emit(
            self.observer,
            "pin_start",
            ticket,
            local_tokens=record.result.query.local_hit_tokens,
            hit_tokens=record.result.hit_tokens,
        )
        record.pin_future = self._control.pin(record.endpoint, request)
        response = record.pin_future.result(timeout=self._rpc_timeout)
        if not self._profile.pin_matches(response, request):
            raise ValueError("Pin reply does not match exact request")
        record.response = response
        if self._profile.prepared(response) and self.observer is not None:
            record.receipt = lease_receipt(
                request.reference, response.lease_id, response.pages
            )
            emit_receipt(self.observer, "target_pin_validated", ticket, record.receipt)
        emit(
            self.observer,
            "pin_reply",
            ticket,
            status=response.status.value,
            duration_ns=time.monotonic_ns() - start,
        )
        if self.group_coordinated:
            decision = record.group.arrive("PIN", self._profile.prepared(response))
            if decision == "ABORT":
                record.cancel.set()
            elif decision != "COPY":
                raise RuntimeError("group did not authorize Pin commit")
        if not self._profile.prepared(response):
            if (
                self.group_coordinated
                and record.group.arrive("DRAINED", False) != "RELEASE"
            ):
                raise RuntimeError("group did not authorize release")
            self._publish(record, PrefetchStatus.FAILED_SAFE, done=True)
            return
        if record.cancel.is_set():
            self._cleanup(record, success=False)
            return
        try:
            metadata = decode_export(
                response, self._adapter.layout, self._engine.owner_id
            )
            plan = self._engine.plan_pages(
                metadata.layout,
                self._adapter.layout,
                response.source_page_ids,
                record.slots.page_ids,
                source_capacity=metadata.capacity,
                target_capacity=self._adapter.capacity,
            )
        except Exception:  # noqa: BLE001 - malformed wire input before any CUDA call
            # Malformed metadata before import: exact lease, zero DMA/imports.
            self._cleanup(record, success=False)
            return
        start = time.monotonic_ns()
        for descriptor in metadata.descriptors:
            if record.cancel.is_set():
                self._cleanup(record, success=False)
                return
            # Any exception may retain engine-native cleanup without a public
            # handle. Outer worker quarantines; empty journal is NOT unmap proof.
            record.imports[descriptor.region_id] = self._engine.import_memory(
                descriptor
            )
            self._resource_owners(imports=1)
        emit(
            self.observer,
            "import_done",
            ticket,
            duration_ns=time.monotonic_ns() - start,
        )
        try:
            requests = self._engine.bind_pages(
                plan, metadata.bind(record.imports), self._adapter.bindings
            )
        except Exception:  # noqa: BLE001 - pure planning, known imports, no DMA
            self._cleanup(record, success=False)
            return
        if record.cancel.is_set():
            self._cleanup(record, success=False)
            return
        self._publish(record, PrefetchStatus.COPYING)
        start = time.monotonic_ns()
        record.batch = self._engine.submit_transfer(requests)
        self._resource_owners(batches=1)
        emit(
            self.observer,
            "copy_submitted",
            ticket,
            requested_bytes=requests.requested_bytes,
            ranges=len(plan.src_pages) * requests.num_buffers,
        )
        deadline = time.monotonic() + self._copy_timeout
        while True:
            status = self._engine.get_transfer_status(record.batch)
            if status.state is not TransferState.IN_FLIGHT:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("DMA has not proven drain")
            time.sleep(self._interval)
        if not status.drained:
            raise RuntimeError("DMA completion uncertain")
        emit_receipt(self.observer, "target_dma_drained", ticket, record.receipt)
        emit(
            self.observer,
            "copy_done",
            ticket,
            status=status.state.value,
            requested_bytes=status.requested_bytes,
            completed_bytes=status.completed_bytes or 0,
            duration_ns=time.monotonic_ns() - start,
        )
        if self._verify_bytes and status.state is TransferState.SUCCESS:
            from .transport.kv_peer_engine import verify_pages

            start = time.monotonic_ns()
            verify_pages(self._engine, requests)
            emit(
                self.observer,
                "verify_buffer",
                ticket,
                buffers=requests.num_buffers,
                requested_bytes=requests.requested_bytes,
                verified=True,
            )
            emit(
                self.observer,
                "verify_done",
                ticket,
                verified=True,
                duration_ns=time.monotonic_ns() - start,
            )
        self._cleanup(record, success=status.state is TransferState.SUCCESS)

    def _cleanup(self, record, *, success):
        ticket = record.result.transfer_id
        self._publish(record, PrefetchStatus.RELEASING)
        if record.batch is not None:
            self._engine.release_batch(record.batch)
            record.batch = None
            self._resource_owners(batches=-1)
        while record.imports:
            region_id = next(reversed(record.imports))
            self._engine.unregister_memory(record.imports[region_id])
            del record.imports[region_id]
            self._resource_owners(imports=-1)
        emit(self.observer, "imports_released", ticket)
        emit_receipt(self.observer, "target_imports_unmapped", ticket, record.receipt)
        if (
            self.group_coordinated
            and record.group.arrive("DRAINED", success) != "RELEASE"
        ):
            raise RuntimeError("group did not authorize Unpin")
        request = self._profile.make_unpin(record.response)
        if self._async_unpin and success:
            # DMA has drained and every import is unmapped, so the private pages
            # are final; only the source lease release remains.
            self._publish(
                record,
                PrefetchStatus.ACKED
                if self.group_coordinated
                else PrefetchStatus.READY,
                done=True,
            )
            emit(self.observer, "unpin_deferred", ticket)
            if not self._send_unpin(record, ticket, request):
                # The record may already be READY and retired. Keep the
                # unreleased source lease visible to lifecycle checks instead.
                emit(
                    self.observer,
                    "unpin_failed",
                    ticket,
                    status=record.failure or "NOT_RELEASED",
                )
                self._resource_owners(unknown=1)
            return
        if self._send_unpin(record, ticket, request):
            self._publish(
                record,
                (
                    (
                        PrefetchStatus.ACKED
                        if self.group_coordinated
                        else PrefetchStatus.READY
                    )
                    if success
                    else PrefetchStatus.FAILED_SAFE
                ),
                done=True,
            )
            return
        raise TimeoutError("no matching release ACK; retain attempt for reconciliation")

    def _send_unpin(self, record, ticket, request):
        for _ in range(self._attempts):
            start = time.monotonic_ns()
            try:
                future = self._control.unpin(record.endpoint, request)
                record.unpin_futures.append(future)
                response = future.result(timeout=self._rpc_timeout)
            except Exception as exc:  # noqa: BLE001 - transport errors are retryable
                record.failure = type(exc).__name__
                continue
            if self._profile.released(response, request):
                emit_receipt(self.observer, "target_unpin_ack", ticket, record.receipt)
                emit(
                    self.observer,
                    "unpin_ack",
                    ticket,
                    status=response.status.value,
                    duration_ns=time.monotonic_ns() - start,
                )
                return True
        return False
