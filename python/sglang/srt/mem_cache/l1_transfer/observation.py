"""Opt-in bounded scalar evidence; no Scheduler disk I/O or CUDA work.

The authenticated control endpoint reads snapshots independently of the JSONL
writer. A dropped event, ring overrun or writer error invalidates smoke evidence.
"""

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path


class TransferObserver:
    FIELDS = frozenset(
        {
            "local_tokens",
            "hit_tokens",
            "reserved_tokens",
            "pinned_tokens",
            "requested_bytes",
            "completed_bytes",
            "duration_ns",
            "status",
            "ranges",
            "buffers",
            "verified",
            "remaining_prefill_tokens",
            "published_tokens",
            "iteration_ns",
            "stage_ns",
            "free_tokens",
            "prefill_tokens",
            "prefill_start",
            # l1-opt measurement diagnostics (GIL and stage attribution).
            "thread_cpu_ns",
            "active",
            "admission_records",
            "source_owners",
            "build_ns",
            "selfcheck_ns",
            "size_ns",
            "pin_cpu_ns",
            "alloc_ns",
            "cpu_ns",
            "materialize_ns",
            "kind",
        }
    )

    def __init__(
        self, path, engine_id, *, capacity=4096, tp_rank=None, tp_group_id=None
    ):
        if not Path(path).is_absolute() or not 1 <= capacity <= 16384:
            raise ValueError("absolute trace path and bounded capacity required")
        if not isinstance(engine_id, str) or len(engine_id) > 128:
            raise ValueError("trace engine ID exceeds bound")
        # Fail at startup, never silently run without the requested evidence.
        self._file = open(path, "x", encoding="utf-8")  # noqa: SIM115 -- lifecycle owned by writer thread
        self.engine_id = engine_id
        if tp_rank is not None and (
            type(tp_rank) is not int
            or not 0 <= tp_rank < 4
            or not isinstance(tp_group_id, str)
            or len(tp_group_id) > 128
        ):
            self._file.close()
            raise ValueError("invalid bounded TP trace identity")
        self.rank_fields = (
            {} if tp_rank is None else {"tp_rank": tp_rank, "tp_group_id": tp_group_id}
        )
        self._queue = queue.Queue(maxsize=capacity)
        self._ring = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = self._written = self._dropped = self._errors = 0
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="l1-trace", daemon=True)
        self._worker.start()

    def emit(self, phase, ticket=None, **fields):
        # Reject unsupported payloads rather than serializing arbitrary objects.
        if (
            not isinstance(phase, str)
            or len(phase) > 64
            or (ticket is not None and len(ticket.rid) > 256)
            or set(fields) - self.FIELDS
            or any(
                type(v) not in (str, int, bool, float)
                or (isinstance(v, str) and len(v) > 128)
                for v in fields.values()
            )
        ):
            with self._lock:
                self._errors += 1
            return
        with self._lock:
            if self._closed:
                self._dropped += 1
                return
            self._seq += 1
            event = dict(
                seq=self._seq,
                monotonic_ns=time.monotonic_ns(),
                engine=self.engine_id,
                phase=phase,
                **self.rank_fields,
                **fields,
            )
            if ticket is not None:
                event.update(rid=ticket.rid, attempt=ticket.attempt)
            self._ring.append(event)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped += 1

    def snapshot(self):
        with self._lock:
            return {
                "schema": 1,
                "engine": self.engine_id,
                "seq": self._seq,
                "written": self._written,
                "dropped": self._dropped,
                "errors": self._errors,
                "closed": self._closed,
                "capacity": self._ring.maxlen,
                "first_seq": self._ring[0]["seq"] if self._ring else None,
                "ring_overruns": self._seq - len(self._ring),
                "good": not (self._dropped or self._errors)
                and self._written == self._seq
                and self._seq == len(self._ring),
                "events": list(self._ring),
            }

    def emit_slotguard(self, phase, ticket=None, **fields):
        """Finite SlotGuard receipts share the existing bounded sink/counters.

        No arbitrary object, nested metadata or newly invented phase is accepted.
        Callers must precompute digests on publication/service/backend workers.
        The legacy emit allowlist and its semantics are intentionally unchanged.
        """
        from .slotguard_receipts import valid_receipt

        if not valid_receipt(phase, fields, ticket):
            with self._lock:
                self._errors += 1
            return
        with self._lock:
            if self._closed:
                self._dropped += 1
                return
            self._seq += 1
            event = dict(
                seq=self._seq,
                monotonic_ns=time.monotonic_ns(),
                engine=self.engine_id,
                phase="slotguard_" + phase,
                **self.rank_fields,
                **fields,
            )
            if ticket is not None:
                event.update(rid=ticket.rid, attempt=ticket.attempt)
            self._ring.append(event)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped += 1

    def emit_destination_prefill(self, rid, **fields):
        """Explicit native RID, without inventing a transfer ticket or Req owner.

        Keep the transfer emitter's allowlist unchanged. Both APIs use the same
        bounded queue/ring, sequence and externally visible loss/error counters.
        """
        integers = {
            "input_tokens",
            "prefill_start",
            "prefill_tokens",
            "cached_tokens",
            "cached_tokens_device",
            "cached_tokens_host",
            "cached_tokens_storage",
            "retraction_count",
            "cache_attempt",
        }
        base = integers | {"sharing_enabled", "record_valid", "record_status"}
        record = {"record_local_tokens", "record_hit_tokens", "transfer_attempt"}
        valid = fields.get("record_valid") is True
        provenance = (
            record
            if valid
            else {"miss_attempt"}
            if fields.get("record_status") == "miss"
            else set()
        )
        if (
            type(rid) is not str
            or not 1 <= len(rid) <= 256
            or set(fields) != base | provenance
            or type(fields.get("record_valid")) is not bool
            or type(fields.get("sharing_enabled")) is not bool
            or type(fields.get("record_status")) is not str
            or fields.get("record_status")
            not in (("valid",) if valid else ("missing", "invalid", "miss"))
            or any(
                type(fields.get(k)) is not int or not 0 <= fields[k] < 2**63
                for k in integers | provenance
            )
        ):
            with self._lock:
                self._errors += 1
            return
        with self._lock:
            if self._closed:
                self._dropped += 1
                return
            self._seq += 1
            event = dict(
                seq=self._seq,
                monotonic_ns=time.monotonic_ns(),
                engine=self.engine_id,
                phase="destination_prefill_dispatch",
                rid=rid,
                **self.rank_fields,
                **fields,
            )
            self._ring.append(event)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped += 1

    def _run(self):
        while True:
            event = self._queue.get()
            if event is None:
                self._file.close()
                return
            try:
                self._file.write(json.dumps(event, separators=(",", ":")) + "\n")
                self._file.flush()
                with self._lock:
                    self._written += 1
            except Exception:  # noqa: BLE001 -- sink errors must be observable without changing transfer safety
                with self._lock:
                    self._errors += 1

    def close(self):
        # Lifecycle thread only. Scheduler never waits for filesystem I/O.
        with self._lock:
            self._closed = True
        self._queue.put(None)
        self._worker.join()


def emit(observer, phase, ticket=None, **fields):
    if observer is not None:
        observer.emit(phase, ticket, **fields)
