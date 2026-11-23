# ruff: noqa: BLE001
# Publication failures are safe misses; unproven producer readiness retains owners.
"""Canonical Full sidecars and a bounded, independent CPU publication worker.

The Tree owner freezes only dirty CPU bytes and stages immutable views during
its existing walk. It commits after cache actions succeed. No TreeNode, Req,
device tensor or allocator is handed to the worker. Cursors are local opaque
content coordinates, never native ROOT keys or wire identities.
"""

import threading
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass, replace

from .protocol import _integer
from .slot_guard import PageUnavailable
from .slotguard_catalog import CatalogBusy, Publication, PublicationScope


@dataclass(frozen=True)
class ContentCursor:
    context: object
    token_offset: int


_MISSING_PARENT = ContentCursor(object(), 0)


@dataclass(frozen=True)
class ContentSpan:
    """A view of one immutable dirty chunk; splits do not copy token bytes."""

    context: object
    parent: ContentCursor | None
    token_bytes: bytes
    start: int
    length: int
    page_span: object

    @property
    def cursor(self):
        return ContentCursor(self.context, self.start + self.length)

    @property
    def parent_cursor(self):
        return ContentCursor(self.context, self.start) if self.start else self.parent

    def slice(self, start, end):
        if not 0 <= start <= end <= self.length:
            raise ValueError("content view is outside its chunk")
        return replace(
            self,
            start=self.start + start,
            length=end - start,
            page_span=self.page_span.slice(start, end),
        )


@dataclass(frozen=True)
class ProducerEvents:
    """Certificates cover forward and capture streams; query never waits."""

    events: tuple = ()
    uncertain: bool = False

    @classmethod
    def record(cls, forward_stream, device):
        import torch

        if torch.device(device).type != "cuda":
            return cls()
        events = []
        try:
            if forward_stream is None:
                raise ValueError("actual forward stream is required")
            for stream in (forward_stream, torch.cuda.current_stream(device)):
                event = torch.cuda.Event()
                events.append(event)
                event.record(stream)
        except Exception:
            # Do not turn a failed/partial event recording into a drain proof.
            return cls(tuple(events), uncertain=True)
        return cls(tuple(events))

    def ready(self):
        if self.uncertain:
            raise RuntimeError("producer certificate is uncertain")
        results = tuple(event.query() for event in self.events)
        if any(type(result) is not bool for result in results):
            raise RuntimeError("producer query did not return a readiness proof")
        return all(results)


@dataclass(frozen=True)
class CaptureOwners:
    queued: int
    preparing: int
    queued_entries: int
    preparing_entries: int
    queued_bytes: int
    preparing_bytes: int
    pending: int
    uncertain: int
    progressing: bool
    context_entries: int
    context_bytes: int
    revision: tuple[int, int]
    pool_epoch: int
    status: str
    dropped: int
    closed: bool

    @property
    def owners(self):
        return self.queued + self.preparing + self.pending + int(self.progressing)


@dataclass(frozen=True)
class CaptureTransactionOwners:
    active: bool
    staged_entries: int
    staged_bytes: int
    revision: int
    worker: CaptureOwners


@dataclass(frozen=True)
class _Capture:
    scope: PublicationScope
    span: ContentSpan
    pages: tuple
    producer_ready: object

    @property
    def retained_bytes(self):
        # A suffix view retains the original bytes; account the entire backing.
        return len(self.span.token_bytes) + len(self.pages) * 24


@dataclass(frozen=True)
class _CursorReceipt:
    cid: int
    full_hash: str
    page: object
    parent_cid: int | None


@dataclass(frozen=True)
class _CaptureBatch:
    captures: tuple[_Capture, ...]

    @property
    def retained_bytes(self):
        return sum(item.retained_bytes for item in self.captures)


class PublicationCaptureWorker:
    """Owns only CPU captures/catalog receipts, with autonomous readiness polls.

    ``autostart=False`` and ``progress_once`` are deterministic CPU test seams.
    Production users start the thread; Scheduler must never call progress_once.
    Bounds include bytes retained by sliced chunk views, not just view lengths.
    """

    # Conservative fixed metadata charge for a cursor/CID/PageRef/SHA256 and
    # bounded Python container overhead; not a process RSS measurement.
    # An independent entry bound also caps every retained Python object.
    RECEIPT_BYTES = 1024

    def __init__(
        self,
        catalog,
        *,
        capacity=128,
        max_batch_entries=128,
        max_capture_bytes=32 * 1024 * 1024,
        max_chunk_bytes=2 * 1024 * 1024,
        max_context_entries=65536,
        max_context_bytes=8 * 1024 * 1024,
        poll_interval_s=0.01,
        autostart=True,
    ):
        for value in (
            capacity,
            max_batch_entries,
            max_capture_bytes,
            max_chunk_bytes,
            max_context_entries,
            max_context_bytes,
        ):
            _integer("capture limit", value, 1)
        if not 0.001 <= poll_interval_s <= 1:
            raise ValueError("bounded non-spinning producer polling required")
        self.catalog = catalog
        self._pool_epoch = catalog.guard.pool_epoch
        self.capacity, self.max_capture_bytes = capacity, max_capture_bytes
        self.max_batch_entries = max_batch_entries
        self.max_chunk_bytes = min(max_chunk_bytes, max_capture_bytes)
        self.max_context_entries = min(
            max_context_entries, max_context_bytes // self.RECEIPT_BYTES
        )
        if not self.max_context_entries:
            raise ValueError("context byte budget cannot hold a receipt")
        self.max_context_bytes = max_context_bytes
        self.poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._queue = deque()
        self._contexts = OrderedDict()
        self._queued_bytes = self._preparing_bytes = self._preparing = 0
        self._queued_entries = self._preparing_entries = 0
        self._revision = self._dropped = 0
        self._progressing = self._closed = False
        self._status = "IDLE"
        self._wake = threading.Event()
        self._thread = None
        if autostart:
            self._thread = threading.Thread(
                target=self._run, name="l1-publication-capture", daemon=True
            )
            self._thread.start()

    def miss(self, status):
        with self._lock:
            self._status = status
            self._dropped += 1
            self._revision += 1

    def enqueue(self, scope, span, pages, producer_ready):
        return self.enqueue_batch((_Capture(scope, span, pages, producer_ready),))

    def enqueue_batch(self, captures):
        if (
            not isinstance(captures, (tuple, list))
            or not 1 <= len(captures) <= self.max_batch_entries
        ):
            self.miss("CAPTURE_LIMIT")
            return False
        batch = _CaptureBatch(tuple(captures))
        size = batch.retained_bytes
        with self._lock:
            if (
                self._closed
                or not batch.captures
                or len(batch.captures) > self.max_batch_entries
                or len(self._queue) + self._preparing >= self.capacity
                or any(
                    item.retained_bytes > self.max_chunk_bytes
                    for item in batch.captures
                )
                or self._queued_bytes + self._preparing_bytes + size
                > self.max_capture_bytes
            ):
                self._status = "QUEUE_FULL" if not self._closed else "CLOSED"
                self._dropped += 1
                self._revision += 1
                return False
            self._queue.append(batch)
            self._queued_bytes += size
            self._queued_entries += len(batch.captures)
            self._revision += 1
            self._wake.set()
            return True

    def _prepare(self, capture):
        from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

        scope, span, pages = capture.scope, capture.span, capture.pages
        ps = scope.page_size
        if (
            scope.cache_salt is not None
            or scope.extra_key is not None
            or span.start % ps
            or span.length % ps
            or span.page_span.token_offset
            or span.length != len(pages) * ps
            or not pages
        ):
            self.miss("INVALID_CONTENT")
            return
        with self._lock:
            parent = self._contexts.get(span.parent_cursor)
        if span.parent_cursor is not None and (
            parent is None or not self.catalog.contains(parent.cid, scope)
        ):
            self.miss("MISSING_PARENT")
            return
        parent_cid = parent.cid if parent else None
        cursors = tuple(
            ContentCursor(span.context, end)
            for end in range(span.start + ps, span.start + span.length + 1, ps)
        )
        with self._lock:
            existing = tuple(self._contexts.get(cursor) for cursor in cursors)
        previous = parent_cid
        duplicate = True
        for receipt, page in zip(existing, pages):
            if (
                receipt is None
                or receipt.page != page
                or receipt.parent_cid != previous
                or not self.catalog.contains(receipt.cid, scope)
            ):
                duplicate = False
                break
            previous = receipt.cid
        if duplicate:
            return
        tokens = array("q")
        tokens.frombytes(
            span.token_bytes[span.start * 8 : (span.start + span.length) * 8]
        )
        hashes = get_hash_str(
            tokens, parent.full_hash if parent else None, page_size=ps
        )
        publication = Publication(
            scope,
            parent_cid,
            tuple(
                tuple(tokens[start : start + ps]) for start in range(0, len(tokens), ps)
            ),
            tuple(hash_str_to_int64(value) for value in hashes),
            pages,
            capture.producer_ready,
        )
        receipt = self.catalog.submit_with_receipt(publication)
        previous = parent_cid
        with self._lock:
            for cursor, cid, value, page in zip(
                cursors, receipt.page_cids, hashes, pages
            ):
                self._contexts[cursor] = _CursorReceipt(cid, value, page, previous)
                self._contexts.move_to_end(cursor)
                previous = cid
                while len(self._contexts) > self.max_context_entries:
                    self._contexts.popitem(last=False)
            self._status = "SUBMITTED"
            self._revision += 1

    def progress_once(self, *, limit=16):
        _integer("capture progress limit", limit, 1)
        with self._lock:
            if self._closed:
                return
            if self._progressing:
                raise RuntimeError("publication worker progress is single flight")
            self._progressing = True
            self._revision += 1
        try:
            for _ in range(limit):
                with self._lock:
                    if not self._queue:
                        break
                    batch = self._queue.popleft()
                    self._queued_bytes -= batch.retained_bytes
                    self._queued_entries -= len(batch.captures)
                    self._preparing += 1
                    self._preparing_bytes += batch.retained_bytes
                    self._preparing_entries += len(batch.captures)
                    self._revision += 1
                try:
                    for capture in batch.captures:
                        try:
                            self._prepare(capture)
                        except CatalogBusy:
                            self.miss("CATALOG_FULL")
                        except Exception:
                            self.miss("INVALID_CONTENT")
                finally:
                    with self._lock:
                        self._preparing -= 1
                        self._preparing_bytes -= batch.retained_bytes
                        self._preparing_entries -= len(batch.captures)
                        self._revision += 1
            self.catalog.progress(limit=limit)
        finally:
            with self._lock:
                self._progressing = False
                self._revision += 1

    def _run(self):
        while True:
            snapshot = self.catalog.snapshot()
            queryable = snapshot.pending - snapshot.uncertain
            self._wake.wait(self.poll_interval_s if queryable else None)
            self._wake.clear()
            with self._lock:
                if self._closed:
                    return
            self.progress_once()
            with self._lock:
                if self._queue:
                    self._wake.set()

    def owner_snapshot(self):
        with self._lock:
            local = {
                "queued": len(self._queue),
                "preparing": self._preparing,
                "queued_entries": self._queued_entries,
                "preparing_entries": self._preparing_entries,
                "queued_bytes": self._queued_bytes,
                "preparing_bytes": self._preparing_bytes,
                "progressing": self._progressing,
                "context_entries": len(self._contexts),
                "context_bytes": len(self._contexts) * self.RECEIPT_BYTES,
                "status": self._status,
                "dropped": self._dropped,
                "closed": self._closed,
            }
            revision = self._revision
        # Do not hold the enqueue lock while waiting for a catalog submission.
        catalog = self.catalog.snapshot()
        return CaptureOwners(
            **local,
            pending=catalog.pending,
            uncertain=catalog.uncertain,
            revision=(revision, catalog.revision),
            pool_epoch=self._pool_epoch,
        )

    def assert_quiescent(self):
        if self.owner_snapshot().owners:
            raise RuntimeError("capture, preparation or producer ownership remains")

    def close(self):
        self.seal_if_empty()
        if self._thread is not None:
            self._thread.join()

    def seal_if_empty(self):
        """Nonblocking atomic enqueue fence; lifecycle thread performs join."""
        with self._lock:
            if (
                self._queue
                or self._preparing
                or self._progressing
                or self.catalog.snapshot().pending
            ):
                raise RuntimeError("publication close requires proven quiescence")
            self._closed = True
            self._revision += 1
            self._wake.set()


class CanonicalPublicationHook:
    """Tree owner only. Opt-in, all-rank hook, independent of KVEventRecorder.

    ``attach`` is an integration seam, not allocator activation or a serving
    factory. Serving must validate its exact mode/topology/guard capability.
    The hook itself rejects non-Python/Full, namespaces and incomplete pages.
    """

    def __init__(
        self,
        worker,
        identity,
        scope,
        *,
        device,
        forward_stream=None,
        readiness_factory=None,
    ):
        import torch

        from sglang.srt.mem_cache.allocator.slotguard_paged import (
            SlotGuardPagedTokenToKVPoolAllocator,
        )

        if (
            type(scope) is not PublicationScope
            or scope.pool_epoch != worker.catalog.guard.pool_epoch
            or scope.cache_salt is not None
            or scope.extra_key is not None
        ):
            raise ValueError("matching publication scope/epoch required")
        if (
            not isinstance(identity, SlotGuardPagedTokenToKVPoolAllocator)
            or identity.cpu_page_identity() is not identity
            or identity.slot_guard is not worker.catalog.guard
            or identity.page_size != scope.page_size
        ):
            raise ValueError("same allocator guard capability required")
        actual_device, capture_device = (
            torch.device(identity.device),
            torch.device(device),
        )
        if (
            actual_device.type != capture_device.type
            or capture_device.type not in ("cpu", "cuda")
            or (
                actual_device.index is not None
                and capture_device.index is not None
                and actual_device.index != capture_device.index
            )
            or (
                capture_device.type == "cuda"
                and (forward_stream is None or readiness_factory is not None)
            )
        ):
            raise ValueError(
                "producer certificate must cover the actual pool device/stream"
            )
        if capture_device.type == "cuda":
            # Events on another GPU never cover this pool's KV writes. A bare
            # "cuda" means the current device, so bind its ordinal now.
            ordinal = capture_device.index
            if ordinal is None:
                ordinal = actual_device.index
            if ordinal is None:
                ordinal = torch.cuda.current_device()
            device = torch.device("cuda", ordinal)
            if torch.device(forward_stream.device) != device:
                raise ValueError(
                    "producer certificate must cover the actual pool device/stream"
                )
        self.worker, self.identity, self.scope = worker, identity, scope
        self.device, self.forward_stream = device, forward_stream
        # Factories are CPU test seams only. CUDA always records its actual
        # forward/capture streams, including at the commit call site.
        self._cuda_producer = capture_device.type == "cuda"
        self._readiness_factory = readiness_factory
        self._owner = threading.get_ident()
        self._staged = None
        self._staged_bytes = 0
        self._revision = 0
        self._closed = False
        self._paused = False

    def pause(self):
        """Mark the ingress fence; already admitted Tree commits still drain.

        The Scheduler blocks new requests before setting this marker. It must
        not reject a legal begin/commit from an already admitted request.
        """
        self._check_owner()
        if self._closed:
            raise RuntimeError("canonical capture is closed")
        self._paused = True
        self._revision += 1

    def resume(self):
        self._check_owner()
        if self._closed:
            raise RuntimeError("canonical capture is closed")
        self._paused = False
        self._revision += 1

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("canonical capture belongs to the Tree owner")

    def attach(self, cache):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        self._check_owner()
        core = cache.tree_core
        if (
            type(core) is not UnifiedTreeCore
            or cache.tree_components != (ComponentType.FULL,)
            # A registry override may install another Full; only this one splits
            # content_span with the node.
            or type(cache.components[ComponentType.FULL]) is not FullComponent
            or core.is_eagle
            or core.enable_hicache
            or core.root_node.children
            or cache.token_to_kv_pool_allocator is not self.identity
            or cache.publication_hook is not None
            or core.publication_hook is not None
        ):
            raise ValueError(
                "canonical capture requires an empty Python Full-only Tree"
            )
        cache.publication_hook = self
        core.publication_hook = self

    def begin(self):
        self._check_owner()
        if self._closed:
            raise RuntimeError("canonical capture is closed")
        if self.scope.pool_epoch != self.worker.catalog.guard.pool_epoch:
            raise RuntimeError(
                "publication epoch changed; sealed quiescent rebind required"
            )
        if self._staged is not None:
            raise RuntimeError("capture transaction already active")
        self._staged = OrderedDict()
        self._staged_bytes = 0
        self._revision += 1

    def materialize(self, key, page_span, parent, previous=None):
        self._check_owner()
        if (
            key.extra_key is not None
            or key.cache_salt is not None
            or key.is_bigram
            or page_span is None
            or page_span.token_offset
            or page_span.page_size != self.scope.page_size
            or len(key) != page_span.token_length
            or len(key) % self.scope.page_size
            or len(key) * 8 + len(page_span.page_ids) * 24 > self.worker.max_chunk_bytes
        ):
            self.worker.miss("UNSUPPORTED_CONTENT")
            return None
        if previous is not None:
            # Resurrection changes physical generations, not immutable content.
            return replace(previous, page_span=page_span)
        raw = key.raw_token_ids()
        if not isinstance(raw, array) or raw.typecode != "q":
            raw = array("q", raw)
        return ContentSpan(object(), parent, raw.tobytes(), 0, len(key), page_span)

    def stage(self, span):
        self._check_owner()
        if span is None or self._staged is None:
            return
        key = (span.context, span.start, span.length)
        if key in self._staged:
            return
        size = len(span.token_bytes) + len(span.page_span.page_ids) * 24
        if (
            len(self._staged) >= self.worker.max_batch_entries
            or self._staged_bytes + size > self.worker.max_capture_bytes
        ):
            self.worker.miss("STAGING_FULL")
            return
        self._staged[key] = span
        self._staged_bytes += size
        self._revision += 1

    @staticmethod
    def parent_cursor(parent, is_root):
        if is_root:
            return None
        return parent.cursor if parent is not None else _MISSING_PARENT

    def abort(self):
        self._check_owner()
        self._staged = None
        self._staged_bytes = 0
        self._revision += 1

    def commit(self):
        self._check_owner()
        staged = self._staged
        self.abort()
        if not staged:
            return
        try:
            ready = (
                self._readiness_factory()
                if not self._cuda_producer and self._readiness_factory is not None
                else ProducerEvents.record(self.forward_stream, self.device)
            )
        except Exception:
            ready = ProducerEvents(uncertain=True)
        captures = []
        for span in staged.values():
            try:
                pages = self.identity.page_refs(span.page_span)
            except (PageUnavailable, ValueError):
                self.worker.miss("STALE_PAGES")
                continue
            captures.append(_Capture(self.scope, span, pages, ready))
        if captures:
            self.worker.enqueue_batch(captures)

    def owner_snapshot(self):
        """Owner-thread observation; global immutable Admin fencing is separate."""
        self._check_owner()
        return CaptureTransactionOwners(
            self._staged is not None,
            len(self._staged) if self._staged is not None else 0,
            self._staged_bytes,
            self._revision,
            self.worker.owner_snapshot(),
        )

    def assert_quiescent(self):
        self._check_owner()
        if self._staged is not None:
            raise RuntimeError("Tree publication transaction remains")
        self.worker.assert_quiescent()

    def close(self):
        self.freeze()
        self.worker.close()

    def freeze(self):
        """Owner-only fence, without joining a producer or backend thread.

        Rejection preserves accepted ownership. A successful fence is permanent;
        Task4 must construct a new scope/hook for reset rather than reopen it.
        """
        self._check_owner()
        if self._staged is not None:
            raise RuntimeError("Tree publication transaction remains")
        self.worker.seal_if_empty()
        self._closed = True
        self._revision += 1
