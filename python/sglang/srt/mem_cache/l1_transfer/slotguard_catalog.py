# ruff: noqa: BLE001
# Producer readiness failures retain uncertainty rather than authorize cleanup.
"""Bounded immutable CPU publications, indexed by untrusted ROOT-key hints.

Producers submit only dirty page chunks linked by catalog IDs. No tree objects,
request objects, allocator handles or tensors belong in publication content.
Evicting an ancestor causes a safe MISS; active leases own their own PageRefs.
Native V1 supports only absent salt/extra_key. Injected CPU hash verifiers are
a test seam for scope isolation, not an extension of serving model support.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from .hash_pin import HashLookupResult, page_hash
from .metadata import root_event_keys
from .protocol import _integer, _integers
from .slot_guard import PageRef, PageUnavailable, _text
from .slotguard_receipts import emit_receipt, publication_receipt


class CatalogBusy(RuntimeError):
    pass


class ProducerReady(Protocol):
    """Nonblocking readiness certificate; failures are diagnostic uncertainty."""

    def ready(self) -> bool: ...


@dataclass(frozen=True)
class PublicationScope:
    model_id: str
    layout_digest: str
    page_size: int
    pool_epoch: int
    cache_salt: str | None = None
    extra_key: str | None = None

    def __post_init__(self):
        for name in ("model_id", "layout_digest"):
            _text(name, getattr(self, name))
        _integer("page_size", self.page_size, 1)
        _integer("pool_epoch", self.pool_epoch)
        for value in (self.cache_salt, self.extra_key):
            if value is not None and (type(value) is not str or len(value) > 4096):
                raise ValueError("invalid publication namespace")


@dataclass(frozen=True)
class Publication:
    scope: PublicationScope
    parent_cid: int | None
    page_tokens: tuple[tuple[int, ...], ...]
    root_event_keys: tuple[int, ...]
    pages: tuple[PageRef, ...]
    producer_ready: ProducerReady

    def __post_init__(self):
        if type(self.scope) is not PublicationScope:
            raise ValueError("publication scope required")
        if self.parent_cid is not None:
            _integer("parent_cid", self.parent_cid, 1)
        if type(self.page_tokens) is not tuple or not self.page_tokens:
            raise ValueError("nonempty immutable dirty chunk required")
        for tokens in self.page_tokens:
            _integers("page_tokens", tokens)
            if len(tokens) != self.scope.page_size:
                raise ValueError("only complete pages may be published")
        _integers("root_event_keys", self.root_event_keys, -(1 << 63))
        if (
            type(self.pages) is not tuple
            or len(self.pages) != len(self.page_tokens)
            or len(self.root_event_keys) != len(self.pages)
            or len(set(self.pages)) != len(self.pages)
            or any(
                type(ref) is not PageRef or ref.pool_epoch != self.scope.pool_epoch
                for ref in self.pages
            )
        ):
            raise ValueError("publication page/key binding mismatch")
        if not callable(getattr(self.producer_ready, "ready", None)):
            raise TypeError("explicit nonblocking producer readiness required")


@dataclass(frozen=True)
class ContentPage:
    cid: int
    parent_cid: int | None
    scope: PublicationScope
    position: int
    tokens: tuple[int, ...]
    root_event_key: int
    page: PageRef
    prefix_hash: str | None = None


@dataclass(frozen=True)
class CatalogSnapshot:
    records: int
    pending: int
    uncertain: int
    revision: int
    token_ids: int


@dataclass(frozen=True)
class SubmissionReceipt:
    """Explicit page-boundary IDs; callers must not infer CID allocation order."""

    page_cids: tuple[int, ...]

    @property
    def terminal_cid(self):
        return self.page_cids[-1]


@dataclass
class _Pending:
    publication: Publication
    nodes: tuple[ContentPage, ...]
    evaluating: bool = False
    uncertain: bool = False
    receipt: object = None


class PublicationCatalog:
    def __init__(
        self,
        guard,
        *,
        max_records=65536,
        max_pending=128,
        max_publication_pages=4096,
        max_token_ids=4 * 1024 * 1024,
        event_keys=root_event_keys,
        observer=None,
    ):
        for value in (max_records, max_pending, max_publication_pages, max_token_ids):
            _integer("catalog limit", value, 1)
        self.guard, self.event_keys = guard, event_keys
        self.observer = observer
        self._native_keys = event_keys is root_event_keys
        self.max_records, self.max_pending = max_records, max_pending
        self.max_publication_pages = min(max_publication_pages, guard.max_pin_pages)
        self.max_token_ids = max_token_ids
        self._lock = threading.Lock()
        self._nodes = OrderedDict()
        self._strong_index = {}
        self._pending = {}
        self._pending_nodes = {}
        self._next_cid = 1
        self._pending_records = self._revision = 0
        self._uncertain_count = 0
        self._token_count = self._pending_tokens = 0
        self._sealed = False
        self._pool_epoch = guard.pool_epoch

    def seal_if_quiescent(self):
        with self._lock:
            if self._pending or self._uncertain_count:
                raise CatalogBusy("pending or uncertain publication remains")
            self._sealed = True
            self._revision += 1

    def invalidate_epoch(self, pool_epoch):
        """Only the sealed, drained owner may retire the previous pool epoch."""
        with self._lock:
            if (
                not self._sealed
                or self._pending
                or self._uncertain_count
                or pool_epoch <= self._pool_epoch
                or pool_epoch != self.guard.pool_epoch
            ):
                raise CatalogBusy("epoch invalidation requires sealed quiescence")
            self._nodes.clear()
            self._strong_index.clear()
            self._pending_nodes.clear()
            self._token_count = 0
            self._pool_epoch = pool_epoch
            self._revision += 1

    def resume_epoch(self, pool_epoch):
        with self._lock:
            if (
                not self._sealed
                or self._pending
                or self._uncertain_count
                or pool_epoch != self._pool_epoch
                or pool_epoch != self.guard.pool_epoch
            ):
                raise CatalogBusy("catalog epoch mismatch")
            self._sealed = False
            self._revision += 1

    def submit(self, publication):
        """Compatibility entrypoint returning the terminal content ID."""
        return self.submit_with_receipt(publication).terminal_cid

    def contains(self, cid, scope):
        """A bounded cursor may outlive catalog eviction; absence is a safe MISS."""
        with self._lock:
            node = self._nodes.get(cid) or self._pending_nodes.get(cid)
            return node is not None and node.scope == scope

    def submit_with_receipt(self, publication):
        if type(publication) is not Publication:
            raise ValueError("immutable publication required")
        if self._native_keys and (
            publication.scope.cache_salt is not None
            or publication.scope.extra_key is not None
        ):
            raise ValueError("native V1 publication does not support salt/extra_key")
        count = len(publication.pages)
        token_count = sum(len(page) for page in publication.page_tokens)
        if count > self.max_publication_pages or token_count > self.max_token_ids:
            raise CatalogBusy("publication exceeds bounded catalog limits")
        # Existing publication workers own this complete hash. Do not extend
        # the catalog installation critical section with hashing/token encoding.
        receipt = (
            publication_receipt(publication) if self.observer is not None else None
        )
        # Verify the immutable dirty suffix once, off the installation lock.
        # A predecessor may be pending but must carry the same verified chain.
        proof_parent, strong = None, [None] * count
        with self._lock:
            proof_parent = self._nodes.get(
                publication.parent_cid
            ) or self._pending_nodes.get(publication.parent_cid)
        prior = proof_parent.prefix_hash if proof_parent else None
        eligible = publication.parent_cid is None or prior is not None
        for index, (tokens, key) in enumerate(
            zip(publication.page_tokens, publication.root_event_keys)
        ):
            if not eligible:
                break
            current = page_hash(tokens, prior)
            native_key = int.from_bytes(bytes.fromhex(current)[:8], "big", signed=True)
            if native_key != key:
                if self._native_keys:
                    raise ValueError(
                        "publication ROOT key does not match native full hash"
                    )
                break  # Injected CPU key seam is not strong evidence.
            strong[index] = current
            prior = current
        with self._lock:
            if self._sealed or publication.scope.pool_epoch != self._pool_epoch:
                raise CatalogBusy("publication epoch is sealed or stale")
            if (
                count > self.max_publication_pages
                or count > self.max_records
                or len(self._pending) >= self.max_pending
                or self._pending_records + count > self.max_records
                or self._pending_tokens + token_count > self.max_token_ids
            ):
                raise CatalogBusy("publication capacity exhausted")
            parent = self._nodes.get(publication.parent_cid)
            if parent is None:
                parent = self._pending_nodes.get(publication.parent_cid)
            if publication.parent_cid is not None and (
                parent is None
                or parent.scope != publication.scope
                or parent is not proof_parent
            ):
                raise ValueError("publication parent is absent or incompatible")
            position = 0 if parent is None else parent.position
            parent_cid = publication.parent_cid
            nodes = []
            for tokens, key, page, prefix_hash in zip(
                publication.page_tokens,
                publication.root_event_keys,
                publication.pages,
                strong,
            ):
                cid = self._next_cid
                self._next_cid += 1
                position += publication.scope.page_size
                nodes.append(
                    ContentPage(
                        cid,
                        parent_cid,
                        publication.scope,
                        position,
                        tokens,
                        key,
                        page,
                        prefix_hash,
                    )
                )
                parent_cid = cid
            while (
                len(self._nodes) + self._pending_records + count > self.max_records
                or self._token_count + token_count > self.max_token_ids
            ):
                _, old = self._nodes.popitem(last=False)
                self._token_count -= len(old.tokens)
                strong_key = (old.scope, old.position, old.prefix_hash)
                if self._strong_index.get(strong_key) == old.cid:
                    del self._strong_index[strong_key]
            self._pending[parent_cid] = _Pending(
                publication,
                tuple(nodes),
                receipt=receipt,
            )
            self._pending_nodes.update((node.cid, node) for node in nodes)
            self._pending_records += count
            self._pending_tokens += token_count
            self._token_count += token_count
            self._revision += 1
            return SubmissionReceipt(tuple(node.cid for node in nodes))

    def progress(self, *, limit=16):
        _integer("progress limit", limit, 1)
        with self._lock:
            pending = [
                (cid, item)
                for cid, item in self._pending.items()
                if not item.evaluating and not item.uncertain
            ][:limit]
            for _, item in pending:
                item.evaluating = True
        completed = 0
        for cid, item in pending:
            terminal = False
            try:
                ready = item.publication.producer_ready.ready()
                if type(ready) is not bool:
                    raise ValueError("producer readiness must return bool")
                if ready:
                    # Free and publication use the guard lock; no late revival.
                    try:
                        self.guard.publish(item.publication.pages)
                        completed += 1
                    except PageUnavailable:
                        terminal = True
                        with self._lock:
                            self._token_count -= sum(
                                len(node.tokens) for node in item.nodes
                            )
                    else:
                        with self._lock:
                            for node in item.nodes:
                                self._nodes[node.cid] = node
                                if node.prefix_hash is not None:
                                    self._strong_index[
                                        (node.scope, node.position, node.prefix_hash)
                                    ] = node.cid
                        # Visible only after producer query, exact guard publish
                        # and catalog installation. Submission is never readiness.
                        emit_receipt(
                            self.observer, "publication_ready", None, item.receipt
                        )
                        terminal = True
            except Exception:
                # Failed event query is not a proof of producer drain.
                with self._lock:
                    item.uncertain = True
                    self._uncertain_count += 1
            finally:
                with self._lock:
                    item.evaluating = False
                    if terminal:
                        del self._pending[cid]
                        for node in item.nodes:
                            del self._pending_nodes[node.cid]
                        self._pending_records -= len(item.nodes)
                        self._pending_tokens -= sum(
                            len(node.tokens) for node in item.nodes
                        )
                    self._revision += 1
        return completed

    def lookup(self, result):
        """Resolve only publication-verified content; no token work on Pin.

        The catalog lock snapshots immutable page references, not ownership.
        SlotGuard.try_pin still atomically checks generations and acquires them.
        """
        if type(result) is not HashLookupResult:
            raise ValueError("exact compact hash result required")
        scope = PublicationScope(
            result.source.model_id,
            result.source.layout_digest,
            result.page_size,
            self.guard.pool_epoch,
        )
        with self._lock:
            if self._sealed:
                return None
            cid = self._strong_index.get((scope, result.hit_tokens, result.prefix_hash))
            expected = result.hit_tokens
            reverse = []
            # Walk through ROOT, including local prefix, to fail closed if an
            # ancestor was evicted. No source tree or Scheduler is accessed.
            while cid is not None:
                node = self._nodes.get(cid)
                if (
                    node is None
                    or node.scope != scope
                    or node.position != expected
                    or node.prefix_hash is None
                ):
                    return None
                if expected > result.local_hit_tokens:
                    reverse.append(node.page)
                expected -= result.page_size
                cid = node.parent_cid
            if expected != 0:
                return None
        pages = tuple(reversed(reverse))
        if len(pages) != result.restore_pages or len(pages) > self.guard.max_pin_pages:
            return None
        return (
            pages
            if len(set(pages)) == len(pages)
            and self.guard.published_refs_current(pages)
            else None
        )

    def snapshot(self):
        with self._lock:
            return CatalogSnapshot(
                len(self._nodes) + self._pending_records,
                len(self._pending),
                self._uncertain_count,
                self._revision,
                self._token_count,
            )
