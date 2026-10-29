# SPDX-License-Identifier: Apache-2.0
"""Scheduler-thread source leases for fixed-layout FULL KV transfers.

No transport dependency. Caller must serialize all cache/allocator mutations on
this thread, synchronize producer writes in ready(), and attest transfer drain.
With a tree access lock installed, a lock holder on another thread may lease too.
"""
from collections import Counter
from dataclasses import dataclass, field
import threading
import uuid

# Source leases (possibly off the scheduler thread) and target reservations share
# the allocator's pin counter and the pool's lease count.
_PIN_LOCK = threading.Lock()


def pin_pages(allocator, pages):
    with _PIN_LOCK:
        if not hasattr(allocator, '_l1_pinned_pages'):
            allocator._l1_pinned_pages = Counter()
        allocator._l1_pinned_pages.update(pages)


def unpin_pages(allocator, pages):
    with _PIN_LOCK:
        counter = allocator._l1_pinned_pages
        counter.subtract(pages)
        for page in set(pages):
            if counter[page] <= 0:
                del counter[page]


def count_pool_lease(pool, delta):
    with _PIN_LOCK:
        pool._l1_transfer_lease_count = getattr(pool, '_l1_transfer_lease_count', 0) + delta


@dataclass(frozen=True)
class SourceLease:
    incarnation: str
    epoch: int
    lease_id: str
    fingerprint: str
    # Token slots of the leased range only: [0, H) by token key, [L, H) by page key.
    token_indices: tuple[int, ...]
    page_indices: tuple[int, ...]
    # Insert-time events of the leased path; wait on them before exporting.
    ready_events: tuple = field(default=(), compare=False)


def page_runs(pages):
    """[[first, count], ...] for consecutive page IDs, in order."""
    runs = []
    for page in pages:
        if runs and runs[-1][0] + runs[-1][1] == page:
            runs[-1][1] += 1
        else:
            runs.append([page, 1])
    return runs


class SourceLeaseManager:
    def __init__(self, cache):
        self.cache = cache
        self.incarnation = uuid.uuid4().hex
        self.epoch = 0
        self._thread = threading.get_ident()
        self._active = {}
        self._released = set()
        self._targets = set()
        self.key_index = None
        self.tree_lock = None

    def _check_thread(self):
        if threading.get_ident() == self._thread:
            return
        if self.tree_lock is None or not self.tree_lock._is_owned():
            raise RuntimeError('source leases require the scheduler thread or the tree access lock')

    @property
    def busy(self):
        return bool(self._active or self._targets)

    def enable_key_index(self, ready_stream=None):
        """Hash and index every device node so targets can name a prefix by page key.

        With ``ready_stream`` each new node records an event there at insert.
        """
        from sglang.srt.mem_cache.source_index import SourceKeyIndex
        self._check_thread()
        if self.key_index is None:
            allocator, _ = self.validate_cache()
            self.key_index = SourceKeyIndex(self.cache.tree_core, allocator.page_size)
        if ready_stream is not None:
            self.key_index.ready_stream = ready_stream
        return self.key_index

    def reset(self):
        self._check_thread()
        if self.busy:
            raise RuntimeError('cache has active source leases')
        self.epoch += 1
        self._released.clear()

    def validate_cache(self):
        cache = self.cache
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        self._check_thread()
        allocator = cache.token_to_kv_pool_allocator
        pool = allocator.get_kvcache()
        if (cache.disable or cache.tree_components != (ComponentType.FULL,)
                or cache.enable_session_radix_cache or cache.linker is not None
                or cache.enable_storage or cache.cache_controller is not None
                or type(allocator).__name__ != 'PagedTokenToKVPoolAllocator'
                or type(pool).__name__ != 'MHATokenToKVPool'
                or pool.kv_cache_layout != 'nhd' or pool._post_capture_owner is None
                or pool._post_capture_owner._final_num_tokens is None):
            raise ValueError('source leases require plain FULL paged NHD VMM cache')
        return allocator, pool

    def acquire(self, key, *, fingerprint, ready):
        """Lease [0, len(key)) found by token match."""
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        self._check_thread()
        allocator, pool = self.validate_cache()
        page_size = allocator.page_size
        if (not len(key) or len(key) % page_size or key.is_bigram
                or not isinstance(fingerprint, str) or not fingerprint or not callable(ready)):
            raise ValueError('complete pages, fingerprint and producer-ready callback required')
        match = self.cache.match_prefix(MatchPrefixParams(key=key))
        return self._lease(match.last_device_node, 0, len(key), fingerprint, ready, allocator, pool)

    def acquire_by_key(self, page_key, *, end, start, fingerprint, ready):
        """Lease [start, end) of the prefix whose last page key is ``page_key``."""
        self._check_thread()
        allocator, pool = self.validate_cache()
        page_size = allocator.page_size
        if self.key_index is None:
            raise RuntimeError('source key index not enabled')
        if (not isinstance(page_key, str) or type(end) is not int or type(start) is not int
                or not 0 <= start < end or end % page_size or start % page_size
                or not isinstance(fingerprint, str) or not fingerprint or not callable(ready)):
            raise ValueError('page key, page-aligned 0 <= start < end, fingerprint and ready required')
        node = self.key_index.find(page_key, end)
        return self._lease(node.id, start, end, fingerprint, ready, allocator, pool,
                           events=self.key_index.ready_events(node))

    def _lease(self, node_id, start, end, fingerprint, ready, allocator, pool, events=()):
        cache = self.cache
        page_size = allocator.page_size
        lock = cache.lock(node_id)
        try:
            path = cache.path_device_indices(node_id)
            if len(path) < end:
                raise ValueError('requested prefix is not fully resident')
            indices = tuple(int(x) for x in path[start:end].tolist())
            pages = []
            for first in range(0, len(indices), page_size):
                row = indices[first:first + page_size]
                if row[0] % page_size or row != tuple(range(row[0], row[0] + page_size)):
                    raise ValueError('source page is not physically contiguous/aligned')
                pages.append(row[0] // page_size)
            if len(set(pages)) != len(pages):
                raise ValueError('duplicate source pages')
            lease = SourceLease(self.incarnation, self.epoch, uuid.uuid4().hex,
                                fingerprint, indices, tuple(pages), events)
        except BaseException:
            cache.unlock(lock)
            raise
        pin_pages(allocator, lease.page_indices)
        count_pool_lease(pool, 1)
        self._active[lease.lease_id] = (lease, lock, allocator, pool)
        try:
            ready()  # pin first; callback must finish the producer before returning
        except BaseException:
            self.release(lease, drained=True)  # no descriptor has escaped acquire
            raise
        return lease

    def release(self, lease, *, drained):
        self._check_thread()
        if not isinstance(lease, SourceLease) or (lease.incarnation, lease.epoch) != (self.incarnation, self.epoch):
            raise ValueError('stale source lease')
        if drained is not True:
            raise RuntimeError('timeout/disconnect is not proof of drain')
        if lease in self._released:
            return False
        item = self._active.get(lease.lease_id)
        if item is None or item[0] != lease:
            raise ValueError('unknown or altered source lease')
        _, lock, allocator, pool = item
        self.cache.unlock(lock)
        unpin_pages(allocator, lease.page_indices)
        count_pool_lease(pool, -1)
        del self._active[lease.lease_id]
        self._released.add(lease)
        return True
