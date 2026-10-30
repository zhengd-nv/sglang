# SPDX-License-Identifier: Apache-2.0
"""Scheduler-thread source leases for fixed-layout FULL KV transfers.

No transport dependency. Caller must serialize all cache/allocator mutations on
this thread, synchronize producer writes in ready(), and attest transfer drain.
With a tree access lock installed, a lock holder on another thread may lease too.
"""
from collections import Counter
from dataclasses import dataclass, field
import threading
import time
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


class PageGenerations:
    """Per-page free counter; a page whose count is unchanged was never reused.

    Bumped on CPU before the free becomes visible to any allocation, so a copy
    that finished before an unchanged read cannot have seen a later owner's KV.
    """
    def __init__(self, num_pages):
        import numpy
        self.values = numpy.zeros(num_pages + 1, dtype=numpy.int64)

    def bump(self, page_ids):
        import numpy
        numpy.add.at(self.values, page_ids.cpu().numpy(), 1)

    def bump_all(self):
        self.values += 1

    def snapshot(self, pages):
        return tuple(int(x) for x in self.values[list(pages)])


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
    # hbm: device pages under the radix lock; egm: host pages under host_lock_ref.
    tier: str = 'hbm'
    # tree: locked before the offer. step: locked at the next scheduler step, read
    # optimistically and valid only if the page generations are unchanged at release.
    protection: str = 'tree'
    generations: tuple = ()


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
        # Set by an L2 (HiCache) source; such an Engine never restores as a target.
        self.allow_host_pool = False
        self.generations = None
        # lease_id -> step-lock state of an optimistic lease; released ones move to step_log.
        self._step = {}
        self.step_log = {}

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

    def validate_cache(self, *, allow_host_pool=None):
        """``allow_host_pool``: a HiCache (L2, no L3 storage) source may lease host pages."""
        if allow_host_pool is None:
            allow_host_pool = self.allow_host_pool
        cache = self.cache
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        self._check_thread()
        allocator = cache.token_to_kv_pool_allocator
        pool = allocator.get_kvcache()
        if (cache.disable or cache.tree_components != (ComponentType.FULL,)
                or cache.enable_session_radix_cache or cache.linker is not None
                or cache.enable_storage or (cache.cache_controller is not None and not allow_host_pool)
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

    def enable_step_protection(self):
        """Track page generations so a lease may be read before its lock applies (c')."""
        self._check_thread()
        allocator, _ = self.validate_cache()
        if self.generations is None:
            self.generations = PageGenerations(allocator.size // allocator.page_size)
            allocator._l1_page_generations = self.generations
        return self.generations

    def acquire_by_key(self, page_key, *, end, start, fingerprint, ready, tier='hbm', protection='tree'):
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
        if tier not in ('hbm', 'egm'):
            raise ValueError('tier must be hbm or egm')
        node = self.key_index.find(page_key, end, tier)
        self.key_index.touch(node)
        if tier == 'egm':
            return self._lease_host(node, start, end, fingerprint, ready, page_size)
        if protection == 'step':
            return self._lease_optimistic(node, page_key, start, end, fingerprint, allocator, pool)
        if protection != 'tree':
            raise ValueError('protection must be tree or step')
        return self._lease(node.id, start, end, fingerprint, ready, allocator, pool,
                           events=self.key_index.ready_events(node))

    def _path_pages(self, node_id, start, end, page_size):
        path = self.cache.path_device_indices(node_id)
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
        return indices, tuple(pages)

    def _lease_optimistic(self, node, page_key, start, end, fingerprint, allocator, pool):
        """No lock and no pin: the scheduler locks at its next step (apply_step_locks)."""
        if self.generations is None:
            raise RuntimeError('step protection not enabled')
        indices, pages = self._path_pages(node.id, start, end, allocator.page_size)
        lease = SourceLease(self.incarnation, self.epoch, uuid.uuid4().hex, fingerprint, indices, pages,
                            self.key_index.ready_events(node), 'hbm', 'step',
                            self.generations.snapshot(pages))
        count_pool_lease(pool, 1)
        self._active[lease.lease_id] = (lease, None, allocator, pool)
        self._step[lease.lease_id] = dict(key=page_key, end=end, granted=time.time(), locked=None,
                                          outcome=None, pages=len(pages))
        return lease

    def _generations_unchanged(self, lease):
        return self.generations.snapshot(lease.page_indices) == lease.generations

    def apply_step_locks(self):
        """Lock every optimistic lease still unlocked; call at a scheduler step boundary.

        A lease whose pages were freed meanwhile is left unlocked and fails at release.
        With TP>1 this runs on every rank at the same step (ridden on the input broadcast).
        """
        self._check_thread()
        for lease_id, state in list(self._step.items()):
            if state['outcome'] is not None:
                continue
            lease, _, allocator, pool = self._active[lease_id]
            if not self._generations_unchanged(lease):
                state['outcome'] = 'stale_at_lock'
                continue
            try:
                node = self.key_index.find(state['key'], state['end'], 'hbm')
            except ValueError:
                state['outcome'] = 'stale_at_lock'  # unreachable while no page was freed
                continue
            lock = self.cache.lock(node.id)
            pin_pages(allocator, lease.page_indices)
            self._active[lease_id] = (lease, lock, allocator, pool)
            state['locked'] = time.time()
            state['outcome'] = 'locked'

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

    def _lease_host(self, node, start, end, fingerprint, ready, page_size):
        """Host pages [start, end): host eviction is leaf-only, so locking the last
        node protects the whole path (as load-back anchors do)."""
        import torch
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        cache = self.cache
        params = cache.inc_host_lock_ref(node.id).to_dec_params()
        try:
            values, cur = [], node
            while cur is not None and cur is not cache.tree_core.root_node:
                values.append(cur.component_data[ComponentType.FULL].host_value)
                cur = cur.parent
            path = torch.cat(values[::-1])
            if len(path) < end:
                raise ValueError('requested prefix is not fully host resident')
            indices = tuple(int(x) for x in path[start:end].tolist())
            pages = []
            for first in range(0, len(indices), page_size):
                row = indices[first:first + page_size]
                if row[0] % page_size or row != tuple(range(row[0], row[0] + page_size)):
                    raise ValueError('host page is not contiguous/aligned')
                pages.append(row[0] // page_size)
            lease = SourceLease(self.incarnation, self.epoch, uuid.uuid4().hex,
                                fingerprint, indices, tuple(pages), (), 'egm')
        except BaseException:
            cache.dec_host_lock_ref(node.id, params)
            raise
        self._active[lease.lease_id] = (lease, ('host', node.id, params), None, None)
        try:
            ready()  # host pages are committed only after their backup completed
        except BaseException:
            self.release(lease, drained=True)
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
        if lease.protection == 'step':
            state = self._step.pop(lease.lease_id)
            if state['outcome'] is None:
                state['outcome'] = 'released_unlocked'
            # A lock only stops future frees; the generations cover the unlocked window.
            state['valid'] = state['outcome'] != 'stale_at_lock' and self._generations_unchanged(lease)
            state['released'] = time.time()
            if lock is not None:
                self.cache.unlock(lock)
                unpin_pages(allocator, lease.page_indices)
            count_pool_lease(pool, -1)
            self.step_log[lease.lease_id] = state
            while len(self.step_log) > 8192:
                del self.step_log[next(iter(self.step_log))]
        elif isinstance(lock, tuple) and lock[0] == 'host':
            self.cache.dec_host_lock_ref(lock[1], lock[2])
        else:
            self.cache.unlock(lock)
            unpin_pages(allocator, lease.page_indices)
            count_pool_lease(pool, -1)
        del self._active[lease.lease_id]
        self._released.add(lease)
        return True

    def lease_valid(self, lease):
        """After release: was the optimistic read intact (always True for a locked lease)."""
        if lease.protection != 'step':
            return True
        return self.step_log[lease.lease_id]['valid']
