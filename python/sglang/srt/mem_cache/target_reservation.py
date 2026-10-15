# SPDX-License-Identifier: Apache-2.0
"""Serialized FULL/NHD target ownership for a synchronous peer restore.

No transport dependency. A reservation protects its pool until release(); on
publication failure ownership may be partial and the Engine must fail-stop.
"""
from collections import Counter

import torch
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams


class TargetReservation:
    def __init__(self, cache, key):
        self.cache = cache
        self.manager = cache.source_leases
        self.allocator, self.pool = self.manager.validate_cache()
        page = self.allocator.page_size
        if not len(key) or len(key) % page or key.is_bigram:
            raise ValueError('target requires nonempty complete pages')
        self.key = key[:len(key)]
        self.state = 'PREPARING'
        self._pins = ()
        self._lock = None
        match = cache.match_prefix(MatchPrefixParams(key=self.key))
        self._lock = cache.lock(match.last_device_node)
        try:
            prefix = cache.path_device_indices(match.last_device_node).clone()
            self.prefix_len = len(prefix)
            if self.prefix_len > len(key) or self.prefix_len % page:
                raise ValueError('unaligned local prefix')
            self.private = self.allocator.alloc(len(key) - self.prefix_len) if self.prefix_len < len(key) else prefix[:0]
            if self.private is None:
                raise MemoryError('insufficient private target pages')
            self.indices = torch.cat((prefix, self.private))
            self.pages = tuple(int(x) for x in (self.private[::page] // page).tolist())
            self._set_pins(self.indices)
        except BaseException:
            # Allocation succeeded only if private is a tensor; no insert yet.
            if getattr(self, 'private', None) is not None:
                self.allocator.free(self.private)
            cache.unlock(self._lock)
            raise
        self.manager._targets.add(self)
        self.pool._l1_transfer_lease_count = getattr(self.pool, '_l1_transfer_lease_count', 0) + 1
        self.allocator._l1_target_count = getattr(self.allocator, '_l1_target_count', 0) + 1
        self.state = 'PRIVATE'

    def _check(self, expected):
        self.manager._check_thread()
        if self.state not in expected:
            raise RuntimeError(f'target state {self.state} does not allow this operation')

    def _set_pins(self, indices):
        pages = tuple(int(x) for x in (indices[::self.allocator.page_size] // self.allocator.page_size).tolist())
        if not hasattr(self.allocator, '_l1_pinned_pages'):
            self.allocator._l1_pinned_pages = Counter()
        self.allocator._l1_pinned_pages.update(pages)
        self._pins = pages

    def _unpin(self):
        self.allocator._l1_pinned_pages.subtract(self._pins)
        self.allocator._l1_pinned_pages += Counter()
        self._pins = ()

    def publish(self):
        self._check(('PRIVATE', 'PUBLISHED'))
        if self.state == 'PUBLISHED':
            return self.indices
        try:
            # insert frees duplicate private slices itself. Keep the matched
            # path locked and pool reset guarded, but permit these exact frees.
            self._unpin()
            self.cache.insert(InsertParams(key=self.key, value=self.indices,
                                           prev_prefix_len=self.prefix_len))
            match = self.cache.match_prefix(MatchPrefixParams(key=self.key))
            canonical = self.cache.path_device_indices(match.last_device_node).clone()
            if len(canonical) != len(self.key):
                raise RuntimeError('incomplete target publication')
            new_lock = self.cache.lock(match.last_device_node)
            self.cache.unlock(self._lock)
            self._lock = new_lock
            self.indices = canonical
            self._set_pins(canonical)
            self.state = 'PUBLISHED'
            return self.indices
        except BaseException:
            self.state = 'QUARANTINED'
            raise  # never free private after a possibly partial insert

    def discard(self):
        self._check(('PRIVATE', 'DISCARDED'))
        if self.state == 'DISCARDED':
            return
        try:
            self._unpin()
            self.allocator.free(self.private)
            self.state = 'DISCARDED'
            self.release()
        except BaseException:
            self.state = 'QUARANTINED'
            raise

    def release(self):
        """Consumer finished using published pages; tree retains ownership."""
        self._check(('PUBLISHED', 'DISCARDED', 'RELEASED'))
        if self.state == 'RELEASED' or self not in self.manager._targets:
            return
        self._unpin()
        self.cache.unlock(self._lock)
        self.manager._targets.remove(self)
        self.pool._l1_transfer_lease_count -= 1
        self.allocator._l1_target_count -= 1
        if self.state == 'PUBLISHED':
            self.state = 'RELEASED'
