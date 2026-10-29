# SPDX-License-Identifier: Apache-2.0
"""Serialized FULL/NHD target ownership for a synchronous peer restore.

No transport dependency. A reservation protects its pool until release(); on
publication failure ownership may be partial and the Engine must fail-stop.
"""

import torch
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.source_lease import count_pool_lease, pin_pages, unpin_pages


class TargetReservation:
    def __init__(self, cache, key, *, required_capacity=None, headroom=0):
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
        self._match = match
        self._lock = cache.lock(match.last_device_node)
        try:
            prefix = cache.path_device_indices(match.last_device_node).clone()
            self.prefix_len = len(prefix)
            if self.prefix_len > len(key) or self.prefix_len % page:
                raise ValueError('unaligned local prefix')
            if required_capacity is not None:
                # The caller serializes admission until consumer completion.
                # Protect the matched prefix while evicting unrelated cache;
                # reserve budget for the entire prompt and maximum output.
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams
                if (type(required_capacity) is not int or required_capacity < len(key)
                        or required_capacity % page or required_capacity > self.allocator.size):
                    raise ValueError('invalid serialized request capacity')
                if type(headroom) is not int or headroom < 0:
                    raise ValueError("invalid concurrent headroom")
                needed = required_capacity - self.prefix_len + headroom
                shortfall = needed - self.allocator.available_size()
                if shortfall > 0:
                    cache.evict_for_alloc(EvictParams(num_tokens=shortfall))
                if self.allocator.available_size() < needed:
                    raise MemoryError('insufficient request headroom')
            self.private = self.allocator.alloc(len(key) - self.prefix_len) if self.prefix_len < len(key) else prefix[:0]
            if self.private is None:
                raise MemoryError('insufficient private target pages')
            self.indices = torch.cat((prefix, self.private))
            self._set_pins(self.indices)
            self.pages = self._pins[self.prefix_len // page:]
        except BaseException:
            # Allocation succeeded only if private is a tensor; no insert yet.
            if getattr(self, 'private', None) is not None:
                self.allocator.free(self.private)
            cache.unlock(self._lock)
            raise
        self.manager._targets.add(self)
        count_pool_lease(self.pool, 1)
        self.allocator._l1_target_count = getattr(self.allocator, '_l1_target_count', 0) + 1
        self.state = 'PRIVATE'

    def _check(self, expected):
        self.manager._check_thread()
        if self.state not in expected:
            raise RuntimeError(f'target state {self.state} does not allow this operation')

    def _set_pins(self, indices):
        pages = tuple(int(x) for x in (indices[::self.allocator.page_size] // self.allocator.page_size).tolist())
        pin_pages(self.allocator, pages)
        self._pins = pages

    def _unpin(self):
        unpin_pages(self.allocator, self._pins)
        self._pins = ()

    def publish(self):
        self._check(('PRIVATE', 'PUBLISHED'))
        if self.state == 'PUBLISHED':
            return self.indices
        try:
            if self.prefix_len == len(self.key):
                # The entire canonical path is already locked and pinned.
                # No private pages exist to insert or reconcile.
                self.state = 'PUBLISHED'
                return self.indices
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

    def ready_private(self):
        """Copy is drained; keep pages private until normal request allocation."""
        self._check(('PRIVATE', 'READY_PRIVATE'))
        self.state = 'READY_PRIVATE'

    def request_match(self, max_prefix_len):
        self._check(('READY_PRIVATE',))
        if max_prefix_len is not None and max_prefix_len < len(self.indices):
            raise ValueError('request cannot consume the entire restored prefix')
        return self._match._replace(device_prefix_len=len(self.indices),
                                    cache_protected_len=self.prefix_len)

    def adopt(self, req):
        """The normal KV row now owns the private suffix, with its own tree lock.

        Called after writing the row, before any forward/checkpoint. The request
        releases/deduplicates these pages via the ordinary cache lifecycle.
        """
        self._check(('READY_PRIVATE',))
        if not req.kv.holds_kv or req.lock is None or req.prefix_len != len(self.indices):
            raise RuntimeError('private prefix has no admitted request owner')
        self._unpin()
        self.cache.unlock(self._lock)
        self._drop_guard()
        self.state = 'ADOPTED'
        req._l1_private_reservation = None

    def _drop_guard(self):
        self.manager._targets.remove(self)
        count_pool_lease(self.pool, -1)
        self.allocator._l1_target_count -= 1

    def discard(self):
        self._check(('PRIVATE', 'READY_PRIVATE', 'DISCARDED'))
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
        self._drop_guard()
        if self.state == 'PUBLISHED':
            self.state = 'RELEASED'
