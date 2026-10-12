# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Bounded immutable objects in one EGM slab, with explicit reader pins."""
import threading
import uuid
from contextlib import contextmanager

from .region import Region


class EGMStore:
    def __init__(self, capacity=64 << 20, numa=0):
        self.region = Region(capacity, numa)
        self._lock = threading.RLock()
        self._objects = {}
        self._leases = {}
        self._cursor = 0
        self._closed = False
        self._failed = False

    def _open(self):
        if self._closed or self._failed:
            raise RuntimeError("store closed or quarantined")

    def exists(self, key):
        with self._lock:
            self._open()
            return key in self._objects

    def put(self, key, ranges, device=False):
        """Publish only after every (pointer, bytes) has been copied and drained.

        Callers keep endpoints alive and synchronize producers before GPU put.
        An existing key is immutable; conflicts in object size are errors.
        """
        if not isinstance(key, str) or not key or not ranges:
            raise ValueError("nonempty key/ranges required")
        if any(p <= 0 or n <= 0 for p, n in ranges):
            raise ValueError("invalid source range")
        size = sum(n for _, n in ranges)
        with self._lock:
            self._open()
            if key in self._objects:
                if self._objects[key][1] != size:
                    raise ValueError("immutable key size conflict")
                return True
            offset = (self._cursor + 63) // 64 * 64
            if size > self.region.logical_size - offset:
                return False
            pos = offset
            try:
                for ptr, n in ranges:
                    self.region.copy(pos, ptr, n, 2 if device else 0)
                    pos += n
            except BaseException:
                self._failed = True
                raise
            self._objects[key] = (offset, size)
            self._cursor = pos
            return True

    def get(self, key, ranges, device=False):
        """ranges are (target_pointer, size, object_offset); misses do not write."""
        with self._lock:
            self._open()
            obj = self._objects.get(key)
            if obj is None:
                return False
            base, size = obj
            validate_ranges(ranges, size)
            try:
                for ptr, n, off in ranges:
                    self.region.copy(base + off, ptr, n, 3 if device else 1)
            except BaseException:
                self._failed = True
                raise
            return True

    def acquire(self, keys):
        with self._lock:
            self._open()
            if not keys or any(k not in self._objects for k in keys):
                raise KeyError("one or more objects missing")
            descriptor = self.region.descriptor()
            token = uuid.uuid4().hex
            self._leases[token] = tuple(keys)
            return dict(token=token, region=descriptor, objects={k: self._objects[k] for k in keys})

    def release(self, token, epoch):
        with self._lock:
            if epoch != self.region.epoch or token not in self._leases:
                raise ValueError("stale/unknown reader lease")
            del self._leases[token]

    @contextmanager
    def reader(self, keys):
        """Pin the entire restore across layer reads, including gaps between calls."""
        lease = self.acquire(keys)
        yield self
        # Exceptions conservatively leave the pin in place. No timeout reclaim.
        self.release(lease["token"], lease["region"]["epoch"])

    def clear(self):
        with self._lock:
            self._open()
            if self._leases:
                raise RuntimeError("store pinned by readers")
            self._objects.clear()
            self._cursor = 0

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._leases:
                raise RuntimeError("store pinned by readers; drain/release first")
            self.region.close()
            self._closed = True


def validate_ranges(ranges, size):
    if not ranges or any(p <= 0 or n <= 0 or off < 0 or off > size or n > size - off
                         for p, n, off in ranges):
        raise ValueError("invalid object range")
