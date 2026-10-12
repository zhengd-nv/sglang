# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Immutable objects in HBM/EGM slabs; one namespace and reader-pin protocol."""
import threading
import uuid
from contextlib import contextmanager

from .region import Region


class EGMStore:
    def __init__(self, capacity=64 << 20, numa=0, hbm_capacity=0, device=0,
                 default_tier="dram"):
        capacities = {"dram": capacity, "hbm": hbm_capacity}
        if any(not isinstance(n, int) or n < 0 for n in capacities.values()):
            raise ValueError("invalid tier capacities")
        if default_tier not in capacities or capacities[default_tier] == 0:
            raise ValueError("default tier must have nonzero capacity")
        self.regions = {}
        try:
            for tier, size in capacities.items():
                if size:
                    self.regions[tier] = Region(size, numa, tier=tier, device=device)
        except BaseException:
            for region in self.regions.values():
                region.close()
            raise
        self.region = self.regions[default_tier]  # Single-tier caller compatibility.
        self.default_tier = default_tier
        self.epoch = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._objects = {}
        self._leases = {}
        self._cursors = {tier: 0 for tier in self.regions}
        self._closed = self._failed = False

    def _open(self):
        if self._closed or self._failed:
            raise RuntimeError("store closed or quarantined")

    def exists(self, key):
        with self._lock:
            self._open()
            return key in self._objects

    def put(self, key, ranges, device=False, tier=None):
        """Publish after copies drain; caller owns/synchronizes source endpoints.

        A key has one immutable placement. No implicit demotion or eviction.
        """
        tier = self.default_tier if tier is None else tier
        if tier not in self.regions:
            raise ValueError("tier unavailable")
        if not isinstance(key, str) or not key or not ranges:
            raise ValueError("nonempty key/ranges required")
        if any(p <= 0 or n <= 0 for p, n in ranges):
            raise ValueError("invalid source range")
        size = sum(n for _, n in ranges)
        with self._lock:
            self._open()
            if key in self._objects:
                placement, _, existing_size = self._objects[key]
                if existing_size != size or placement != tier:
                    raise ValueError("immutable key size/tier conflict")
                return True
            region = self.regions[tier]
            offset = (self._cursors[tier] + 63) // 64 * 64
            if size > region.logical_size - offset:
                return False
            pos = offset
            try:
                for ptr, n in ranges:
                    region.copy(pos, ptr, n, 2 if device else 0)
                    pos += n
            except BaseException:
                self._failed = True
                raise
            self._objects[key] = (tier, offset, size)
            self._cursors[tier] = pos
            return True

    def get(self, key, ranges, device=False):
        """Same read API for either tier. Misses do not modify the target."""
        with self._lock:
            self._open()
            obj = self._objects.get(key)
            if obj is None:
                return False
            tier, base, size = obj
            validate_ranges(ranges, size)
            try:
                for ptr, n, off in ranges:
                    self.regions[tier].copy(base + off, ptr, n, 3 if device else 1)
            except BaseException:
                self._failed = True
                raise
            return True

    def acquire(self, keys):
        with self._lock:
            self._open()
            keys = list(dict.fromkeys(keys))
            if not keys or any(k not in self._objects for k in keys):
                raise KeyError("one or more objects missing")
            objects = {k: dict(zip(("tier", "offset", "size"), self._objects[k])) for k in keys}
            tiers = {obj["tier"] for obj in objects.values()}
            regions = {t: self.regions[t].descriptor() for t in sorted(tiers)}
            token = uuid.uuid4().hex
            self._leases[token] = tuple(keys)
            return dict(version=2, token=token, epoch=self.epoch,
                        regions=regions, objects=objects)

    def release(self, token, epoch):
        with self._lock:
            if epoch != self.epoch or token not in self._leases:
                raise ValueError("stale/unknown reader lease")
            del self._leases[token]

    @contextmanager
    def reader(self, keys):
        lease = self.acquire(keys)
        yield self
        # Exceptions conservatively retain the entire mixed-tier pin.
        self.release(lease["token"], lease["epoch"])

    def clear(self):
        with self._lock:
            self._open()
            if self._leases:
                raise RuntimeError("store pinned by readers")
            self._objects.clear()
            self._cursors = {tier: 0 for tier in self.regions}
            self.epoch = uuid.uuid4().hex

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._leases:
                raise RuntimeError("store pinned by readers; drain/release first")
            self._failed = True  # Refuse reads after partial resource cleanup.
            for region in self.regions.values():
                region.close()
            self._closed = True


def validate_ranges(ranges, size):
    if not ranges or any(p <= 0 or n <= 0 or off < 0 or off > size or n > size - off
                         for p, n, off in ranges):
        raise ValueError("invalid object range")
