"""Validated int64 tuples: check a prompt-sized vector once, not per message.

A 512K-token prompt is one tuple that flows through LookupQuery, its replaces,
PinReq and the wire codec; each used to re-check every element in Python.
The cache holds a strong reference, so an id cannot be reused while cached.
"""

import sys
import threading
from array import array
from collections import OrderedDict

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
# Each entry may also keep the little-endian bytes (8 B/element); 16 bounds it.
_CAPACITY = 16
# Small vectors (per-page tokens, page tables) are cheap to recheck and would
# evict prompt-sized entries from the LRU.
_MIN_CACHED_LEN = 4096

_lock = threading.Lock()
# id(tuple) -> (tuple, smallest element, little-endian bytes or None); bounded LRU.
_validated: "OrderedDict[int, tuple]" = OrderedDict()


def _remember(values: tuple, smallest: int, raw: bytes | None = None) -> None:
    if len(values) < _MIN_CACHED_LEN:
        return
    with _lock:
        _validated[id(values)] = (values, smallest, raw)
        _validated.move_to_end(id(values))
        while len(_validated) > _CAPACITY:
            _validated.popitem(last=False)


def _cached(values: tuple):
    with _lock:
        entry = _validated.get(id(values))
        if entry is None or entry[0] is not values:
            return None
        _validated.move_to_end(id(values))
        return entry


def from_int64_unpacked(values: tuple, raw: bytes) -> tuple:
    """Register struct.unpack('<Nq', raw), which is int64 by construction."""
    _remember(values, min(values) if values else INT64_MAX, raw)
    return values


def from_trusted_ints(values: tuple, smallest: int) -> tuple:
    """Register a tuple the caller built from int64 storage, with its known minimum."""
    if type(values) is not tuple:
        raise ValueError("tuple required")
    _remember(values, smallest)
    return values


def is_int64_tuple(values: object, minimum: int = INT64_MIN) -> bool:
    """Exact tuple of exact ints (no bool) within [minimum, INT64_MAX]."""
    if type(values) is not tuple:
        return False
    entry = _cached(values)
    if entry is not None:
        smallest = entry[1]
    else:
        if values and set(map(type, values)) != {int}:
            return False
        if values and (min(values) < INT64_MIN or max(values) > INT64_MAX):
            return False
        smallest = min(values) if values else INT64_MAX
        _remember(values, smallest)
    return smallest >= minimum


def pack_int64_le(values: tuple) -> bytes:
    """Little-endian int64 bytes of a tuple already accepted by is_int64_tuple."""
    entry = _cached(values)
    if entry is not None and entry[2] is not None:
        return entry[2]
    raw = array("q", values)
    if sys.byteorder != "little":
        raw.byteswap()
    return raw.tobytes()
