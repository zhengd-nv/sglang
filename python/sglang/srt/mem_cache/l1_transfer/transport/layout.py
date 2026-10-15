# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""Same-layout page planning, independent of CUDA and serving frameworks.

Paired contiguous coalescing follows SGLang PD common/utils.py's
 group_concurrent_contiguous algorithm (31ff9b5d42a6898b33d379a787ab0cd07d15ba73).
This implementation uses stdlib only and validates both inputs before planning.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

from .api import MemoryHandle, Opcode, TransferRequest, checked_int


@dataclass(frozen=True)
class BufferLayout:
    name: str  # stable semantic identity, e.g. layer0.k, layer0.v
    page_bytes: int
    layout: str  # canonical physical ordering, including head shape/shard identity


@dataclass(frozen=True)
class PoolLayout:
    dtype: str
    page_size: int
    buffers: tuple[BufferLayout, ...]

    def validate(self):
        checked_int(self.page_size, "page_size", 1)
        if not isinstance(self.dtype, str) or not self.dtype or not self.buffers:
            raise ValueError("dtype and buffers are required")
        names = set()
        for b in self.buffers:
            if not isinstance(b.name, str) or not b.name or b.name in names:
                raise ValueError("buffer names must be unique and nonempty")
            names.add(b.name)
            checked_int(b.page_bytes, "page_bytes", 1)
            if not isinstance(b.layout, str) or not b.layout:
                raise ValueError("canonical layout identity is required")


@dataclass(frozen=True)
class BufferRange:
    buffer: str
    source_offset: int
    target_offset: int
    length: int


@dataclass(frozen=True)
class RegionSpan:
    """A borrowed mapping of buffer bytes onto a registered memory handle.

    No allocation, registration, export, or ownership transfer occurs here.
    Bindings cover the full buffer without holes; region offsets may be nonzero.
    """
    buffer_offset: int
    length: int
    memory: MemoryHandle
    region_offset: int = 0


def plan_transfer(source: PoolLayout, target: PoolLayout,
                  source_pages: Sequence[int], target_pages: Sequence[int], *,
                  source_capacity: int, target_capacity: int) -> tuple[BufferRange, ...]:
    source.validate(); target.validate()
    if source != target:
        raise ValueError("same-layout transfer requires identical layout contracts")
    checked_int(source_capacity, "source_capacity")
    checked_int(target_capacity, "target_capacity")
    src, dst = tuple(source_pages), tuple(target_pages)
    if len(src) != len(dst):
        raise ValueError("page lists must have equal lengths")
    for pages, capacity in ((src, source_capacity), (dst, target_capacity)):
        for page in pages:
            if checked_int(page, "page") >= capacity:
                raise ValueError("page outside pool capacity")
    if len(set(dst)) != len(dst):
        raise ValueError("duplicate target pages")
    groups = []
    for s, t in zip(src, dst):
        if groups and s == groups[-1][0] + groups[-1][2] and t == groups[-1][1] + groups[-1][2]:
            a, b, count = groups[-1]; groups[-1] = (a, b, count + 1)
        else:
            groups.append((s, t, 1))
    result = []
    for buffer in source.buffers:
        checked_int(max(source_capacity, target_capacity) * buffer.page_bytes, "buffer size")
        for s, t, count in groups:
            result.append(BufferRange(buffer.name, s * buffer.page_bytes,
                                      t * buffer.page_bytes, count * buffer.page_bytes))
    return tuple(result)


def _validate_bindings(bindings):
    normalized = {}
    for name, spans in bindings.items():
        spans = tuple(spans)
        end = 0
        for span in spans:
            checked_int(span.buffer_offset, "buffer_offset")
            checked_int(span.region_offset, "region_offset")
            checked_int(span.length, "span length", 1)
            if span.buffer_offset != end:
                raise ValueError("bindings must be ordered, contiguous, and start at zero")
            end = checked_int(end + span.length, "buffer extent")
            if span.region_offset + span.length > span.memory.size:
                raise ValueError("binding exceeds registered region")
        normalized[name] = spans
    return normalized


def bind_regions(plan: Sequence[BufferRange],
                 source: Mapping[str, Sequence[RegionSpan]],
                 target: Mapping[str, Sequence[RegionSpan]],
                 opcode: Opcode) -> tuple[TransferRequest, ...]:
    """Split at BOTH endpoints' region boundaries; READ source is remote.

    The engine remains responsible for live-handle validation and lease checks.
    Empty plans return no requests; callers should not submit an empty batch.
    """
    if not isinstance(opcode, Opcode):
        raise ValueError("opcode must be Opcode.READ or Opcode.WRITE")
    source, target = _validate_bindings(source), _validate_bindings(target)
    requests = []
    for r in plan:
        checked_int(r.source_offset, "source_offset")
        checked_int(r.target_offset, "target_offset")
        checked_int(r.length, "range length", 1)
        try:
            src, dst = source[r.buffer], target[r.buffer]
        except KeyError:
            raise ValueError(f"missing binding: {r.buffer}") from None
        if (not src or not dst or r.source_offset + r.length > src[-1].buffer_offset + src[-1].length
                or r.target_offset + r.length > dst[-1].buffer_offset + dst[-1].length):
            raise ValueError("range exceeds buffer binding")
        si = di = 0
        s, t, left = r.source_offset, r.target_offset, r.length
        while left:
            while s >= src[si].buffer_offset + src[si].length: si += 1
            while t >= dst[di].buffer_offset + dst[di].length: di += 1
            a, b = src[si], dst[di]
            n = min(left, a.buffer_offset + a.length - s, b.buffer_offset + b.length - t)
            so, to = a.region_offset + s - a.buffer_offset, b.region_offset + t - b.buffer_offset
            if a.memory.remote != (opcode == Opcode.READ) or b.memory.remote != (opcode == Opcode.WRITE):
                raise ValueError("binding locality disagrees with opcode")
            local, remote, lo, ro = ((b.memory, a.memory, to, so) if opcode == Opcode.READ
                                      else (a.memory, b.memory, so, to))
            requests.append(TransferRequest(opcode, local, remote, lo, ro, n))
            s += n; t += n; left -= n
    # Reject destination aliases even across different logical buffers/regions views.
    writes = {}
    for request in requests:
        memory, offset = ((request.local, request.local_offset) if opcode == Opcode.READ
                          else (request.remote, request.remote_offset))
        writes.setdefault(memory.region_id, []).append((offset, offset + request.length))
    for intervals in writes.values():
        intervals.sort()
        if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("overlapping destination region ranges")
    return tuple(requests)
