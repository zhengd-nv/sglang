"""Strict JSON v1 exports for the typed Pin control protocol (no pickle/address).

Each binding carries a small shared pool header and its own BufferLayout/spans;
the ordered binding tuple reconstructs the full PoolLayout in linear space.
BufferBinding.region_id is ONLY the first-span anchor, not a whole-buffer map.
Source producer workers use encode_export after exporting their registered pool.
The control plane is trusted to authenticate engine identities; JSON is not an
authentication mechanism. No CUDA imports or calls occur in this module.
"""

import base64
import json
from dataclasses import asdict, dataclass
from itertools import pairwise

from .protocol import BufferBinding, EngineIdentity, RegionDescriptor
from .transport.api import RegionDescriptor as TransportDescriptor
from .transport.api import checked_int
from .transport.layout import BufferLayout, PoolLayout, RegionSpan

MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_REGIONS = 4096
MAX_SPANS = 16384


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _load(value, fields):
    if type(value) is not bytes or len(value) > MAX_METADATA_BYTES:
        raise ValueError("invalid metadata size/type")
    obj = json.loads(value, object_pairs_hook=_object)
    _fields(obj, fields)
    if type(obj["version"]) is not int or obj["version"] != 1:
        raise ValueError("unsupported export codec version")
    return obj


def _fields(obj, fields):
    if type(obj) is not dict or set(obj) != set(fields):
        raise ValueError("unexpected metadata fields")


@dataclass(frozen=True)
class SpanSpec:
    region_id: str
    buffer_offset: int
    length: int
    region_offset: int


@dataclass(frozen=True)
class DecodedExport:
    layout: PoolLayout
    capacity: int
    descriptors: tuple[TransportDescriptor, ...]
    bindings: tuple[tuple[str, tuple[SpanSpec, ...]], ...]

    def bind(self, imports):
        return {
            name: tuple(
                RegionSpan(
                    s.buffer_offset, s.length, imports[s.region_id], s.region_offset
                )
                for s in spans
            )
            for name, spans in self.bindings
        }


def encode_export(source, lease_id, adapter, descriptors):
    """Encode existing exports; never register/export or allocate memory here."""
    regions = tuple(
        RegionDescriptor(
            region_id,
            _dump(
                {
                    "version": 1,
                    "source": asdict(source),
                    "lease_id": lease_id,
                    "transport": descriptor.to_dict(),
                }
            ),
        )
        for region_id, descriptor in descriptors.items()
    )
    bindings = tuple(
        BufferBinding(
            buffer.name,
            adapter.bindings[buffer.name][0].memory.region_id,
            _dump(
                {
                    "version": 1,
                    "dtype": adapter.layout.dtype,
                    "page_size": adapter.layout.page_size,
                    "capacity": adapter.capacity,
                    "buffer": asdict(buffer),
                    "spans": [
                        {
                            "region_id": s.memory.region_id,
                            "buffer_offset": s.buffer_offset,
                            "length": s.length,
                            "region_offset": s.region_offset,
                        }
                        for s in adapter.bindings[buffer.name]
                    ],
                }
            ),
        )
        for buffer in adapter.layout.buffers
    )
    return regions, bindings


def decode_export(response, target_layout, target_owner_id):
    """Validate ALL metadata and source pages before the first CUDA import.

    The protocol's opaque layout_digest is already peer-checked; we additionally
    compare the actual full layout, so a matching digest alone grants no access.
    """
    if not 0 < len(response.regions) <= MAX_REGIONS:
        raise ValueError("region count exceeds metadata bound")
    if not 0 < len(response.buffer_bindings) <= MAX_SPANS:
        raise ValueError("buffer count exceeds metadata bound")
    if (
        sum(len(x.descriptor) for x in (*response.regions, *response.buffer_bindings))
        > MAX_METADATA_BYTES
    ):
        raise ValueError("export metadata exceeds byte bound")
    descriptors, owners, leases, handles = {}, set(), set(), set()
    for region in response.regions:
        obj = _load(region.descriptor, ("version", "source", "lease_id", "transport"))
        if (
            EngineIdentity(**obj["source"]) != response.request.result.source
            or obj["lease_id"] != response.lease_id
        ):
            raise ValueError("descriptor source/control lease mismatch")
        descriptor = TransportDescriptor.from_dict(obj["transport"])
        raw = base64.b64decode(descriptor.transport_handle, validate=True)
        if (
            descriptor.region_id != region.region_id
            or descriptor.region_id in descriptors
            or descriptor.owner_id == target_owner_id
            or descriptor.transport != "nvlink-fabric"
            or descriptor.access != "read"
            or len(raw) != 64
            or descriptor.lease_id in leases
            or raw in handles
        ):
            raise ValueError("invalid, aliased, or incompatible transport export")
        descriptors[region.region_id] = descriptor
        owners.add(descriptor.owner_id)
        leases.add(descriptor.lease_id)
        handles.add(raw)
    if len(owners) != 1:
        raise ValueError("all regions must have one transport owner")
    header, layouts, bindings, used, intervals = None, [], [], set(), {}
    span_count = 0
    for binding in response.buffer_bindings:
        obj = _load(
            binding.descriptor,
            ("version", "dtype", "page_size", "capacity", "buffer", "spans"),
        )
        checked_int(obj["capacity"], "source capacity", 2)
        checked_int(obj["page_size"], "source page_size", 1)
        current = (obj["dtype"], obj["page_size"], obj["capacity"])
        if header is not None and header != current:
            raise ValueError("inconsistent pool headers")
        header = current
        _fields(obj["buffer"], ("name", "page_bytes", "layout"))
        layout = BufferLayout(**obj["buffer"])
        checked_int(layout.page_bytes, "page_bytes", 1)
        if layout.name != binding.buffer_name:
            raise ValueError("buffer name mismatch")
        if type(obj["spans"]) is not list or not obj["spans"]:
            raise ValueError("buffer must have spans")
        spans, end = [], 0
        for value in obj["spans"]:
            span_count += 1
            if span_count > MAX_SPANS:
                raise ValueError("span count exceeds bound")
            _fields(value, ("region_id", "buffer_offset", "length", "region_offset"))
            span = SpanSpec(**value)
            checked_int(span.buffer_offset, "buffer_offset")
            checked_int(span.length, "span length", 1)
            checked_int(span.region_offset, "region_offset")
            if type(span.region_id) is not str or span.region_id not in descriptors:
                raise ValueError("unknown span region")
            region_end = checked_int(span.region_offset + span.length, "region end")
            if (
                span.buffer_offset != end
                or region_end > descriptors[span.region_id].size
            ):
                raise ValueError("invalid span extent or buffer gap/overlap")
            end = checked_int(end + span.length, "buffer extent")
            intervals.setdefault(span.region_id, []).append(
                (span.region_offset, region_end)
            )
            used.add(span.region_id)
            spans.append(span)
        if (
            end != checked_int(header[2] * layout.page_bytes, "full buffer extent")
            or binding.region_id != spans[0].region_id
        ):
            raise ValueError("full-buffer capacity or first-span anchor mismatch")
        layouts.append(layout)
        bindings.append((binding.buffer_name, tuple(spans)))
    for ranges in intervals.values():
        ranges.sort()
        if any(a[1] > b[0] for a, b in pairwise(ranges)):
            raise ValueError("aliased source spans")
    if used != set(descriptors):
        raise ValueError("unreferenced export region")
    source_layout = PoolLayout(header[0], header[1], tuple(layouts))
    source_layout.validate()
    if source_layout != target_layout:
        raise ValueError("source and target full layouts differ")
    pages = response.source_page_ids
    if len(set(pages)) != len(pages) or any(
        type(page) is not int or not 0 < page < header[2] for page in pages
    ):
        raise ValueError("invalid source pages")
    return DecodedExport(
        source_layout, header[2], tuple(descriptors.values()), tuple(bindings)
    )
