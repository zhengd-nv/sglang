"""Strict SlotGuard hash-Pin registry; data ABI/export metadata remain version 1."""

import base64
import hashlib
import json
import re
import struct
import threading
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from json.decoder import scanstring as _scanstring
from typing import get_args, get_origin, get_type_hints

from . import protocol as abi
from . import slotguard_protocol as p
from .hash_pin import HashLookupResult
from .int_vectors import from_int64_unpacked, is_int64_tuple, pack_int64_le
from .slot_guard import LeaseIdentity, PageRef
from .slotguard_limits import MAX_PACKED_BASE64_BYTES, MAX_PACKED_ITEMS, MAX_PIN_PAGES

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 16
MAX_ITEMS = 300_000
MAX_STRING = 4 * 1024 * 1024
_MESSAGES = (p.PinReq, p.PinPreparedResp, p.UnpinReq, p.UnpinResp)
_CLASSES = (
    *_MESSAGES,
    p.PinReference,
    HashLookupResult,
    LeaseIdentity,
    PageRef,
    abi.EngineIdentity,
    abi.TransferId,
    abi.RegionDescriptor,
    abi.BufferBinding,
)
_SCHEMAS = {
    cls: {f.name: get_type_hints(cls)[f.name] for f in fields(cls) if f.init}
    for cls in _CLASSES
}
# The ledger keeps transfer_id opaque; on the wire it is always a TransferId.
_SCHEMAS[LeaseIdentity]["transfer_id"] = abi.TransferId
# Responses carry a compact binding. Only a trusted local call context can
# restore the in-memory .request used by the unchanged export metadata decoder.
_SCHEMAS[p.PinPreparedResp] = {
    "reference" if name == "request" else name: (
        p.PinReference if name == "request" else typ
    )
    for name, typ in _SCHEMAS[p.PinPreparedResp].items()
}
_STRUCTURAL = re.compile(r'["\[\]{}]')
_PREPARED_ENCODE_LOCK = threading.Lock()


class WireError(ValueError):
    """Messages deliberately omit payload data."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WireError("duplicate wire field")
        result[key] = value
    return result


def _depth_check(body):
    # Match json.loads(bytes)'s encoding detection, including BOM and unpaired
    # surrogates. A byte scanner would miss nesting in UTF-16/32 input.
    text = body.decode(json.detect_encoding(body), "surrogatepass")
    stack = []
    offset = 0
    while match := _STRUCTURAL.search(text, offset):
        token = match.group()
        offset = match.end()
        if token == '"':
            # CPython's C string scanner skips the entire (often huge base64)
            # string, validating escapes and termination without Python loops.
            _, offset = _scanstring(text, offset, True)
        elif token in "[{":
            stack.append(token)
            if len(stack) > MAX_DEPTH:
                raise WireError("wire depth exceeded")
        elif not stack or stack.pop() != ("[" if token == "]" else "{"):
            raise WireError("invalid wire nesting")
    if stack:
        raise WireError("invalid wire nesting")
    return text


def _pack_ints(value):
    if type(value) is not tuple or len(value) > MAX_PACKED_ITEMS:
        raise WireError("invalid v2 integer vector")
    if not is_int64_tuple(value):
        raise WireError("v2 integer vector exceeds signed int64 or is not int")
    raw = pack_int64_le(value)
    return {
        "codec": "i64le",
        "count": len(value),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _unpack_ints(value, depth):
    if (
        type(value) is not dict
        or value.keys() != {"codec", "count", "data"}
        or type(value["codec"]) is not str
        or value["codec"] != "i64le"
        or type(value["count"]) is not int
        or not 0 <= value["count"] <= MAX_PACKED_ITEMS
        or type(value["data"]) is not str
        or len(value["data"]) > MAX_PACKED_BASE64_BYTES
    ):
        raise WireError("invalid v2 packed integer vector")
    count, encoded = value["count"], value["data"]
    if len(encoded) != 4 * ((count * 8 + 2) // 3) or (count and depth + 1 > MAX_DEPTH):
        raise WireError("invalid v2 packed vector length/depth")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != count * 8 or base64.b64encode(raw).decode("ascii") != encoded:
        raise WireError("noncanonical v2 packed vector")
    # Explicit width, signedness and byte order, independent of native C sizes.
    # ABI constructors still validate semantic ranges (e.g. nonnegative tokens).
    return from_int64_unpacked(struct.unpack(f"<{count}q", raw), raw)


def _pack_pages(pages):
    if (
        type(pages) is not tuple
        or len(pages) > MAX_PIN_PAGES
        or any(type(page) is not PageRef for page in pages)
        or len({page.pool_epoch for page in pages}) > 1
    ):
        raise WireError("invalid v2 page table")
    return {
        "codec": "pageref-i64le-v1",
        "pool_epoch": pages[0].pool_epoch if pages else 0,
        "count": len(pages),
        "page_ids": _pack_ints(tuple(page.page_id for page in pages))["data"],
        "generations": _pack_ints(tuple(page.generation for page in pages))["data"],
    }


def _unpack_pages(value, depth, expected_count):
    if (
        type(value) is not dict
        or value.keys() != {"codec", "pool_epoch", "count", "page_ids", "generations"}
        or type(value["codec"]) is not str
        or value["codec"] != "pageref-i64le-v1"
        or type(value["pool_epoch"]) is not int
        or not 0 <= value["pool_epoch"] < 1 << 63
        or type(value["count"]) is not int
        or not 0 <= value["count"] <= MAX_PIN_PAGES
        or value["count"] != expected_count
        or (not value["count"] and value["pool_epoch"] != 0)
    ):
        raise WireError("invalid v2 page table")
    vectors = tuple(
        _unpack_ints(
            {"codec": "i64le", "count": value["count"], "data": value[name]}, depth
        )
        for name in ("page_ids", "generations")
    )
    ids, generations = vectors
    if len(set(ids)) != len(ids) or any(
        item < 1 for vector in vectors for item in vector
    ):
        raise WireError("invalid v2 page identities")
    return tuple(
        PageRef(value["pool_epoch"], page_id, generation)
        for page_id, generation in zip(ids, generations)
    )


def _plain(value, expected=None):
    if is_dataclass(value) and type(value) in _SCHEMAS:
        return {
            name: _plain(
                (
                    value.request.reference
                    if type(value) is p.PinPreparedResp and name == "reference"
                    else getattr(value, name)
                ),
                _SCHEMAS[type(value)][name],
            )
            for name in _SCHEMAS[type(value)]
        }
    if isinstance(value, Enum):
        return value.value
    if type(value) is bytes:
        return base64.b64encode(value).decode("ascii")
    if type(value) is tuple:
        if expected == tuple[PageRef, ...]:
            return _pack_pages(value)
        if expected == tuple[int, ...]:
            return _pack_ints(value)
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    raise WireError("unsupported v2 value")


def _typed(value, expected, depth=0, request_context=None):
    if depth > MAX_DEPTH:
        raise WireError("wire depth exceeded")
    origin, args = get_origin(expected), get_args(expected)
    if origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        return _typed(
            value, next(t for t in args if t is not type(None)), depth, request_context
        )
    if origin is tuple:
        if args == (int, Ellipsis):
            return _unpack_ints(value, depth)
        if type(value) is not list or len(value) > MAX_ITEMS:
            raise WireError("invalid v2 array")
        return tuple(
            _typed(item, args[0], depth + 1, request_context) for item in value
        )
    if expected in _SCHEMAS:
        schema = _SCHEMAS[expected]
        if type(value) is not dict or value.keys() != schema.keys():
            raise WireError("invalid v2 fields")
        if expected is p.PinPreparedResp:
            # Validate correlation BEFORE processing any unsolicited exports.
            if (
                type(value["protocol_version"]) is not int
                or value["protocol_version"] != p.CONTROL_VERSION
                or value["capability"] != p.SLOTGUARD_CAPABILITY
            ):
                raise WireError("invalid v2 response capability")
            reference = _typed(value["reference"], p.PinReference, depth + 1)
            if (
                type(request_context) is not p.PinReq
                or reference != request_context.reference
            ):
                raise WireError("v2 response requires exact local Pin context")
            values = {"request": request_context}
            values.update(
                {
                    k: (
                        _typed(value[k], t, depth + 1, request_context)
                        if k != "pages"
                        else _unpack_pages(
                            value[k],
                            depth + 1,
                            (
                                request_context.result.restore_pages
                                if value["status"] == p.PinStatus.PREPARED.value
                                else 0
                            ),
                        )
                    )
                    for k, t in schema.items()
                    if k != "reference"
                }
            )
            return expected(**values)
        return expected(
            **{
                k: _typed(value[k], t, depth + 1, request_context)
                for k, t in schema.items()
            }
        )
    if expected in (p.PinStatus, p.UnpinStatus):
        if type(value) is not str:
            raise WireError("invalid v2 status")
        return expected(value)
    if expected is bytes:
        if type(value) is not str or len(value) > MAX_STRING:
            raise WireError("invalid v2 descriptor")
        decoded = base64.b64decode(value, validate=True)
        if base64.b64encode(decoded).decode("ascii") != value:
            raise WireError("noncanonical v2 descriptor")
        return decoded
    if type(value) is not expected or (expected is str and len(value) > MAX_STRING):
        raise WireError("invalid v2 type")
    return value


def _body(message):
    return json.dumps(
        {
            "version": message.protocol_version,
            "type": type(message).__name__,
            "payload": _plain(message),
        },
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pin_binding(request):
    """SHA256 of canonical ASCII JSON envelope (declaration field order,
    compact separators, ASCII escapes, canonical base64, i64le vectors). Includes
    every public Pin and embedded ABI field, including namespace, identities,
    tokens, ROOT keys and L/H/page size (which uniquely derive restore pages).
    Private derived fields never participate.
    This is a correlation binding, not an authority for matching source pages.
    """
    body = _body(request)
    if len(body) > MAX_BODY_BYTES:
        raise WireError("bounded canonical Pin body exceeded")
    return hashlib.sha256(body).hexdigest(), body


def reference_bytes(reference):
    """Compact immutable history accounting, without revisiting the prompt."""
    if type(reference) is not p.PinReference:
        raise WireError("exact reference required")
    return json.dumps(
        _plain(reference), separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def encode(message, *, max_body_bytes=MAX_BODY_BYTES):
    if type(message) not in _MESSAGES:
        raise WireError("unsupported v2 message")
    if type(message) is p.PinReq:
        # Fully validated and canonicalized once at construction. Incoming Pin
        # decoding independently validates every vector/ABI field and computes
        # its own canonical body. No untrusted cache/hash field is accepted.
        body = message._wire_body
        if len(body) > max_body_bytes:
            raise WireError("bounded v2 Pin body exceeded")
        return body
    if type(message) is p.PinPreparedResp and message.status is p.PinStatus.PREPARED:
        # Only strict outgoing validation may publish a cache. A single cold
        # encoder wins even under concurrent calls; warm calls need no lock.
        body = message._wire_body
        if body is None:
            with _PREPARED_ENCODE_LOCK:
                body = message._wire_body
                if body is None:
                    body = _body(message)
                    decode(body, type(message), request_context=message.request)
                    object.__setattr__(message, "_wire_body", body)
        if len(body) > max_body_bytes:
            raise WireError("bounded v2 prepared body exceeded")
        return body
    body = _body(message)
    decode(
        body,
        type(message),
        max_body_bytes=max_body_bytes,
        request_context=message.request if type(message) is p.PinPreparedResp else None,
    )
    return body


def decode(body, expected, *, max_body_bytes=MAX_BODY_BYTES, request_context=None):
    if (
        type(body) is not bytes
        or len(body) > max_body_bytes
        or expected not in _MESSAGES
    ):
        raise WireError("invalid bounded v2 body/type")
    try:
        text = _depth_check(body)
        root = json.loads(text, object_pairs_hook=_pairs)
        if (
            type(root) is not dict
            or root.keys() != {"version", "type", "payload"}
            or type(root["version"]) is not int
            or root["version"] != p.CONTROL_VERSION
            or type(root["type"]) is not str
            or root["type"] != expected.__name__
        ):
            raise WireError("invalid v2 envelope")
        result = _typed(root["payload"], expected, request_context=request_context)
        if result.protocol_version != root["version"]:
            raise WireError("control envelope/payload mismatch")
        if (
            expected is p.UnpinResp
            and request_context is not None
            and result.request != request_context
        ):
            raise WireError("v2 Unpin context mismatch")
        return result
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise WireError("invalid SlotGuard v2 message") from None
