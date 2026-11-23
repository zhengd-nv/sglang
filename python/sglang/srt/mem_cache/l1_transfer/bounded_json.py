# SPDX-License-Identifier: Apache-2.0
"""Bound JSON nesting and value count before json.loads allocates anything."""

import json
import re
from json.decoder import scanstring as _scanstring

_STRUCTURAL = re.compile(r'["\[\]{},]')


def checked_json_text(body, *, max_depth, max_items, error=ValueError):
    """Return ``body`` decoded as json.loads(bytes) would, after bounding its shape.

    Containers plus separators outside strings bound every array element and
    object member, so an oversized body fails before any value is built.
    """
    # Match json.loads(bytes)'s encoding detection, including BOM and unpaired
    # surrogates. A byte scanner would miss nesting in UTF-16/32 input.
    text = body.decode(json.detect_encoding(body), "surrogatepass")
    stack, offset, items = [], 0, 0
    while match := _STRUCTURAL.search(text, offset):
        token = match.group()
        offset = match.end()
        if token == '"':
            # CPython's C string scanner skips the entire (often huge base64)
            # string, validating escapes and termination without Python loops.
            _, offset = _scanstring(text, offset, True)
            continue
        if token in "[{,":
            items += 1
            if items > max_items:
                raise error("JSON item budget exceeded")
            if token == ",":
                continue
            stack.append(token)
            if len(stack) > max_depth:
                raise error("JSON depth exceeded")
        elif not stack or stack.pop() != ("[" if token == "]" else "{"):
            raise error("invalid JSON nesting")
    if stack:
        raise error("invalid JSON nesting")
    return text
