"""Finite v2 content bounds; generic control/data ABI limits stay unchanged."""

MAX_PREFIX_TOKENS = 524288
MAX_PIN_PAGES = 8192
MAX_PACKED_ITEMS = MAX_PREFIX_TOKENS
# Canonical base64 for at most MAX_PACKED_ITEMS signed int64 values.
MAX_PACKED_BASE64_BYTES = 4 * ((MAX_PACKED_ITEMS * 8 + 2) // 3)
