# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import base64
import ctypes as C
import os
import threading
import uuid


class Region:
    """Explicit owner. No raw tensor views escape; each copy drains before return."""

    def __init__(self, size=None, numa=0, descriptor=None):
        self._lock = threading.RLock()
        self._handle = C.c_void_p()
        self._lib = lib = C.CDLL(os.environ["EGM_NATIVE_LIB"])
        lib.egm_error.restype = C.c_char_p
        lib.egm_fabric_size.restype = C.c_size_t
        lib.egm_create.argtypes = [C.c_size_t, C.c_int, C.POINTER(C.c_void_p)]
        lib.egm_import.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.POINTER(C.c_void_p)]
        lib.egm_export.argtypes = [C.c_void_p, C.c_void_p]
        for name in ["egm_size", "egm_granularity"]:
            fn = getattr(lib, name); fn.argtypes = [C.c_void_p]; fn.restype = C.c_size_t
        lib.egm_copy.argtypes = [C.c_void_p, C.c_size_t, C.c_uint64, C.c_size_t, C.c_int]
        lib.egm_close.argtypes = [C.c_void_p]
        self.imported = descriptor is not None
        if descriptor is None:
            if not isinstance(size, int) or not 0 < size < 2**63 or numa < 0:
                raise ValueError("invalid size/NUMA")
            self._check(lib.egm_create(size, numa, C.byref(self._handle)))
            self.epoch = uuid.uuid4().hex
            self.logical_size = size
        else:
            if descriptor["version"] != 1 or descriptor["kind"] != "HOST_NUMA_FABRIC":
                raise ValueError("unsupported EGM descriptor")
            raw = base64.b64decode(descriptor["fabric"], validate=True)
            if len(raw) != lib.egm_fabric_size():
                raise ValueError("invalid Fabric handle length")
            n, gran = descriptor["allocation_bytes"], descriptor["granularity"]
            if not 0 < n < 2**63 or not 0 < gran <= n or n % gran:
                raise ValueError("invalid descriptor size")
            self.logical_size = descriptor["logical_bytes"]
            if not 0 < self.logical_size <= n:
                raise ValueError("invalid logical size")
            self._check(lib.egm_import(C.create_string_buffer(raw), n, gran, C.byref(self._handle)))
            self.epoch = descriptor["epoch"]
        self.allocation_bytes = lib.egm_size(self._handle)
        self.granularity = lib.egm_granularity(self._handle)

    def _check(self, rc):
        if rc:
            raise RuntimeError(self._lib.egm_error().decode())

    def _open(self):
        if not self._handle.value:
            raise RuntimeError("region is closed")

    def descriptor(self):
        with self._lock:
            self._open()
            if self.imported:
                raise ValueError("cannot re-export imported region")
            raw = C.create_string_buffer(self._lib.egm_fabric_size())
            self._check(self._lib.egm_export(self._handle, raw))
            return dict(version=1, kind="HOST_NUMA_FABRIC", epoch=self.epoch,
                        logical_bytes=self.logical_size, allocation_bytes=self.allocation_bytes,
                        granularity=self.granularity, fabric=base64.b64encode(raw.raw).decode())

    def copy(self, offset, pointer, size, direction):
        with self._lock:
            self._open()
            if not (0 <= offset <= self.logical_size and 0 < size <= self.logical_size - offset):
                raise ValueError("out-of-bounds or empty range")
            if not 0 < pointer < 2**64 or direction not in range(4):
                raise ValueError("invalid endpoint/direction")
            if self.imported and direction in (0, 2):
                raise ValueError("imported regions are read-only to clients")
            self._check(self._lib.egm_copy(self._handle, offset, pointer, size, direction))

    def close(self):
        with self._lock:
            if self._handle.value:
                self._check(self._lib.egm_close(self._handle))
                self._handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
