# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import base64
import ctypes as C
import os
import threading
import uuid


class Region:
    """Explicit owner. No raw tensor views escape; each copy drains before return."""

    def __init__(self, size=None, numa=0, descriptor=None, tier="dram", device=0):
        self._lock = threading.RLock()
        self._handle = C.c_void_p()
        self._lib = lib = C.CDLL(os.environ["EGM_NATIVE_LIB"])
        if not hasattr(lib, "egm_abi_version") or lib.egm_abi_version() != 3:
            raise RuntimeError("rebuild EGM_NATIVE_LIB: native ABI 3 required")
        lib.egm_error.restype = C.c_char_p
        lib.egm_fabric_size.restype = C.c_size_t
        lib.egm_create.argtypes = [
            C.c_size_t,
            C.c_int,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        lib.egm_import.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_size_t,
            C.c_int,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        lib.egm_export.argtypes = [C.c_void_p, C.c_void_p]
        for name in ["egm_size", "egm_granularity"]:
            fn = getattr(lib, name)
            fn.argtypes = [C.c_void_p]
            fn.restype = C.c_size_t
        lib.egm_copy.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_uint64,
            C.c_size_t,
            C.c_int,
        ]
        lib.egm_copy_batch.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.POINTER(C.c_size_t),
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_size_t),
            C.c_int,
            C.c_int,
            C.POINTER(C.c_size_t),
        ]
        lib.egm_batch_submit.argtypes = [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_size_t,
            C.c_void_p,
            C.POINTER(C.c_void_p),
        ]
        lib.egm_batch_wait.argtypes = [C.c_void_p, C.c_size_t]
        lib.egm_batch_query.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        lib.egm_batch_free.argtypes = [C.c_void_p]
        lib.egm_close.argtypes = [C.c_void_p]
        self.imported = descriptor is not None
        self.device = device
        if descriptor is not None:
            tier = descriptor.get("tier")
        if tier not in ("dram", "hbm") or not isinstance(device, int) or device < 0:
            raise ValueError("invalid tier/local device")
        self.tier = tier
        kind = "HOST_NUMA_FABRIC" if tier == "dram" else "DEVICE_FABRIC"
        tier_id = int(tier == "hbm")
        if descriptor is None:
            if not isinstance(size, int) or not 0 < size < 2**63 or numa < 0:
                raise ValueError("invalid size/NUMA")
            self._check(
                lib.egm_create(
                    size,
                    tier_id,
                    numa if tier == "dram" else device,
                    device,
                    C.byref(self._handle),
                )
            )
            self.epoch = uuid.uuid4().hex
            self.logical_size = size
        else:
            if descriptor["version"] != 2 or descriptor["kind"] != kind:
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
            self._check(
                lib.egm_import(
                    C.create_string_buffer(raw),
                    n,
                    gran,
                    tier_id,
                    device,
                    C.byref(self._handle),
                )
            )
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
            return dict(
                version=2,
                tier=self.tier,
                kind="HOST_NUMA_FABRIC" if self.tier == "dram" else "DEVICE_FABRIC",
                epoch=self.epoch,
                logical_bytes=self.logical_size,
                allocation_bytes=self.allocation_bytes,
                granularity=self.granularity,
                fabric=base64.b64encode(raw.raw).decode(),
            )

    def copy(self, offset, pointer, size, direction):
        with self._lock:
            self._open()
            if not (
                0 <= offset <= self.logical_size
                and 0 < size <= self.logical_size - offset
            ):
                raise ValueError("out-of-bounds or empty range")
            if not 0 < pointer < 2**64 or direction not in range(4):
                raise ValueError("invalid endpoint/direction")
            if self.imported and direction in (0, 2):
                raise ValueError("imported regions are read-only to clients")
            self._check(
                self._lib.egm_copy(self._handle, offset, pointer, size, direction)
            )

    def copy_batch(self, offsets, pointers, sizes, direction, method=0):
        """GPU<->region only; drains before return. method 0 async per run, 1 cuMemcpyBatchAsync, 2 gather kernel."""
        with self._lock:
            self._open()
            if direction not in (2, 3):
                raise ValueError("batch copy supports GPU<->region only")
            if self.imported and direction == 2:
                raise ValueError("imported regions are read-only to clients")
            n = len(offsets)
            if not n or len(pointers) != n or len(sizes) != n:
                raise ValueError("empty or mismatched batch")
            submitted = C.c_size_t()
            self._check(
                self._lib.egm_copy_batch(
                    self._handle,
                    n,
                    (C.c_size_t * n)(*offsets),
                    (C.c_uint64 * n)(*pointers),
                    (C.c_size_t * n)(*sizes),
                    direction,
                    method,
                    C.byref(submitted),
                )
            )
            return submitted.value

    def submit(
        self, offsets, pointers, sizes, direction, group_ends, method=2, writable=False
    ):
        """Asynchronous grouped GPU<->region copy; returns a :class:`Batch` that must be freed.

        ``offsets`` / ``pointers`` / ``sizes`` are uint64 numpy arrays; ``group_ends`` are exclusive ends.
        An imported region accepts writes (direction 2) only when the caller holds an allocation
        (``writable=True``); the owner hands out disjoint ranges.
        """
        import numpy as np

        with self._lock:
            self._open()
            if direction not in (2, 3):
                raise ValueError("batch copy supports GPU<->region only")
            if self.imported and direction == 2 and not writable:
                raise ValueError("imported regions are read-only without an allocation")
            arrays = [
                np.ascontiguousarray(a, dtype=np.uint64)
                for a in (offsets, pointers, sizes, group_ends)
            ]
            n = len(arrays[0])
            if (
                not n
                or len(arrays[1]) != n
                or len(arrays[2]) != n
                or not len(arrays[3])
            ):
                raise ValueError("empty or mismatched batch")
            handle = C.c_void_p()
            self._check(
                self._lib.egm_batch_submit(
                    self._handle,
                    n,
                    arrays[0].ctypes.data,
                    arrays[1].ctypes.data,
                    arrays[2].ctypes.data,
                    direction,
                    method,
                    len(arrays[3]),
                    arrays[3].ctypes.data,
                    C.byref(handle),
                )
            )
            return Batch(self, handle, len(arrays[3]))

    def close(self):
        with self._lock:
            if self._handle.value:
                self._check(self._lib.egm_close(self._handle))
                self._handle = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Batch:
    """In-flight grouped copy. ``wait(g)`` blocks until group ``g`` landed; ``free()`` drains and releases."""

    def __init__(self, region, handle, groups):
        self.region, self._handle, self.groups = region, handle, groups

    def wait(self, group):
        self.region._check(self.region._lib.egm_batch_wait(self._handle, group))

    def done(self):
        pending = C.c_int()
        self.region._check(
            self.region._lib.egm_batch_query(self._handle, C.byref(pending))
        )
        return not pending.value

    def free(self):
        if self._handle is not None:
            handle, self._handle = self._handle, None
            self.region._check(self.region._lib.egm_batch_free(handle))
