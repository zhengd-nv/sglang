# SPDX-License-Identifier: Apache-2.0
"""Thin ctypes binding; only libcuda is linked by the native backend."""
import base64
import ctypes as C
from pathlib import Path
from .api import TransferError


class _Range(C.Structure):
    _fields_ = [("source", C.c_void_p), ("source_offset", C.c_uint64),
                ("target", C.c_void_p), ("target_offset", C.c_uint64),
                ("length", C.c_uint64)]


class Driver:
    def __init__(self, library: str):
        self.lib = C.CDLL(str(Path(library).resolve()))
        definitions = {
            "nvt_abi_version": ([], C.c_int), "nvt_fabric_size": ([], C.c_size_t),
            "nvt_error": ([], C.c_char_p),
            "nvt_register_vmm": ([C.c_int, C.c_uint64, C.c_size_t, C.POINTER(C.c_void_p)], C.c_int),
            "nvt_create": ([C.c_int, C.c_size_t, C.POINTER(C.c_void_p)], C.c_int),
            "nvt_import": ([C.c_int, C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.POINTER(C.c_void_p)], C.c_int),
            "nvt_export": ([C.c_void_p, C.c_void_p], C.c_int),
            "nvt_info": ([C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)], C.c_int),
            "nvt_host_copy": ([C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int], C.c_int),
            "nvt_submit": ([C.POINTER(_Range), C.c_size_t, C.POINTER(C.c_void_p)], C.c_int),
            "nvt_wait": ([C.c_void_p, C.c_double], C.c_int),
            "nvt_poll": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
            "nvt_batch_error": ([C.c_void_p], C.c_char_p),
            "nvt_release_batch": ([C.c_void_p], C.c_int),
            "nvt_close": ([C.c_void_p], C.c_int),
        }
        for name, (args, result) in definitions.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result
        if self.lib.nvt_abi_version() != 2 or self.lib.nvt_fabric_size() != 64:
            raise TransferError("native ABI mismatch")

    def check(self, code):
        if code != 0:
            raise TransferError(self.lib.nvt_error().decode())

    def create(self, device, size):
        out = C.c_void_p(); self.check(self.lib.nvt_create(device, size, C.byref(out)))
        return out.value

    def register_vmm(self, device, address, physical):
        out = C.c_void_p()
        self.check(self.lib.nvt_register_vmm(device, address, physical, C.byref(out)))
        return out.value

    def info(self, handle):
        ptr, physical, gran = C.c_uint64(), C.c_uint64(), C.c_uint64()
        self.check(self.lib.nvt_info(handle, C.byref(ptr), C.byref(physical), C.byref(gran)))
        return ptr.value, physical.value, gran.value

    def export(self, handle):
        raw = C.create_string_buffer(64); self.check(self.lib.nvt_export(handle, raw))
        return base64.b64encode(raw.raw).decode()

    def import_region(self, device, desc):
        raw = C.create_string_buffer(base64.b64decode(desc.transport_handle), 64)
        out = C.c_void_p()
        self.check(self.lib.nvt_import(device, raw, desc.size, desc.physical_size, desc.granularity, C.byref(out)))
        return out.value

    def upload(self, handle, offset, data):
        raw = C.create_string_buffer(data, len(data))
        self.check(self.lib.nvt_host_copy(handle, offset, raw, len(data), 0))

    def download(self, handle, offset, length):
        raw = C.create_string_buffer(length)
        self.check(self.lib.nvt_host_copy(handle, offset, raw, length, 1))
        return raw.raw

    def submit(self, ranges):
        rows = (_Range * len(ranges))(*[_Range(*r) for r in ranges])
        out = C.c_void_p(); self.check(self.lib.nvt_submit(rows, len(rows), C.byref(out)))
        return out.value

    def poll(self, handle):
        state = C.c_int(); self.check(self.lib.nvt_poll(handle, C.byref(state)))
        return state.value, self.lib.nvt_batch_error(handle).decode()

    def wait(self, handle, timeout):
        self.check(self.lib.nvt_wait(handle, -1.0 if timeout is None else timeout))

    def release_batch(self, handle):
        self.check(self.lib.nvt_release_batch(handle))

    def close(self, handle):
        self.check(self.lib.nvt_close(handle))
