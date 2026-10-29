# SPDX-License-Identifier: Apache-2.0
"""HiCache host pool in exportable Grace DRAM (EGM): route A of L2 sharing.

The Engine owns the DRAM: the host pool is one HOST_NUMA + FABRIC allocation per
buffer, mapped for this GPU and for the CPU, so the same VA is a CPU tensor for
HiCache and a GPU address for its copy kernels. HiCache must not host-register
it (``pin_memory`` is forced off): the kernels then take ``data_ptr()`` as is.
Remote readers import the same allocation through the L1 transport.

Enable with ``SGLANG_L1_EGM_HOST_POOL=1``; ``SGLANG_L1_EGM_NUMA`` overrides the
NUMA node, which defaults to the GPU's own.
"""
import ctypes
import math
import os
from pathlib import Path

import torch

from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator


def egm_host_pool_enabled():
    return os.environ.get('SGLANG_L1_EGM_HOST_POOL', '0') not in ('0', '', 'false', 'False')


def gpu_numa_node(device):
    override = os.environ.get('SGLANG_L1_EGM_NUMA')
    if override:
        return int(override)
    p = torch.cuda.get_device_properties(device)
    bus = f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0'
    path = Path('/sys/bus/pci/devices') / bus / 'numa_node'
    node = int(path.read_text()) if path.is_file() else -1
    if node < 0:
        raise ValueError(f'cannot tell the NUMA node of GPU {device}; set SGLANG_L1_EGM_NUMA')
    return node


class EGMHostTensorAllocator(HostTensorAllocator):
    """Allocations live for the process lifetime; nothing is host-registered."""

    # HostKVCache turns pin_memory off for this allocator: cudaHostRegister and
    # cudaHostGetDevicePointer reject VMM memory, and the VA is GPU-mapped anyway.
    requires_host_register = False

    def __init__(self):
        super().__init__()
        from sglang.srt.mem_cache.l1_transfer.transport._driver import Driver
        from sglang.srt.mem_cache.l1_transfer.transport.engine import default_library
        self.driver = Driver(default_library())
        # One entry per buffer: address, allocation bytes, NUMA node, native handle.
        self.regions = []

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == 'cpu', f'EGM host pools are CPU tensors; got device={device!r}'
        self.dtype, self.dims = dtype, dims
        gpu = torch.cuda.current_device()
        numa = gpu_numa_node(gpu)
        numel = math.prod(dims)
        nbytes = numel * dtype.itemsize
        handle = self.driver.create_host_numa(gpu, numa, nbytes)
        address, physical, _ = self.driver.info(handle)
        array = (ctypes.c_uint8 * nbytes).from_address(address)
        tensor = torch.frombuffer(array, dtype=dtype, count=numel).reshape(dims)
        self.regions.append(dict(address=address, physical=physical, numa=numa, handle=handle))
        return tensor


class EGMHostPoolAdapter:
    """Export an EGM host pool so peers read its pages exactly like a device pool's.

    A layer_first host pool keeps each layer's K (then V) as one [size, heads, dim]
    slab, the same rows as the device's NHD buffers, so the bindings reuse the
    device buffer names and page sizes and the target plans with its own layout.
    """

    def __init__(self, host_pool, engine, layout):
        from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan
        allocator = host_pool.allocator
        if not isinstance(allocator, EGMHostTensorAllocator) or host_pool.layout != 'layer_first':
            raise ValueError('L2 sharing requires an EGM layer_first host pool')
        if len(allocator.regions) != 1 or allocator.regions[0]['address'] != host_pool.kv_buffer.data_ptr():
            raise ValueError('expected the host pool as one EGM allocation')
        region = allocator.regions[0]
        token_bytes = host_pool.token_stride_size
        slab = host_pool.size * token_bytes
        self.engine, self.layout = engine, layout
        self.numa = region['numa']
        self.memory = engine.register_vmm(region['address'], region['physical'], owner=host_pool)
        self.memories = [self.memory]
        self.capacity = host_pool.page_num  # host page IDs start at 0
        self.bindings = {}
        for buffer in layout.buffers:
            kind, layer = buffer.name[0], int(buffer.name[1:])
            if kind not in 'kv' or buffer.page_bytes != host_pool.page_size * token_bytes:
                raise ValueError(f'host pool does not match device buffer {buffer.name}')
            index = layer if kind == 'k' else host_pool.layer_num + layer
            self.bindings[buffer.name] = (RegionSpan(0, slab, self.memory, index * slab),)

