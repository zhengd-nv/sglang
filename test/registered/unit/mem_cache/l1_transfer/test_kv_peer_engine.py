"""SlotGuard's READ path on KvPeerEngine (upstream kv_peer_read), on a fake CUDA layer.

GLM-5.2 geometry goes through the real adapter and wire codec; the bytes that land must
be exactly the ranges the native planner (plan_transfer + bind_regions) would copy.
"""

import importlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from sglang.srt.mem_cache.l1_transfer import wire_codec as wire
from sglang.srt.mem_cache.l1_transfer.transport import api
from sglang.srt.mem_cache.l1_transfer.transport import kv_peer_engine as kpe
from sglang.srt.mem_cache.l1_transfer.transport import layout
from sglang.srt.mem_cache.l1_transfer.transport import sglang_adapter as adapter_mod
from sglang.srt.utils import cuda_vmm_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.target import glm_pool
from sglang.test.kits.slotguard.wire import identity
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


_spec = importlib.util.spec_from_file_location(
    "_kv_peer_read_contract",
    Path(__file__).resolve().parents[1] / "test_kv_peer_read.py",
)
_reader_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reader_tests)

SIZE = 1024  # tokens; 17 pages including the PAD page 0
GRANULARITY = 1 << 16
SENTINEL = 0xEE
SHARD = "replicated-mla"


class _Ops(_reader_tests.FakeOps):
    def alloc_at(self, base, data, device):
        self.allocs[base] = (bytearray(data), device)


class _Vmm:
    def __init__(self):
        self.retained = set()

    def retain(self, address, physical_size):
        self.retained.add(address)
        return address, GRANULARITY

    def release(self, handle):
        self.retained.remove(handle)


class _Driver:
    """The driver calls of the real _Vmm.retain; the handle-properties query fails."""

    class CUresult:
        CUDA_SUCCESS = 0
        CUDA_ERROR_INVALID_VALUE = 1

    CUdeviceptr = int

    def __init__(self):
        self.retained, self.released = [], []

    def cuMemGetAddressRange(self, address):
        return self.CUresult.CUDA_SUCCESS, address, GRANULARITY

    def cuMemRetainAllocationHandle(self, address):
        self.retained.append(address)
        return self.CUresult.CUDA_SUCCESS, address

    def cuMemGetAllocationPropertiesFromHandle(self, handle):
        return self.CUresult.CUDA_ERROR_INVALID_VALUE, None

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return (self.CUresult.CUDA_SUCCESS,)


def _pattern(n, seed):
    # Random bytes per allocation: a misplaced page or region cannot match by period.
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


class KvPeerEngineGLMTest(CustomTestCase):
    def _side(self, ops, bases, fill):
        pool = glm_pool(SIZE, granularity=GRANULARITY, bases=bases)
        owners = (pool._fixed_fabric_owner, pool.index_key_cache._fixed_fabric_owner)
        allocations = [owner.allocation for owner in owners]
        for i, allocation in enumerate(allocations):
            data = (
                bytes([SENTINEL]) * allocation.size
                if fill is None
                else _pattern(allocation.size, (fill, i))
            )
            ops.alloc_at(allocation.base, data, 0)
        vmm = _Vmm()
        engine = kpe.KvPeerEngine(
            0,
            # The fake's FABRIC handle is the owner's base address (the _Vmm handle).
            _export=lambda handle, size, *, device: _reader_tests.describe(
                handle, size
            ),
            _reader=_reader_tests.KvPeerReader(0, _ops=ops),
            _vmm=vmm,
        )
        adapter = adapter_mod.SGLangPoolAdapter(
            pool, engine, tp_rank=0, tp_size=4, shard_id=SHARD
        )
        return engine, adapter, vmm

    def _round_trip(self):
        ops = _Ops()
        src_engine, src_adapter, src_vmm = self._side(ops, (1 << 40, 2 << 40), fill=3)
        dst_engine, dst_adapter, dst_vmm = self._side(
            ops, (3 << 40, 4 << 40), fill=None
        )
        src_pages, dst_pages = [5, 1, 16, 2], [3, 4, 9, 16]

        exports = {
            m.region_id: src_engine.export_memory(m) for m in src_adapter.memories
        }
        source = identity("source")
        regions, bindings = wire.encode_export(source, "lease-1", src_adapter, exports)
        response = NS(
            regions=regions,
            buffer_bindings=bindings,
            request=NS(result=NS(source=source)),
            lease_id="lease-1",
            source_page_ids=tuple(src_pages),
        )
        metadata = wire.decode_export(response, dst_adapter.layout, dst_engine.owner_id)
        plan = dst_engine.plan_pages(
            metadata.layout,
            dst_adapter.layout,
            src_pages,
            dst_pages,
            source_capacity=metadata.capacity,
            target_capacity=dst_adapter.capacity,
        )
        imports = {
            d.region_id: dst_engine.import_memory(d) for d in metadata.descriptors
        }
        read = dst_engine.bind_pages(plan, metadata.bind(imports), dst_adapter.bindings)
        batch = dst_engine.submit_transfer(read)
        # Unmapping or unregistering under a live read would let the owner reuse pages.
        with self.assertRaises(api.BusyError):
            dst_engine.unregister_memory(next(iter(imports.values())))
        with self.assertRaises(api.BusyError):
            src_engine.unregister_memory(src_adapter.memories[0])
        with self.assertRaises(api.BusyError):
            dst_engine.unregister_memory(dst_adapter.memories[0])
        status = dst_engine.get_transfer_status(batch)
        self.assertIs(status.state, api.TransferState.SUCCESS)

        native = layout.bind_regions(
            layout.plan_transfer(
                metadata.layout,
                dst_adapter.layout,
                src_pages,
                dst_pages,
                source_capacity=metadata.capacity,
                target_capacity=dst_adapter.capacity,
            ),
            metadata.bind(imports),
            dst_adapter.bindings,
            api.Opcode.READ,
        )
        self.assertEqual(sum(r.length for r in native), read.requested_bytes)
        self.assertEqual(status.completed_bytes, read.requested_bytes)
        for r in native:
            self.assertEqual(
                ops.read(r.local.address + r.local_offset, r.length),
                ops.read(r.remote.address + r.remote_offset, r.length),
            )
        untouched = [p for p in range(dst_adapter.capacity) if p not in dst_pages]
        for buf in dst_adapter.layout.buffers:
            (span,) = dst_adapter.bindings[buf.name]
            base = span.memory.address + span.region_offset
            for page in (untouched[0], untouched[-1]):
                row = ops.read(base + page * buf.page_bytes, buf.page_bytes)
                self.assertEqual(row, bytes([SENTINEL]) * buf.page_bytes, buf.name)

        dst_engine.release_batch(batch)
        for memory in imports.values():
            dst_engine.unregister_memory(memory)
        for desc in exports.values():
            src_engine.release_export(desc.lease_id)
        for engine, vmm in ((src_engine, src_vmm), (dst_engine, dst_vmm)):
            engine.close()
            self.assertEqual(vmm.retained, set())

    def test_glm_reads_the_native_planner_bytes(self):
        self._round_trip()

    def test_multi_span_buffer_is_rejected_before_any_read(self):
        """A split buffer would be read at the wrong offsets by a single-span PeerBuffer."""
        ops = _Ops()
        src_engine, src_adapter, _ = self._side(ops, (1 << 40, 2 << 40), fill=3)
        dst_engine, dst_adapter, _ = self._side(ops, (3 << 40, 4 << 40), fill=None)
        exports = [src_engine.export_memory(m) for m in src_adapter.memories]
        imports = {d.region_id: dst_engine.import_memory(d) for d in exports}
        plan = dst_engine.plan_pages(
            dst_adapter.layout,
            dst_adapter.layout,
            [1],
            [1],
            source_capacity=dst_adapter.capacity,
            target_capacity=dst_adapter.capacity,
        )
        source = {
            name: tuple(
                layout.RegionSpan(
                    s.buffer_offset,
                    s.length,
                    imports[s.memory.region_id],
                    s.region_offset,
                )
                for s in spans
            )
            for name, spans in src_adapter.bindings.items()
        }
        target = dict(dst_adapter.bindings)
        name = dst_adapter.layout.buffers[0].name
        (whole,) = target[name]
        half = whole.length // 2
        target[name] = (
            layout.RegionSpan(0, half, whole.memory, whole.region_offset),
            layout.RegionSpan(
                half, whole.length - half, whole.memory, whole.region_offset + half
            ),
        )
        with self.assertRaises(ValueError):
            dst_engine.bind_pages(plan, source, target)
        self.assertEqual(ops.launches, 0)

    def test_target_unregistered_after_bind_is_never_written(self):
        """A read must not launch once its local target is unregistered; the pages
        behind that address are no longer this engine's to write."""
        ops = _Ops()
        src_engine, src_adapter, _ = self._side(ops, (1 << 40, 2 << 40), fill=3)
        dst_engine, dst_adapter, _ = self._side(ops, (3 << 40, 4 << 40), fill=None)
        exports = [src_engine.export_memory(m) for m in src_adapter.memories]
        imports = {d.region_id: dst_engine.import_memory(d) for d in exports}
        plan = dst_engine.plan_pages(
            dst_adapter.layout,
            dst_adapter.layout,
            [1],
            [1],
            source_capacity=dst_adapter.capacity,
            target_capacity=dst_adapter.capacity,
        )
        source = {
            name: tuple(
                layout.RegionSpan(
                    s.buffer_offset,
                    s.length,
                    imports[s.memory.region_id],
                    s.region_offset,
                )
                for s in spans
            )
            for name, spans in src_adapter.bindings.items()
        }
        read = dst_engine.bind_pages(plan, source, dst_adapter.bindings)
        dst_engine.unregister_memory(dst_adapter.memories[0])
        with self.assertRaises(ValueError):
            dst_engine.submit_transfer(read)
        self.assertEqual(ops.launches, 0)

    def test_failed_handle_query_releases_the_retained_allocation(self):
        """A driver error after the allocation handle is retained must release it;
        no caller ever receives that handle."""
        driver = _Driver()
        vmm = object.__new__(kpe._Vmm)  # __init__ only imports the CUDA driver
        vmm._drv = driver
        with patch.object(cuda_vmm_utils, "_drv", driver):
            with self.assertRaises(RuntimeError):
                vmm.retain(1 << 40, GRANULARITY)
        self.assertEqual(driver.retained, [1 << 40])
        self.assertEqual(driver.released, driver.retained)


if __name__ == "__main__":
    unittest.main()
