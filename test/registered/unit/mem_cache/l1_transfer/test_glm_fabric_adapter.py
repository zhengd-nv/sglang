"""Actual GLM-5.2 geometry goldens without allocating GPU memory."""

import gc
import unittest
import weakref
from math import prod
from types import SimpleNamespace as NS

from test_layout import plan_transfer
from test_protocol import register_cpu_ci
from test_sglang_adapter import BorrowDriver, NvlinkEngine, SGLangPoolAdapter

from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def glm_pool(
    size=2619584,
    *,
    coalesced=False,
    granularity=2 << 20,
    bases=(1 << 40, 2 << 40),
):
    pool = type("DSATokenToKVPool", (), {})()
    pool.__dict__.update(
        fixed_fabric_pool=True,
        size=size,
        page_size=64,
        layer_num=78,
        start_layer=0,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dim=576,
        dtype="torch.float8_e4m3fn",
        store_dtype="torch.uint8",
        dsa_kv_cache_store_fp8=False,
        dcp_replicated=False,
        index_head_dim=128,
        index_kpool=1,
        index_kpool_compress=False,
        tail_extra_slots=0,
        index_page_size=64,
        index_buf_size=size,
        skip_topk_layers=[False] * 78,
    )

    def buffers(shape, base):
        tensors, allocations = [], []
        extent = prod(shape)
        gran = granularity
        physical = (extent + gran - 1) // gran * gran
        stride = (extent + 255) // 256 * 256
        root_extent = 77 * stride + extent
        root_size = (root_extent + gran - 1) // gran * gran
        slices = []
        for i in range(78):
            ptr = base + i * (stride if coalesced else physical)
            slices.append(
                NS(
                    allocation_index=0 if coalesced else i,
                    offset=i * stride if coalesced else 0,
                    nbytes=extent,
                )
            )
            tensors.append(
                NS(
                    device="cuda:0",
                    shape=shape,
                    dtype="torch.uint8",
                    nbytes=extent,
                    is_contiguous=lambda: True,
                    data_ptr=lambda ptr=ptr: ptr,
                )
            )
            if coalesced and i:
                continue
            allocation_base = base if coalesced else ptr
            allocation_size = root_size if coalesced else physical
            allocations.append(
                NS(
                    base=allocation_base,
                    size=allocation_size,
                    nbytes=root_extent if coalesced else extent,
                    granularity=gran,
                    device_id=0,
                    reservation=NS(
                        _closed=False,
                        base=allocation_base,
                        size=allocation_size,
                        _mappings=[(allocation_base, allocation_size, i + 1)],
                    ),
                )
            )
        return NS(
            closed=False,
            device_id=0,
            dtype="torch.uint8",
            tensors=tensors,
            allocations=allocations,
            coalesced=coalesced,
            buffer_slices=slices,
        )

    def infos(tensors, item):
        return (
            [t.data_ptr() for t in tensors],
            [t.nbytes for t in tensors],
            [item] * 78,
        )

    pool._fixed_fabric_owner = buffers((size + 64, 1, 576), bases[0])
    pool.kv_buffer = pool._fixed_fabric_owner.tensors.copy()
    index_owner = buffers((size // 64 + 1, 8448), bases[1])
    pool.index_key_cache = NS(_fixed_fabric_owner=index_owner)
    pool.index_k_with_scale_buffer = index_owner.tensors.copy()
    pool.get_contiguous_buf_infos = lambda: infos(pool.kv_buffer, 36864)
    pool.get_state_buf_infos = lambda: infos(pool.index_k_with_scale_buffer, 8448)
    return pool


class GLMAdapterTests(CustomTestCase):
    def prepare(self, pool=None, rank=0, tp=4, shard="replicated-mla"):
        return SGLangPoolAdapter._prepare_glm(pool or glm_pool(), 0, rank, tp, shard)

    def test_full_glm_golden(self):
        capacity, layout, buffers = self.prepare()
        self.assertEqual(capacity, 40932)
        self.assertEqual(len(layout.buffers), 156)
        self.assertEqual(len(buffers), 156)
        self.assertEqual(sum(b.page_bytes for b in layout.buffers), 3534336)
        self.assertEqual(
            [b.name for b in layout.buffers[:2]], ["layer0.mla", "layer1.mla"]
        )
        self.assertEqual(layout.buffers[-1].name, "layer77.dsa_index")
        self.assertEqual(layout.buffers[0].page_bytes, 36864)
        self.assertEqual(layout.buffers[-1].page_bytes, 8448)
        plan = plan_transfer(
            layout,
            layout,
            [capacity - 1],
            [capacity - 1],
            source_capacity=capacity,
            target_capacity=capacity,
        )
        self.assertEqual(len(plan), 156)
        self.assertEqual(plan[0].target_offset + plan[0].length, (2619584 + 64) * 576)
        with self.assertRaises(ValueError):
            plan_transfer(
                layout,
                layout,
                [capacity],
                [1],
                source_capacity=capacity,
                target_capacity=capacity,
            )

    def test_matching_rank_and_shard_mandatory(self):
        for rank, tp, shard in (
            (None, 4, "a"),
            (4, 4, "a"),
            (True, 4, "a"),
            (0, None, "a"),
            (0, 4, None),
        ):
            with (
                self.subTest(rank=rank, tp=tp, shard=shard),
                self.assertRaises(ValueError),
            ):
                self.prepare(rank=rank, tp=tp, shard=shard)
        c, a, _ = self.prepare(rank=0)
        for b in (self.prepare(rank=1)[1], self.prepare(shard="other")[1]):
            with self.assertRaises(ValueError):
                plan_transfer(a, b, [1], [1], source_capacity=c, target_capacity=c)

    def test_all_index_layers_and_exact_backing_required(self):
        for mutate in (
            lambda p: p.index_k_with_scale_buffer.pop(),
            lambda p: p.skip_topk_layers.__setitem__(77, True),
            lambda p: setattr(p, "kv_cache_dim", 656),
            lambda p: setattr(p, "dcp_replicated", True),
            lambda p: setattr(p._fixed_fabric_owner, "closed", True),
            lambda p: setattr(p.index_key_cache._fixed_fabric_owner, "device_id", 1),
            lambda p: setattr(p.index_k_with_scale_buffer[0], "shape", (40932, 8192)),
            lambda p: p._fixed_fabric_owner.allocations[
                0
            ].reservation._mappings.clear(),
        ):
            pool = glm_pool()
            mutate(pool)
            with self.assertRaises(ValueError):
                self.prepare(pool)

    def test_register_all_buffers_and_cleanup(self):
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        adapter = SGLangPoolAdapter(
            glm_pool(64), engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
        )
        self.assertEqual(len(adapter.memories), 156)
        adapter.close()
        self.assertFalse(engine._memories)
        engine.close()

    def test_coalesced_geometry_and_logical_identity(self):
        for size in (64, 128, 2560000, 2619584):
            with self.subTest(size=size):
                old = self.prepare(glm_pool(size))
                pool = glm_pool(size, coalesced=True)
                capacity, layout, buffers = self.prepare(pool)
                self.assertEqual((capacity, layout), old[:2])
                self.assertEqual(len(buffers), 156)
                plan = plan_transfer(
                    layout,
                    layout,
                    [capacity - 1],
                    [capacity - 1],
                    source_capacity=capacity,
                    target_capacity=capacity,
                )
                for piece, transfer in zip(buffers, plan):
                    self.assertEqual(
                        transfer.source_offset + transfer.length, piece[1][0][1]
                    )
                with self.assertRaises(ValueError):
                    plan_transfer(
                        layout,
                        layout,
                        [capacity],
                        [1],
                        source_capacity=capacity,
                        target_capacity=capacity,
                    )
                for kind, owner in (
                    ("mla", pool._fixed_fabric_owner),
                    ("dsa_index", pool.index_key_cache._fixed_fabric_owner),
                ):
                    root = owner.allocations[0]
                    for i, span in enumerate(owner.buffer_slices):
                        piece = buffers[i + (78 if kind == "dsa_index" else 0)][1][0]
                        self.assertEqual(
                            piece, (0, span.nbytes, root.base, root.size, span.offset)
                        )
                        self.assertLessEqual(span.offset + span.nbytes, root.nbytes)
                    self.assertEqual(span.offset + span.nbytes, root.nbytes)
                if size <= 128:
                    engine = NvlinkEngine(0, _driver=BorrowDriver())
                    adapter = SGLangPoolAdapter(
                        pool, engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
                    )
                    self.assertEqual(len(adapter.memories), 2)
                    adapter.close()
                    engine.close()

    def test_two_mib_rounding_goldens(self):
        # Pure arithmetic: no corresponding allocation or GPU result is claimed.
        goldens = {
            64: (327155712, 8388608),
            128: (327155712, 12582912),
            2560000: (141658423296, 141379502080),
            2619584: (144766402560, 144669933568),
        }
        for size, expected in goldens.items():
            totals = []
            for packed in (False, True):
                pool = glm_pool(size, coalesced=packed)
                self.prepare(pool)
                totals.append(
                    sum(
                        a.size
                        for o in (
                            pool._fixed_fabric_owner,
                            pool.index_key_cache._fixed_fabric_owner,
                        )
                        for a in o.allocations
                    )
                )
            self.assertEqual(tuple(totals), expected)
        self.assertEqual((goldens[2560000][0] - goldens[2560000][1]) // (1 << 20), 266)

    def test_missing_mode_is_legacy_and_adapter_snapshots_survive_clear(self):
        pool = glm_pool(64)
        del pool._fixed_fabric_owner.coalesced
        del pool.index_key_cache._fixed_fabric_owner.coalesced
        self.prepare(pool)
        pool = glm_pool(64, coalesced=True)
        tensor = pool.kv_buffer[-1]
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        adapter = SGLangPoolAdapter(
            pool, engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
        )
        ref = weakref.ref(pool)
        pool._fixed_fabric_owner.tensors.clear()
        pool._fixed_fabric_owner.allocations.clear()
        pool.kv_buffer.clear()
        del pool
        gc.collect()
        self.assertIsNotNone(ref())
        snapshots = engine._memory(adapter.memories[0]).owner
        self.assertIs(snapshots[1][-1], tensor)
        del snapshots
        adapter.close()
        del adapter
        gc.collect()
        self.assertIsNone(ref())
        engine.close()

    def test_coalesced_rejects_invalid_before_registration(self):
        def offset(p, value):
            p._fixed_fabric_owner.buffer_slices[1].offset = value

        for mutate in (
            lambda p: setattr(p._fixed_fabric_owner, "coalesced", 1),
            lambda p: setattr(
                p.index_key_cache._fixed_fabric_owner, "coalesced", False
            ),
            lambda p: offset(p, 0),
            lambda p: offset(p, 73729),
            lambda p: offset(p, 73984),
            lambda p: setattr(
                p._fixed_fabric_owner.buffer_slices[0], "allocation_index", 1
            ),
            lambda p: setattr(p._fixed_fabric_owner.buffer_slices[0], "nbytes", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocations[0], "nbytes", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocations[0], "size", 1),
            lambda p: p._fixed_fabric_owner.allocations.append(
                p._fixed_fabric_owner.allocations[0]
            ),
            lambda p: p.kv_buffer.__setitem__(0, NS(**vars(p.kv_buffer[0]))),
            lambda p: setattr(p._fixed_fabric_owner, "closed", True),
            lambda p: p._fixed_fabric_owner.allocations[
                0
            ].reservation._mappings.clear(),
            lambda p: setattr(p._fixed_fabric_owner.allocations[0], "device_id", 1),
            lambda p: setattr(p._fixed_fabric_owner, "device_id", False),
            lambda p: setattr(
                p._fixed_fabric_owner.allocations[0].reservation, "_closed", True
            ),
            lambda p: setattr(
                p._fixed_fabric_owner.allocations[0].reservation, "size", 1
            ),
            lambda p: setattr(
                p._fixed_fabric_owner.allocations[0].reservation, "base", 1
            ),
            lambda p: setattr(p._fixed_fabric_owner.allocations[0], "base", 1),
            lambda p: setattr(
                p._fixed_fabric_owner.allocations[0], "granularity", True
            ),
            lambda p: setattr(p.kv_buffer[1], "nbytes", 1),
            lambda p: setattr(p.kv_buffer[1], "dtype", "torch.float16"),
            lambda p: setattr(p.kv_buffer[1], "is_contiguous", lambda: False),
            lambda p: setattr(p.kv_buffer[1], "data_ptr", lambda: 1),
            lambda p: p._fixed_fabric_owner.buffer_slices.pop(),
        ):
            with self.subTest(mutate=mutate):
                pool = glm_pool(64, coalesced=True)
                mutate(pool)
                driver = BorrowDriver()
                engine = NvlinkEngine(0, _driver=driver)
                with self.assertRaises(ValueError):
                    SGLangPoolAdapter(
                        pool, engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
                    )
                self.assertEqual(driver.next, 0)
                engine.close()

    def test_coalesced_cross_owner_overlap_rejected_pre_registration(self):
        pool = glm_pool(64, coalesced=True)
        mla, index = pool._fixed_fabric_owner, pool.index_key_cache._fixed_fabric_owner
        delta = mla.allocations[0].base - index.allocations[0].base
        root = index.allocations[0]
        root.base += delta
        root.reservation.base += delta
        root.reservation._mappings = [(root.base, root.size, 1)]
        for t in index.tensors:
            ptr = t.data_ptr() + delta
            t.data_ptr = lambda ptr=ptr: ptr
        driver = BorrowDriver()
        engine = NvlinkEngine(0, _driver=driver)
        with self.assertRaisesRegex(ValueError, "overlapping GLM physical"):
            SGLangPoolAdapter(
                pool, engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
            )
        self.assertEqual(driver.next, 0)
        engine.close()

    def test_coalesced_registration_failure_rolls_back_first_root(self):
        class FailingDriver(BorrowDriver):
            def register_vmm(self, device, address, physical):
                if self.next:
                    raise RuntimeError("second root registration failed")
                return super().register_vmm(device, address, physical)

        driver = FailingDriver()
        engine = NvlinkEngine(0, _driver=driver)
        with self.assertRaisesRegex(RuntimeError, "second root"):
            SGLangPoolAdapter(
                glm_pool(64, coalesced=True),
                engine,
                tp_rank=0,
                tp_size=4,
                shard_id="replicated-mla",
            )
        self.assertEqual(driver.closed, [1])
        self.assertFalse(driver.memories)
        self.assertFalse(engine._memories)
        engine.close()


if __name__ == "__main__":
    unittest.main()
