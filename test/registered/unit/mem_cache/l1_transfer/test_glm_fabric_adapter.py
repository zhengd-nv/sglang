"""Actual GLM-5.2 geometry goldens without allocating GPU memory."""

import gc
import unittest
import weakref
from types import SimpleNamespace as NS

from sglang.srt.mem_cache.l1_transfer.transport.layout import plan_transfer
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.target import NvlinkEngine, glm_pool
from sglang.test.kits.slotguard.wire import BorrowDriver
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


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
            lambda p: setattr(p._fixed_fabric_owner.allocation, "reservation", None),
        ):
            pool = glm_pool()
            mutate(pool)
            with self.assertRaises(ValueError):
                self.prepare(pool)

    def test_fp4_indexer_is_rejected_by_name(self):
        """FP4-indexer pools (68 index bytes/token) are rejected by name, not shape."""
        pool = glm_pool()
        pool.use_fp4_indexer = True
        with self.assertRaisesRegex(ValueError, "FP4 indexer"):
            self.prepare(pool)

    def test_register_all_buffers_and_cleanup(self):
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        adapter = SGLangPoolAdapter(
            glm_pool(64), engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
        )
        self.assertEqual(len(adapter.memories), 2)
        adapter.close()
        self.assertFalse(engine._memories)
        engine.close()

    def test_slice_geometry(self):
        for size in (64, 128, 2560000, 2619584):
            with self.subTest(size=size):
                pool = glm_pool(size)
                capacity, layout, buffers = self.prepare(pool)
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
                    root = owner.allocation
                    for i, span in enumerate(owner.buffer_slices):
                        piece = buffers[i + (78 if kind == "dsa_index" else 0)][1][0]
                        self.assertEqual(
                            piece, (0, span.nbytes, root.base, root.size, span.offset)
                        )
                        self.assertLessEqual(span.offset + span.nbytes, root.nbytes)
                    self.assertEqual(span.offset + span.nbytes, root.nbytes)

    def test_two_mib_rounding_goldens(self):
        # Pure arithmetic: no corresponding allocation or GPU result is claimed.
        goldens = {
            64: 8388608,
            128: 12582912,
            2560000: 141379502080,
            2619584: 144669933568,
        }
        for size, expected in goldens.items():
            pool = glm_pool(size)
            self.prepare(pool)
            total = sum(
                o.allocation.size
                for o in (
                    pool._fixed_fabric_owner,
                    pool.index_key_cache._fixed_fabric_owner,
                )
            )
            self.assertEqual(total, expected)

    def test_adapter_snapshots_survive_clear(self):
        pool = glm_pool(64)
        tensor = pool.kv_buffer[-1]
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        adapter = SGLangPoolAdapter(
            pool, engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
        )
        ref = weakref.ref(pool)
        pool._fixed_fabric_owner.tensors.clear()
        pool._fixed_fabric_owner.allocation = None
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

    def test_rejects_invalid_before_registration(self):
        def offset(p, value):
            p._fixed_fabric_owner.buffer_slices[1].offset = value

        for mutate in (
            lambda p: offset(p, 0),
            lambda p: offset(p, 73729),
            lambda p: offset(p, 73984),
            lambda p: setattr(p._fixed_fabric_owner.buffer_slices[0], "nbytes", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocation, "nbytes", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocation, "size", 1),
            lambda p: setattr(p._fixed_fabric_owner, "allocation", None),
            lambda p: p.kv_buffer.__setitem__(0, NS(**vars(p.kv_buffer[0]))),
            lambda p: setattr(p._fixed_fabric_owner, "closed", True),
            lambda p: setattr(p._fixed_fabric_owner.allocation, "reservation", None),
            lambda p: setattr(p._fixed_fabric_owner.allocation, "device_id", 1),
            lambda p: setattr(p._fixed_fabric_owner, "device_id", False),
            lambda p: setattr(p._fixed_fabric_owner.allocation.reservation, "size", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocation.reservation, "base", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocation, "base", 1),
            lambda p: setattr(p._fixed_fabric_owner.allocation, "granularity", True),
            lambda p: setattr(p.kv_buffer[1], "nbytes", 1),
            lambda p: setattr(p.kv_buffer[1], "dtype", "torch.float16"),
            lambda p: setattr(p.kv_buffer[1], "is_contiguous", lambda: False),
            lambda p: setattr(p.kv_buffer[1], "data_ptr", lambda: 1),
            lambda p: p._fixed_fabric_owner.buffer_slices.pop(),
        ):
            with self.subTest(mutate=mutate):
                pool = glm_pool(64)
                mutate(pool)
                driver = BorrowDriver()
                engine = NvlinkEngine(0, _driver=driver)
                with self.assertRaises(ValueError):
                    SGLangPoolAdapter(
                        pool, engine, tp_rank=0, tp_size=4, shard_id="replicated-mla"
                    )
                self.assertEqual(driver.next, 0)
                engine.close()

    def test_cross_owner_overlap_rejected_pre_registration(self):
        pool = glm_pool(64)
        mla, index = pool._fixed_fabric_owner, pool.index_key_cache._fixed_fabric_owner
        delta = mla.allocation.base - index.allocation.base
        root = index.allocation
        root.base += delta
        root.reservation.base += delta
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

    def test_registration_failure_rolls_back_first_root(self):
        class FailingDriver(BorrowDriver):
            def register_vmm(self, device, address, physical):
                if self.next:
                    raise RuntimeError("second root registration failed")
                return super().register_vmm(device, address, physical)

        driver = FailingDriver()
        engine = NvlinkEngine(0, _driver=driver)
        with self.assertRaisesRegex(RuntimeError, "second root"):
            SGLangPoolAdapter(
                glm_pool(64),
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
