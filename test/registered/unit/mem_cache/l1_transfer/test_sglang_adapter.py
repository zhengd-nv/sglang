import gc
import unittest
import weakref
from types import SimpleNamespace as NS

from sglang.srt.mem_cache.l1_transfer.transport.api import BusyError
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.target import NvlinkEngine
from sglang.test.kits.slotguard.wire import BorrowDriver, FakeDriver
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class AdapterTests(CustomTestCase):
    def pool(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool = object.__new__(MHATokenToKVPool)
        pool.use_hnd = False
        pool.kv_cache_layout = "nhd"
        pool.size = 16
        pool.page_size = 4
        pool.store_dtype = NS(itemsize=2)
        pool.start_layer = 0
        pool._kv_buffer_descs = [
            NS(name="k0", shape=(20, 2, 4), row_bytes=16, tokens_per_row=1)
        ]
        pool.get_contiguous_buf_infos = lambda: ([1024], [320], [64])
        pool._post_capture_owner = NS(
            _final_num_tokens=16,
            page_size=4,
            store_dtype=pool.store_dtype,
            device_id=0,
            _specs=[NS(desc=pool._kv_buffer_descs[0], offset=0)],
            _arena=NS(
                base=1024,
                reserved=1024,
                granularity=64,
                _closed=False,
                _allocation=NS(
                    _closed=False, _mappings=[(1024, 128, 1), (1152, 192, 2)]
                ),
            ),
        )
        return pool

    def assert_invalid(self, pool):
        driver = BorrowDriver()
        engine = NvlinkEngine(0, _driver=driver)
        with self.assertRaises(ValueError):
            SGLangPoolAdapter(pool, engine)
        self.assertFalse(engine._memories)
        self.assertEqual(
            driver.next, 0, "metadata must be validated before registration"
        )

    def test_exact_metadata_arity(self):
        for metadata in (
            ([], [320], [64]),
            ([1024], [], [64]),
            ([1024], [320], []),
            ([1024, 2048], [320], [64]),
            ([1024], [320, 320], [64]),
            ([1024], [320], [64, 64]),
        ):
            with self.subTest(metadata=metadata):
                pool = self.pool()
                pool.get_contiguous_buf_infos = lambda metadata=metadata: metadata
                self.assert_invalid(pool)
        pool = self.pool()
        pool._kv_buffer_descs = []
        self.assert_invalid(pool)

    def test_positive_page_aligned_capacity(self):
        for field, values in (
            ("size", (0, -4, 15, True, 16.0)),
            ("page_size", (0, -4, True, 4.0)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    pool = self.pool()
                    setattr(pool, field, value)
                    self.assert_invalid(pool)

    def test_finalized_owner_contract(self):
        for field, value in (
            ("_final_num_tokens", 12),
            ("_final_num_tokens", True),
            ("page_size", 8),
            ("store_dtype", "int8"),
            ("device_id", 1),
            ("_specs", []),
        ):
            with self.subTest(field=field):
                pool = self.pool()
                setattr(pool._post_capture_owner, field, value)
                self.assert_invalid(pool)

    def test_exact_buffer_extent_and_page_bytes(self):
        for size, item in (
            (256, 64),
            (384, 64),
            (320, 32),
            (320, 0),
            (True, 64),
            (320, True),
        ):
            with self.subTest(size=size, item=item):
                pool = self.pool()
                pool.get_contiguous_buf_infos = lambda size=size, item=item: (
                    [1024],
                    [size],
                    [item],
                )
                self.assert_invalid(pool)

    def test_descriptor_layout_must_match_nhd(self):
        for field, value in (
            ("tokens_per_row", 4),
            ("row_bytes", 8),
            ("shape", (20, 8)),
            ("shape", (16, 2, 4)),
            ("shape", (20, 0, 4)),
        ):
            with self.subTest(field=field):
                pool = self.pool()
                setattr(pool._kv_buffer_descs[0], field, value)
                self.assert_invalid(pool)

    def test_pointer_must_match_owned_buffer(self):
        pool = self.pool()
        pool.get_contiguous_buf_infos = lambda: ([1088], [320], [64])
        pool._post_capture_owner._arena._allocation._mappings = [(1024, 512, 1)]
        self.assert_invalid(pool)

    def test_owner_descriptor_must_match_pool(self):
        pool = self.pool()
        pool._post_capture_owner._specs[0].desc = NS(
            name="k0", shape=(20, 4, 2), row_bytes=16, tokens_per_row=1
        )
        self.assert_invalid(pool)

    def test_reserved_tail_and_mapping_padding_are_not_transfer_capacity(self):
        pool = self.pool()
        pool._kv_buffer_descs[0].shape = (64, 2, 4)
        pool._post_capture_owner._arena._allocation._mappings = [(1024, 384, 1)]
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        adapter = SGLangPoolAdapter(pool, engine)
        self.assertEqual(adapter.capacity, 5)
        self.assertEqual(adapter.memories[0].size, 384)
        self.assertEqual(adapter.bindings["k0"][0].length, 320)
        from sglang.srt.mem_cache.l1_transfer.transport.layout import plan_transfer

        with self.assertRaises(ValueError):
            plan_transfer(
                adapter.layout,
                adapter.layout,
                [5],
                [5],
                source_capacity=adapter.capacity,
                target_capacity=adapter.capacity,
            )
        adapter.close()
        engine.close()

    def test_invalid_physical_mappings(self):
        for mappings in (
            [(1024, 320, 1), (1088, 64, 2)],
            [(1024, 320, 1), (1024, 320, 2)],
            [(1024, 128, 1), (1216, 128, 2)],
            [(1024, 0, 1)],
            [(1025, 320, 1)],
            [(1024, 319, 1)],
            [(1024, -64, 1)],
            [(1024, 320, 1), (1984, 128, 2)],
        ):
            with self.subTest(mappings=mappings):
                pool = self.pool()
                pool._post_capture_owner._arena._allocation._mappings = mappings
                self.assert_invalid(pool)

    def test_closed_owner_is_rejected(self):
        for target in ("_arena", "_allocation"):
            pool = self.pool()
            arena = pool._post_capture_owner._arena
            (arena if target == "_arena" else arena._allocation)._closed = True
            self.assert_invalid(pool)

    def test_registration_failure_releases_previous_alias(self):
        class FailingDriver(BorrowDriver):
            def register_vmm(self, device, address, physical):
                if self.next:
                    raise RuntimeError("second registration failed")
                return super().register_vmm(device, address, physical)

        driver = FailingDriver()
        engine = NvlinkEngine(0, _driver=driver)
        with self.assertRaisesRegex(RuntimeError, "second registration"):
            SGLangPoolAdapter(self.pool(), engine)
        self.assertFalse(engine._memories)
        self.assertFalse(driver.memories)
        self.assertEqual(driver.closed, [1])

    def test_failed_registration_cleanup_keeps_pool_alive(self):
        class FailingDriver(BorrowDriver):
            def info(self, native):
                raise RuntimeError("metadata unavailable")

            def close(self, native):
                raise RuntimeError("alias unmap failed")

        driver = FailingDriver()
        engine = NvlinkEngine(0, _driver=driver)
        pool = self.pool()
        ref = weakref.ref(pool)
        with self.assertRaisesRegex(RuntimeError, "alias unmap failed"):
            SGLangPoolAdapter(pool, engine)
        del pool
        gc.collect()
        self.assertIsNotNone(ref())
        driver.close = lambda native: FakeDriver.close(driver, native)
        engine.close()
        gc.collect()
        self.assertIsNone(ref())
        self.assertFalse(driver.memories)

    def test_split_and_owner_lifetime(self):
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        pool = self.pool()
        adapter = SGLangPoolAdapter(pool, engine)
        self.assertEqual([s.length for s in adapter.bindings["k0"]], [128, 192])
        h = adapter.memories[0]
        desc = engine.export_memory(h)
        with self.assertRaises(BusyError):
            engine.unregister_memory(h)
        engine.release_export(desc.lease_id)
        ref = weakref.ref(pool)
        del pool
        gc.collect()
        self.assertIsNotNone(ref())
        adapter.close()
        self.assertFalse(engine._memories)
        del adapter
        gc.collect()
        self.assertIsNone(ref())
        engine.close()

    def test_last_real_page_is_inside_capacity(self):
        from sglang.srt.mem_cache.l1_transfer.transport.layout import plan_transfer

        engine = NvlinkEngine(0, _driver=BorrowDriver())
        adapter = SGLangPoolAdapter(self.pool(), engine)
        self.assertEqual(adapter.capacity, 5)
        plan = plan_transfer(
            adapter.layout,
            adapter.layout,
            [4],
            [4],
            source_capacity=adapter.capacity,
            target_capacity=adapter.capacity,
        )
        self.assertEqual(plan[0].target_offset + plan[0].length, 320)
        adapter.close()
        engine.close()

    def test_partial_registration_cleanup(self):
        pool = self.pool()
        pool._post_capture_owner._arena._allocation._mappings.pop()
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        with self.assertRaises(ValueError):
            SGLangPoolAdapter(pool, engine)
        self.assertFalse(engine._memories)

    def test_reject_unsupported(self):
        engine = NvlinkEngine(0, _driver=BorrowDriver())
        for attr, value in [
            ("use_hnd", True),
            ("_post_capture_owner", None),
            ("kv_cache_layout", "hnd"),
        ]:
            pool = self.pool()
            setattr(pool, attr, value)
            with self.assertRaises(ValueError):
                SGLangPoolAdapter(pool, engine)
        with self.assertRaises(ValueError):
            engine.register_vmm(100, 128, owner=None)


if __name__ == "__main__":
    unittest.main()
