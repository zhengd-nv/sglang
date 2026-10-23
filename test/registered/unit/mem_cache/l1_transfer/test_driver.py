"""Exercise the real ctypes marshaling path without loading CUDA."""
import ctypes as C
import unittest
from types import SimpleNamespace
from sglang.srt.mem_cache.l1_transfer.transport._driver import Driver


class DriverSubmitTests(unittest.TestCase):
    def test_ranges_and_producer_are_forwarded_without_truncation(self):
        driver = Driver.__new__(Driver)
        captured = []
        def submit(rows, count, stream, has_stream, output):
            captured.append(([(r.source, r.source_offset, r.target, r.target_offset, r.length)
                              for r in rows], count, stream, has_stream))
            C.cast(output, C.POINTER(C.c_void_p))[0] = 1234
            return 0
        driver.lib = SimpleNamespace(nvt_submit=submit)
        ranges = [(0x100000001, 2**33+7, 0x200000003, 2**34+9, 2**32+11),
                  (1, 0, 2, 13, 17)]
        for stream in (None, 0, 0x300000001):
            self.assertEqual(driver.submit(ranges, producer_stream=stream), 1234)
            self.assertEqual(captured[-1], (ranges, 2, stream, int(stream is not None)))
        # Exercise packed and fallback layouts, including maximum uint64 values.
        from unittest.mock import patch
        for packed in (False, True):
            with patch('sglang.srt.mem_cache.l1_transfer.transport._driver._PACKED_RANGES', packed):
                many = ranges * 8 + [(3, 2**64-1, 7, 2**64-1, 2**64-1)]
                driver.submit(many)
                self.assertEqual(captured[-1][0], many)
        ranges[0] = (3, 5, 7, 11, 19)
        driver.submit(ranges)
        self.assertEqual(captured[-1][0], ranges)
        self.assertEqual(captured[0][0][0][1], 2**33+7)

    def test_native_error_is_not_suppressed(self):
        from sglang.srt.mem_cache.l1_transfer.transport.api import TransferError
        driver = Driver.__new__(Driver)
        driver.lib = SimpleNamespace(nvt_submit=lambda *args: 1, nvt_error=lambda: b'failure')
        with self.assertRaisesRegex(TransferError, 'failure'):
            driver.submit([(1, 0, 2, 0, 8)])
