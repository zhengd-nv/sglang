"""CPU-only contract tests with failure injection, not a CUDA correctness substitute."""
import base64
import unittest
from dataclasses import replace
from sglang.srt.mem_cache.l1_transfer.transport import (NvlinkEngine, Opcode, TransferRequest, RegionDescriptor,
                             TransferState, BusyError, TransferError)


class FakeDriver:
    def __init__(self):
        self.memories = {}; self.next = 0; self.submits = 0
        self.state = 0; self.closed = []; self.raise_submit = False; self.raise_poll = False

    def create(self, device, size):
        self.next += 1; self.memories[self.next] = bytearray(size)
        return self.next

    def info(self, h):
        size = len(self.memories[h]); return h * 4096, ((size + 4095)//4096)*4096, 4096

    def export(self, h):
        return base64.b64encode(h.to_bytes(64, "little")).decode()

    def import_region(self, device, desc):
        h = int.from_bytes(base64.b64decode(desc.transport_handle), "little")
        self.next += 1; self.memories[self.next] = self.memories[h]; return self.next

    def close(self, h):
        self.closed.append(h); del self.memories[h]

    def upload(self, h, offset, data):
        self.memories[h][offset:offset+len(data)] = data

    def download(self, h, offset, length):
        return bytes(self.memories[h][offset:offset+length])

    def submit(self, ranges):
        self.submits += 1
        if self.raise_submit:
            raise TransferError("unknown native submission error")
        for s, so, t, to, n in ranges:
            self.memories[t][to:to+n] = self.memories[s][so:so+n]
        return self.submits

    def poll(self, h):
        if self.raise_poll:
            raise TransferError("lost native completion state")
        return self.state, "injected" if self.state in (2, 3) else ""

    def release_batch(self, h):
        pass


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeDriver()
        self.a = NvlinkEngine(0, _driver=self.driver)
        self.b = NvlinkEngine(1, _driver=self.driver)
        self.source = self.a.allocate_memory(64)
        self.a.upload(self.source, bytes(range(64)))
        self.desc = self.a.export_memory(self.source)
        self.remote = self.b.import_memory(self.desc)
        self.local = self.b.allocate_memory(64)

    def request(self, **kwargs):
        return replace(TransferRequest(Opcode.READ, self.local, self.remote, 4, 8, 12), **kwargs)

    def test_fragment_read_and_sentinel(self):
        self.b.upload(self.local, b"\xa5"*64)
        handle = self.b.submit_transfer([self.request(), self.request(local_offset=40, remote_offset=2, length=3)])
        self.driver.state = 1
        status = self.b.wait(handle, 0)
        self.assertEqual(status.completed_bytes, 15)
        self.b.release_batch(handle)
        result = bytearray(b"\xa5"*64); result[4:16] = bytes(range(8,20)); result[40:43] = bytes(range(2,5))
        self.assertEqual(self.b.download(self.local), result)

    def test_bad_second_range_is_atomic_rejection(self):
        before = self.b.download(self.local)
        with self.assertRaises(ValueError):
            self.b.submit_transfer([self.request(), self.request(remote_offset=63)])
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual(self.b.download(self.local), before)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            self.b.submit_transfer([self.request(), self.request(local_offset=8)])

    def test_invalid_offsets_lengths_and_opcode(self):
        for req in (self.request(length=0), self.request(length=-1), self.request(local_offset=-1),
                    self.request(length=True), self.request(length=2**64), self.request(opcode="READ")):
            with self.subTest(req=req), self.assertRaises(ValueError):
                self.b.submit_transfer([req])

    def test_write_requires_write_export(self):
        with self.assertRaises(ValueError):
            self.b.submit_transfer([self.request(opcode=Opcode.WRITE)])

    def test_remote_write(self):
        self.b.unregister_memory(self.remote); self.a.release_export(self.desc.lease_id)
        desc = self.a.export_memory(self.source, "write")
        remote = self.b.import_memory(desc)
        self.b.upload(self.local, b"Q"*64)
        h = self.b.submit_transfer([self.request(opcode=Opcode.WRITE, remote=remote)])
        self.driver.state = 1; self.b.wait(h); self.b.release_batch(h)
        self.b.unregister_memory(remote); self.a.release_export(desc.lease_id)
        expected = bytearray(range(64)); expected[8:20] = b"Q"*12
        self.assertEqual(self.a.download(self.source), expected)

    def test_exports_freeze_local_access_and_close(self):
        for op in (lambda: self.a.upload(self.source,b"X"), lambda: self.a.download(self.source),
                   lambda: self.a.unregister_memory(self.source), self.a.close,
                   lambda: self.a.export_memory(self.source,"write")):
            with self.assertRaises(BusyError): op()

    def test_timeout_does_not_release(self):
        h = self.b.submit_transfer([self.request()])
        with self.assertRaises(TimeoutError): self.b.wait(h,0)
        for op in (lambda: self.b.unregister_memory(self.local), lambda: self.b.download(self.local),
                   lambda: self.b.release_batch(h), self.b.close):
            with self.assertRaises(BusyError): op()
        self.driver.state = 1; self.b.wait(h); self.b.release_batch(h)
        self.b.unregister_memory(self.local)

    def test_memory_stays_owned_until_batch_released(self):
        h = self.b.submit_transfer([self.request()]); self.driver.state = 1; self.b.wait(h)
        with self.assertRaises(BusyError): self.b.unregister_memory(self.remote)
        self.b.release_batch(h); self.b.unregister_memory(self.remote)

    def test_failed_drained_cleanup(self):
        h = self.b.submit_transfer([self.request()]); self.driver.state = 2
        result = self.b.wait(h)
        self.assertTrue(result.drained); self.assertIsNone(result.completed_bytes)
        self.b.release_batch(h); self.b.unregister_memory(self.remote)

    def test_failed_uncertain_quarantines(self):
        h = self.b.submit_transfer([self.request()]); self.driver.state = 3
        result = self.b.wait(h); self.assertFalse(result.drained)
        for op in (lambda: self.b.release_batch(h), lambda: self.b.unregister_memory(self.local), self.b.close):
            with self.assertRaises(BusyError): op()

    def test_native_submit_exception_quarantines(self):
        self.driver.raise_submit = True
        with self.assertRaises(TransferError): self.b.submit_transfer([self.request()])
        with self.assertRaises(BusyError): self.b.unregister_memory(self.local)

    def test_native_poll_exception_quarantines(self):
        h = self.b.submit_transfer([self.request()]); self.driver.raise_poll = True
        self.assertEqual(self.b.wait(h).state, TransferState.FAILED_UNCERTAIN)
        with self.assertRaises(BusyError): self.b.close()

    def test_native_wait_timeout_preserves_ownership(self):
        self.driver.wait = lambda handle, timeout: None
        h = self.b.submit_transfer([self.request()])
        with self.assertRaises(TimeoutError): self.b.wait(h, 0)
        with self.assertRaises(BusyError): self.b.release_batch(h)
        with self.assertRaises(BusyError): self.b.unregister_memory(self.local)
        self.driver.state = 1
        self.assertEqual(self.b.wait(h, 1).state, TransferState.SUCCESS)
        self.b.release_batch(h)

    def test_native_wait_exception_quarantines(self):
        def fail(handle, timeout): raise RuntimeError("wait failed")
        self.driver.wait = fail
        h = self.b.submit_transfer([self.request()])
        status = self.b.wait(h, 1)
        self.assertEqual(status.state, TransferState.FAILED_UNCERTAIN)
        self.assertEqual(status.error, "wait failed")
        with self.assertRaises(BusyError): self.b.unregister_memory(self.local)
        with self.assertRaises(BusyError): self.b.release_batch(h)

    def test_native_wait_completion_and_terminal_idempotence(self):
        calls = []
        def complete(handle, timeout):
            calls.append(timeout); self.driver.state = 1
        self.driver.wait = complete
        h = self.b.submit_transfer([self.request()])
        self.assertEqual(self.b.wait(h).state, TransferState.SUCCESS)
        self.assertEqual(self.b.wait(h, 0).state, TransferState.SUCCESS)
        self.assertEqual(calls, [None])
        self.b.release_batch(h)

    def test_foreign_and_forged_handles(self):
        for req in (self.request(local=self.source), self.request(local=replace(self.local))):
            with self.assertRaises(ValueError): self.b.submit_transfer([req])

    def test_descriptor_roundtrip_and_validation(self):
        self.assertEqual(RegionDescriptor.from_dict(self.desc.to_dict()), self.desc)
        for change in ({"version":2}, {"version":True}, {"transport":"tcp"}, {"size":-1},
                       {"physical_size":3}, {"transport_handle":"broken"}, {"access":"rw"}, {"owner_id":"bad"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.b.import_memory(replace(self.desc, **change))

    def test_duplicate_import_and_stale_ack_rejected(self):
        with self.assertRaises(ValueError): self.b.import_memory(self.desc)
        self.b.unregister_memory(self.remote)
        with self.assertRaises(ValueError): self.b.import_memory(self.desc)
        self.a.release_export(self.desc.lease_id)
        with self.assertRaises(ValueError): self.a.release_export(self.desc.lease_id)
        self.a.unregister_memory(self.source)

    def test_multiple_read_exports(self):
        other = self.a.export_memory(self.source)
        self.a.release_export(self.desc.lease_id)
        with self.assertRaises(BusyError): self.a.unregister_memory(self.source)
        self.a.release_export(other.lease_id); self.a.unregister_memory(self.source)

    def test_empty_batch_and_bad_timeout(self):
        with self.assertRaises(ValueError): self.b.submit_transfer([])
        h=self.b.submit_transfer([self.request()])
        for timeout in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError): self.b.wait(h,timeout)

    def test_common_descriptor_does_not_require_cuda_metadata(self):
        other = replace(self.desc, transport="nixl-adapter", transport_handle="opaque-peer-metadata",
                        physical_size=self.desc.size, granularity=1)
        self.assertEqual(RegionDescriptor.from_dict(other.to_dict()), other)
        with self.assertRaises(ValueError): self.b.import_memory(other)


if __name__ == "__main__":
    unittest.main()
