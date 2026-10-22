"""Private mapping lifetime is independent of imported data leases."""
import unittest
from test_contract import FakeDriver
from sglang.srt.mem_cache.l1_transfer.transport import NvlinkEngine, BusyError, TransferError, Opcode, TransferRequest


class GuardDriver(FakeDriver):
    fail_guard_close = False

    def create_mapping_guard(self, device, peer):
        self.guard_args = (device, peer)
        self.guard = self.create(peer, 1)
        return self.guard

    def close(self, handle):
        if self.fail_guard_close and handle == self.guard:
            raise TransferError("injected guard unmap failure")
        super().close(handle)


class MappingGuardTests(unittest.TestCase):
    def test_opt_in_and_idempotent_close(self):
        d = GuardDriver(); e = NvlinkEngine(1, mapping_peer_device=2, _driver=d)
        self.assertEqual(d.guard_args, (1, 2))
        self.assertFalse(e._memories or e._exports)
        e.close(); e.close()
        self.assertEqual(d.closed, [d.guard])

    def test_default_does_not_allocate_peer_memory(self):
        d = GuardDriver(); e = NvlinkEngine(0, _driver=d)
        e.close(); self.assertFalse(d.memories or d.closed)

    def test_invalid_peer(self):
        for peer in (1, -1, True, '2'):
            with self.assertRaises(ValueError): NvlinkEngine(1, mapping_peer_device=peer, _driver=GuardDriver())

    def test_kv_mapping_still_unmaps_and_fresh_lease_is_required(self):
        d = GuardDriver(); source = NvlinkEngine(0, _driver=d)
        target = NvlinkEngine(1, mapping_peer_device=2, _driver=d)
        local = source.allocate_memory(32)
        for _ in range(2):
            desc = source.export_memory(local)
            imported = target.import_memory(desc); native = target._memory(imported).native
            target.unregister_memory(imported)
            self.assertIn(native, d.closed)
            self.assertNotIn(d.guard, d.closed)
            self.assertFalse(target._memories)
            source.release_export(desc.lease_id)
            with self.assertRaises(ValueError): target.import_memory(desc)
        source.close(); target.close()
        self.assertFalse(d.memories)

    def test_busy_close_keeps_guard_until_drain(self):
        d = GuardDriver(); a = NvlinkEngine(0, _driver=d)
        b = NvlinkEngine(1, mapping_peer_device=2, _driver=d)
        src = a.allocate_memory(8); desc = a.export_memory(src)
        remote = b.import_memory(desc); dst = b.allocate_memory(8)
        batch = b.submit_transfer([TransferRequest(Opcode.READ, dst, remote, 0, 0, 8)])
        with self.assertRaises(BusyError): b.close()
        self.assertNotIn(d.guard, d.closed)
        d.state = 1; b.release_batch(batch); b.close()
        a.release_export(desc.lease_id); a.close()
        self.assertFalse(d.memories)

    def test_cleanup_failure_quarantines_engine(self):
        d = GuardDriver(); e = NvlinkEngine(1, mapping_peer_device=2, _driver=d)
        d.fail_guard_close = True
        with self.assertRaises(TransferError): e.close()
        d.fail_guard_close = False
        for operation in (e.close, lambda:e.allocate_memory(8)):
            with self.assertRaises(BusyError): operation()
        self.assertIn(d.guard, d.memories)
        self.assertFalse(d.closed)


if __name__ == '__main__': unittest.main()
