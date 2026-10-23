"""Regression cases for call-local validation reuse: no weaker acceptance."""
import unittest
from dataclasses import replace
import test_contract
from sglang.srt.mem_cache.l1_transfer.transport import BusyError, Opcode, TransferRequest


class SubmitValidationTests(unittest.TestCase):
    setUp = test_contract.ContractTests.setUp
    request = test_contract.ContractTests.request

    def reject(self, requests, error=ValueError):
        before = self.b.download(self.local)
        with self.assertRaises(error):
            self.b.submit_transfer(requests)
        self.assertEqual(self.driver.submits, 0)
        self.assertEqual(self.b.download(self.local), before)

    def test_equal_but_forged_second_handle(self):
        for field, original in [('local', self.local), ('remote', self.remote)]:
            forged = replace(original)
            self.assertEqual(original, forged)
            self.assertIsNot(original, forged)
            self.reject([self.request(), self.request(**{field: forged}, local_offset=40)])

    def test_opcode_switch_and_untyped_opcode(self):
        for op in (Opcode.WRITE, 'READ', None):
            self.reject([self.request(), self.request(opcode=op, local_offset=40)])

    def test_invalid_first_pair_does_not_hit_cache(self):
        self.reject([TransferRequest(None, None, None, 0, 0, 1)])

    def test_every_range_still_checked(self):
        for kwargs in [dict(length=True), dict(local_offset=63), dict(remote_offset=-1),
                       dict(length=0), dict(local_offset=8)]:
            self.reject([self.request(), self.request(**kwargs)])

    def test_overlap_after_pair_switch(self):
        other = self.b.allocate_memory(64)
        self.reject([self.request(), self.request(local=other), self.request(local_offset=8)])

    def test_alias_imports_share_destination_overlap_check(self):
        self.b.unregister_memory(self.remote)
        self.a.release_export(self.desc.lease_id)
        write = self.a.export_memory(self.source, 'write')
        first = self.b.import_memory(write)
        # Model two import identities for the same owner/region. The public
        # descriptor accepts distinct leases; destination overlap uses region ID.
        alias = self.b.import_memory(replace(write, lease_id='f'*32))
        self.reject([self.request(opcode=Opcode.WRITE, remote=first),
                     self.request(opcode=Opcode.WRITE, remote=alias, remote_offset=10)])

    def test_no_reuse_across_batches(self):
        requests = [self.request(), self.request(local_offset=40)]
        batch = self.b.submit_transfer(requests)
        with self.assertRaises(BusyError): self.b.submit_transfer(requests)
        self.driver.state = 1
        self.b.wait(batch, 0)
        self.b.release_batch(batch)
        lease = self.b.export_memory(self.local)
        with self.assertRaises(BusyError): self.b.submit_transfer(requests)
        self.b.release_export(lease.lease_id)
        self.b.unregister_memory(self.remote)
        with self.assertRaises(ValueError): self.b.submit_transfer(requests)
        self.assertEqual(self.driver.submits, 1)
