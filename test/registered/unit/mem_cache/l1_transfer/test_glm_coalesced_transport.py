"""Full-import CPU fake oracles. No CUDA, RDMA or serving performance proof."""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS

from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer import wire_codec as codec
from sglang.srt.mem_cache.l1_transfer.config import layout_digest as digest
from sglang.srt.mem_cache.l1_transfer.transport.api import BusyError, Opcode
from sglang.srt.mem_cache.l1_transfer.transport.layout import (
    bind_regions,
    plan_transfer,
)
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.target import NvlinkEngine, glm_pool
from sglang.test.kits.slotguard.wire import BorrowDriver, ExportResponse, identity
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class FullImportDriver(BorrowDriver):
    # Deliberately no size_limit argument: this campaign must map full roots.
    def import_region(self, device, desc):
        return super().import_region(device, desc)


def response(adapter, descriptors, pages):
    source = identity("exporter", layout_digest=digest(adapter.layout))
    target = identity(layout_digest=source.layout_digest)
    count = len(pages)
    query = p.LookupQuery(
        target,
        p.TransferId(target.incarnation, "coalesced", 0),
        tuple(range(count)),
        tuple(range(count * 64 + 1)),
        0,
        count * 64,
        64,
    )
    result = p.LookupResult(query, source, count * 64, query.event_block_keys)
    regions, bindings = codec.encode_export(
        source, "control-lease", adapter, descriptors
    )
    return ExportResponse(
        NS(result=result), "control-lease", regions, bindings, tuple(pages)
    )


def roots(adapter):
    """Fixture physical order, independent of layout planner and bindings."""
    owners = (
        adapter.pool._fixed_fabric_owner,
        adapter.pool.index_key_cache._fixed_fabric_owner,
    )
    return [(owner, memory) for owner, memory in zip(owners, adapter.memories)]


def page_pattern(kind, layer, page, length):
    # 156 unique layer/kind shifts; bytes also vary within each physical page.
    pattern = bytes((kind * 78 + layer + page * 31 + i) % 251 for i in range(251))
    return (pattern * ((length + 250) // 251))[:length]


class CoalescedTransportTests(CustomTestCase):
    def adapter(self, engine):
        return SGLangPoolAdapter(
            glm_pool(384, granularity=4096),
            engine,
            tp_rank=0,
            tp_size=4,
            shard_id="replicated-mla",
        )

    def export(self, adapter, access="read"):
        return {
            m.region_id: adapter.engine.export_memory(m, access)
            for m in adapter.memories
        }

    def test_full_import_read_and_write(self):
        for opcode in Opcode:
            with self.subTest(opcode=opcode):
                self.roundtrip(opcode)

    def roundtrip(self, opcode):
        driver = FullImportDriver()
        se, te = NvlinkEngine(0, _driver=driver), NvlinkEngine(0, _driver=driver)
        source, target = self.adapter(se), self.adapter(te)
        self.assertEqual((len(source.memories), len(target.memories)), (2, 2))
        self.assertEqual(len(source.bindings), 156)
        self.assertEqual(digest(source.layout), digest(target.layout))
        source_pages, target_pages = (5, 1, 6), (2, 6, 3)
        # Each layer/kind/page has independent bytes. Source PAD and physical
        # padding also have deterministic sentinels, checked after the transfer.
        source_original, expected = {}, {}
        for kind, (owner, m) in enumerate(roots(source)):
            page_bytes = (36864, 8448)[kind]
            se.upload(m, b"\x6c" * m.size)
            for layer, tensor in enumerate(owner.tensors):
                offset = tensor.data_ptr() - owner.allocation.base
                for page in range(7):
                    se.upload(
                        m,
                        page_pattern(kind, layer, page, page_bytes),
                        offset + page * page_bytes,
                    )
            source_original[m.region_id] = se.download(m)
        for m in target.memories:
            te.upload(m, b"\xa5" * m.size)
            expected[m.region_id] = bytearray(b"\xa5" * m.size)
        # Byte oracle uses only original geometry and selected physical pages,
        # never BufferRange/TransferRequest offsets or lengths from the planner.
        for kind, (owner, m) in enumerate(roots(target)):
            page_bytes = (36864, 8448)[kind]
            for layer, tensor in enumerate(owner.tensors):
                offset = tensor.data_ptr() - owner.allocation.base
                for sp, tp in zip(source_pages, target_pages):
                    start = offset + tp * page_bytes
                    expected[m.region_id][start : start + page_bytes] = page_pattern(
                        kind, layer, sp, page_bytes
                    )

        exported, importer = (source, te) if opcode == Opcode.READ else (target, se)
        descriptors = self.export(exported)
        resp = response(exported, descriptors, source_pages)
        metadata = codec.decode_export(resp, exported.layout, importer.owner_id)
        self.assertEqual(len(metadata.descriptors), len(exported.memories))
        if opcode == Opcode.WRITE:
            for desc in descriptors.values():
                exported.engine.release_export(desc.lease_id)
            descriptors = self.export(exported, "write")
            # v1 Pin is READ-only. WRITE uses the unchanged generic transport
            # permission path with the same validated region-relative geometry.
            with self.assertRaises(ValueError):
                codec.decode_export(
                    response(exported, descriptors, source_pages),
                    exported.layout,
                    importer.owner_id,
                )
        imports = {
            key: importer.import_memory(desc) for key, desc in descriptors.items()
        }
        remote = metadata.bind(imports)
        sb, tb = (
            (remote, target.bindings)
            if opcode == Opcode.READ
            else (source.bindings, remote)
        )
        plan = plan_transfer(
            source.layout,
            target.layout,
            source_pages,
            target_pages,
            source_capacity=7,
            target_capacity=7,
        )
        requests = bind_regions(plan, sb, tb, opcode)
        self.assertEqual(sum(r.length for r in requests), 3 * 64 * 55224)
        with self.assertRaises(BusyError):
            exported.close()
        handle = importer.submit_transfer(requests)
        local = target if opcode == Opcode.READ else source
        with self.assertRaises(BusyError):
            local.close()
        for m in imports.values():
            with self.assertRaises(BusyError):
                importer.unregister_memory(m)
        driver.state = 1
        self.assertEqual(importer.wait(handle, 0).completed_bytes, 3 * 64 * 55224)
        importer.release_batch(handle)
        for key, m in imports.items():
            importer.unregister_memory(m)
            with self.assertRaises(ValueError):
                importer.import_memory(descriptors[key])  # tombstoned lease
        for desc in descriptors.values():
            exported.engine.release_export(desc.lease_id)
        for m in source.memories:
            self.assertEqual(se.download(m), source_original[m.region_id])
        for m in target.memories:
            # Whole physical root includes PAD, unused pages, between-view gaps
            # (zero for this GLM geometry), and final allocation padding.
            self.assertEqual(te.download(m), expected[m.region_id])
        source.close()
        target.close()
        se.close()
        te.close()
        self.assertFalse(driver.memories)

    def test_corrupt_export_metadata_rejected_before_import_or_submit(self):
        driver = FullImportDriver()
        se, te = NvlinkEngine(0, _driver=driver), NvlinkEngine(0, _driver=driver)
        source, target = self.adapter(se), self.adapter(te)
        descriptors = self.export(source)
        resp = response(source, descriptors, (6,))
        before = driver.next
        for field, value in (
            ("length", 258049),
            ("region_offset", 1),
            ("region_offset", 2**63 - 1),
            ("buffer_offset", 1),
            ("region_id", "foreign"),
        ):
            bindings = list(resp.buffer_bindings)
            data = json.loads(bindings[1].descriptor)
            data["spans"][0][field] = value
            bindings[1] = replace(bindings[1], descriptor=json.dumps(data).encode())
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                codec.decode_export(
                    replace(resp, buffer_bindings=tuple(bindings)),
                    target.layout,
                    te.owner_id,
                )
        # A second region with a fresh lease cannot alias the first FABRIC handle.
        regions = list(resp.regions)
        data = json.loads(regions[1].descriptor)
        first = json.loads(regions[0].descriptor)["transport"]
        data["transport"]["transport_handle"] = first["transport_handle"]
        regions[1] = replace(regions[1], descriptor=json.dumps(data).encode())
        with self.assertRaises(ValueError):
            codec.decode_export(
                replace(resp, regions=tuple(regions)), target.layout, te.owner_id
            )
        self.assertEqual((driver.next, driver.submits), (before, 0))
        metadata = codec.decode_export(resp, target.layout, te.owner_id)
        imports = {d.region_id: te.import_memory(d) for d in metadata.descriptors}
        bindings = metadata.bind(imports)
        plan = plan_transfer(
            source.layout, target.layout, [6], [6], source_capacity=7, target_capacity=7
        )
        for bad in (
            (replace(plan[0], source_offset=7 * 36864),),
            (replace(plan[0], length=36865),),
            (plan[0], plan[0]),
        ):
            with self.assertRaises(ValueError):
                te.submit_transfer(
                    bind_regions(bad, bindings, target.bindings, Opcode.READ)
                )
        name = "layer0.mla"
        broken = bindings | {
            name: (
                replace(
                    bindings[name][0], region_offset=imports[next(iter(imports))].size
                ),
            )
        }
        with self.assertRaises(ValueError):
            bind_regions(plan, broken, target.bindings, Opcode.READ)
        self.assertEqual(driver.submits, 0)
        for m in imports.values():
            te.unregister_memory(m)
        for d in descriptors.values():
            se.release_export(d.lease_id)
        source.close()
        target.close()
        se.close()
        te.close()
        self.assertFalse(driver.memories)


if __name__ == "__main__":
    unittest.main()
