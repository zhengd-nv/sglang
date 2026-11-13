"""Contract tests for the peer KV page reader, against a fake CUDA layer."""

import contextlib
import threading
import time
import unittest

import numpy as np

from sglang.srt.mem_cache.kv_peer_read import (
    KvPeerExporter,
    KvPeerReader,
    PeerBuffer,
    PeerReadBusy,
    PeerReadError,
    PeerRegion,
    PeerRegionDescriptor,
    PeerWriteBuffer,
    ReadGroup,
    ReadState,
    WriteGroup,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NOT_READY, OK, ILLEGAL_ADDRESS, NON_STICKY = 600, 0, 700, 999
PAGE = 256
OWNER_DEV, READER_DEV = 0, 1


class FakeOps:
    """Byte-addressed fake: launches queue page copies that land when a later event is queried."""

    not_ready = NOT_READY

    def __init__(self, clique=(b"c" * 16, 1)):
        self.allocs = {}  # base -> (bytearray, device)
        self.handles = {}  # fabric handle -> base
        self.clique = clique
        self.pending = []  # copies not yet landed
        self.event_marks = {}  # recorded event -> number of copies queued before it
        self.num_events = 0
        self.landed = 0
        self.log = []
        self.launches = 0
        self.fail_launch_at = None
        self.event_codes = {}  # event -> error code reported instead of completion
        self.sync_code = OK
        self.fail_record = False
        self.fail_event_at = None  # new_event call that raises
        self.hold = False
        self.hold_from = None  # events from this id on stay in flight until synced
        self.synced = set()  # events a sync_event saw fire
        self.gate = None  # a threading.Event that sync_event blocks on
        self.staged = []  # the spans of every stage() call
        self.maps = []  # read_only of every map_import() call
        self.streams = 0
        self._next = 1 << 20

    def alloc(self, size, device, fill=0):
        base, self._next = self._next, self._next + (size + (1 << 20))
        self.allocs[base] = (bytearray([fill]) * size, device)
        return base

    def _locate(self, addr):
        for base, (buf, _) in self.allocs.items():
            if base <= addr < base + len(buf):
                return buf, addr - base
        raise AssertionError(f"wild address {addr:#x}")

    def read(self, addr, n):
        buf, off = self._locate(addr)
        return bytes(buf[off : off + n])

    def fabric_clique(self, device):
        return self.clique

    def export_handle(self, base):
        handle = base.to_bytes(8, "little") * 8
        self.handles[handle] = base
        return handle

    def map_import(self, fabric_handle, size, device, *, read_only):
        self.log.append("map")
        self.maps.append(read_only)
        return self.handles[fabric_handle]  # alias the owner's bytes

    def unmap_import(self, address, size):
        self.log.append("unmap")

    def pointer_device(self, pointer):
        for base, (buf, device) in self.allocs.items():
            if base <= pointer < base + len(buf):
                return device
        return None

    def new_stream(self, device):
        self.streams += 1
        return f"stream-{self.streams}"

    def wait_stream(self, stream, producer):
        self.log.append(("wait", producer))

    def warmup(self, stream, *, load):
        self.log.append("warmup" if load else "warmup-write")

    def current(self, stream):
        return contextlib.nullcontext()

    def stage(self, stream, meta, spans):
        self.log.append("stage")
        self.staged.append(list(spans))
        meta = meta.copy()
        return meta, [meta[o : o + n] for o, n, _ in spans]

    def launch(self, view):
        if self.fail_launch_at == self.launches:
            raise RuntimeError("launch failed")
        self.launches += 1
        self.log.append("launch")
        load, page_bytes, stride, peer_bases, local_bases, peer_pages, local_pages = (
            view
        )
        for peer, local in zip(peer_bases.tolist(), local_bases.tolist()):
            for pp, lp in zip(peer_pages.tolist(), local_pages.tolist()):
                at_peer, at_local = peer + pp * stride, local + lp * page_bytes
                copy = (at_peer, at_local) if load else (at_local, at_peer)
                self.pending.append((*copy, page_bytes))

    def new_event(self, stream):
        if self.fail_event_at == self.num_events:
            raise RuntimeError("event creation failed")
        self.num_events += 1
        return self.num_events - 1

    def record(self, event, stream):
        if self.fail_record:
            raise RuntimeError("record failed")
        self.event_marks[event] = len(self.pending)
        self.synced.discard(event)
        self.log.append(("record", event, stream))

    def _land(self, upto):
        while self.landed < upto:
            src, dst, n = self.pending[self.landed]
            data = self.read(src, n)
            buf, off = self._locate(dst)
            buf[off : off + n] = data
            self.landed += 1

    def query_event(self, event):
        if event in self.event_codes:
            return self.event_codes[event]
        held = self.hold or (self.hold_from is not None and event >= self.hold_from)
        if held and event not in self.synced:
            return NOT_READY
        self._land(self.event_marks[event])
        return OK

    def sync_event(self, event):
        self.log.append(("sync", event))
        if self.gate is not None:
            assert self.gate.wait(10), "gate never opened"
        if self.sync_code == OK:
            self._land(self.event_marks[event])
            self.synced.add(event)
        return self.sync_code

    def destroy_event(self, event):
        self.log.append(("destroy", event))

    def is_sticky(self, code):
        return code == ILLEGAL_ADDRESS


class PeerReadTestBase(unittest.TestCase):
    NUM_PAGES = 8

    def setUp(self):
        self.ops = FakeOps()
        self.layers = 2
        self.region_bytes = self.layers * self.NUM_PAGES * PAGE
        self.src = self.ops.alloc(self.region_bytes, OWNER_DEV)
        buf, _ = self.ops.allocs[self.src]
        buf[:] = bytes(range(256)) * (self.region_bytes // 256)
        for layer in range(self.layers):
            for page in range(self.NUM_PAGES):
                off = (layer * self.NUM_PAGES + page) * PAGE
                buf[off : off + 2] = bytes([layer, page])
        self.dst = [
            self.ops.alloc(self.NUM_PAGES * PAGE, READER_DEV, fill=0xEE)
            for _ in range(self.layers)
        ]
        self.exporter = KvPeerExporter(OWNER_DEV, _ops=self.ops)
        self.reader = KvPeerReader(READER_DEV, _ops=self.ops)
        self.desc = self.exporter.export(self.src, self.region_bytes, region_id="kv")
        self.region = self.reader.import_region(self.desc)

    def group(self, src_pages, dst_pages, layers=None, dst=None, regions=None):
        dst = dst or self.dst
        regions = regions or [self.region] * self.layers
        layers = range(self.layers) if layers is None else layers
        buffers = tuple(
            PeerBuffer(
                src_region=regions[layer],
                src_offset=layer * self.NUM_PAGES * PAGE,
                dst_base=dst[layer],
                page_bytes=PAGE,
                dst_num_pages=self.NUM_PAGES,
            )
            for layer in layers
        )
        return ReadGroup(
            buffers=buffers,
            src_pages=np.array(src_pages),
            dst_pages=np.array(dst_pages),
        )

    def dst_page(self, layer, page):
        return self.ops.read(self.dst[layer] + page * PAGE, PAGE)

    def src_page(self, layer, page):
        return self.ops.read(self.src + (layer * self.NUM_PAGES + page) * PAGE, PAGE)


class TestReadData(PeerReadTestBase):
    def test_pages_land_and_untouched_pages_keep_sentinel(self):
        batch = self.reader.read([self.group([5, 1], [0, 3]), self.group([7], [6])])
        status = self.reader.wait(batch)
        self.assertEqual(status.state, ReadState.SUCCESS)
        self.assertEqual(status.groups_done, 2)
        for layer in range(self.layers):
            for sp, dp in ((5, 0), (1, 3), (7, 6)):
                self.assertEqual(self.dst_page(layer, dp), self.src_page(layer, sp))
            for dp in (1, 2, 4, 5, 7):
                self.assertEqual(self.dst_page(layer, dp), b"\xee" * PAGE)
        self.reader.release(batch)

    def test_kernel_is_warmed_before_the_first_read(self):
        # A first launch inside a (possibly gated) group could load the module there.
        self.assertLess(self.ops.log.index("warmup"), self.ops.log.index("map"))

    def test_producer_stream_is_waited_before_any_launch(self):
        batch = self.reader.read([self.group([0], [0])], producer_stream="prod")
        self.assertLess(
            self.ops.log.index(("wait", "prod")), self.ops.log.index("launch")
        )
        self.reader.wait(batch)
        self.reader.release(batch)


class TestValidationIsAtomic(PeerReadTestBase):
    def assert_rejected(self, groups, error=ValueError):
        before = [bytes(self.ops.allocs[d][0]) for d in self.dst]
        with self.assertRaises(error):
            self.reader.read(groups)
        self.assertEqual(self.ops.launches, 0)
        self.assertEqual([bytes(self.ops.allocs[d][0]) for d in self.dst], before)

    def test_source_page_beyond_region(self):
        self.assert_rejected([self.group([0], [0]), self.group([self.NUM_PAGES], [1])])

    def test_destination_page_beyond_buffer(self):
        self.assert_rejected([self.group([0], [0]), self.group([1], [self.NUM_PAGES])])

    def test_duplicate_destination_page_in_group(self):
        self.assert_rejected([self.group([0, 1], [2, 2])])

    def test_same_destination_page_in_two_groups(self):
        self.assert_rejected([self.group([0], [2]), self.group([1], [2])])

    def test_two_buffers_of_a_group_share_a_destination(self):
        self.assert_rejected([self.group([0], [2], dst=[self.dst[0], self.dst[0]])])

    def test_partially_overlapping_destination_buffers(self):
        shifted = [self.dst[0] + PAGE, self.dst[1]]
        self.assert_rejected([self.group([0], [0]), self.group([1], [5], dst=shifted)])

    def test_destination_on_another_device(self):
        foreign = self.ops.alloc(self.NUM_PAGES * PAGE, OWNER_DEV)
        self.assert_rejected([self.group([0], [0], dst=[foreign, self.dst[1]])])

    def test_destination_inside_an_imported_region(self):
        local = self.ops.alloc(self.region_bytes, READER_DEV)
        peer = self.reader.import_region(
            self.exporter.export(local, self.region_bytes, region_id="local")
        )
        self.assert_rejected([self.group([0], [0], dst=[peer.address, self.dst[1]])])

    def test_destination_running_past_its_allocation(self):
        short = self.ops.alloc(PAGE, READER_DEV)
        self.assert_rejected([self.group([0], [0], dst=[short, self.dst[1]])])

    def test_misaligned_destination(self):
        self.assert_rejected(
            [self.group([0], [0], dst=[self.dst[0] + 16, self.dst[1]])]
        )

    def test_mismatched_page_lists_and_empty_batch(self):
        self.assert_rejected([self.group([0, 1], [0])])
        self.assert_rejected([])


class TestRegions(PeerReadTestBase):
    def test_one_batch_reads_several_regions(self):
        second = self.reader.import_region(
            self.exporter.export(self.src, self.region_bytes, region_id="kv")
        )
        self.ops.hold = True
        batch = self.reader.read([self.group([2], [4], regions=[self.region, second])])
        for region in (self.region, second):
            with self.assertRaises(PeerReadBusy):
                self.reader.unmap(region)
        self.ops.hold = False
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        for layer in range(self.layers):
            self.assertEqual(self.dst_page(layer, 4), self.src_page(layer, 2))
        self.reader.release(batch)
        self.reader.unmap(second)
        self.reader.unmap(self.region)

    def test_unmap_evicts_cached_buffers_that_read_the_region(self):
        group = self.group([0], [0])
        batch = self.reader.read([group])
        self.reader.wait(batch)
        self.reader.release(batch)
        self.reader.unmap(self.region)
        with self.assertRaises(ValueError):  # same tuple, its region is gone
            self.reader.read([group])


class TestInFlightRanges(PeerReadTestBase):
    def test_conflict_only_while_in_flight(self):
        self.ops.hold = True
        first = self.reader.read([self.group([0, 1], [0, 1])])
        with self.assertRaises(PeerReadBusy):  # retry later, not a caller bug
            self.reader.read([self.group([2], [1])])
        disjoint = self.reader.read([self.group([2], [2])])  # same region, other pages
        self.ops.hold = False
        for batch in (first, disjoint):
            self.reader.wait(batch)
            self.reader.release(batch)
        again = self.reader.read([self.group([3], [1])])
        self.reader.wait(again)
        self.reader.release(again)

    def single(self, dst_base, capacity, page):
        buffer = PeerBuffer(
            src_region=self.region,
            src_offset=0,
            dst_base=dst_base,
            page_bytes=PAGE,
            dst_num_pages=capacity,
        )
        return ReadGroup(
            buffers=(buffer,), src_pages=np.array([0]), dst_pages=np.array([page])
        )

    def test_growing_a_destination_over_an_in_flight_one(self):
        self.ops.hold = True
        base = self.dst[0]
        self.reader.read([self.single(base, 1, 0)])
        self.reader.read([self.single(base + 2 * PAGE, 1, 0)])
        # Same (base, page size) as the first, but now reaching into the second.
        with self.assertRaises(PeerReadBusy):
            self.reader.read([self.single(base, 4, 2)])

    def test_self_overlap_is_a_caller_error_behind_an_in_flight_batch(self):
        """A batch whose own destinations overlap raises ValueError even when it also
        overlaps a batch in flight: an uncertain one would make PeerReadBusy permanent."""
        self.ops.hold = True
        base = self.dst[0]
        self.reader.read([self.single(base, 4, 0)])
        with self.assertRaises(ValueError):
            self.reader.read(
                [self.single(base, 4, 1), self.single(base + 2 * PAGE, 2, 0)]
            )

    def test_finished_but_unpolled_batch_does_not_block(self):
        first = self.reader.read([self.group([0], [0])])  # done, nobody asked
        second = self.reader.read([self.group([1], [0])])
        for batch in (first, second):
            self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
            self.reader.release(batch)
        self.assertEqual(self.dst_page(0, 0), self.src_page(0, 1))

    def test_timeout_keeps_batch_and_ranges(self):
        self.ops.hold = True
        batch = self.reader.read([self.group([0], [0])])
        with self.assertRaises(TimeoutError):
            self.reader.wait(batch, timeout=0)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)
        with self.assertRaises(PeerReadBusy):
            self.reader.unmap(self.region)
        with self.assertRaises(PeerReadBusy):
            self.reader.read([self.group([1], [0])])
        for bad in (-1, float("inf"), float("nan"), "1"):
            with self.assertRaises(ValueError):
                self.reader.wait(batch, timeout=bad)


class TestFailures(PeerReadTestBase):
    def test_completed_group_survives_later_failure(self):
        self.ops.event_codes[1] = NON_STICKY
        batch = self.reader.read([self.group([4], [0]), self.group([5], [1])])
        status = self.reader.wait(batch)
        self.assertEqual(status.state, ReadState.FAILED_DRAINED)
        self.assertTrue(self.reader.group_done(batch, 0))
        self.assertFalse(self.reader.group_done(batch, 1))
        self.assertEqual(self.dst_page(0, 0), self.src_page(0, 4))
        self.reader.release(batch)
        self.reader.unmap(self.region)

    def test_undrained_failure_is_uncertain_and_pins_everything(self):
        self.ops.event_codes[0] = NON_STICKY
        self.ops.sync_code = NON_STICKY
        batch = self.reader.read([self.group([0], [0])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.FAILED_UNCERTAIN)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)
        with self.assertRaises(PeerReadBusy):
            self.reader.unmap(self.region)
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([1], [1])])

    def test_sticky_error_fail_stops_every_in_flight_batch(self):
        self.ops.hold = True
        other = self.reader.read([self.group([1], [1])])
        self.ops.hold = False
        self.ops.event_codes[1] = ILLEGAL_ADDRESS
        failing = self.reader.read([self.group([0], [0])])
        self.ops.event_codes[0] = NOT_READY  # the other batch has not finished either
        self.assertEqual(self.reader.wait(failing).state, ReadState.FAILED_UNCERTAIN)
        self.assertEqual(self.reader.status(other).state, ReadState.FAILED_UNCERTAIN)
        with self.assertRaises(PeerReadError):
            self.reader.import_region(
                self.exporter.export(self.src, self.region_bytes, region_id="kv")
            )

    def test_enqueue_failure_with_clean_drain_leaves_nothing_held(self):
        self.ops.fail_launch_at = 1
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        self.ops.fail_launch_at = None
        batch = self.reader.read([self.group([2], [0]), self.group([3], [1])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        self.reader.release(batch)
        self.reader.unmap(self.region)
        self.reader.close()

    def test_drain_covers_a_partially_enqueued_group(self):
        # Two page sizes make two launches in one group; the second one fails.
        half = [
            PeerBuffer(
                src_region=self.region,
                src_offset=0,
                dst_base=self.dst[0],
                page_bytes=PAGE,
                dst_num_pages=self.NUM_PAGES,
            ),
            PeerBuffer(
                src_region=self.region,
                src_offset=self.NUM_PAGES * PAGE,
                dst_base=self.dst[1],
                page_bytes=PAGE // 2,
                dst_num_pages=self.NUM_PAGES,
            ),
        ]
        group = ReadGroup(
            buffers=tuple(half), src_pages=np.array([0]), dst_pages=np.array([0])
        )
        self.ops.fail_launch_at = 1
        with self.assertRaises(RuntimeError):
            self.reader.read([group])
        self.assertIn(
            ("sync", 0), self.ops.log
        )  # the group's own event, recorded after the failure
        self.ops.fail_launch_at = None
        batch = self.reader.read([self.group([1], [0])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)

    def test_unrecorded_launch_fail_stops(self):
        self.ops.fail_record = True
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0])])
        self.assertNotIn(
            ("sync", 0), self.ops.log
        )  # nothing covers the launch, so no drain is claimed
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([1], [1])])
        with self.assertRaises(PeerReadBusy):  # a launch may still be reading it
            self.reader.unmap(self.region)

    def test_failed_event_creation_destroys_the_events_made(self):
        self.ops.fail_event_at = 1
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        self.assertIn(("destroy", 0), self.ops.log)
        self.assertEqual(self.ops.launches, 0)
        self.ops.fail_event_at = None
        batch = self.reader.read([self.group([0], [0])])  # nothing was held
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)

    def test_enqueue_failure_without_drain_fail_stops(self):
        self.ops.fail_launch_at = 1  # group 0 is enqueued, group 1 fails, no drain
        self.ops.sync_code = ILLEGAL_ADDRESS
        with self.assertRaises(RuntimeError):
            self.reader.read([self.group([0], [0]), self.group([1], [1])])
        with self.assertRaises(PeerReadError):
            self.reader.read([self.group([2], [2])])
        with self.assertRaises(PeerReadBusy):  # group 0 may still be reading it
            self.reader.unmap(self.region)


class TestRegionsAndLeases(PeerReadTestBase):
    def test_descriptor_round_trip_and_malformed_input(self):
        self.assertEqual(
            PeerRegionDescriptor.from_bytes(self.desc.to_bytes()), self.desc
        )
        for data in (b"", b"\x01garbage", self.desc.to_bytes()[:-3]):
            with self.assertRaises(ValueError):
                PeerRegionDescriptor.from_bytes(data)

    def test_lease_maps_once_even_after_unmap(self):
        with self.assertRaises(ValueError):
            self.reader.import_region(self.desc)
        self.reader.unmap(self.region)
        with self.assertRaises(ValueError):
            self.reader.import_region(self.desc)
        fresh = self.exporter.export(self.src, self.region_bytes, region_id="kv")
        self.reader.import_region(fresh)

    def test_other_fabric_clique_rejected_before_mapping(self):
        other = KvPeerReader(READER_DEV, _ops=FakeOps(clique=(b"x" * 16, 1)))
        maps = self.ops.log.count("map")
        with self.assertRaises(PeerReadError):
            other.import_region(
                self.exporter.export(self.src, self.region_bytes, region_id="kv")
            )
        self.assertEqual(self.ops.log.count("map"), maps)

    def test_forged_region_handle(self):
        forged = PeerRegion(
            self.region.region_id, self.region.size, self.region.address
        )
        with self.assertRaises(ValueError):
            self.reader.read([self.group([0], [0], regions=[forged] * self.layers)])
        with self.assertRaises(ValueError):
            KvPeerReader(READER_DEV, _ops=self.ops).unmap(self.region)

    def test_lease_release_is_exact(self):
        second = self.exporter.export(self.src, self.region_bytes, region_id="kv")
        self.assertEqual(self.exporter.outstanding("kv"), 2)
        self.exporter.release_export(self.desc.lease_id)
        with self.assertRaises(ValueError):
            self.exporter.release_export(self.desc.lease_id)
        self.assertEqual(self.exporter.outstanding("kv"), 1)
        self.exporter.release_export(second.lease_id)
        self.assertEqual(self.exporter.outstanding(), 0)


class TestGroupWait(PeerReadTestBase):
    def test_returns_once_that_group_landed(self):
        self.ops.hold_from = 1  # group 1 stays in flight until a wait blocks on it
        batch = self.reader.read(
            [self.group([4], [0], layers=[0]), self.group([5], [1], layers=[1])]
        )
        status = self.reader.wait_group(batch, 0)
        self.assertEqual((status.state, status.groups_done), (ReadState.IN_FLIGHT, 1))
        self.assertNotIn(("sync", 0), self.ops.log)  # it had landed: no blocking
        self.assertEqual(self.dst_page(0, 0), self.src_page(0, 4))
        self.assertEqual(self.dst_page(1, 1), b"\xee" * PAGE)
        status = self.reader.wait_group(batch, 1)
        self.assertIn(("sync", 1), self.ops.log)
        self.assertEqual((status.state, status.groups_done), (ReadState.SUCCESS, 2))
        self.assertEqual(self.dst_page(1, 1), self.src_page(1, 5))
        self.reader.release(batch)

    def test_blocked_waiter_holds_no_lock_and_pins_its_batch(self):
        """A thread blocked on a group must not stop other threads from submitting or
        polling, and the batch must not be released (its event recycled) under it."""
        self.ops.hold_from = 0
        self.ops.gate = threading.Event()
        batch = self.reader.read([self.group([0], [0])])
        results = []
        waiter = threading.Thread(
            target=lambda: results.append(self.reader.wait_group(batch, 0))
        )
        waiter.start()
        deadline = time.monotonic() + 10
        while ("sync", 0) not in self.ops.log:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(1e-3)
        other = self.reader.read([self.group([1], [1])])  # takes the lock
        self.ops.hold_from = None
        self.assertEqual(self.reader.status(batch).state, ReadState.SUCCESS)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)
        self.ops.gate.set()
        waiter.join(10)
        self.assertEqual(results[0].state, ReadState.SUCCESS)
        self.reader.release(batch)
        self.assertEqual(self.reader.wait(other).state, ReadState.SUCCESS)
        self.reader.release(other)

    def test_a_failed_earlier_group_ends_the_wait(self):
        self.ops.event_codes[0] = NON_STICKY
        batch = self.reader.read([self.group([0], [0]), self.group([1], [1])])
        status = self.reader.wait_group(batch, 1)
        self.assertEqual(
            (status.state, status.groups_done), (ReadState.FAILED_DRAINED, 0)
        )
        self.reader.release(batch)

    def test_timeout_polls_and_keeps_everything(self):
        self.ops.hold = True
        batch = self.reader.read([self.group([0], [0]), self.group([1], [1])])
        with self.assertRaises(TimeoutError):
            self.reader.wait_group(batch, 0, timeout=0)
        self.assertNotIn(("sync", 0), self.ops.log)  # never blocks in the driver
        with self.assertRaises(IndexError):
            self.reader.wait_group(batch, 2)
        with self.assertRaises(PeerReadBusy):
            self.reader.release(batch)


class TestPerBatchCost(PeerReadTestBase):
    @staticmethod
    def pointer_spans(spans):
        return sum(pointer for _, _, pointer in spans)

    def test_address_tables_go_up_once_per_buffer_tuple(self):
        """Once a buffer tuple's address tables are resident, a batch stages only its
        page arrays: layer-wise reads must not pay a table upload per group."""
        group = self.group([0], [0])
        for src, dst in ((0, 0), (1, 1)):
            batch = self.reader.read(
                [ReadGroup(group.buffers, np.array([src]), np.array([dst]))]
            )
            self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
            self.reader.release(batch)
        self.assertEqual([self.pointer_spans(s) for s in self.ops.staged], [2, 0])
        self.assertEqual(self.dst_page(1, 1), self.src_page(1, 1))

    def test_tables_go_up_again_after_a_failed_enqueue(self):
        """An enqueue that failed may not have run its upload; the next batch must not
        launch on those tables."""
        group = self.group([0], [0])
        self.ops.fail_launch_at = 0
        with self.assertRaises(RuntimeError):
            self.reader.read([group])
        self.ops.fail_launch_at = None
        batch = self.reader.read(
            [ReadGroup(group.buffers, np.array([2]), np.array([2]))]
        )
        self.assertEqual([self.pointer_spans(s) for s in self.ops.staged], [2, 2])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        self.assertEqual(self.dst_page(1, 2), self.src_page(1, 2))

    def test_events_are_recycled_only_from_successful_batches(self):
        first = self.reader.read([self.group([0], [0])])
        self.reader.wait(first)
        self.reader.release(first)
        made = self.ops.num_events
        failing = self.reader.read([self.group([1], [1])])
        self.assertEqual(self.ops.num_events, made)  # the released event came back
        self.ops.event_codes[0] = NON_STICKY
        self.assertEqual(self.reader.wait(failing).state, ReadState.FAILED_DRAINED)
        self.reader.release(failing)
        self.assertIn(("destroy", 0), self.ops.log)  # a failed event is never reused
        self.reader.read([self.group([2], [2])])
        self.assertEqual(self.ops.num_events, made + 1)


# Page objects as an EGM store lays them out: each slot holds one page of every
# buffer, at that buffer's item offset; GAP bytes after each item belong to nobody.
ITEMS = (256, 128)
GAP = 128
ITEM_OFF = (0, ITEMS[0] + GAP)
OBJECT = sum(ITEMS) + GAP * len(ITEMS)
INSIDE = (ITEM_OFF[0], ITEM_OFF[0] + 128)  # item 1 placed inside item 0


class PageObjectTestBase(unittest.TestCase):
    SLOTS = 6

    def setUp(self):
        self.ops = FakeOps()
        self.slab_bytes = self.SLOTS * OBJECT
        self.slab = self.ops.alloc(self.slab_bytes, OWNER_DEV, fill=0x5A)
        buf, _ = self.ops.allocs[self.slab]
        for slot in range(self.SLOTS):
            for item, size in enumerate(ITEMS):
                at = slot * OBJECT + ITEM_OFF[item]
                buf[at : at + size] = bytes([slot, item]) * (size // 2)
        self.local = [
            self.ops.alloc(self.SLOTS * size, READER_DEV, fill=0xEE) for size in ITEMS
        ]
        self.exporter = KvPeerExporter(OWNER_DEV, _ops=self.ops)
        self.reader = KvPeerReader(READER_DEV, _ops=self.ops)

    def import_slab(self, writable=False, size=None):
        desc = self.exporter.export(
            self.slab, size or self.slab_bytes, region_id="slab", writable=writable
        )
        return self.reader.import_region(desc, writable=writable)

    def local_page(self, item, page):
        size = ITEMS[item]
        return self.ops.read(self.local[item] + page * size, size)

    def fill_local(self):
        for item, size in enumerate(ITEMS):
            buf, _ = self.ops.allocs[self.local[item]]
            for page in range(self.SLOTS):
                buf[page * size : (page + 1) * size] = bytes([0xA0 + page, item]) * (
                    size // 2
                )

    def read_group(self, region, src_pages, dst_pages, stride=OBJECT):
        buffers = tuple(
            PeerBuffer(
                src_region=region,
                src_offset=ITEM_OFF[item],
                dst_base=self.local[item],
                page_bytes=size,
                dst_num_pages=self.SLOTS,
                src_stride=stride,
            )
            for item, size in enumerate(ITEMS)
        )
        return ReadGroup(buffers, np.array(src_pages), np.array(dst_pages))

    def write_group(self, region, src_pages, dst_pages, items=(0, 1), offsets=ITEM_OFF):
        buffers = tuple(
            PeerWriteBuffer(
                dst_region=region,
                dst_offset=offsets[item],
                src_base=self.local[item],
                page_bytes=ITEMS[item],
                src_num_pages=self.SLOTS,
                dst_stride=OBJECT,
            )
            for item in items
        )
        return WriteGroup(buffers, np.array(src_pages), np.array(dst_pages))


class TestSourceStride(PageObjectTestBase):
    def test_items_of_page_objects_land_in_their_buffers(self):
        """Page p of a buffer is read at p * object size from the buffer's item offset:
        items of one object go to different buffers and gap bytes are not copied."""
        region = self.import_slab()
        batch = self.reader.read([self.read_group(region, [4, 0], [1, 5])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        for item, size in enumerate(ITEMS):
            for slot, page in ((4, 1), (0, 5)):
                self.assertEqual(
                    self.local_page(item, page), bytes([slot, item]) * (size // 2)
                )
            for page in (0, 2, 3, 4):
                self.assertEqual(self.local_page(item, page), b"\xee" * size)

    def test_each_buffer_bounds_the_source_pages(self):
        """The bound is per buffer: when the region ends inside the last object, its
        first item still fits but the slot is out of range for the batch."""
        short = (self.SLOTS - 1) * OBJECT + ITEM_OFF[1]
        region = self.import_slab(size=short)
        batch = self.reader.read([self.read_group(region, [self.SLOTS - 2], [0])])
        self.reader.wait(batch)
        self.reader.release(batch)
        launches = self.ops.launches
        with self.assertRaises(ValueError):
            self.reader.read([self.read_group(region, [self.SLOTS - 1], [1])])
        for stride in (ITEMS[0] // 2, OBJECT + 64):  # shorter than a page; misaligned
            with self.assertRaises(ValueError):
                self.reader.read([self.read_group(region, [0], [1], stride=stride)])
        self.assertEqual(self.ops.launches, launches)


class TestWrites(PageObjectTestBase):
    def test_pages_land_in_their_slots_and_spare_everything_else(self):
        """A write scatters each local page into its item of the destination object;
        other slots and the gaps between items keep their bytes."""
        self.fill_local()
        expected = bytearray(self.ops.allocs[self.slab][0])
        region = self.import_slab(writable=True)
        batch = self.reader.write([self.write_group(region, [2, 5], [3, 0])])
        self.assertEqual(self.reader.wait(batch).state, ReadState.SUCCESS)
        for item, size in enumerate(ITEMS):
            for page, slot in ((2, 3), (5, 0)):
                at = slot * OBJECT + ITEM_OFF[item]
                expected[at : at + size] = bytes([0xA0 + page, item]) * (size // 2)
        self.assertEqual(bytes(self.ops.allocs[self.slab][0]), bytes(expected))
        self.reader.release(batch)
        self.reader.unmap(region)

    def test_writing_needs_a_writable_lease_and_import(self):
        read_only = self.exporter.export(self.slab, self.slab_bytes, region_id="slab")
        with self.assertRaises(ValueError):
            self.reader.import_region(read_only, writable=True)
        region = self.reader.import_region(read_only)
        with self.assertRaises(ValueError):
            self.reader.write([self.write_group(region, [0], [0])])
        self.assertNotIn("warmup-write", self.ops.log)
        granted = self.exporter.export(
            self.slab, self.slab_bytes, region_id="slab", writable=True
        )
        self.reader.import_region(granted)  # least privilege unless asked
        start = len(self.ops.log)
        self.import_slab(writable=True)
        self.assertEqual(self.ops.log[start:], ["warmup-write", "map"])
        self.import_slab(writable=True)
        self.assertEqual(self.ops.maps, [True, True, False, False])
        self.assertEqual(self.ops.log.count("warmup-write"), 1)

    def test_in_flight_writes_conflict_on_the_allocation_not_the_mapping(self):
        """Two leases of one allocation are one destination: the same item of the same
        slot is busy through either mapping; another slot or item is not."""
        self.fill_local()
        first, second = self.import_slab(writable=True), self.import_slab(writable=True)
        self.ops.hold = True
        self.reader.write([self.write_group(first, [0], [3])])
        with self.assertRaises(PeerReadBusy):
            self.reader.write([self.write_group(second, [1], [3])])
        self.reader.write([self.write_group(second, [1], [4])])
        self.reader.write([self.write_group(first, [2], [5], items=(0,))])
        self.reader.write([self.write_group(second, [3], [5], items=(1,))])
        with self.assertRaises(PeerReadBusy):
            self.reader.write([self.write_group(second, [4], [5], items=(0,))])
        with self.assertRaises(
            PeerReadBusy
        ):  # another key, but inside item 0 of slot 5
            self.reader.write(
                [self.write_group(first, [4], [5], items=(1,), offsets=INSIDE)]
            )

    def test_overlapping_writes_within_a_batch_are_caller_errors(self):
        region = self.import_slab(writable=True)
        with self.assertRaises(ValueError):
            self.reader.write(
                [
                    self.write_group(region, [0], [2]),
                    self.write_group(region, [1], [2]),
                ]
            )
        with self.assertRaises(ValueError):
            self.reader.write([self.write_group(region, [0], [1], offsets=INSIDE)])
        with self.assertRaises(ValueError):  # a slot past the region
            self.reader.write([self.write_group(region, [0], [self.SLOTS])])
        self.assertEqual(self.ops.launches, 0)

    def test_reads_and_writes_run_on_separate_streams(self):
        """A long write-through must not delay a later read: each direction has its
        own stream, so neither queues behind the other."""
        self.fill_local()
        region = self.import_slab(writable=True)
        self.ops.hold = True
        self.reader.write([self.write_group(region, [0], [1])])
        self.reader.read([self.read_group(region, [2], [3])])
        streams = [entry[2] for entry in self.ops.log if entry[0] == "record"]
        self.assertEqual(len(set(streams)), 2)


if __name__ == "__main__":
    unittest.main()
