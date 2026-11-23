"""SlotGuard target-side fixtures: the kv_peer-typed CPU engine, fake pools and a TP4 cohort.

`NvlinkEngine` keeps the CPU byte model but has the KvPeerEngine type that the
NVLink backend requires. The FABRIC helpers need CUDA and import it lazily.
"""

import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from math import prod
from types import SimpleNamespace as NS

import msgspec

from sglang.srt.mem_cache.l1_transfer import directory as d
from sglang.srt.mem_cache.l1_transfer import kvcm_client as k
from sglang.srt.mem_cache.l1_transfer import nvlink_backend as b
from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer import role_gate as l
from sglang.srt.mem_cache.l1_transfer import slotguard_protocol as v
from sglang.srt.mem_cache.l1_transfer import slotguard_service as svc
from sglang.srt.mem_cache.l1_transfer import tp_group as g
from sglang.srt.mem_cache.l1_transfer.control_profile import ControlProfile
from sglang.srt.mem_cache.l1_transfer.generated import (
    tair_kvcache_meta_service_pb2 as pb,
)
from sglang.srt.mem_cache.l1_transfer.generated import (
    tair_kvcache_meta_service_pb2_grpc as rpc,
)
from sglang.srt.mem_cache.l1_transfer.transport.kv_peer_engine import KvPeerEngine
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.test.kits.slotguard.source import ServiceFixture
from sglang.test.kits.slotguard.wire import CpuEngine, identity

V3 = ControlProfile.SLOTGUARD_HASH_V3


class NvlinkEngine(CpuEngine, KvPeerEngine):
    """CPU byte model typed as the kv_peer engine the NVLink backend accepts."""


class Handle(msgspec.Struct, frozen=True):
    """Stand-in for the scheduler's request handle."""

    rid: str
    attempt_id: int


def pool(device):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    # The real class without __init__: only the fields the adapter reads.
    result = object.__new__(MHATokenToKVPool)
    result.use_hnd, result.kv_cache_layout = False, "nhd"
    result.size, result.page_size = 2048, 64
    result.store_dtype, result.start_layer = NS(itemsize=1), 0
    names = ("k0", "v0", "k1", "v1")
    extent, stride, base = 2112, 4096, 65536
    result._kv_buffer_descs = [
        NS(name=name, shape=(2112, 1, 1), row_bytes=1, tokens_per_row=1)
        for name in names
    ]
    result.get_contiguous_buf_infos = lambda: (
        [base + i * stride for i in range(4)],
        [extent] * 4,
        [64] * 4,
    )
    result._post_capture_owner = NS(
        _final_num_tokens=2048,
        page_size=64,
        store_dtype=result.store_dtype,
        device_id=device,
        _specs=[
            NS(desc=desc, offset=i * stride)
            for i, desc in enumerate(result._kv_buffer_descs)
        ],
        _arena=NS(
            base=base,
            reserved=16384,
            granularity=64,
            _closed=False,
            _allocation=NS(
                _closed=False,
                _mappings=[
                    (base + i * stride + offset, length, 0)
                    for i in range(4)
                    for offset, length in ((0, 704), (704, 1408))
                ],
            ),
        ),
    )
    return result


def glm_pool(
    size=2619584,
    *,
    granularity=2 << 20,
    bases=(1 << 40, 2 << 40),
):
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    pool = object.__new__(DSATokenToKVPool)
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
        # Mirrors FixedFabricBufferOwner: 78 slices, 256-byte aligned, of one allocation.
        extent = prod(shape)
        stride = (extent + 255) // 256 * 256
        root_extent = 77 * stride + extent
        root_size = (root_extent + granularity - 1) // granularity * granularity
        tensors, slices = [], []
        for i in range(78):
            ptr = base + i * stride
            slices.append(NS(offset=i * stride, nbytes=extent))
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
        return NS(
            closed=False,
            device_id=0,
            dtype="torch.uint8",
            tensors=tensors,
            buffer_slices=slices,
            allocation=NS(
                base=base,
                size=root_size,
                nbytes=root_extent,
                granularity=granularity,
                device_id=0,
                reservation=NS(base=base, size=root_size),
            ),
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
    # index_k_with_scale_buffer is a property reading index_key_cache.buffer.
    pool.index_key_cache = NS(
        _fixed_fabric_owner=index_owner, buffer=index_owner.tensors.copy()
    )
    pool.get_contiguous_buf_infos = lambda: infos(pool.kv_buffer, 36864)
    pool.get_state_buf_infos = lambda: infos(pool.index_k_with_scale_buffer, 8448)
    return pool


class KvcmService(rpc.MetaServiceServicer):
    def __init__(self):
        self.requests = []
        self.response = pb.GetHostCacheStateResponse(
            header=pb.CommonResponseHeader(status=pb.Status(code=pb.OK))
        )
        self.error = None
        self.wait = False

    def GetHostCacheState(self, request, context):
        self.requests.append(request)
        if self.error:
            context.abort(self.error, "test RPC error")
        if self.wait:
            done = threading.Event()
            context.add_callback(done.set)
            done.wait(2)
        return self.response


class DirectClient:
    def __init__(self, service):
        self.service = service
        self.hold_pin = self.hold_ack = False
        self.held = self.held_ack = None
        self.calls = []

    def pin(self, endpoint, request):
        self.calls.append(("pin", request))
        out = Future()

        def done(future):
            value = future.result()
            if self.hold_pin:
                self.held = (out, value)
            else:
                out.set_result(value)

        self.service.submit(request).add_done_callback(done)
        return out

    def unpin(self, endpoint, request):
        self.calls.append(("unpin", request))
        out = Future()

        def done(future):
            value = future.result()
            if self.hold_ack:
                self.held_ack = (out, value)
            else:
                out.set_result(value)

        self.service.submit(request).add_done_callback(done)
        return out


def wait_source_quiescent(source, *, timeout_s=2):
    """ACK can wake a caller before the service releases accepted-work ownership.

    Read the authoritative proof before fixture resource teardown. Never issue
    another Unpin, mutate ownership, or accept a nonempty service on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            source.assert_quiescent()
            return
        except svc.ServiceBusy:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.001, remaining))


class Rank:
    def __init__(self, rank):
        self.base = ServiceFixture()
        self.base.setUp()
        self.base.service.close()
        source = identity("source", tp_size=4, tp_rank=rank)
        target = identity(tp_size=4, tp_rank=rank)
        self.base.service = svc.SlotGuardService(
            guard=self.base.g,
            catalog=self.base.catalog,
            local_identity=source,
            exact_peers=(target,),
            engine=self.base.engine,
            adapter=self.base.adapter,
            arbiter=self.base.arbiter,
        )
        self.source = self.base.service
        initial = self.base.lookup()
        self.result = replace(
            initial, source=source, query=replace(initial.query, target=target)
        )
        self.driver = self.base.engine._driver
        self.driver.state = 1
        self.engine = NvlinkEngine(1, _driver=self.driver)
        self.adapter = SGLangPoolAdapter(pool(1), self.engine)
        directory = d.EngineDirectory(
            target, (d.EngineDirectoryEntry("source", source, "http://source:1234"),)
        )
        self.client = DirectClient(self.source)
        self.gate = l.RoleGate()
        self.gate.acquire(l.EngineRole.INBOUND, self.result.transfer_id)
        self.identity = None
        self.backend = b.NVLinkTransferBackend(
            self.engine,
            self.adapter,
            directory,
            k.KVCMClient(None, directory),
            self.client,
            self.gate,
            control_profile=V3,
            group_coordinated=True,
            rpc_timeout_s=0.5,
        )
        self.slots = p.PrivateSlotView(
            self.result.transfer_id, 64, 192, 64, tuple(range(256, 384)), (4, 5)
        )

    def status(self):
        return self.backend.poll_prefetch(self.result.transfer_id)

    def close(self):
        status = self.status()
        if status is p.PrefetchStatus.ACKED:
            status = self.backend.cancel_prefetch(self.result.transfer_id)
        if status in (p.PrefetchStatus.READY, p.PrefetchStatus.FAILED_SAFE):
            self.backend.retire(self.result.transfer_id)
            self.backend.close()
        else:
            # Local test reconciliation uses the authoritative source query.
            # It does not retire or reuse the quarantined backend's slots.
            for kind, req in self.client.calls:
                if kind == "pin":
                    state = self.source.query(req.identity)
                    if state is not None and state.state == "PREPARED":
                        self.source.submit(
                            v.UnpinReq(
                                req.reference,
                                state.lease_id,
                                True,
                                True,
                                req.protocol_version,
                                req.capability,
                            )
                        ).result(2)
        wait_source_quiescent(self.source)
        self.adapter.close()
        self.engine.close()
        self.base.tearDown()


class BackendCohort:
    """Four target ranks, each with an independent source service, driven to agreement."""

    def setUp(self):
        self.ranks = [Rank(rank) for rank in range(4)]
        self.decisions = []

    def tearDown(self):
        for rank in self.ranks:
            rank.close()

    def start(self):
        for rank in self.ranks:
            rank.backend.prefetch(rank.result, rank.slots, identity=rank.identity)

    def tick(self):
        snapshots = [
            rank.backend.group_snapshot(rank.result.transfer_id) for rank in self.ranks
        ]
        decision = g.worker_decision(snapshots)
        if decision == "QUARANTINE":
            return
        if decision:
            if not self.decisions or self.decisions[-1] != decision:
                self.decisions.append(decision)
            if decision in ("COPY", "ABORT"):
                assert [r.driver.submits for r in self.ranks] == [0] * 4
                assert all(s[0] == "PIN" for s in snapshots)
            if decision == "RELEASE":
                assert not any(
                    kind == "unpin"
                    for rank in self.ranks
                    for kind, _ in rank.client.calls
                )
                assert all(
                    not rank.backend._records[rank.result.transfer_id].imports
                    for rank in self.ranks
                )
            for rank in self.ranks:
                rank.backend.group_permit(rank.result.transfer_id, decision)
        statuses = [rank.status() for rank in self.ranks]
        if all(status is p.PrefetchStatus.ACKED for status in statuses):
            for rank in self.ranks:
                rank.backend.group_commit_ready(rank.result.transfer_id)

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.tick()
            if predicate():
                return
            time.sleep(0.001)
        raise AssertionError(
            str(
                [
                    (r.status(), r.backend.group_snapshot(r.result.transfer_id))
                    for r in self.ranks
                ]
            )
        )

    def assert_rejected_cohort(self):
        self.start()
        self.until(
            lambda: all(r.status() is p.PrefetchStatus.FAILED_SAFE for r in self.ranks)
        )
        assert "ABORT" in self.decisions
        assert "COPY" not in self.decisions
        assert [r.driver.submits for r in self.ranks] == [0] * 4
        assert [
            sum(kind == "unpin" for kind, _ in r.client.calls) for r in self.ranks
        ] == [1, 1, 1, 0]
        assert all(r.base.g.owner_counts().active_leases == 0 for r in self.ranks)


# Two-buffer FABRIC region shaped like one GLM-5.2 layer: MLA and DSA index pages.
FABRIC_CAPACITY = 64
FABRIC_PAGE_BYTES = (36864, 8448)
FABRIC_SENTINEL = 0xEE


def fabric_offsets():
    first = FABRIC_CAPACITY * FABRIC_PAGE_BYTES[0]
    return (0, (first + 255) // 256 * 256)


def fabric_allocation(device):
    from cuda.bindings import driver as drv

    from sglang.srt.utils.cuda_vmm_utils import (
        align_up,
        check_drv,
        get_allocation_granularity,
        make_device_allocation_prop,
        make_rw_access_desc,
    )

    fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = make_device_allocation_prop(device, handle_types=int(fabric))
    need = fabric_offsets()[1] + FABRIC_CAPACITY * FABRIC_PAGE_BYTES[1]
    size = align_up(need, get_allocation_granularity(prop))
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = int(check_drv(drv.cuMemAddressReserve(size, 0, 0, 0), "reserve"))
    check_drv(drv.cuMemMap(va, size, 0, handle, 0), "map")
    check_drv(drv.cuMemSetAccess(va, size, [make_rw_access_desc(device)], 1), "access")
    return handle, va, size


def free_fabric_allocation(handle, va, size):
    from cuda.bindings import driver as drv

    from sglang.srt.utils.cuda_vmm_utils import check_drv

    check_drv(drv.cuMemUnmap(va, size), "unmap")
    check_drv(drv.cuMemAddressFree(va, size), "free")
    check_drv(drv.cuMemRelease(handle), "release")


def fabric_layout():
    from sglang.srt.mem_cache.l1_transfer.transport.layout import (
        BufferLayout,
        PoolLayout,
    )

    return PoolLayout(
        "test",
        64,
        tuple(
            BufferLayout(f"b{i}", pb, "test") for i, pb in enumerate(FABRIC_PAGE_BYTES)
        ),
    )


def fabric_bindings(memory):
    from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan

    return {
        f"b{i}": (RegionSpan(0, FABRIC_CAPACITY * pb, memory, off),)
        for i, (pb, off) in enumerate(zip(FABRIC_PAGE_BYTES, fabric_offsets()))
    }


def require_fabric_pair():
    """Skip unless two GPUs share an NVLink fabric clique; CI must have them."""
    import torch

    from sglang.srt.utils.cuda_vmm_utils import get_gpu_fabric_clique
    from sglang.test.test_utils import is_in_ci

    if torch.cuda.device_count() >= 2:
        clique = get_gpu_fabric_clique(0)
        if clique is not None and get_gpu_fabric_clique(1) == clique:
            return
    reason = "needs two GPUs in an NVLink fabric clique"
    if is_in_ci():
        raise RuntimeError(reason)
    raise unittest.SkipTest(reason)
