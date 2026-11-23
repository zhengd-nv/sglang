"""SlotGuard source-service fixtures: catalog, service and a CPU exporter.

Nothing here reaches the target side; `cpu_exporter` stands in for the pool
adapter and engine through the transport API only.
"""

from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.slotguard_paged import (
    SlotGuardPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.l1_transfer import hash_pin as hp
from sglang.srt.mem_cache.l1_transfer import protocol as p
from sglang.srt.mem_cache.l1_transfer import slot_guard as s
from sglang.srt.mem_cache.l1_transfer import slotguard_catalog as c
from sglang.srt.mem_cache.l1_transfer import slotguard_protocol as v
from sglang.srt.mem_cache.l1_transfer import slotguard_service as svc
from sglang.srt.mem_cache.l1_transfer import transport_role_arbiter as a
from sglang.srt.mem_cache.l1_transfer.slotguard_group import (
    AllocationEpochError,
    SlotGuardAllocationGroup,
)
from sglang.srt.mem_cache.l1_transfer.transport.layout import (
    BufferLayout,
    PoolLayout,
    RegionSpan,
)
from sglang.test.kits.slotguard.ledger import guard, seed_credit
from sglang.test.kits.slotguard.wire import (
    BorrowDriver,
    CpuEngine,
    hashes,
    identity,
    keys,
)


class Ready:
    value = False

    def ready(self):
        return self.value


# The four-buffer NHD test pool: 33 pages of 64 one-byte tokens per buffer, each
# backed by two VMM mappings. PoolExport binds it exactly as the MHA pool adapter does.
POOL_BUFFERS = ("k0", "v0", "k1", "v1")
POOL_BASE, POOL_STRIDE, POOL_PAGE, POOL_CAPACITY = 65536, 4096, 64, 33
POOL_MAPPINGS = ((0, 704), (704, 1408))


class PoolExport:
    """Exported pool bindings over `engine` without a framework pool object."""

    def __init__(self, engine):
        self.engine = engine
        self.capacity = POOL_CAPACITY
        self.layout = PoolLayout(
            str(SimpleNamespace(itemsize=1)),
            POOL_PAGE,
            tuple(
                BufferLayout(name, POOL_PAGE, "NHD:shape=(1, 1):layer_start=0")
                for name in POOL_BUFFERS
            ),
        )
        self.layout.validate()
        self.memories, self.bindings = [], {}
        try:
            for index, name in enumerate(POOL_BUFFERS):
                start = POOL_BASE + index * POOL_STRIDE
                spans = []
                for offset, length in POOL_MAPPINGS:
                    memory = engine.register_vmm(start + offset, length, owner=self)
                    self.memories.append(memory)
                    spans.append(RegionSpan(offset, length, memory))
                self.bindings[name] = tuple(spans)
        except BaseException:
            self.close()
            raise

    def close(self):
        while self.memories:
            self.engine.unregister_memory(self.memories[-1])
            self.memories.pop()


def cpu_exporter(device=0):
    engine = CpuEngine(device, _driver=BorrowDriver())
    return engine, PoolExport(engine)


class ServiceFixture:
    """One source service over a real guard and catalog; call setUp/tearDown."""

    exporter = staticmethod(cpu_exporter)

    def setUp(self):
        self.g = guard(shared_headroom_credit=True)
        seed_credit(self.g, 32)
        self.refs = self.g.register_allocation((1, 2, 3))
        self.catalog = c.PublicationCatalog(self.g, event_keys=keys)
        ready = Ready()
        ready.value = True
        self.catalog.submit(
            c.Publication(
                c.PublicationScope("model-1", "layout-1", 64, 1),
                None,
                tuple(tuple(range(start, start + 64)) for start in (0, 64, 128)),
                keys(tuple(range(192)), 64),
                self.refs,
                ready,
            )
        )
        self.catalog.progress()
        self.engine, self.adapter = self.exporter()
        self.arbiter = a.TransportRoleArbiter()
        self.service = svc.SlotGuardService(
            guard=self.g,
            catalog=self.catalog,
            local_identity=identity("source"),
            exact_peers=(identity(),),
            engine=self.engine,
            adapter=self.adapter,
            arbiter=self.arbiter,
            max_requests=2,
            max_records=8,
        )

    def tearDown(self):
        self.service.assert_quiescent()
        self.service.close()
        self.adapter.close()
        self.engine.close()

    def lookup(self, attempt=0):
        """Target discovery result; Pin carries only its compact hash proof."""
        target = identity()
        tokens = tuple(range(193))
        query = p.LookupQuery(
            target,
            p.TransferId(target.incarnation, "rid", attempt),
            keys(tokens, 64),
            tokens,
            64,
            192,
            64,
            prefix_hashes=hashes(tokens, 64),
        )
        return p.LookupResult(query, identity("source"), 192, query.event_block_keys)

    def pin(self, attempt=0):
        result = self.lookup(attempt)
        identity_ = s.LeaseIdentity(
            identity("source").incarnation,
            result.query.target.incarnation,
            result.transfer_id,
            f"nonce-{attempt}",
        )
        return v.PinReq(hp.HashLookupResult.from_lookup(result), identity_)

    def unpin(self, response):
        return v.UnpinReq(response.request.reference, response.lease_id, True, True)


def group_fixture(group, *, pages=100, reserve=0):
    """Real allocator, guard, service and allocation group over a CPU exporter."""
    slot_guard = s.SlotGuard(
        1,
        identity("source").incarnation,
        max_remote_pinned_pages=pages,
        max_active_leases=100,
        max_pending_reclaim_pages=pages,
        max_pin_pages=pages,
        max_pages=pages,
        shared_headroom_credit=True,
    )
    alloc = SlotGuardPagedTokenToKVPoolAllocator(
        pages * 64, 64, torch.float16, "cpu", None, False, slot_guard=slot_guard
    )
    alloc.use_triton_kernels = False
    alloc.dcp_enabled = False
    engine, adapter = cpu_exporter()
    catalog = c.PublicationCatalog(slot_guard, event_keys=keys)
    service = svc.SlotGuardService(
        guard=slot_guard,
        catalog=catalog,
        local_identity=identity("source"),
        exact_peers=(identity(),),
        engine=engine,
        adapter=adapter,
        arbiter=a.TransportRoleArbiter(),
        max_records=100,
    )
    state = SimpleNamespace(
        alloc=alloc,
        guard=slot_guard,
        service=service,
        engine=engine,
        adapter=adapter,
        evictable=0,
        data=torch.full((pages + 1, 64), -7, dtype=torch.int64),
    )
    state.controller = SlotGuardAllocationGroup.from_process_group(
        group,
        alloc,
        service,
        members=(0, 1, 2, 3),
        evictable_tokens=lambda: state.evictable,
        scope="cpu-test-pool",
        min_headroom_tokens=reserve * 64,
    )
    state.requests = ServiceFixture()
    return state


def group_owned(state, count):
    indices = state.alloc.alloc(count * 64)
    assert indices is not None
    span = state.alloc.last_allocation_spans[0]
    refs = state.alloc.page_refs(span)
    state.guard.publish(refs)
    return indices, span, refs


def group_lease(state, attempt):
    return s.LeaseIdentity(
        state.guard.source_incarnation,
        "target",
        p.TransferId("target", "cpu-test", attempt),
        f"nonce-{attempt}",
    )


def close_group_fixture(state):
    state.service.close()
    state.adapter.close()
    state.engine.close()


def expect_group_fault(state, function):
    try:
        function()
        raise AssertionError("common failure allowed successful epoch")
    except AllocationEpochError as error:
        assert str(error).replace("_", "").isupper()
    assert state.controller.faulted
    assert state.service.owner_snapshot().fenced
    assert state.alloc._epoch == state.controller.status().epoch
    assert state.guard.owner_counts().frozen_epoch == state.controller.status().epoch
    for operation in (
        lambda: state.controller.assert_ready_for_forward(),
        lambda: state.controller.maintenance(),
        lambda: state.alloc.alloc(64),
        lambda: type(state.alloc).install_allocation_epoch(
            state.alloc, state.controller.status().epoch, (), reclaim_refs=()
        ),
        lambda: state.alloc.clear(),
    ):
        try:
            operation()
            raise AssertionError("faulted owner admitted mutation/forward")
        except (AllocationEpochError, s.EpochError, s.PinsFrozen, s.UnsafeReset):
            pass
    refs = state.alloc.local_page_refs(((1, 1, 0),))
    try:
        state.guard.try_pin(group_lease(state, 98), refs)
        raise AssertionError("faulted guard admitted a fresh direct Pin")
    except s.PinsFrozen:
        pass
