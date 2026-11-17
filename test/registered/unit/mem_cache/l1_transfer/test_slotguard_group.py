"""Exact TP4 allocation epochs on an owned, dedicated CPU/Gloo group."""

import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

from sglang.srt.mem_cache.l1_transfer.slotguard_group import (
    AllocationEpochError,
    SlotGuardAllocationGroup,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


class GroupAPITest(CustomTestCase):
    def test_real_four_process_gloo_safety(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="slotguard-tp4-") as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            context = mp.spawn(_worker, args=(rendezvous,), nprocs=4, join=False)
            deadline = time.monotonic() + 100
            try:
                while not context.join(timeout=0.2):
                    if time.monotonic() > deadline:
                        self.fail("owned TP4 test workers exceeded bounded deadline")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3)


def _fixture(group, *, pages=100, reserve=0):
    """Real allocator/guard/service; borrowed transport is explicitly CPU fake."""
    from types import SimpleNamespace

    import torch
    from test_sglang_adapter import BorrowDriver, SGLangPoolAdapter
    from test_slotguard_service import ServiceTest, a, c, identity, keys, pool, svc

    from sglang.srt.mem_cache.allocator.slotguard_paged import (
        SlotGuardPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.l1_transfer.slot_guard import SlotGuard
    from sglang.test.kits.l1_transfer_cpu import NvlinkEngine

    guard = SlotGuard(
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
        pages * 64, 64, torch.float16, "cpu", None, False, slot_guard=guard
    )
    alloc.use_triton_kernels = False
    alloc.dcp_enabled = False
    engine = NvlinkEngine(0, _driver=BorrowDriver())
    adapter = SGLangPoolAdapter(pool(0), engine)
    catalog = c.PublicationCatalog(guard, event_keys=keys)
    service = svc.SlotGuardService(
        guard=guard,
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
        guard=guard,
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
    state.requests = ServiceTest()
    return state


def _owned(state, count):
    indices = state.alloc.alloc(count * 64)
    assert indices is not None
    span = state.alloc.last_allocation_spans[0]
    refs = state.alloc.page_refs(span)
    state.guard.publish(refs)
    return indices, span, refs


def _lease(state, attempt):
    from sglang.srt.mem_cache.l1_transfer.protocol import TransferId
    from sglang.srt.mem_cache.l1_transfer.slot_guard import LeaseIdentity

    return LeaseIdentity(
        state.guard.source_incarnation,
        "target",
        TransferId("target", "cpu-test", attempt),
        f"nonce-{attempt}",
    )


def _release(state, lease, refs):
    state.guard.release(lease, refs)


def _close(state):
    state.service.close()
    state.adapter.close()
    state.engine.close()


def _expect_fault(state, function):
    from sglang.srt.mem_cache.l1_transfer.slot_guard import (
        EpochError,
        PinsFrozen,
        UnsafeReset,
    )

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
        except (AllocationEpochError, EpochError, PinsFrozen, UnsafeReset):
            pass
    refs = state.alloc.local_page_refs(((1, 1, 0),))
    try:
        state.guard.try_pin(_lease(state, 98), refs)
        raise AssertionError("faulted guard admitted a fresh direct Pin")
    except PinsFrozen:
        pass


def _physical_and_capacity(rank, group):
    """F10+E90-R40 realizes demand60 past a protected-first LRU in one epoch."""
    from collections import deque
    from types import SimpleNamespace

    import torch

    from sglang.srt.mem_cache.l1_transfer.slot_guard import PinsFrozen

    state = _fixture(group)
    alloc, guard, controller = state.alloc, state.guard, state.controller
    # Physical page queues and old registration generations legitimately differ.
    ids = tuple(range(1, 101))
    ids = ids[rank:] + ids[:rank]
    alloc._free_cpu = deque(ids)
    alloc.free_pages = torch.tensor(ids, dtype=torch.int64)
    if rank:
        old = guard.register_allocation(ids[:rank])
        guard.on_free(old)
        guard.begin_epoch(0)
        guard.install_epoch(0, ())
        guard.drain_reclaim(0)
        guard.finish_epoch(0, verified_headroom_pages=0)
    indices, span, refs = _owned(state, 90)
    physical = controller.collective.gather(
        tuple((r.page_id, r.generation) for r in refs)
    )
    assert len(set(physical)) == 4
    assert len(set(controller.collective.gather(alloc.allocation_signature()))) == 1
    # Initial credit remains zero; only a verified common epoch publishes it.
    assert guard.owner_counts().credit_remaining == 0
    state.evictable = 90 * 64
    controller.maintenance("initial-credit")
    lease = _lease(state, 0)
    if rank == 0:
        guard.try_pin(lease, refs[:40])  # PREPARED/partial source visibility
    state.data[list(span.page_ids[:40])] = 123
    cursor = 0

    class Tree:
        def supports_prefix_sharing(self):
            return True

        def evict_for_alloc(self, params):
            nonlocal cursor
            count = min(
                (params.full_tokens - alloc.available_size()) // 64, 90 - cursor
            )
            start, cursor = cursor, cursor + count
            alloc.free_segment(
                indices[start * 64 : cursor * 64],
                start_pos=0,
                page_span=span.slice(start * 64, cursor * 64),
            )
            state.evictable -= count * 64
            return SimpleNamespace(num_tokens_evicted=count * 64)

    def body(decision):
        assert decision.capacity_tokens == 60 * 64
        assert len(decision.protected_pages) == 40
        assert state.service.owner_snapshot().fenced
        try:
            guard.try_pin(_lease(state, 1), refs[40:41])
            raise AssertionError("fresh Pin entered CPU body")
        except PinsFrozen:
            pass
        if rank == 0:
            _release(state, lease, refs[:40])  # late Unpin cannot shrink protection
        assert alloc.evict_to_free_tokens(Tree(), 60 * 64)
        allocated = alloc.alloc(60 * 64)
        assert allocated is not None
        captured = alloc.last_allocation_spans[0]
        state.data[list(captured.page_ids)] = 999
        assert bool(torch.all(state.data[list(span.page_ids[:40])] == 123))
        state.evictable = 60 * 64  # fixture models the completed CPU tree commit
        return captured

    result = controller.execute(
        "prefill", body, request_key="demand60", demand_tokens=60 * 64
    )
    assert result.decision.admitted and cursor == 90
    assert alloc.conservation_counts() == (60, 0, 40, 100)
    assert not state.service.owner_snapshot().fenced
    controller.assert_ready_for_forward()  # simulated forward is outside the fence
    new_refs = alloc.page_refs(result.value)
    guard.publish(new_refs)
    forwarded_lease = _lease(state, 2)
    guard.try_pin(forwarded_lease, new_refs[:1])  # fresh Pin during simulated forward
    _release(state, forwarded_lease, new_refs[:1])
    state.evictable = 60 * 64
    controller.maintenance("late-unpin-reclaim")
    assert alloc.conservation_counts() == (60, 40, 0, 100)
    assert not alloc.verify_byte_accounting()
    _close(state)


def _owner_and_factory(rank, group):
    import threading

    import torch.distributed as dist

    from sglang.srt.mem_cache.l1_transfer.slot_guard import OwnerThreadViolation

    state = _fixture(group, pages=8)

    def factory(selected_group=group, members=(0, 1, 2, 3)):
        return SlotGuardAllocationGroup.from_process_group(
            selected_group,
            state.alloc,
            state.service,
            members=members,
            evictable_tokens=lambda: 0,
            scope="cpu-test-pool",
        )

    for arguments in ((dist.group.WORLD, (0, 1, 2, 3)), (group, (3, 2, 1, 0))):
        try:
            factory(*arguments)
            raise AssertionError("invalid dedicated TP4 group was accepted")
        except ValueError:
            pass
    errors = []

    def other_owner():
        for operation in (factory, state.controller.maintenance):
            try:
                operation()
                errors.append(None)
            except (OwnerThreadViolation, AllocationEpochError) as error:
                errors.append(type(error))

    thread = threading.Thread(target=other_owner)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == [OwnerThreadViolation, AllocationEpochError]
    assert not state.service.owner_snapshot().fenced
    assert not state.controller.faulted
    _close(state)


def _pin_union_and_reclaim_intersection(rank, group):
    """A Pin on one rank protects that page on every rank until it is released."""
    state = _fixture(group, pages=8)
    indices, span, refs = _owned(state, 4)
    state.evictable = 4 * 64
    state.controller.maintenance()
    lease = _lease(state, 0)
    if rank == 0:
        state.guard.try_pin(lease, refs[:1])
    state.alloc.free(indices, page_span=span)
    state.evictable = 0
    result = state.controller.maintenance("rank0-pinned")
    assert result.decision.protected_pages == ((1, 1, 0),)
    assert result.decision.reclaim_pages == ((1, 1, 1), (1, 1, 2), (1, 1, 3))
    assert state.alloc.conservation_counts() == (0, 7, 1, 8)
    if rank == 0:
        _release(state, lease, refs[:1])
    assert state.alloc.available_size() == 7 * 64  # Unpin alone never frees
    result = state.controller.maintenance("rank0-unpinned")
    assert result.decision.reclaim_pages == ((1, 1, 0),)
    assert state.alloc.last_allocated_page_ids == span.page_ids
    assert state.alloc.conservation_counts() == (0, 8, 0, 8)
    _close(state)


def _private_denial(rank, group):
    state = _fixture(group, pages=8)
    _owned(state, 4)
    state.evictable = 0 if rank == 3 else 4 * 64
    before = state.alloc.allocation_signature()
    calls = []
    result = state.controller.execute(
        "target-private",
        lambda _: calls.append(state.alloc.alloc(5 * 64)),
        request_key="same-target-demand",
        demand_tokens=5 * 64,
    )
    assert not result.decision.admitted and result.decision.capacity_tokens == 4 * 64
    assert not calls and state.alloc.allocation_signature() == before
    assert state.guard.owner_counts().credit_remaining == 4
    assert not state.service.owner_snapshot().fenced
    _close(state)


def _history_and_failures(rank, group):
    for failure in (
        "history",
        "body",
        "finish",
        "afterfinish",
        "prevalidate",
        "install",
        "begin",
        "stage",
        "scope",
        "request",
        "posthistory",
        "counter",
        "reentry",
    ):
        _run_failure(rank, group, failure)


def _run_failure(rank, group, failure):
    from unittest.mock import patch

    state = _fixture(group, pages=16)
    sequence = (2, 1, 3) if failure == "history" and rank == 1 else (1, 2, 3)
    for count in sequence:
        _owned(state, count)
    state.evictable = 6 * 64
    if failure == "counter" and rank == 2:
        state.evictable = 7 * 64
    if failure == "scope" and rank == 2:
        state.controller.scope = "another-scope"
    calls = []

    def body(_):
        calls.append(True)
        if failure == "body" and rank == 2:
            # Executed allocation remains real, never a pretend rollback.
            state.alloc.alloc(64)
            raise RuntimeError("secret-request-must-not-escape")
        if failure == "posthistory":
            state.alloc.alloc((2 if rank == 1 else 1) * 64)
        if failure == "reentry" and rank == 2:
            state.controller.maintenance()

    method = {
        "finish": "finish_allocation_epoch",
        "afterfinish": "finish_allocation_epoch",
        "prevalidate": "prevalidate_finish_allocation_epoch",
        "install": "install_allocation_epoch",
        "begin": "begin_allocation_epoch",
    }.get(failure)
    original_finish = state.alloc.finish_allocation_epoch

    def after_finish(*args, **kwargs):
        original_finish(*args, **kwargs)
        assert state.service.owner_snapshot().fenced
        raise RuntimeError("failed after real local commit")

    if method and rank == 2:
        effect = after_finish if failure == "afterfinish" else RuntimeError("private")
        with patch.object(state.alloc, method, side_effect=effect):
            _expect_fault(state, lambda: state.controller.execute("failing", body))
    else:
        stage = "different" if failure == "stage" and rank == 2 else "failing"
        kwargs = {}
        if failure == "request":
            kwargs = {
                "request_key": "different" if rank == 2 else "same",
                "demand_tokens": 64,
            }
        _expect_fault(state, lambda: state.controller.execute(stage, body, **kwargs))
    if failure in (
        "history",
        "stage",
        "scope",
        "request",
        "begin",
        "install",
        "counter",
    ):
        assert not calls
    assert not state.alloc.verify_byte_accounting()
    if failure == "body" and rank == 2:
        assert state.alloc.conservation_counts()[0] == 7
    _close(state)


def _collective_failure(rank, group):
    """A Python collective failure after a real gather permanently fences TP4."""
    from unittest.mock import patch

    import torch.distributed as dist

    state = _fixture(group, pages=8)
    _owned(state, 1)
    gather = state.controller.collective.gather

    def fail_after_gather(value):
        gather(value)
        raise RuntimeError("collective failed")

    with patch.object(
        state.controller.collective, "gather", side_effect=fail_after_gather
    ):
        _expect_fault(state, lambda: state.controller.maintenance())
    _close(state)
    # WORLD remains independent and healthy. A controller which observed a
    # failed allocation collective must never enter another allocation epoch.
    dist.barrier()


def _idle_service_credit(rank, group):
    from test_slotguard_service import Ready, c, keys, v

    state = _fixture(group, pages=8, reserve=6)
    _, _, refs = _owned(state, 3)
    state.evictable = 3 * 64
    # Same bounded publication/CPU fake engine as the adjacent service fixtures.
    state.service.catalog.event_keys = keys
    ready = Ready()
    ready.value = True
    state.service.catalog.submit(
        c.Publication(
            c.PublicationScope("model-1", "layout-1", 64, 1),
            None,
            tuple(tuple(range(start, start + 64)) for start in (0, 64, 128)),
            keys(tuple(range(192)), 64),
            refs,
            ready,
        )
    )
    state.service.catalog.progress()
    assert (
        state.service.submit(state.requests.pin()).result(2).status == v.PinStatus.BUSY
    )
    state.controller.maintenance("initial-credit")
    assert state.guard.owner_counts().credit_remaining == 2
    for attempt in range(4):
        request = state.requests.pin(attempt + 1)
        response = state.service.submit(request).result(2)
        assert response.status == v.PinStatus.PREPARED
        assert state.guard.owner_counts().credit_remaining == 0
        if rank != 0:
            assert (
                state.service.submit(state.requests.unpin(response)).result(2).status
                == v.UnpinStatus.RELEASED
            )
        # All ranks participate even if only one has not completed Unpin.
        state.controller.maintenance("rank-skew-unpin")
        assert state.guard.owner_counts().credit_remaining == 0
        if rank == 0:
            assert (
                state.service.submit(state.requests.unpin(response)).result(2).status
                == v.UnpinStatus.RELEASED
            )
        hints = state.controller.collective.gather(
            state.controller.maintenance_needed()
        )
        assert any(hints) and not all(hints)
        assert state.guard.owner_counts().credit_remaining == 0
        state.controller.maintenance("idle-refresh")
        assert state.guard.owner_counts().credit_remaining == 2
        assert not state.controller.maintenance_needed()
        assert not state.service.owner_snapshot().sealed
    _close(state)


def _worker(rank, rendezvous):
    import importlib

    import test_protocol as contract
    import torch
    import torch.distributed as dist

    # Existing CPU fake transport helpers normally use a hermetic module alias.
    # In these isolated spawned workers bind them to the production package so
    # strict PageRef/LeaseIdentity/wire type checks exercise the same classes as
    # the actual allocator. No parent process/module state is changed.
    contract.PACKAGE = "sglang.srt.mem_cache.l1_transfer"
    contract.p = importlib.import_module(f"{contract.PACKAGE}.protocol")
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    group = dist.new_group(
        ranks=[0, 1, 2, 3], backend="gloo", timeout=timedelta(seconds=15)
    )
    try:
        for scenario in (
            _owner_and_factory,
            _physical_and_capacity,
            _pin_union_and_reclaim_intersection,
            _private_denial,
            _history_and_failures,
            _idle_service_credit,
            _collective_failure,
        ):
            scenario(rank, group)
    finally:
        dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
