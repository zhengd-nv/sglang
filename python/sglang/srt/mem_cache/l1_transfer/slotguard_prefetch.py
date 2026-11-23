"""Scheduler-owned bridge from external prefetch to SlotGuard page ownership.

This manages allocation and publication only. The linker still owns Pin, device
I/O, drain proofs and Unpin; no transport or second admission queue lives here.
"""

import hashlib
import json

from sglang.srt.mem_cache.allocator.page_identity import cpu_page_identity
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    PrefetchAllocationOwner,
)


class SlotGuardPrefetchOwner(PrefetchAllocationOwner):
    def __init__(self, cache, allocation_group, *, headroom_tokens):
        allocator = cache.token_to_kv_pool_allocator
        if (
            cpu_page_identity(allocator) is not allocator
            or allocation_group.allocator is not allocator
            or cache.tree_components != (ComponentType.FULL,)
            or not callable(headroom_tokens)
        ):
            raise ValueError("prefetch needs the exact guarded Full cache and owner")
        self.cache, self.group = cache, allocation_group
        self.headroom_tokens = headroom_tokens

    def allocating(self):
        return self.group.status().active

    def faulted(self):
        return self.group.faulted

    @staticmethod
    def _request_key(request):
        return hashlib.sha256(
            json.dumps(
                (request.rid, request.attempt_id), separators=(",", ":")
            ).encode()
        ).hexdigest()

    def allocate(self, request, num_tokens, body):
        """Commonly reserve FREE pages, without evicting a source's local cache.

        Capture every allocation's PageSpan in body before returning. An
        admitted demand must succeed; a local surprise faults all ranks through
        the controller rather than letting their allocator histories diverge.
        """
        key, demand = "", 0
        try:
            headroom = self.headroom_tokens()
            page = self.cache.page_size
            if type(headroom) is not int or headroom < 0:
                raise ValueError("nonnegative headroom required")
            if type(num_tokens) is not int or num_tokens <= 0 or num_tokens % page:
                raise ValueError("positive whole-page demand required")
            headroom = (headroom + page - 1) // page * page
            demand = num_tokens + max(0, headroom - self.group.min_headroom_tokens)
            key = self._request_key(request)
        except Exception:  # noqa: BLE001 -- invalid local inputs fail in the common vote
            pass

        def allocate(_decision):
            if body() is not True:
                raise RuntimeError("admitted prefetch allocation did not complete")

        result = self.group.execute(
            "external-prefetch-allocate",
            allocate,
            request_key=key,
            demand_tokens=demand,
            private_free_only=True,
        )
        return result.decision.admitted

    def finish(self, request, body):
        """Publish or retire drained slots, reusing a scheduler epoch if active.

        check_prefetch_progress can consume a completed read during batch
        selection. Starting another epoch there would nest the allocator's
        freeze and deadlock/fault the group. Its outer CPU body already owns
        the protection union and commonly checks the final allocator history.
        """
        status = self.group.status()
        if status.faulted:
            self.group.assert_ready_for_forward()
        if status.active:
            return body()
        stage = "external-prefetch-finish-" + self._request_key(request)
        return self.group.execute(stage, lambda _decision: body()).value
