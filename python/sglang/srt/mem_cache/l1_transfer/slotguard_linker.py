"""Dynamic Full-only SlotGuard prefetch backend.

Construction deliberately defers network ownership to the cache builder's
owner-thread startup. The shared serving composition creates source/runtime and
transport resources, but the overridden target hook creates no legacy admission
or PrefetchOnlyLinkerWrapper.
"""

import threading

from sglang.srt.mem_cache.allocator.slotguard_paged import (
    SlotGuardPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)

from .serving_runtime import SlotGuardServingRuntime
from .slotguard_linker_transport import SlotGuardLinkerTransport
from .slotguard_prefetch import SlotGuardPrefetchOwner
from .slotguard_runtime import SlotGuardSchedulerHooks, create_startup_groups


class _Runtime(SlotGuardServingRuntime):
    def _initialize_target(self, cache, config, grouped, collective):
        self.collective = collective
        self.transport = SlotGuardLinkerTransport(
            self.backend,
            collective,
            forward_stream=self.publication_hook.forward_stream,
            capacity=config.prefetch_pending_capacity or 64,
        )
        self.headroom = config.min_headroom_tokens
        self.prefetch_owner = SlotGuardPrefetchOwner(
            cache, self.allocation_group, headroom_tokens=lambda: self.headroom
        )
        self.scheduler_hooks = SlotGuardSchedulerHooks(self)
        self._paused = False
        self._ticks = 0
        self.resetting = False

    def _stage(self, body):
        result, ok = None, False
        try:
            result = body()
            ok = True
        except Exception:
            pass
        if not all(self.collective.gather(ok)):
            self.transport.disabled = True
            self.allocation_group.fault_control_stage()
            raise RuntimeError("SlotGuard lifecycle stage failed")
        return result

    def scheduler_tick(self, *, paused, batches, chunked_req):
        def prepare():
            page = self.cache.page_size
            running = {}
            for batch in batches:
                if batch is not None:
                    running.update((id(req), req) for req in batch.reqs)
            if chunked_req is not None:
                running[id(chunked_req)] = chunked_req
            self.headroom = self.min_headroom_tokens + sum(
                (
                    max(0, req.sampling_params.max_new_tokens - len(req.output_ids))
                    + max(0, len(req.origin_input_ids) - req.kv.kv_allocated_len)
                    + page
                    - 1
                )
                // page
                * page
                for req in running.values()
                if not req.finished()
            )
            return (
                paused,
                self.allocation_group.maintenance_needed(),
                self._paused,
                self._ticks,
            )

        local = self._stage(prepare)
        try:
            votes = self.collective.gather(local)
            if any(v[2:] != votes[0][2:] for v in votes):
                raise RuntimeError("source lifecycle state diverged")
            paused = any(v[0] for v in votes)
            if paused != self._paused:
                generation = self.source_service.owner_snapshot().generation + 1
                if paused:
                    self._stage(
                        lambda: self.source_service.pause_for_maintenance(generation)
                    )
                else:
                    self._stage(
                        lambda: self.source_service.resume_after_maintenance(
                            self.source_service.owner_snapshot().generation
                        )
                    )
                self._paused = paused
            if not paused and self._ticks % 8 == 0 and any(v[1] for v in votes):
                self.allocation_group.maintenance("linker-source-maintenance")
            self._ticks += 1
            self.transport.poll()
        except Exception:
            self.transport.disabled = True
            self.allocation_group.fault_control_stage()
            raise

    def assert_quiescent(self):
        self.transport.assert_quiescent()
        self.source_service.assert_quiescent()
        self.publication_hook.assert_quiescent()

    def flush(self, *, idle, paused, reset_pools):
        """Bounded production reset; busy owners leave the old epoch intact."""
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        from .publication_capture import (
            CanonicalPublicationHook,
            PublicationCaptureWorker,
        )
        from .slotguard_catalog import PublicationScope

        allocator = self.cache.token_to_kv_pool_allocator
        votes = self.collective.gather((idle, paused))
        if not all(v[0] for v in votes):
            return False
        self._paused = any(v[1] for v in votes)
        generation = self.source_service.owner_snapshot().generation + 1
        self._stage(lambda: self.source_service.pause_for_maintenance(generation))
        ready = True
        try:
            self.assert_quiescent()
        except RuntimeError:
            ready = False
        if not all(self.collective.gather(ready)):
            if not self._paused:
                self._stage(
                    lambda: self.source_service.resume_after_maintenance(generation)
                )
            return False
        self.resetting = True
        try:
            # Discard drained wrapper journals/holds before evicting. No new
            # request is dispatched while Scheduler executes this transaction.
            self._stage(self.cache.linker.reset)
            self.allocation_group.execute(
                "linker-reset-evict",
                lambda _: self.cache.evict(
                    EvictParams(num_tokens=self.cache.evictable_size())
                ),
            )
            self.allocation_group.maintenance("linker-reset-reclaim")
            safe = True
            try:
                allocator.assert_reset_safe()
                self.assert_quiescent()
            except RuntimeError:
                safe = False
            if not all(self.collective.gather(safe)):
                # The eviction above is kept; the pools and epoch are not reset.
                if not self._paused:
                    self._stage(
                        lambda: self.source_service.resume_after_maintenance(
                            self.source_service.owner_snapshot().generation
                        )
                    )
                return False
            old_hook, old_worker = self.publication_hook, self.publication_worker

            def freeze():
                old_hook.freeze()
                self.catalog.seal_if_quiescent()
                self.cache.publication_hook = self.cache.tree_core.publication_hook = (
                    None
                )

            self._stage(freeze)
            ok = False
            try:
                ok = reset_pools() is True
            except Exception:
                pass  # all ranks must vote before retaining a failed reset
            if not all(self.collective.gather(ok)):
                raise RuntimeError("partial Scheduler pool reset")

            def renew():
                epoch = self.slot_guard.pool_epoch
                self.catalog.invalidate_epoch(epoch)
                self.catalog.resume_epoch(epoch)
                worker = PublicationCaptureWorker(
                    self.catalog,
                    capacity=old_worker.capacity,
                    max_batch_entries=old_worker.max_batch_entries,
                    max_capture_bytes=old_worker.max_capture_bytes,
                    max_chunk_bytes=old_worker.max_chunk_bytes,
                    max_context_entries=old_worker.max_context_entries,
                    max_context_bytes=old_worker.max_context_bytes,
                    poll_interval_s=old_worker.poll_interval_s,
                )
                hook = CanonicalPublicationHook(
                    worker,
                    allocator,
                    PublicationScope(
                        old_hook.scope.model_id,
                        old_hook.scope.layout_digest,
                        old_hook.scope.page_size,
                        epoch,
                    ),
                    device=old_hook.device,
                    forward_stream=old_hook.forward_stream,
                    readiness_factory=old_hook._readiness_factory,
                )
                self.publication_worker, self.publication_hook = worker, hook
                self.source_runtime.publication_worker = worker
                self.source_runtime.publication_hook = hook
                hook.attach(self.cache)

            self._stage(renew)
            self.allocation_group.maintenance("linker-reset-new-pool")
            if not self._paused:
                self._stage(
                    lambda: self.source_service.resume_after_maintenance(
                        self.source_service.owner_snapshot().generation
                    )
                )
            threading.Thread(target=old_worker.close, name="l1-old-capture").start()
            return True
        except Exception:
            self.transport.disabled = True
            self.allocation_group.fault_control_stage()
            raise
        finally:
            self.resetting = False

    def close(self):
        if self.closed:
            return
        generation = self.source_service.owner_snapshot().generation + 1
        self.source_service.fence_pins(generation)
        self.transport.closed = True
        self.assert_quiescent()
        self.publication_hook.freeze()
        if not self.source_service.seal_if_empty(generation, close_admission=True):
            raise RuntimeError("source owners prevent linker close")
        self.transport.close()
        self.closed = True
        threading.Thread(target=self._cleanup, name="l1-linker-cleanup").start()


class SlotGuardLinker(UnifiedCacheLinker):
    request_prefetch = True

    def __init__(
        self, server_args, params, *, components, extra_config, forward_stream=None
    ):
        if components != {ComponentType.FULL} or extra_config:
            raise ValueError(
                "SlotGuard linker requires Full-only and frozen serving config"
            )
        self.params = params
        self.forward_stream = forward_stream
        self.pools = {PoolName.KV: params.token_to_kv_pool_allocator.get_kvcache()}
        self.layer_done_counter = LayerWiseLoadCounter(78)
        self.runtime = None

    def bind_tree_cache(self, ctx, cache):
        allocator = self.params.token_to_kv_pool_allocator
        if self.load_mode != "prefetch" or not isinstance(
            allocator, SlotGuardPagedTokenToKVPoolAllocator
        ):
            raise ValueError("SlotGuard linker requires prefetch and a SlotGuard pool")
        config = allocator._frozen_serving_config
        groups = create_startup_groups(ctx, config)
        self.runtime = _Runtime(ctx, cache, _config=config, _startup_groups=groups)

    def prefetch_allocation_owner(self):
        return None if self.runtime is None else self.runtime.prefetch_owner

    def scheduler_hooks(self):
        return None if self.runtime is None else self.runtime.scheduler_hooks

    def submit_prefetch_lookup(self, context, transfers):
        if self.runtime._paused:
            return None
        return self.runtime.transport.submit_lookup(context)

    def prefetch_request(self, request, transfers):
        if self.runtime._paused:
            return None
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("SlotGuard requires exactly one Full transfer")
        span = transfers[0].page_span
        if (
            span is None
            or span.token_offset != 0
            or span.token_length != len(transfers[0].device_indices)
        ):
            raise ValueError("SlotGuard transfer lacks exact CPU page identity")
        return self.runtime.transport.prefetch(request, transfers[0])

    def cancel_prefetch_request(self, request):
        self.runtime.transport.cancel(request)

    def release_prefetch_request(self, request):
        self.runtime.transport.release(request)

    def poll_prefetch(self):
        try:
            self.runtime.transport.poll()
        except Exception:
            self.runtime.allocation_group.fault_control_stage()
            raise

    def lookup(self, rid, transfers):
        raise RuntimeError("SlotGuard requires request-aware asynchronous lookup")

    def load(self, rid, transfers):
        return False

    def start_layer_wise_loading(self):
        return -1

    def finish_layer_wise_loading(self, counter_index):
        return True

    def cancel_queued_load(self, rid):
        return False

    def offload(self, transfers, session_id=None):
        # Canonical publication hooks, not external-store offload, advertise L1.
        return False

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        raise RuntimeError("SlotGuard does not enqueue offloads")

    def reset(self):
        self.runtime.assert_quiescent()
        if not self.runtime.resetting:
            raise RuntimeError(
                "SlotGuard reset must enter through Scheduler.flush_cache"
            )

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
