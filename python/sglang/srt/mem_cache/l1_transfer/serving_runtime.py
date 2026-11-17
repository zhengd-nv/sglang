"""SlotGuard startup and shared resource lifetime, independent of legacy targets."""

import logging
import os
import threading

from .config import layout_digest, validate_serving_context
from .control_http import HTTPControlGateway, HTTPPrefixControlClient
from .kvcm_client import KVCMClient
from .nvlink_backend import NVLinkTransferBackend
from .observation import TransferObserver
from .role_gate import RoleGate
from .transport.kv_peer_engine import KvPeerEngine
from .transport.sglang_adapter import SGLangPoolAdapter

logger = logging.getLogger(__name__)


class SlotGuardServingRuntime:
    def __init__(self, ctx, cache, *, _config=None, _startup_groups=None):
        self.cleanup_done = threading.Event()
        self.cleanup_error = None
        self._startup_stage = 0
        reported = False
        try:
            error = None
            try:
                self._initialize(
                    ctx, cache, config=_config, startup_groups=_startup_groups
                )
            except BaseException as exc:  # noqa: BLE001 -- every rank reaches startup ACK
                error = exc
                if _startup_groups is not None:
                    self._report_startup_failure(exc)
                    reported = True
            if _startup_groups is not None:
                self._startup_stage = 13
                collective = _startup_groups.collective(ctx.tp_rank)
                votes = collective.gather(error is None)
                if not all(votes):
                    raise RuntimeError("SlotGuard startup composition failed") from None
            if error is not None:
                raise error
            if _startup_groups is not None:
                # No HTTP Pin is reachable until the first authoritative empty
                # pool epoch finishes on all four allocation owners.
                self._startup_stage = 14
                self.allocation_group.maintenance("startup-empty-pool")
                self._startup_stage = 15
                self._start_gateway_common(collective)
            else:
                self.gateway.start()
        except BaseException as exc:
            if not reported and _startup_groups is not None:
                self._report_startup_failure(exc)
            self.closed = True
            if _startup_groups is not None and hasattr(self, "allocation_group"):
                try:
                    # Also covers a broken startup status collective. Never
                    # release any unknown native ownership while fencing.
                    self.allocation_group.fault_control_stage()
                except Exception:  # noqa: BLE001, S110 -- continue safe lifecycle cleanup
                    pass
            if hasattr(self, "publication_hook"):
                # Construction has not entered the Scheduler/Tree loop. Close
                # this empty owner-only hook here; lifecycle workers must never
                # invoke Tree-owner APIs. Background cleanup performs joins.
                # A rejection retains all backing.
                self.publication_hook.freeze()
            if hasattr(self, "backend"):
                self.backend.close()
            # Startup failure has never accepted a Pin or target reservation.
            threading.Thread(target=self._cleanup, name="l1-startup-cleanup").start()
            raise

    def _report_startup_failure(self, exc):
        """Bounded diagnostics, never exception messages/traceback locals/secrets.

        Logging must not prevent a failed rank reaching the common startup
        vote. Custom exception class names are not trusted log material either.
        Stage numbers refer only to fixed checkpoints in this constructor.
        """
        try:
            kinds = (
                ValueError,
                TypeError,
                RuntimeError,
                OSError,
                ImportError,
                AttributeError,
                AssertionError,
                KeyError,
                MemoryError,
            )
            # Equality/membership can execute an untrusted metaclass hook.
            # Read names only from the exact known builtin classes, by identity.
            kind = next(
                (known.__name__ for known in kinds if type(exc) is known), "Exception"
            )
            logger.error(
                "SlotGuard startup failure stage=%d kind=%s", self._startup_stage, kind
            )
        except BaseException:  # noqa: BLE001, S110 -- isolate diagnostics, not startup
            pass

    def _start_gateway_common(self, collective):
        """Start HTTP outside allocation epochs, with source Pins still fenced.

        No rank returns ready after a peer's partial thread-start failure. The
        source fence is separate from an allocation epoch: starting/joining
        listener/worker threads must never extend an allocator freeze.
        """
        generation, ready = None, False
        try:
            generation = self.source_service.owner_snapshot().generation + 1
            self.source_service.fence_pins(generation)
            self.source_service.prevalidate_unfence_pins(generation)
            ready = True
        except Exception:  # noqa: BLE001, S110 -- all ranks enter the startup vote
            pass
        fences = collective.gather((ready, generation))
        if not all(row[0] for row in fences) or any(row != fences[0] for row in fences):
            self.allocation_group.fault_control_stage()

        started = False
        try:
            self.gateway.start()
            started = True
        except Exception:  # noqa: BLE001, S110 -- report partial Thread.start commonly
            pass
        if not all(collective.gather(started)):
            self.allocation_group.fault_control_stage()

        # Every gateway has now reported successful startup. Publish source
        # admission only after this common commit decision, then acknowledge
        # the local commit before any constructor returns serving-ready.
        ready = False
        try:
            ready = self.source_service.unfence_pins(generation)
        except Exception:  # noqa: BLE001, S110 -- commit failure retains owners
            pass
        if not all(collective.gather(ready)):
            self.allocation_group.fault_control_stage()

    @staticmethod
    def _validate_slotguard(ctx, cache, config, *, inactive=False):
        from sglang.srt.mem_cache.allocator.slotguard_paged import (
            SlotGuardPagedTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        from .slot_guard import SlotGuard

        allocator = ctx.params.token_to_kv_pool_allocator
        if (
            ctx.tp_size != 4
            or not ctx.is_dsa
            or ctx.params.page_size != 64
            or type(allocator) is not SlotGuardPagedTokenToKVPoolAllocator
            or type(allocator.slot_guard) is not SlotGuard
            or allocator.cpu_page_identity() is not (None if inactive else allocator)
            or allocator.page_size != 64
            or cache.page_size != 64
            or cache.token_to_kv_pool_allocator is not allocator
            or not allocator.slot_guard.shared_headroom_credit_enabled
            or allocator.slot_guard.source_incarnation
            != config.directory.target.incarnation
            or type(cache.tree_core) is not UnifiedTreeCore
            or cache.tree_components != (ComponentType.FULL,)
            or cache.tree_core.is_eagle
            or cache.tree_core.enable_hicache
            or cache.tree_core.root_node.children
            or getattr(cache, "publication_hook", None) is not None
            or getattr(getattr(ctx, "server_args", None), "enable_mixed_chunk", False)
            or os.environ.get("SGLANG_DEBUG_MEMORY_POOL", "0").lower() in ("1", "true")
        ):
            raise ValueError(
                "SlotGuard requires exact injected guarded TP4 FullMLA/DSA page64 allocator"
            )
        if (
            config.publication_context_entries < allocator.num_pages
            or config.publication_context_bytes
            < config.publication_context_entries * 1024
        ):
            raise ValueError(
                "publication context must cover the complete physical pool"
            )
        return SlotGuardServingRuntime._slotguard_pool_device(ctx, allocator)

    @staticmethod
    def _slotguard_pool_device(ctx, allocator):
        """Resolve generic ``cuda`` from finalized backing, never a fallback GPU.

        ModelRunner intentionally propagates a device *type*. The fixed FABRIC
        owners and all 156 actual tensors already carry concrete ordinals.
        Require those, runtime current device, and the producer stream to agree
        before activation or the first native registration. The adapter still
        independently validates every physical allocation, slice and pointer.
        """
        import torch

        def reject():
            raise ValueError(
                "SlotGuard requires the actual CUDA pool ordinal/forward stream"
            )

        pool = allocator.get_kvcache()
        if pool is None:
            reject()
        try:
            owners = (
                pool._fixed_fabric_owner,
                pool.index_key_cache._fixed_fabric_owner,
            )
            ordinal = owners[0].device_id
            if type(ordinal) is not int or ordinal < 0:
                reject()
            device = torch.device("cuda", ordinal)
            for owner, tensors in zip(
                owners, (pool.kv_buffer, pool.index_k_with_scale_buffer)
            ):
                if (
                    owner.closed
                    or type(owner.device_id) is not int
                    or owner.device_id != ordinal
                    or len(tensors) != 78
                ):
                    reject()
                if any(torch.device(tensor.device) != device for tensor in tensors):
                    reject()
            for declared in (pool.device, allocator.device):
                declared = torch.device(declared)
                if declared.type != "cuda" or declared.index not in (None, ordinal):
                    reject()
            current = torch.cuda.current_device()
            if type(current) is not int or current != ordinal:
                reject()
            stream = ctx.tp_worker.model_runner.forward_stream
            if stream is None or torch.device(stream.device) != device:
                reject()
        except (AttributeError, TypeError, IndexError):
            reject()
        return device

    def _initialize_slotguard_source(self, ctx, cache, config, pool):
        from .slotguard_source_runtime import SlotGuardSourceRuntime

        self.source_runtime = SlotGuardSourceRuntime()
        try:
            self.source_runtime.initialize(
                ctx,
                cache,
                config,
                pool,
                engine=self.engine,
                adapter=self.adapter,
                observer=getattr(self, "observer", None),
            )
        finally:
            # Preserve partial construction for the existing startup rollback.
            for name in self.source_runtime.serving_fields:
                if hasattr(self.source_runtime, name):
                    setattr(self, name, getattr(self.source_runtime, name))

    def _cleanup(self):
        try:
            for name in (
                "gateway",
                "client",
                "source_service",
                "publication_worker",
                "arbiter",
                "adapter",
                "engine",
                "channel",
                "observer",
            ):
                resource = getattr(self, name, None)
                if resource is not None:
                    resource.close()
        except Exception as exc:  # noqa: BLE001 -- driver failures retain ownership
            # Retain all owners on failure; do not print exception payloads.
            self.cleanup_error = type(exc).__name__
        finally:
            self.cleanup_done.set()

    def _initialize(self, ctx, cache, *, config=None, startup_groups=None):
        self._startup_stage = 1
        validate_serving_context(ctx, cache)
        if config is None or startup_groups is None:
            raise ValueError("SlotGuard requires frozen config and startup groups")
        if os.environ.get("SGLANG_L1_TRANSPORT") != "kv_peer":
            raise ValueError("SlotGuard requires the kv_peer reader")
        if os.environ.get("SGLANG_NVLINK_TEST_ADMIN") == "1":
            raise ValueError("SlotGuard linker does not support test admin")
        if os.environ.get("SGLANG_NVLINK_OBSERVE_PREFILL") == "1":
            raise ValueError("legacy prefill observation is not supported")
        self._config = config
        self._startup_stage = 2
        allocator = ctx.params.token_to_kv_pool_allocator
        if allocator.cpu_page_identity() is None:
            if allocator._frozen_serving_config is not config:
                raise ValueError("inactive SlotGuard requires validated owner startup")
            self._validate_slotguard(ctx, cache, config, inactive=True)
            allocator.activate(ctx.params.req_to_token_pool, cache)
        device = self._validate_slotguard(ctx, cache, config)
        self._startup_stage = 3
        if ctx.tp_size != 4 or config.local_group is None:
            raise ValueError("SlotGuard requires an exact TP4 group manifest")
        collective = startup_groups.collective(ctx.tp_rank)
        self.closed = False
        self.sharing_enabled = True
        self.cache = cache
        self.min_headroom_tokens = config.min_headroom_tokens
        self._startup_stage = 4
        trace_path = os.environ.get("SGLANG_NVLINK_TRACE_PATH")
        self.observer = (
            TransferObserver(
                f"{trace_path}.rank{ctx.tp_rank}",
                config.directory.target.engine_id,
                tp_rank=ctx.tp_rank,
                tp_group_id=config.directory.target.tp_group_id,
            )
            if trace_path
            else None
        )
        verify_bytes = os.environ.get("SGLANG_NVLINK_VERIFY_BYTES") == "1"
        if verify_bytes and self.observer is None:
            raise ValueError("byte verification requires a trace sink")
        pool = allocator.get_kvcache()
        self._startup_stage = 5
        identity = config.directory.target
        if identity.tp_size != ctx.tp_size or identity.tp_rank != ctx.tp_rank:
            raise ValueError("configured NVLink identity must match runtime TP rank")
        adapter_args = dict(
            tp_rank=ctx.tp_rank, tp_size=ctx.tp_size, shard_id=identity.shard_id
        )
        _, actual_layout, _ = SGLangPoolAdapter._prepare_glm(
            pool, device.index, **adapter_args
        )
        if identity.layout_digest != layout_digest(actual_layout):
            raise ValueError("configured layout_digest does not match finalized pool")
        if identity.model_id != ctx.server_args.model_path:
            raise ValueError(
                "configured model_id must equal the exact serving model_path"
            )
        self.gate = RoleGate()
        self._startup_stage = 6
        self.engine = KvPeerEngine(device.index)
        self.adapter = SGLangPoolAdapter(pool, self.engine, **adapter_args)
        self._startup_stage = 7
        import grpc

        self.channel = grpc.insecure_channel(config.kvcm_endpoint)
        self._startup_stage = 8
        self._initialize_slotguard_source(ctx, cache, config, pool)
        self._startup_stage = 9
        self.client = HTTPPrefixControlClient(
            config.directory,
            bearer_token=config.bearer_token,
            trusted_private_domain=True,
            control_profile=config.control_profile,
        )
        self.backend = NVLinkTransferBackend(
            self.engine,
            self.adapter,
            config.directory,
            KVCMClient(self.channel, config.directory),
            self.client,
            self.gate,
            rpc_timeout_s=25,
            observer=self.observer,
            verify_bytes=verify_bytes,
            group_coordinated=True,
            control_profile=config.control_profile,
            transport_arbiter=self.arbiter,
            async_unpin=config.async_unpin,
        )
        self._startup_stage = 10
        self.allocation_group = self.source_runtime.allocation_group(
            startup_groups, allocator, cache, config, collective
        )
        self._initialize_target(cache, config, True, collective)
        self._startup_stage = 12
        self.gateway = HTTPControlGateway(
            identity,
            tuple(entry.identity for entry in config.directory.entries),
            bind_address=config.bind_address,
            bearer_token=config.bearer_token,
            trusted_private_domain=True,
            observer=self.observer,
            control_profile=config.control_profile,
            slotguard_service=self.source_service,
        )
