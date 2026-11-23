"""SlotGuard source composition, independent of target admission and wrappers.

The caller owns the borrowed pool/transport and the startup TP groups. Source
resources are installed incrementally so startup cleanup can retain every owner
when construction fails. No target queue, client or backend is constructed here.
"""


class SlotGuardSourceRuntime:
    # Transitional aliases keep the existing serving lifecycle and observers
    # working while the new linker receives this runtime directly.
    serving_fields = (
        "slot_guard",
        "arbiter",
        "catalog",
        "publication_worker",
        "publication_hook",
        "source_service",
    )

    def __init__(self):
        self._initialized = False
        # initialize() installs these one by one; None marks a step not reached.
        self.engine = self.adapter = self.observer = None
        self.slot_guard = self.arbiter = self.catalog = None
        self.publication_worker = self.publication_hook = None
        self.source_service = None

    def initialize(self, ctx, cache, config, pool, *, engine, adapter, observer=None):
        if self._initialized:
            raise RuntimeError("source runtime can only be initialized once")
        self._initialized = True
        self.engine, self.adapter, self.observer = engine, adapter, observer
        from .publication_capture import (
            CanonicalPublicationHook,
            PublicationCaptureWorker,
        )
        from .slotguard_catalog import PublicationCatalog, PublicationScope
        from .slotguard_service import SlotGuardService
        from .transport_role_arbiter import TransportRoleArbiter

        allocator = ctx.params.token_to_kv_pool_allocator
        self.slot_guard = allocator.slot_guard
        self.arbiter = TransportRoleArbiter()
        self.catalog = PublicationCatalog(
            self.slot_guard,
            observer=self.observer,
            max_records=config.publication_context_entries,
            max_token_ids=config.publication_context_entries * 64,
            max_publication_pages=config.slotguard_limits.max_pin_pages,
        )
        self.publication_worker = PublicationCaptureWorker(
            self.catalog,
            max_context_entries=config.publication_context_entries,
            max_context_bytes=config.publication_context_bytes,
        )
        self.publication_hook = CanonicalPublicationHook(
            self.publication_worker,
            allocator,
            PublicationScope(
                config.directory.target.model_id,
                config.directory.target.layout_digest,
                64,
                self.slot_guard.pool_epoch,
            ),
            device=pool.device,
            forward_stream=ctx.tp_worker.model_runner.forward_stream,
        )
        self.publication_hook.attach(cache)
        self.source_service = SlotGuardService(
            guard=self.slot_guard,
            catalog=self.catalog,
            local_identity=config.directory.target,
            exact_peers=tuple(entry.identity for entry in config.directory.entries),
            engine=self.engine,
            adapter=self.adapter,
            arbiter=self.arbiter,
            max_records=config.source_max_records,
            observer=self.observer,
        )
        # No credit is manufactured here. A fresh guard has zero usable credit
        # and source Pin stays BUSY until the common startup owner finishes an
        # authoritative empty-pool allocation epoch.

    def allocation_group(self, startup_groups, allocator, cache, config, collective):
        from .slotguard_group import SlotGuardAllocationGroup

        group = SlotGuardAllocationGroup.from_process_group(
            startup_groups.allocation,
            allocator,
            self.source_service,
            members=startup_groups.members,
            evictable_tokens=cache.evictable_size,
            scope=config.group_manifest_digest,
            min_headroom_tokens=config.min_headroom_tokens,
        )
        group.runtime_collective = collective
        return group

    def close(self):
        """Close background source owners after owner-thread publication freeze.

        Borrowed engine/adapter/pool teardown belongs to the caller. A failed
        close retains the remaining owners, just as the existing serving path.
        """
        for resource in (self.source_service, self.publication_worker, self.arbiter):
            if resource is not None:
                resource.close()
