# SlotGuard external-linker prefetch

This draft restores peer GPU KV prefixes through SGLang's external-linker
prefetch path. Its tested scope is GLM-5.2 NVFP4, DSA MLA/index, TP4, page64,
Prefill, Python TreeCore, FULL components and fixed FABRIC pools.

Start with [deployment and tests](../../../../../test/manual/slotguard_linker/README.md).
The standalone launcher selects `SlotGuardLinker`, `load_mode=prefetch`,
`prefetch_policy=wait_complete`, and the `kv_peer` reader. Test-admin is disabled.

## Review order

| Concern | Entry points | Contract |
|---|---|---|
| Configuration and startup | `config.py`, `tp_manifest.py`, `slotguard_runtime.py` | Validate matching rank identities and capabilities before dedicated TP groups |
| Source allocation | `../allocator/slotguard_paged.py`, `slot_guard.py`, `slotguard_group.py` | Page generations, held/deferred pages and common allocation epochs prevent premature reuse |
| Source publication | `publication_capture.py`, `slotguard_catalog.py`, `slotguard_source_runtime.py` | Publish only captured, ready page identities; source construction is separate from target admission |
| Source leases | `slotguard_service.py`, `control_http.py`, `slotguard_control_wire.py` | Validate identity and Pin budgets; retain uncertain owners; release through Unpin |
| Target lifecycle | `slotguard_linker.py`, `slotguard_prefetch.py`, `slotguard_linker_transport.py` | Request-aware lookup and TP agreement; target pages remain owned until writers drain |
| Data transfer | `nvlink_backend.py`, `transport/kv_peer_engine.py`, `../kv_peer_read.py` | Reader from #49474; asynchronous completion and Unpin remain part of the ownership contract |
| Framework integration | `../../managers/scheduler.py`, `../unified_cache/unified_cache_linker.py`, `../unified_radix_cache.py` | Pair lookup/load/release, admit completed prefixes and handle scheduler maintenance |
| Directory | `kvcm_client.py`, `generated/`, `../../../../../sgl-router/sgl-kv-indexer/` | KVCM lookup/publication through the included indexer and bridge |

This stack builds on the peer reader (#49474), the generated KVCM bindings from
#50216, and request-aware prefetch for the external linker (#50217). #50217 is
stacked on #48820, and the stack is tested on that combination; whether it works
without #48820 has not been checked. KVCR #45650 is not required.

## Runtime and test boundaries

`serving_runtime.py` owns SlotGuard source startup and shared resource lifetime.
`SlotGuardLinker` composes its target on the generic external-linker lifecycle.
The old serving entry, private admission queue, wrapper, test-admin routes and
native reader/build helper have been removed. The launcher uses only the peer
reader from #49474. Shared backend/layout and control contracts remain.

CPU adapter/lease tests retain an explicit byte-storage fixture in `sglang.test.kits`. It cannot load a native library and is never imported by production.
Real peer-reader tests separately exercise the selected transport implementation.

Historical performance/accuracy campaigns, bootstrap operators and their test
fixtures are excluded. Three old manual helpers remain solely to support
streaming and receipt contract tests; their directory README
lists the consumers. Historical implementation and documents are preserved at
`ad059d3f33d1432a65e0139c886d8f008a025e46`.

## Validation boundaries

The standalone GLM entry has demonstrated remote restores, hit-state output
agreement, per-rank byte equality, and idle pause/flush/continue. The original
cold-versus-hit output comparison still fails; byte agreement is not proof of
cold/hit output equivalence. See the deployment README for precise gate criteria.

Remaining draft work includes in-flight Scheduler abort/retract/flush, source
restart and old-incarnation rejection, memory pressure, real CUDA uncertain
failures, and an explicit clique field in the wire descriptor. CPU/fake-driver
tests cannot establish these GPU and live-lifecycle properties. No performance
or broader model/topology claim is made.

## Transfer record lifetime limit

Each backend/incarnation retains retired attempt records for replay protection.
By default, after 4096 cumulatively accepted transfer records, it rejects new
transfers. This is not a concurrency limit or a count of successful cache hits.
Cache flush does not clear these records or replenish the capacity. Safe record
reclamation for long-running deployments remains unresolved; clearing the records
on flush would discard the existing attempt fence.
