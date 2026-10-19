# L1 sharing implementation

L1 means the live GPU KV cache of an ordinary SGLang Engine. A target Engine
explicitly pulls a cached prefix from a fixed source Engine over NVLink, installs
it in its own radix cache, and then runs ordinary generation. The two Engines
retain separate physical KV pages. There is no shared allocator, external KV
store, PD role, or automatic remote lookup on request admission.

This document describes `feat/l1-sharing`, built on the standalone
`feat/nvlink-transfer` core. All links refer to files in this repository.
See [README](README.md) for native execution and provenance and
[TESTING](TESTING.md) for reproduction and the validation boundary.

## Supported configuration

The integration requires identical FULL/MHA/NHD layouts, plain radix caching,
a finalized CUDA VMM KV pool, TP/PP/DP = 1, no speculation or LoRA, and
`max_running_requests=1`. Cache/transport mutations execute serially on the
scheduler's cache-owner thread. The nonblocking request mode additionally
requires `disable_overlap_schedule=True` and pauses forward while sharing. The tested model setup uses post-capture VMM
sizing. An ordinary cudaMalloc pool is not exportable through this adapter.

Peers are trusted. The control socket is unauthenticated; descriptor permissions
are a cooperative API contract, not a hardware security boundary. No discovery,
reconnection recovery, cross-layout conversion, multi-rank consensus, cache salts
or alternative cache-key namespaces are implemented. Do not change weights after
initializing a peer.

## Code map

| Layer | Files | Responsibility |
| --- | --- | --- |
| Public API | [Engine](../../entrypoints/engine.py) | `Engine.l1_peer(action, **payload)` and `generate(l1_restore=...)` |
| Scheduler integration | [peer/scheduler.py](peer/scheduler.py) | Configuration/ownership checks; explicit controls and nonblocking request state; fail-stop boundary |
| Network workers | [peer/nonblocking.py](peer/nonblocking.py) | Persistent source listener, async target RPC; no cache/CUDA operations |
| Request lifecycle | [io_struct.py](../../managers/io_struct.py), [tokenizer](../../managers/tokenizer_manager.py), [scheduler](../../managers/scheduler.py), [common.py](../common.py) | Carry context, hold requests until READY, pump each turn, release consumer at terminal KV cleanup |
| Control protocol | [peer/control.py](peer/control.py) | Identity/layout contract, request IDs, bounded length-prefixed JSON socket messages |
| Peer endpoints | [peer/peer.py](peer/peer.py) | Source offers/receipts, target local-hit handling, imports and READ plan |
| Transaction | [peer/restore.py](peer/restore.py) | DMA completion, unmapping, publication and source release ordering |
| Source ownership | [source_lease.py](../source_lease.py) | Radix locks, physical page pins, readiness, incarnation/epoch receipts |
| Target ownership | [target_reservation.py](../target_reservation.py) | Local-prefix pins, private missing pages, canonical publication and consumer pin |
| Cache integration | [unified_radix_cache.py](../unified_radix_cache.py), [paged allocator](../allocator/paged.py) | Prevent eviction/free/reset while pages are in use |
| Pool adapter | [transport/sglang_adapter.py](transport/sglang_adapter.py) | Describe finalized MHA/NHD buffers and borrow their VMM allocations |
| Planner | [transport/layout.py](transport/layout.py) | Validate layout/pages; produce bounded, paired-contiguous byte ranges |
| Transport | [transport/engine.py](transport/engine.py), [api.py](transport/api.py), [native.cpp](transport/native.cpp) | FABRIC export/import, region ownership, asynchronous READ/WRITE and completion |

The transport core does not implement Engine scheduling or radix ownership.
Keeping a CUDA allocation alive prevents deallocation; it does **not** prevent
SGLang from recycling a logical KV page. The source and target ownership layers
supply that separate guarantee.

## Successful restore, end to end

```mermaid
sequenceDiagram
    participant T as Target scheduler
    participant S as Source scheduler
    participant N as NVLink transport
    T->>T: Match local prefix; reserve missing pages
    T->>S: Acquire complete prefix + identity/layout contract
    S->>S: Lock radix path; pin source pages; synchronize producer
    S-->>T: Lease receipt + page IDs + FABRIC descriptors
    T->>N: Import regions; READ missing suffix into private pages
    N-->>T: Completed byte count
    T->>N: Release batch; unmap imported regions
    T->>T: Publish and rematch canonical radix pages
    T->>S: Exact release receipt + unmapped acknowledgement
    S->>S: Release exports and source page lease
    T->>T: Generate normally; finish releases consumer pin
```

1. **Initialize identity.** Both callers supply `ModelIdentity`: explicit
   `weights_revision`, `adapter_revision`, SHA-256 digests for model configuration,
   position configuration and context, plus TP size/rank. The layout is included
   in the peer contract. Equality rejects incompatible peers but does not prove
   that a caller's claims match its actual weights; the model test computes its
   own hashes. The protocol has a version, request ID and 8 MiB framing limit.
2. **Match and reserve.** The receiver truncates the supplied prefix to complete
   pages. `TargetReservation` locks the existing local prefix, allocates private
   pages only for the missing suffix, and pins both. A complete local hit publishes
   locally without opening a connection or submitting DMA.
3. **Acquire the source.** The source must have the entire requested complete-page
   prefix resident, including the part already local on the target. It locks and
   pins that full prefix, then makes its producer writes ready. The Engine wrapper
   currently uses `torch.cuda.synchronize`; the native local producer-stream event
   facility alone cannot synchronize a remote writer.
4. **Offer and copy.** The source exports the adapter's memory regions and returns
   buffer bindings, source page IDs and a receipt containing request ID, source
   incarnation, cache epoch, lease ID and contract/token fingerprint. The receiver
   validates these, imports the regions, and plans READs only for the missing
   suffix. Layout, dtype, page size and capacities must match. The planner combines
   ranges only where source and destination are both contiguous, and splits at
   region boundaries. No KV bytes travel over the control socket.
5. **Publish.** Successful completion must report the expected byte count. The
   receiver releases the batch and all imports before publishing the private pages
   into radix. Publication rematches canonical indices, handles duplicate insertion
   and transfers locks/pins to the canonical path. Consumers must use those indices,
   not assume the original private page IDs survive insertion.
6. **Release source, then consume.** Only after unmapping and publication does the
   receiver acknowledge the exact receipt. The source releases exports and its page
   lease. Target consumer pins remain through generation; `finish` releases them.
   Published pages stay cache-owned and can subsequently be evicted or flushed.

For example, with page size 16, a 119-token request can restore a 112-token prefix
and compute the remaining 7 input tokens locally. If 32 prefix tokens are already
local, DMA copies only 80 tokens, although the source leases all 112. Keep the
restored prefix strictly shorter than the generation request so the final input
position is computed locally.

## Ownership and failure contract

Source receipts distinguish process incarnation and cache epoch as well as lease
identity. Exact duplicate active acquisition is idempotent; altered requests,
stale receipts and replay of released acquisition IDs are rejected. Release is
idempotent for the same completed receipt. Source lease/reset guards and target
reservation guards prevent allocator reuse, cache reset and flush while needed.
Receipt tombstones are not currently compacted.

The restore transaction progresses from `PREPARED` to `IN_FLIGHT`, then to
`PUBLISHED` or `DISCARDED`; uncertain states are `QUARANTINED`.

| Outcome | Required handling |
| --- | --- |
| Explicit source rejection before ownership escapes | Discard private target pages; no restore is published |
| Cancellation before submission | Discard safely and release any acquired ownership through the transaction |
| Successful DMA | Release batch, unmap imports, publish, acknowledge source |
| `FAILED_DRAINED` | Release batch, unmap imports, discard target reservation, acknowledge source |
| Timeout | Keep ownership; timeout is not cancellation or proof of drainage |
| Uncertain DMA, cleanup, publication or acknowledgement | Quarantine/retain ownership; no automatic free, retry or fallback |

A lower-level timed-out transaction can be waited on again. The ordinary Engine
wrapper is stricter: an exception with unresolved ownership causes scheduler
fail-stop (`os._exit(70)`). The source also fails stop if its served exchange fails
while a lease remains active. This is not a coordinated peer recovery protocol.
A partially failed publication cannot be treated as a clean, discardable private
reservation. `finish_consumer()` requires source release to have completed.

For the explicit/synchronous modes, do not issue other generation/control calls
concurrently on either Engine. A source `serve` blocks scheduling for one
acquire/release exchange. The native core
can handle disjoint batches; this does not make the Engine integration concurrent.

## Calling the ordinary Engine interface

The control sequence below assumes compatible Engines and truthful identities
have already been constructed. Use the executable
[three-Engine model test](../../../../../test/manual/l1_transfer/test-peer-model.py)
for Engine arguments, identity construction and process orchestration.

1. Call `source.l1_peer("init", identity=identity)` and
   `target.l1_peer("init", identity=identity)`.
2. Generate the source prefix normally to populate its radix cache.
3. Call `address = source.l1_peer("listen")`, then run
   `source.l1_peer("serve")` in the source's owning process while the target runs
   `target.l1_peer("restore", tokens=prefix_ids, **address)`.
   The default listener binds loopback; cross-node callers need a reachable bind
   address on a trusted network. Do not start a blocking `serve` for an all-local
   hit: that restore needs no source connection.
4. Call ordinary `target.generate(input_ids=request_ids, ...)` after restore
   returns successfully. Normal cache matching reuses the published prefix.
5. After the consumer finishes, call `target.l1_peer("finish")`.
   `l1_peer("stats")` reports pins, leases, imports, batches and available tokens.

This is explicit pre-restoration, not a remote-cache hook inside `generate`.
Application orchestration is responsible for selecting the source, establishing
that it holds the prefix, and deciding when to restore.

## Synchronous request convenience API

The Python `Engine.generate` entry now accepts a narrow `l1_restore` option:

```python
# Both peers have already run l1_peer("init", identity=...). The driver
# starts source listen/serve in its owning process before this call.
result = target.generate(
    input_ids=request_ids,
    sampling_params={"temperature": 0, "max_new_tokens": 24},
    l1_restore={"source": source_address, "prefix_tokens": 112},
)
```

This runs existing synchronous restore, ordinary non-streaming generation, then
finish. The scheduler synchronizes local GPU work before releasing consumer pins.
`result["meta_info"]["l1_restore"]` contains restore-time counters and a
`consumer_released` flag; use `l1_peer("stats")` for current counters. An all-local
hit performs no source RPC; the source address is still syntactically required.

Only one tokenized text request with `n=1`, no sessions/cache salt/LoRA/multimodal
inputs or PD bootstrap fields is accepted. `prefix_tokens` must be page-aligned
and strictly shorter than the input. Unsupported fields are rejected before
acquiring ownership. Both Engines still require exclusive serial callers and
explicit peer initialization. Do not start blocking source serve for an all-local
hit. Source startup/serve remains driver-owned.

This is Python API orchestration, not a new scheduler request queue or an
HTTP/async API. It does not reserve tail/decode growth space or implement the full
future request contract. A generate/finish exception after successful restore
shuts down the target Engine instead of freeing potentially live pages. Restore
failures retain the existing scheduler fail-stop rules; no retry/fallback is added.
The original explicit l1_peer workflow remains available.


## Minimal nonblocking scheduler path

This opt-in path admits the request before restore and keeps both schedulers
available for control messages while waiting for peer RPC or device completion:

```python
# Both Engines: ordinary TP/PP/DP1, max_running_requests=1,
# disable_overlap_schedule=True; same model, FULL/MHA/NHD cache and page size.
source.l1_peer("init", identity=identity)
target.l1_peer("init", identity=identity)
source.generate(input_ids=source_prefix, sampling_params={"max_new_tokens": 1})
address = source.l1_peer("start_service")  # once per source Engine lifetime
result = target.generate(
    input_ids=request_ids,
    sampling_params={"temperature": 0, "max_new_tokens": 24},
    l1_restore={"mode": "nonblocking", "source": address, "prefix_tokens": 112},
)
```

The Python call still returns the terminal generation result synchronously. The
nonblocking property applies to scheduler progress, not the caller's API.
Source selection and prefix availability are supplied by the caller; this does
not implement Router discovery or decisions. The default mode and explicit
listen/serve/restore/finish APIs retain their previous behavior.

The request context travels through `GenerateReqInput`, tokenizer and
`TokenizedGenerateReqInput`. After normal request validation, `SchedulerPeer`
keeps the request outside the ordinary waiting queue:

```text
WAIT_ACQUIRE -> WAIT_TRANSFER -> WAIT_RELEASE -> READY -> ordinary forward -> finish
```

- `WAIT_ACQUIRE`: reserve/pin the target prefix and private missing pages on the
  scheduler thread; a network worker connects and requests a source lease.
- Source's persistent `SourceService` only receives/sends bytes. It queues a
  request/future for `SchedulerPeer.pump`, which validates readiness, pins pages
  and exports regions on the cache owner thread. A busy source rejects before
  acquiring ownership. The service handles subsequent connections automatically.
- `WAIT_TRANSFER`: the scheduler imports the offer, builds the missing-page READ
  plan, submits once, and polls `get_transfer_status` each turn. No transport
  `wait()` is used on the successful path.
- `WAIT_RELEASE`: after confirmed completion, release the batch, unmap imports
  and publish once. Send the exact source receipt with `unmapped=True`, then poll
  the network future for the release ACK. Publication alone does not make the
  request schedulable; READY requires that ACK.
- `READY`: put the request in the ordinary queue. Normal radix matching consumes
  the restored prefix and normal prefill computes the remaining tail. At terminal
  request KV cleanup, synchronize outstanding local GPU work and release the
  consumer reservation before ordinary cache/page cleanup. The request result
  exposes restore counters, wait poll counts and `consumer_released`.
- An all-local prefix skips all three wait phases and all source RPCs.

Each normal scheduler turn ingests input, pumps peer progress, then considers
forward. While the target is waiting or the source holds a lease, **ordinary
GPU forward on that Engine is paused**. Control queries and peer messages still
progress. `stats` reports `wait_state`; destructive idle-only operations cannot
flush an owned request. Network workers never call CUDA or mutate the cache.
Readiness synchronization, import/export, cache publication and terminal GPU
synchronization are still synchronous operations: this is not a guarantee that
every scheduler operation has bounded latency.

Relevant code: `peer/nonblocking.py` (network only), `peer/scheduler.py` (request
state and source dispatch), `peer/peer.py::prepare_steps` (serialized preparation),
`peer/restore.py::poll` (READ/release completion), normal scheduler loop/request
admission, and `mem_cache/common.py::release_kv_cache` (consumer release hook).

Limits remain intentional: one shared transaction per Engine, fixed trusted
source, no grammar/session/LoRA/multimodal/PD request context, no transfer/forward
overlap, no tail/decode capacity admission policy, no retries or fallback.
Cancellation/retraction and uncertain ownership fail-stop instead of reclaiming
potentially live pages. There is no independently stoppable source service;
its listener lives with the Engine. Do not mix explicit ownership operations
with an active nonblocking request/service. Only the synchronous Python Engine
entry is covered by the model harness; HTTP/async/multi-rank integrations are not
validated. Router metadata delivery remains separate follow-up work.
