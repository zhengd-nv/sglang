# Experimental L1 KV peer transfer over the NVLink core

This branch layers Engine page ownership and explicit fixed-peer restoration on
`feat/nvlink-transfer` (native ABI 3). Neither branch includes the experimental PD
backend or its tests; `feat/nvlink-transfer-pd` remains an independent, unchanged
branch. Same-layout FULL/MHA/NHD, TP/PP/DP1 and trusted serial callers are required.
Normal request admission does not automatically look up remote prefixes.

- `transport/`: CUDA Driver backend, range/region API, planner and pool adapter.
- `peer/`: control protocol, source/receiver, restore transaction and scheduler RPC.
- `../source_lease.py`, `../target_reservation.py`: page ownership/lifecycle guards.
- `test/registered/unit/mem_cache/l1_transfer/`: transport and Peer CPU contracts.
- `test/manual/l1_transfer/`: GPU transport/safety, real pool/Peer and model tests.

Build from the repository root in a CUDA development environment:

```bash
bash python/sglang/srt/mem_cache/l1_transfer/build-native.sh
```

The default artifact is `$XDG_CACHE_HOME/sglang/l1_transfer/libnvlink_transfer.so`
(or `$HOME/.cache/sglang/l1_transfer/...`). Set `NVT_BUILD` and
`NVLINK_TRANSFER_LIB` for a custom build location. No full sgl-kernel rebuild is
needed. The transport has no Mooncake or EGM runtime dependency; CUDA FABRIC VMM
and compatible NVLink devices are required. Importing through the SGLang package
requires its runtime environment. See the license and provenance section below.

With SGLang runtime dependencies installed:

```bash
PYTHONPATH=python python -m unittest discover \
  -s test/registered/unit/mem_cache/l1_transfer -v
```

Transport GPU tests run with `test/manual/l1_transfer/test-gpu.py --local
--output OUTPUT`, or one process on each of two nodes with `--rank 0/1` and a
shared output directory. Set PYTHONPATH and NVLINK_TRANSFER_LIB first.

## Transfer execution

The native ABI is version 3; rebuild the library after updating Python bindings.
CUDA 13+ builds submit independent ranges with `cuMemcpyBatchAsync`, using
stream-ordered source access. Older toolkit builds use per-range
`cuMemcpyDtoDAsync`. This is a Driver API implementation with no cudart or TE
runtime dependency. By default the caller must make source writes ready before
submission. For local producer work, use
`engine.submit_transfer(requests, producer_stream=stream.cuda_stream)`.
The native layer records a reusable event on that stream and enqueues a wait on
the copy stream, without a host wait. `None` means externally ready; `0` explicitly
selects the default stream. The handle must belong to the engine context and stay
valid until batch release. Do not append overlapping writes after the readiness
marker until the transfer completes. This does not synchronize a remote writer;
remote readiness and export leases remain the peer protocol's responsibility.

Registered regions on a device share a primary-context resource pool. Each batch
exclusively owns a nonblocking stream and completion event until release; at most
eight idle resource sets are retained, each optionally including a readiness event. Failed batches are not returned to the pool.
Idle resources are destroyed when the last associated region/batch is closed.
This preserves batch-local completion when multiple disjoint batches are active.

`wait(timeout=None)` blocks on the batch's native event. Finite timeouts poll in
C++ and periodically yield the CPU, removing the Python 100-microsecond sleep;
ctypes releases the GIL, but the engine's serialized API lock remains held.
Finite waits can consume a CPU core. Timeout does not cancel, drain or release
the transfer. Uncertain failures still quarantine affected regions.

Partial submission, completion-record and asynchronous query/wait errors drain the
copy stream before reporting `FAILED_DRAINED`. If readiness setup failed before
the dependency was enqueued, the producer stream is also drained. Any required
drain failure yields `FAILED_UNCERTAIN`: pins stay held, regions are quarantined,
and resources cannot be reused. Failed cleanup also quarantines the batch and its
regions; successfully destroyed handles are cleared immediately to avoid double
destruction. There is no automatic retry, cancellation or fallback after submission.
Error diagnostics include the original CUDA code and drain results.

For controlled ablations, set `NVT_NATIVE_WAIT=0`, `NVT_RESOURCE_POOL=0`, or
`NVT_BATCH_COPY=0` to disable that optimization (default: enabled). Set these
before starting the process; do not change them during concurrent calls.
Batching preserves the range list, with no implicit coalescing or layout packing.
Mixed READ/WRITE batches retain per-range stream submission because they can
contain local producer/consumer dependencies; CUDA batch copies are unordered.
`test/manual/l1_transfer/test-resource-reuse.py` validates simultaneous batches,
reverse completion waits and repeated resource-pool teardown on two GPUs.

## Safety tests

`test/manual/l1_transfer/test-producer-stream.py` checks delayed producer writes,
byte correctness, timeout ownership and reuse on two GPUs. Run it with the same
`PYTHONPATH` and `NVLINK_TRANSFER_LIB` settings as the GPU range tests.

`test/manual/l1_transfer/test-native-failures.cpp` includes the production native
state machine and overrides CUDA calls in the test executable. Build with C++17,
CUDA headers and `-lcuda`, without `NDEBUG`; run each scenario in a fresh process.
It tests partial issue, event record/query/wait failures, readiness setup failures,
drain failures, timeout followed by completion, and cleanup failure isolation.
These deterministic tests do not inject real GPU resets or validate device recovery.
Python contract tests separately check handle ownership and failure propagation.

## License and provenance

This distribution contains code adapted from the local SGLang experimental
implementation python/sglang/srt/mem_cache/storage/egm/native.cpp.
Copyright contributors to SGLang project
Licensed under the Apache License, Version 2.0 (see the [repository LICENSE](../../../../../LICENSE)).

Reference repository: https://github.com/sgl-project/sglang
Local reference revision: 31ff9b5d42a6898b33d379a787ab0cd07d15ba73
Original local addition: a2276a0ee8b6aee0400d506ce03071f18d617ace
These are local experimental commits, not a claim of upstream acceptance
or availability from that remote.

Adapted portions: CUDA Driver error handling, context/region RAII, VMM
allocation/export/import/mapping and C ABI exception conversion patterns.
Modifications: device-only allocation, single-device context, asynchronous
range batches, CUDA event completion, active-region tracking and nvt_* ABI.

Mooncake Transfer Engine and SGLang PD adapters were design references for
range/batch interfaces and completion contracts. No Mooncake, NIXL or MORI
implementation is bundled in this distribution. This notice records the
local adaptation; it is not an upstream SGLang NOTICE file.

Batch copy submission and resource reuse were informed by Mooncake classic TE
`nvlink_transport.cpp` at `97abb51488208b97b1f19a775df70a03e861241b` (Apache-2.0).
The implementation here uses CUDA Driver APIs, region-scoped resource ownership
and per-batch completion events; no TE source is incorporated.

The paired-contiguous planning in `transport/layout.py` follows SGLang
python/sglang/srt/disaggregation/common/utils.py::group_concurrent_contiguous
at the same reference revision, under Apache-2.0. The new planner uses only
the Python standard library and adds layout/capacity checks, byte ranges and
region binding. Copyright contributors to SGLang project.

## Peer ownership contract

The caller supplies a finalized VMM pool, FULL cache, trusted model/context
identity and a control connection. Source dispatch and receiver operations must
run serially on the cache owner thread. `prepare(tokens)` returns a private target
reservation and optional READ transaction. After successful `start()/wait()`, use
`reservation.indices`; call `finish_consumer()` only when the consumer no longer
accesses those pages. A full local hit needs no RPC or DMA.

Known limitations: trusted fixed peers only, no authentication/discovery or
reconnect recovery; contract fields are caller supplied. Unknown transfer or
publication state requires fail-stop and retaining ownership. Non-page-aligned
tails require local computation. No cross-layout conversion, fallback, multi-rank
consensus, full-logit validation or automatic remote prefix lookup yet.


## Explicit ordinary Engine prefix restoration

`Engine.l1_peer(action, **payload)` executes on the idle scheduler's cache-owner
thread. Initialize both Engines with `init(identity=...)`, where `identity` is a
`ModelIdentity` dictionary covering the actual weights, model/position settings,
adapter and context. The caller is responsible for truthful matching identities.
Use trusted peers with identical FULL MHA/NHD layout and a finalized VMM pool,
TP/PP/DP1, no LoRA/speculation, and `max_running_requests=1`.

1. Generate the source prefix normally so that its full pages enter radix cache.
2. Source: `address = source.l1_peer("listen")`; run
   `source.l1_peer("serve")` in its owning process concurrently with step 3.
   This blocks source scheduling until the one acquire/release exchange finishes.
3. Target: `target.l1_peer("restore", tokens=prefix_ids, **address)`.
   Only complete pages are restored. Existing local prefix pages are retained;
   missing pages are read directly into private HBM, then published/rematched to
   canonical radix indices. A fully local hit requires neither address nor RPC.
4. Target: ordinary `generate(input_ids=request_ids, ...)` reuses the published
   prefix and computes the remaining tail. Keep the restored prefix strictly
   shorter than the request so the last input position is computed locally.
5. After generation completes: `target.l1_peer("finish")` releases the consumer
   pin. `stats` exposes leases, imports, batches, available tokens and page pins.

Do not submit other control/generation requests concurrently on either Engine.
The control connection transfers descriptors and token/page metadata, not KV bytes.
Uncertain ownership or completion terminates the scheduler; there is no automatic
retry/fallback. This is an explicit fixed-peer integration, not automatic remote
prefix discovery. Cache salts, adapters and other cache-key namespaces are not
supported by this interface. Do not update weights after peer initialization.

`test/manual/l1_transfer/run-peer-model.sh OUTPUT MODEL` builds the transport and
runs the combined transport/Peer CPU contracts plus a three-Engine Qwen2-1.5B comparison (source, restored
target, independent reference). The tested configuration uses BF16, page size 16,
FlashInfer and post-capture VMM sizing. Four cases cover cold, partial local,
fully local and repeated cold restoration, generation token equality, observed
output-token logprob differences, flush protection and pool reclamation. The
runner contains a compatibility override for the older 0.5.21 test image; this is
not a claim of compatibility with arbitrary model/runtime configurations.

For component validation on two allocated NVLink-connected nodes with a shared
output directory, run one task per node:

```bash
srun --nodes=2 --ntasks-per-node=1 \
  bash test/manual/l1_transfer/run-tests.sh /shared/unique-output-directory
```

This covers source/target page protection, real pool/Peer transfers and same-/cross-node
range transport. Safety tests above additionally exercise ABI 3 resource reuse,
producer ordering and deterministic native failure cleanup.
