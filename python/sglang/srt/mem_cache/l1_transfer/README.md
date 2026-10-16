# Experimental NVLink transport and PD integration

This base branch contains a CUDA Driver range transport, same-layout page planner,
SGLang MHA/NHD pool adapter, and a synchronous PD backend. It does not contain
Peer prefix restoration, source leases, target reservations, or changes to the
radix cache / allocator / ordinary Engine. The package path `l1_transfer` is kept
stable so the L1 sharing branch can build on this implementation.

- `transport/`: region registration/export/import, READ/WRITE batches, completion,
  layout planning, and finalized post-capture VMM pool adapter.
- `../../disaggregation/nvlink/`: PD backend; PD owns request/page lifetimes.
- `test/registered/unit/mem_cache/l1_transfer/`: CPU transport/planner/adapter tests.
- `test/registered/unit/disaggregation/test_nvlink_pd.py`: PD control lifecycle tests.
- `test/manual/l1_transfer/`: GPU range tests and Qwen2-1.5B PD model comparison.

Build from the repository root in a CUDA development environment:

```bash
bash python/sglang/srt/mem_cache/l1_transfer/build-native.sh
```

The default artifact is `$XDG_CACHE_HOME/sglang/l1_transfer/libnvlink_transfer.so`
(or `$HOME/.cache/sglang/l1_transfer/...`). Set `NVT_BUILD` and
`NVLINK_TRANSFER_LIB` for a custom build location. No full sgl-kernel rebuild is
needed. The transport has no Mooncake or EGM runtime dependency; CUDA FABRIC VMM
and compatible NVLink devices are required. See the license and provenance section below.

With SGLang runtime dependencies installed:

```bash
PYTHONPATH=python python -m unittest discover \
  -s test/registered/unit/mem_cache/l1_transfer -v
PYTHONPATH=python python test/registered/unit/disaggregation/test_nvlink_pd.py -v
bash test/manual/l1_transfer/run-pd.sh /shared/unique-output /path/to/model
```

The PD runner builds the native transport and compares six greedy requests against
an independent ordinary Engine, including multi-chunk prefill. It needs three
GPUs. See `disaggregation/nvlink/README.md` for supported settings and fail-stop
semantics. The runner explicitly carries compatibility overrides for the older
0.5.21/cu130 experimental image; these are not normal framework defaults.

Transport GPU tests can be run with `test/manual/l1_transfer/test-gpu.py --local
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
The PD adapter currently retains its existing explicit producer synchronization.

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
