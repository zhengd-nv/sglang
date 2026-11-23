# Standalone SlotGuard prefetch gate (draft)

This checkout contains the SlotGuard source protection, dynamic prefetch linker,
request lifecycle extensions, and the fixed #49474 peer reader. No checkout of
sglang2 or a separate experiment workspace is required. The branch composition,
PR mapping and test record are in `L1UP-STATUS.md` at the repository root.
This is a runnable draft, not a claim that all lifecycle faults or hardware
configurations have passed.

The supported deployment is GLM-5.2 NVFP4, MLA/index KV, TP4, page64, Python
TreeCore, fixed FABRIC pools, two Prefill nodes and one Decode node.
Each node needs four compatible GPUs; the two Prefill nodes must share a healthy
NVLink Fabric domain with usable IMEX channels. P-to-D uses Mooncake/RDMA.
The default capacity is 2,560,000 tokens and requires the tested GB300 memory
budget. Smaller devices have not been validated.

## Prerequisites

- A compatible SGLang/CUDA environment with PyTorch, FlashInfer, Mooncake,
  transformers, requests, grpcio, msgspec, NumPy, a C++ compiler and CUDA headers.
  The GLM runs on this base used ARM64
  `lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439`, platform digest
  `sha256:763ce3cf6544d1928d6c79359da4fd3d06a116cd79cfcecf8b1527e9f7d4dcb6`,
  whose sgl-deep-gemm 0.2.8, sgl-flash-mla 1.0.5 and sglang-kernel 0.4.11 match
  this base. Using this checkout via PYTHONPATH does not replace compiled dependencies.
- Local GLM-5.2 NVFP4 weights and tokenizer files.
- The indexer and bridge from this checkout. Build from `sgl-router/` with
  `cargo build --release -p sgl-kv-indexer --bins`; binaries are under
  `sgl-router/target/release/` unless CARGO_TARGET_DIR is overridden. The
  workspace also needs the repository's top-level `rust/` crates. Binaries built
  from an older sgl-router reject `KV_INDEXER_BACKEND=tair_kvcache_kvcm`.
- A [Tair KVCache Manager](https://github.com/alibaba/tair-kvcache) binary.
  The existing deployment uses source revision `534df33b`; build its target with
  `bazelisk build //kv_cache_manager:kv_cache_manager_bin` in that repository.
  See the in-tree [KVCM setup instructions](../../../sgl-router/sgl-kv-indexer/README.md#the-tair-kvcache-backend).
- The checkout, weights, binaries and a fresh writable run directory visible at
  the same paths on all three nodes. Run one role per allocated node. Scheduling
  and allocation are supplied by the operator; these scripts do not need Slurm.

Run in a trusted private network. The generated control token protects the
SlotGuard control plane; the model and metadata endpoints are not configured
as a public authenticated service.

## Preflight and launch

From this checkout, in the compatible environment:

```bash
export PYTHONPATH="$PWD/python"
export MODEL=/models/GLM-5.2-NVFP4
python test/manual/slotguard_linker/preflight.py --model "$MODEL"
```

Set the following on each node (absolute paths, values specific to the deployment):

```bash
export MODEL=/models/GLM-5.2-NVFP4
export KVCM_BIN=/opt/kvcm/kv_cache_manager_bin
export INDEXER_BIN=/opt/sglang/sgl-router/target/release/kv-indexer-server
export BRIDGE_BIN=/opt/sglang/sgl-router/target/release/kv-indexer-bridge
export CACHE_KEY=sglang-nightly-dev-cu13-20261117-25c1c439
export CACHE_ROOT=/shared/sglang-cache
export IB_DEVICES=rdma_vf_rail0,rdma_vf_rail1,rdma_vf_rail2,rdma_vf_rail3
```

Launch these concurrently on the three allocated nodes, using one fresh shared
run directory. Each command stays alive until the gate completes or fails:

```bash
# Node P0
bash test/manual/slotguard_linker/launch.sh /shared/runs/unique-id p0
# Node P1
bash test/manual/slotguard_linker/launch.sh /shared/runs/unique-id p1
# Node D0
bash test/manual/slotguard_linker/launch.sh /shared/runs/unique-id d0
```

`HOST_IP` can override the advertised address on each node. The launcher probes
ports, generates both TP manifests and a fresh control token, provisions KVCM,
starts indexer/bridges and servers, then runs the HTTP gate on P0. Cache directories
are separated by CACHE_KEY and role. `WITH_NVIDIA_PEERMEM` defaults to 0 (dma-buf).
Only process groups created by this launcher are terminated on exit. Never reuse
an active or previously completed run directory.

The actual backend configuration is:

```json
{"linker_module_path":"sglang.srt.mem_cache.l1_transfer.slotguard_linker","linker":"SlotGuardLinker","load_mode":"prefetch","prefetch_policy":"wait_complete"}
```

No test-admin endpoint is enabled. Byte verification and trace recording are
enabled for correctness and add overhead. This launch is not a performance run.

`DRIVER=perf` starts the same deployment without byte verification or traces
and runs `perf.py` instead of the gate: per round it flushes every server and
times TTFT for a prompt set restored remotely on P1 against a fresh set
prefilled on P1 itself, plus local hits on both prefills (`driver/perf.json`).
`PERF_PLAN` holds `<tokens>x<concurrency>x<rounds>` items
(default `128000x8x3 256000x2x3 512000x2x3`).

## Results and limits

- `driver/gate.json`: four directional restores; all ten remote requests
  covered on all four TP ranks by successful byte verification; six
  remote/target-hit comparisons on P1. On main 7c21d8c701 two identical
  instances can decode the same short prompt differently, so remote/source-hit
  agreement is recorded in `driver/diag-determinism.json` but not required.
- `driver/transfer-e/summary.json`: the original cold-versus-remote verdict,
  preserved independently. A gate PASS does **not** imply this summary is PASS.
  The existing GLM runs have cold/hit output differences despite byte equality.
- `driver/flush-*.json` and `driver/pause-continue.json`: production HTTP receipts.
- `p*/trace/`, role/server logs and `source-path.txt`: raw evidence.
- `complete.txt` is written only on gate success; `failed.txt` or
  `shared/failed-*` indicates failure. All launcher exit codes must also be zero.

This covers idle pause/flush/continue, not in-flight abort/retract, source restart,
real CUDA UNCERTAIN injection or memory-pressure acceptance. The v1 wire adapter
still lacks the final explicit clique field; startup topology checks are not a
replacement for that protocol migration.

The launcher, HTTP PD dispatcher, KVCM provisioning and transfer workload are
adapted from the existing sglang2 measurement harness. The small `gate_client`
package has no imports of its old admin, performance or workspace helpers.
SlotGuard integration is a layered commit stack; see `L1UP-STATUS.md`.
Upstream refuses `/pause_generation` and `/continue_generation` unless the server has an
admin API key, so the launcher generates one for the prefill servers and the gate sends it.

## CPU and CUDA review checks

Use the SGLang runtime/test environment described above with `pytest` and
`pytest-subtests` installed. Run from this checkout:

```bash
bash test/manual/slotguard_linker/run-tests.sh collect
bash test/manual/slotguard_linker/run-tests.sh cpu --junitxml=/absolute/output/cpu.xml
bash test/manual/slotguard_linker/run-tests.sh cuda --junitxml=/absolute/output/cuda.xml
```

The CPU suite includes the retained L1 contract tests and the peer-reader CPU
suite. Run it on a CPU worker; it excludes the explicit CUDA suites.
`collect` checks the same imports without executing tests. `cuda` requires compatible CUDA devices
and FABRIC support; inspect skips before treating the result as GPU coverage.
Use the three-node launcher above for actual GLM restore/byte/maintenance proof.
The scripts require no checkout of another PR and no experiment-workspace code.

For the production ownership flow and the compatibility modules still included
in this draft, see the [review guide](../../../python/sglang/srt/mem_cache/l1_transfer/README.md).

The CPU commands need neither model weights nor a running KVCM service. Real
URC ownership tests use CPU tensors; CUDA allocator and FABRIC checks belong to
the separate CUDA suite. CPU contract tests use local loopback sockets and Gloo
workers, so allow subprocesses and local communication. The CPU launcher clears
CUDA_VISIBLE_DEVICES because the upstream test utility rejects an empty mask;
use a CPU worker for this entry point.

### Review coverage

All paths below are relative to `test/registered/unit/mem_cache/`. These are
checked-in tests, not patches supplied by a measurement workspace.

| Area | Representative tests | Evidence boundary |
|---|---|---|
| Source ownership and publication | `l1_transfer/test_slotguard_allocator.py`, `test_slotguard_service.py`, `test_slotguard_catalog.py` | CPU page generations, leases, deferred free and publication contracts |
| Target ownership and integration | `l1_transfer/test_slotguard_external_prefetch.py`, `test_slotguard_linker_runtime.py`, `test_slotguard_linker_transport.py` | CPU lifecycle, request queueing and actual Scheduler method bodies with model/IPC ports replaced |
| TP consistency and refusal | `l1_transfer/test_slotguard_linker_gloo.py`, `test_tp4_contract.py` | Four-process CPU collectives; no GPU completion proof |
| Reader and layout | `test_kv_peer_read.py`, `l1_transfer/test_kv_peer_engine.py`, `test_fixed_fabric_pool.py` | CPU protocol/layout and startup allocation ordering with CUDA ports replaced |
| GPU transfer | `test_kv_peer_read.py` (its two-process GPU test; the contract tests also run in CPU mode), `l1_transfer/test_kv_peer_engine_cuda.py`, `test_slotguard_linker_cuda.py` | Separate `cuda` mode; device/FABRIC prerequisites and skips apply |
| Live GLM deployment | This directory's `launch.sh` with `gate.py` or `perf.py` | Actual restores, hit-state output agreement, bytes and idle maintenance; TTFT with byte checks off; limits above |

The AST-based CPU fixtures run the checked-in method bodies without importing
optional GPU modules. They do not instantiate a complete live Scheduler. A
passing CPU suite does not close the live in-flight lifecycle items above.

## Transfer record lifetime limit

Each backend/incarnation retains retired attempt records for replay protection.
By default, after 4096 cumulatively accepted transfer records, it rejects new
transfers. This is not a concurrency limit or a count of successful cache hits.
Cache flush does not clear these records or replenish the capacity. Safe record
reclamation for long-running deployments remains unresolved; clearing the records
on flush would discard the existing attempt fence.
