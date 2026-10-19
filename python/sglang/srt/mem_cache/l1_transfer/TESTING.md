# L1 sharing tests and validated scope

Read [IMPLEMENTATION](IMPLEMENTATION.md) for the data path and ownership rules.
Commands below run from the SGLang repository root in its framework runtime;
no external workspace scripts or documents are required.

## Prerequisites

Use a CUDA development environment with SGLang/PyTorch dependencies, a C++17
compiler and CUDA Driver headers/libraries. GPU tests require FABRIC-capable VMM,
compatible NVLink GPUs and, for cross-node tests, an operational common IMEX/Fabric
domain. Allocate GPU resources before running these commands. Container launches
must expose GPUs, IMEX devices, the repository, model and output paths.

The native library uses ABI 3. Rebuild after changing bindings/native code; this
does not require rebuilding all of sgl-kernel. Use an absolute, fresh output path
for each run. Both nodes need the same source and shared output directory.

## CPU contracts and native build

```bash
export PYTHONPATH="$PWD/python"
export NVT_BUILD=/absolute/output/build
export NVLINK_TRANSFER_LIB="$NVT_BUILD/libnvlink_transfer.so"
bash python/sglang/srt/mem_cache/l1_transfer/build-native.sh
python -m unittest discover -s test/registered/unit/mem_cache/l1_transfer -v
```

The combined branch has 59 transport/Peer CPU contracts. GPU-backed source lease
and target reservation tests are separate; the two-node runner below invokes them.

## Components on two nodes

Within a two-node GPU allocation and configured framework environment:

```bash
srun --nodes=2 --ntasks-per-node=1 \
  bash test/manual/l1_transfer/run-tests.sh /absolute/shared/output/components
```

[run-tests.sh](../../../../../test/manual/l1_transfer/run-tests.sh) builds once,
records commit/source hashes, and runs CPU contracts, source lease and target
reservation checks, real pool mapping, Peer restoration, and local/cross-node
transport tests. Logs are under `rank-0/` and `rank-1/`; structured outputs are
under `cross-node/`, `peer/`, `transport-local/` and `transport-cross-node/`.
Use the runner's failure markers and process exit status, not merely the presence
of output files, to determine success.

For separate two-GPU transport/safety checks, after the build above:

```bash
python test/manual/l1_transfer/test-gpu.py --local --output /absolute/output/ranges
python test/manual/l1_transfer/test-resource-reuse.py > /absolute/output/reuse.log 2>&1
python test/manual/l1_transfer/test-producer-stream.py > /absolute/output/producer.log 2>&1
```

The deterministic native failure harness includes the production native state
machine and substitutes CUDA calls. Compile without `NDEBUG` and run each case in
its own process:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  test/manual/l1_transfer/test-native-failures.cpp \
  -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64/stubs" -lcuda \
  -o "$NVT_BUILD/test-native-failures"
for scenario in validation wrong-context success timeout-late reuse producer-order \
  release-event release-stream partial-seq-drained partial-seq-uncertain \
  partial-batch-drained partial-batch-uncertain record-drained record-uncertain \
  query-drained query-uncertain wait-drained wait-uncertain \
  producer-create-drained producer-record-drained producer-wait-drained \
  producer-wait-uncertain producer-wait-drain-uncertain; do
  "$NVT_BUILD/test-native-failures" "$scenario" || exit 1
done
```

Adjust the CUDA library path for your toolkit layout. These are deterministic
state-machine tests, not real GPU-reset injection or device recovery validation.

## Ordinary Engine generation

On a node with at least three available compatible GPUs:

```bash
bash test/manual/l1_transfer/run-peer-model.sh \
  /absolute/output/model /absolute/models/Qwen2-1.5B
```

[run-peer-model.sh](../../../../../test/manual/l1_transfer/run-peer-model.sh)
builds the library and runs CPU contracts plus
[test-peer-model.py](../../../../../test/manual/l1_transfer/test-peer-model.py).
It starts independent source, target and reference Engines, tests cold/partial/
all-local/repeated-cold restoration, compares generated tokens and output-token
logprobs, and verifies flush protection and reclamation. It records `run.log`,
source hashes and the source commit alongside test outputs. An archive checkout
can carry its originating commit in `.source-commit`.

This runner reflects the tested SGLang 0.5.21 CUDA 13 ARM64 image: it explicitly
sets `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` and installs isolated CUTLASS DSL
4.8.0 packages into its output cache. Inspect these compatibility choices before
using another runtime; they are not general compatibility guarantees. Post-capture
KV sizing is enabled by the runner. Build/JIT/package caches can be regenerated;
retain result files, logs and source hashes as evidence.

## Validation snapshot: 2026-10-19 (Asia/Shanghai)

The following are recorded results, not tests rerun by reading this document.
Hardware was GB200 NVL72 (four GPUs per node; two nodes for component tests).
Core commit: `79b7823c84aac47b5697bfdf877458f1b42b826a`.
L1 component commit: `dc10951ebd63c8998d070fc5c2870cba9d5e8909`.
The model run used `1cadc3b11842552276ed201ecc92f7ddbcdb24af`; the subsequent
`dc10951eb` change only corrected the Peer test's radix-key fixture from a list
to `array('q')`. Runtime code and the model driver were unchanged.

| Check | Result |
| --- | --- |
| Standalone core CPU contracts | 33 passed |
| Native deterministic failure scenarios | 23 passed |
| Core local GPU range tests | 10 passed |
| Resource reuse and dependent READ/WRITE | 360 reuse batches + 20 dependent cases passed |
| Producer stream ordering | 4 passed |
| Combined L1 CPU contracts | 59 passed |
| Source lease / target reservation GPU checks | 5 / 7 passed |
| Real pool / Peer cases | 4 / 6 passed |
| Same-node + cross-node range tests | 20 passed |
| Ordinary Engine | Four restore/generation cases passed; all 96 generated tokens matched reference |

The model run used Qwen2-1.5B BF16, FlashInfer, TP/PP/DP1, NHD, page size 16,
4096 pool tokens, one running request and overlap disabled. The source primed 160
tokens. Each target request had 119 input tokens: 112 restored/cache-hit tokens,
7 locally computed tail tokens, and 24 greedy output tokens.

| Case | Already-local tokens | Transferred bytes |
| --- | ---: | ---: |
| Cold | 0 | 3,211,264 |
| Partial | 32 | 2,293,760 |
| All local | 112 | 0 |
| Repeated cold | 0 | 3,211,264 |

Maximum observed output-token logprob difference was 0.0023036450147628784.
This establishes generated-token equality for these cases, **not full-logit
identity**, general model accuracy or a performance benefit. After restoration,
imports, exports, source leases and batches were zero, while seven target pages
remained consumer-pinned. Flush was rejected while pinned; after `finish`, flush
recovered all 4096 pool tokens on source and target.

Initial validation caught the test fixture's list/array radix-key mismatch; the
final component run passed after that correction. Raw cluster outputs are not
bundled in this repository. The scripts and recorded configuration make the
checks repeatable, but this snapshot does not replace retaining new run outputs.
Automatic remote lookup, concurrent serving, multi-rank models and failure
recovery remain outside the validated scope.

## Synchronous request API validation

Use the same environment as the ordinary Engine test, adding the mode flag:

```bash
bash test/manual/l1_transfer/run-peer-model.sh \
  /absolute/output/request-model /absolute/models/Qwen2-1.5B --request-restore
```

The four cases now call `generate(l1_restore=...)` without driver-side target
restore/finish calls. They compare tokens and cached prefix lengths, inspect
restore-time pins, verify automatic release with live stats and flush the pool.
Run without the flag to retain the explicit API regression, including flush
rejection while a consumer reservation is active. These modes use separate output
paths. CPU request tests also check validation-before-acquire and shutdown rather
than premature finish on ambiguous generate/finish errors.
