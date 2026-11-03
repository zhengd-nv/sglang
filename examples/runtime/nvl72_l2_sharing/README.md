# Sharing prefix KV across servers through Grace memory in an NVL72 domain

This example runs two SGLang servers on two nodes of the same NVL72 domain.
The **writer** computes a long prefix, and its KV pages are written through to
a Mooncake store that lives in the Grace CPU memory (EGM) of one node. The
**reader** starts with an empty GPU cache, finds the prefix in the store, and
copies it over NVLink instead of recomputing it. A client checks two things:
the reader hits the cache, and its output matches the writer's token for token.

```
 node 0                                      node 1 (same NVL72 domain)
 +---------------------------------+         +---------------------------+
 | mooncake_master + HTTP metadata |         |                           |
 | EGM provider (NUMA 0 Grace DRAM)|<------->| reader: sglang server     |
 | writer: sglang server (GPU 0)   | NVLink  |   (GPU 0)                 |
 +---------------------------------+  fabric |   demo_client.py          |
                                             +---------------------------+
```

No host staging and no RDMA: the reader's GPU reads the provider's EGM pages
directly over the NVLink fabric.

## What this branch contains

| Commit / path | Purpose |
|---|---|
| `[HiCache] Direct linker pool group for plain MHA KV` | Lets the direct external linker serve plain MHA models (upstream main only covers DSv4 / DSA / Mamba). Rejects DCP and packed draft KV. |
| `[Feature] Unified Cache: Recompute requests whose external linker load fails` | A failed store read retracts the request and recomputes it (`--unified-cache-external-linker-load-failure`, default `recompute`) instead of returning 503. |
| `examples/runtime/nvl72_l2_sharing/` | This example. |
| `mooncake-nvlink-grouped-scatter.patch` | Mooncake fix: the NVLink transport handles one scatter as one task. Without it every 32 KiB page fragment is polled separately; restoring a 16k-token prefix under load takes about 70 ms instead of 43 ms. |

## Requirements

- Two GB200 / GB300 nodes in the **same NVL72 domain** (same
  `nvidia-smi -q | grep ClusterUUID`), with an IMEX channel available in the
  container (`/dev/nvidia-caps-imex-channels/`). Under Slurm, `--segment=2`
  keeps both nodes in one domain on block-topology clusters.
- The SGLang nightly container this branch targets, for example
  `lmsysorg/sglang:nightly-dev-cu13-20261102-38d9e4a3`, and this branch installed
  or placed first on `PYTHONPATH` (`<repo>/python`).
- A directory both nodes can see (`WORK` below) for the Mooncake build and run
  logs.
- Tested with `Qwen/Qwen2-1.5B` at TP1 (any plain-MHA model should work).

## 1. Build Mooncake (once, inside the container)

The Mooncake wheel bundled in the image (0.3.13.post1) has no EGM store pool,
so build Mooncake from source. The script pins commit `522b716` and applies the
grouped-scatter patch:

```bash
WORK=/shared/l2-sharing ./build_mooncake.sh
```

It writes `$WORK/env.sh`, which the other scripts source. That file puts the
built `mooncake` package ahead of the image's copy on `PYTHONPATH`. If the
image lacks yaml-cpp, zstd, xxHash or msgpack-c, install them with Mooncake's
`dependencies.sh` or pass `DEPS_PREFIX=<prefix>`.

Then make the checkout importable ahead of the image's SGLang (also once,
inside the container). The script copies the Rust extensions that the image
built but git does not track. `run_slurm.sh` runs it for you:

```bash
./prepare_sglang.sh
```

## 2. Run

### Under Slurm (both nodes at once)

```bash
srun -N2 --ntasks-per-node=1 --segment=2 --gpus-per-node=4 \
  --container-image=<sglang image> --container-mounts=/shared:/shared \
  env WORK=/shared/l2-sharing MODEL=Qwen/Qwen2-1.5B \
      PYTHONPATH=<this repo>/python \
  <this repo>/examples/runtime/nvl72_l2_sharing/run_slurm.sh
```

### By hand

```bash
# every node: PYTHONPATH=<this repo>/python, after ./prepare_sglang.sh
# node 0
WORK=/shared/l2-sharing NUMAS=0 ./start_store.sh            # wait for STORE_READY
WORK=/shared/l2-sharing MASTER_HOST=<node0 IP> MODEL=Qwen/Qwen2-1.5B GPU=0 ./start_server.sh
# node 1
WORK=/shared/l2-sharing MASTER_HOST=<node0 IP> MODEL=Qwen/Qwen2-1.5B GPU=0 ./start_server.sh
python3 demo_client.py --writer http://<node0 IP>:30000 --reader http://127.0.0.1:30000
```

Start the store before any server: each server does a warm-up write when it
starts, and that write needs a segment to land in.

## 3. What to look for

`demo_client.py` prints a JSON summary and one line:

```
DEMO_RESULT PASS prefix=16384 writer_cold=...ms reader_from_pool=...ms reader_cached_tokens=16384
```

- `reader_cached_tokens` is the full prefix: the reader did not recompute it.
- `reader output matches the writer's cached output`: the bytes that came back
  are the writer's KV. The reference is the writer answering from its own GPU
  cache, not a cold prefill. A cold prefill takes a different numeric path
  and may legitimately pick another token at a near tie.
- `run_slurm.sh` also prints `READER_NODE_NVLINK_DELTA`, the NVLink bytes each
  GPU on the reader node sent and received during the demo. Only the reader
  GPU should receive anything: `prefix / page_size x bytes per page` per round
  (1.75 MiB per 64-token page for Qwen2-1.5B, so 448 MiB for 16384 tokens),
  and the client runs one untimed warm-up round plus the measured one.

## Measured on GB200

Qwen2-1.5B, TP1, page size 64, CUDA graph on. The writer serves eight
concurrent streams while the reader restores the prefix in a loop for 60 s.
Values are restore latency p50 in ms. All three columns are single-node runs.

| Prefix | This example | Mooncake without the patch | HiCache local host L2 |
|---|---|---|---|
| 2048 | 15.4 | 17.0 | 14.4 |
| 16384 | 42.9 | 70.0 | 17.9 |

Short prefixes restore about as fast as from local host memory. Long prefixes
are still about 2.4x slower, because the cost is per page fragment rather than
per byte: each page has two 32 KiB pieces per layer, so 16384 tokens means
14336 pieces. Restoring whole pages without splitting them by layer reads
16384 tokens in 7 ms in a microbenchmark.

One run of this example across two nodes (idle servers, 16384-token prefix,
16 generated tokens, each value includes the decode):

| Request | Latency |
|---|---|
| Writer, cold prefill | 111 ms |
| Writer, prefix in its own GPU cache | 55 ms |
| Reader, prefix loaded from the other node's EGM | 83 ms |

## Limits

- One NVL72 domain only. Across domains, importing another node's fabric
  memory fails (`cuMemImportFromShareableHandle` returns 800, not permitted);
  use Mooncake over RDMA there.
- Validated only at TP1 with overlap scheduling off. TP > 1, PD
  disaggregation and speculative decoding are untested.
- The direct linker cannot be combined with `--enable-hierarchical-cache`, so
  there is no local host tier: anything evicted from GPU lives only in the
  store.
- Chunked prefill still answers a failed store read with 503 instead of
  recomputing.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `start_store.sh` reports `mooncake_master exited` | The metadata (18080), RPC (50051) or metrics (19003) port is taken on that node. Set `METADATA_PORT` / `MASTER_PORT` / `METRICS_PORT`, and the same `METADATA_PORT` / `MASTER_PORT` for `start_server.sh`. |
| Provider fails with `invalid argument` | It cannot see the entry GPU of its NUMA node. Keep `PROVIDER_GPUS` at all GPUs. |
| A restarted server fails with `Duplicate rpc_meta key` | The previous instance was killed without unregistering its transfer-engine endpoint. Use a new `TE_PORT` or restart the master. |
| `flush_cache` returns 400 | The previous request is still finishing. The client retries. |
| Reader shows `reader_cached_tokens` well below the prefix | Write-through had not finished; raise `--offload-wait`. |
