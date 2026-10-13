# Standalone EGM store (experimental)

CUDA Driver HOST_NUMA/DEVICE + FABRIC storage, without Mooncake. The Python adapter
is loaded through HiCache's existing `dynamic` backend; no factory changes
are required. Build the native library on a CUDA development machine:

```bash
g++ -std=c++17 -O2 -fPIC -shared native.cpp -I"$CUDA_HOME/include" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda -o libegm.so
export EGM_NATIVE_LIB="$PWD/libegm.so"
```

Use `hicache_storage_backend="dynamic"` with extra config:

```json
{"backend_name":"egm","module_path":"sglang.srt.mem_cache.storage.egm.hicache","class_name":"EGMHiCacheStorage","interface_v1":true,"capacity_bytes":67108864,"namespace":"model-weight-revision","numa":0}
```

Requires CUDA HOST_NUMA/FABRIC support and an accessible IMEX channel.
Supports standard FULL MHA, TP/PP/CP=1. Copies are synchronous; objects are
immutable and capacity is fixed. Explicit reader pins block clear/close.
Uncertain copy failures quarantine memory; there is no orphan-pin recovery.
Unix-socket or job-private TCP peers exchange Fabric metadata. The HiCache
path uses ordinary host memory for L2 and EGM for L3. The experimental Direct
Linker Engine path below bypasses that host staging pool.

Manual tests live in `test/manual/egm/` relative to the repository root:

```bash
python test/manual/egm/test_egm_store.py --output "$EGM_RUNROOT"
python test/manual/egm/test_egm_engine.py
```

Set `EGM_RUNROOT` to a fresh output directory and `EGM_NATIVE_LIB` to the built
library. The Engine test generates a tiny dummy-weight model, writes cache,
flushes local cache, and checks actual storage restoration and output equality.

For testing against an installed SGLang runtime, optionally set
`EGM_TEST_STORAGE_PATH="$PWD/python/sglang/srt/mem_cache/storage"`.
The manual-test bootstrap overlays only this backend package (also in spawned
children), preserving the installed runtime. Such a run validates the backend
against that installed version, not the complete checkout. Production use must
install the package in the intended runtime and does not use this bootstrap.

## HBM / DRAM tiers and remote reads

`EGMStore(capacity=dram_bytes, hbm_capacity=hbm_bytes, device=0)` owns two
optional slabs: `dram` is strict CUDA HOST_NUMA/FABRIC (EGM), and `hbm` is
CUDA DEVICE/FABRIC. At least the default tier must have nonzero capacity.
`put(key, ranges, device=True, tier="hbm")` explicitly selects placement;
`get(key, ranges, device=True)` is identical for either source tier. An existing
key cannot change size or tier. There is no automatic demotion/eviction.

```python
store = EGMStore(capacity=64 << 20, hbm_capacity=64 << 20, device=0)
# Caller retains/synchronizes source tensors before device writes.
store.put("hot", hot_ranges, device=True, tier="hbm")
store.put("warm", warm_ranges, device=True, tier="dram")
server = PeerServer(store, (bind_address, 0), secret=job_secret)
client = PeerClient(server.path, device=1, secret=job_secret)
with client.reader(["hot", "warm"]) as reader:
    reader.get("hot", hot_target_ranges, device=True)
    reader.get("warm", warm_target_ranges, device=True)
```

TCP is a job-private metadata endpoint requiring a secret of at least 32
characters. Use a trusted isolated job network: it is not TLS and does not
provide a public multi-tenant service. Unix sockets remain supported. Neither
protocol carries KV payloads. One mixed-tier lease pins the whole store until
all mappings drain and close. Errors retain the lease conservatively.

The native ABI and descriptors are version 2; rebuild `libegm.so`. The Store
lease has `version`, `epoch`, `token`, `regions[tier]` descriptors, and
`objects[key] = {tier, offset, size}`. `clear()` rotates the Store epoch, while
region epochs identify the underlying allocations. Version-1 peers are not
wire compatible. `release(token, epoch)` is explicit, not TTL based/idempotent.

HiCache extra config supports `hbm_capacity_bytes`, `write_tier`, `device`,
and `peer_secret`. `listen_path` / `peer_path` can be a Unix path or JSON
`[host, port]`. Direct Linker uses the Store's default tier without API changes.
This is a Store-owned L1/L2 data-plane prototype, not zero-copy export of a
live Engine KV pool. Via HiCache it remains an external-storage/L3 interface.

`test/manual/egm/test_tiered_fabric.py` runs two ranks with a shared fresh output
directory and a private `control-secret` file. Set native library and runtime
as above; run `--rank 0|1 --output ...` on two nodes in one Fabric/IMEX domain,
or add `--local` for two processes on one four-GPU node. It checks both tiers,
both directions, canaries, mixed pins, injected pre-copy failure, and collects
NVLink/RDMA counters in separate windows. The shared files are rendezvous and
evidence only. `test_egm_engine.py --tier hbm` exercises HBM placement through
HiCache; the default remains DRAM.

## Two Engines through Direct Linker

Both Engines require this full source checkout, the native library above,
identical model weights and KV settings, FULL MHA, TP/PP/CP/DP=1, no LoRA or
speculative decoding. Enable:

```text
--enable-unified-cache-external-linker
--unified-cache-external-linker-backend egm
--disable-cuda-graph --disable-overlap-schedule
```

Pass JSON via `--hicache-storage-backend-extra-config`. Provider example:

```json
{"namespace":"model-weight-revision-and-kv-settings","listen_path":["127.0.0.1",29590],"peer_secret":"replace-with-job-private-secret-at-least-32-characters","capacity_bytes":67108864,"hbm_capacity_bytes":67108864,"write_tier":"hbm"}
```

Consumer example (read-only remote Store):

```json
{"namespace":"model-weight-revision-and-kv-settings","peer_path":["127.0.0.1",29590],"peer_secret":"replace-with-job-private-secret-at-least-32-characters"}
```

Use the provider's reachable job-network address for another host; both hosts
must share a supported Fabric/IMEX domain. Same-host Unix paths also work.
`base_gpu_id` selects each Engine's local GPU; do not override Slurm's device
visibility. `audit_path` optionally writes JSONL operation metadata. Namespace
is a caller assertion of compatibility, not an automatic weight checksum.

The provider's write-through path copies completed KV pages into its owned
Store. The consumer matches a remote prefix during request admission, allocates
private device slots, and calls `load_private`. All copies and source reads
drain before the slots are inserted into the tree and become a request prefix.
There is no transfer/compute overlap. The normal queued path for other external
linkers is unchanged.

A lookup miss recomputes normally. Failure after a remote hit is fail-stop:
unpublished destination slots are quarantined, reset/close refuse recovery, and
the Engine must terminate. Unknown copy completion keeps the source lease and
import mappings too. Cancellation can drop a queued lookup before allocation;
there is no in-flight DMA cancellation or timeout-based pin reclamation. A
successful restore is counted as storage KV, with no CPU HostPool staging.

Run the real-Engine test on an allocated three-or-more-GPU node:

```bash
python test/manual/egm/test_dual_engine.py --tier hbm
# Use a fresh EGM_RUNROOT for the second run.
python test/manual/egm/test_dual_engine.py --tier dram
```

The test shares one saved tiny random-weight checkpoint between Engines,
compares restored generation against a cold recompute, verifies a miss and
subsequent local reuse, and injects a malformed second-page range through a
test-only metadata proxy. Request-tree tests additionally check that eviction
cannot reclaim private slots and failed loads never publish. This is prefix
reuse through Store-owned memory, not direct export of live Engine KV pools,
decode-request migration, a global directory, or an Orchestrator.
