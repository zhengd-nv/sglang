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
Unix-socket or job-private TCP peers exchange Fabric metadata. The Direct Linker
adapter is fixture-tested and is not connected to an Engine CLI selector.
The Engine path uses ordinary host memory for L2 and EGM for L3.

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
