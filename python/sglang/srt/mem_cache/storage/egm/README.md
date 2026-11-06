# EGM store: prefix KV in Grace memory, shared over NVLink (experimental)

A small KV store for GB200 / GB300 NVL72 that keeps radix-cache pages in Grace
CPU memory (EGM: CUDA `HOST_NUMA` allocations exported as FABRIC handles) and
lets any SGLang Engine in the same NVLink domain read and write them directly
with its own GPU. It needs only the CUDA driver, an IMEX channel, the Python
standard library and numpy: no Mooncake, etcd or metadata server. A domain
master (`master.py`) is optional: it adds a prefix directory, membership and
placement; without it a lookup asks every store.

An end-to-end two-node example lives in
`examples/runtime/nvl72_l2_sharing/` (`BACKEND=egm`).

## Pieces

| File | Role |
|---|---|
| `service.py` | Store service, one process per NUMA node. Owns the slab, hands out slots, pins objects for readers, evicts LRU unpinned objects. Control only: it never copies KV. Serves a Unix socket (same node) and optionally TCP with a job secret (other nodes). |
| `linker.py` | `EGMDirectLinker`, loaded through the `dynamic` external-linker backend. Imports every configured store's slab once at startup, then copies pages between its GPU pool and the slabs. |
| `master.py` | Optional domain master, one per NVLink domain. Metadata only: store registration and liveness, key -> store directory reported by the stores, and placement for pages that do not fit the local store. Not on the data path; the linker falls back to asking every store when it is down. |
| `region.py`, `native.cpp`, `egm_gather.cu` | ctypes wrapper over `libegm.so`: allocate / export / import the slab, and one gather kernel launch per group of segments. |

## Build the native library

```bash
cd python/sglang/srt/mem_cache/storage/egm
nvcc -O2 -std=c++17 -shared -Xcompiler -fPIC -DEGM_WITH_GATHER \
  -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100 \
  native.cpp egm_gather.cu -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64/stubs" -lcuda -o libegm.so
export EGM_NATIVE_LIB="$PWD/libegm.so"   # both the service and the Engine read this
```

## Run

```bash
# Each node: one service per NUMA node, seeing all GPUs (the slab is reached through the NUMA's entry GPU).
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.srt.mem_cache.storage.egm.service \
  --numa 0 --capacity-gib 64 --listen /tmp/egm-numa0.sock \
  --tcp 0.0.0.0:29600 --secret-file /shared/job/egm.secret

# Each Engine: local store first (it is the write target), then the other nodes' stores.
python -m sglang.launch_server ... --page-size 64 --hicache-write-policy write_through \
  --enable-unified-cache-external-linker --unified-cache-external-linker-backend dynamic \
  --unified-cache-external-linker-config '{"linker": "EGMDirectLinker",
    "linker_module_path": "sglang.srt.mem_cache.storage.egm.linker",
    "linker_extra_config": {"stores": ["/tmp/egm-numa0.sock", "tcp://<other node>:29600"],
                            "secret_file": "/shared/job/egm.secret", "namespace": "<model + weights id>"}}'
```

Linker settings (`linker_extra_config`):

| Key | Meaning |
|---|---|
| `stores` | Store addresses: a Unix socket path or `tcp://host:port`. Required. |
| `write_store` | Index into `stores` that offloads go to (default 0). |
| `secret_file` | Job secret for TCP stores. |
| `namespace` | Part of every key; must differ between models or weight versions. Required. |
| `read_only` | Restore only, never offload. |
| `lease_ttl_ms` | Read pin deadline; a load that outlives it is abandoned and the request is recomputed. |
| `audit_path` | Append per-lookup / load / offload timing as JSON lines. |

## Semantics

- **Write**: `alloc` skips keys any store already holds (domain-wide dedup) and
  reserves slots in the write store; the Engine copies with one gather launch
  and `commit`s. Reservations have a deadline; an expired slot is quarantined
  before reuse so a late copy cannot land in someone else's object.
- **Read**: a lookup asks every store in parallel and takes each page from the
  first holder (write store first). `acquire` pins the whole batch or nothing;
  the Engine then gathers layer by layer on a private stream so the forward
  overlaps the remaining layers. Any failure (missing key, store down, expired
  pin) is reported to the Unified Cache wrapper, which recomputes the request.
- **Liveness**: pins and reservations carry deadlines computed by the client
  before the RPC; the service reaps expired ones, so a stuck or vanished client
  cannot hold memory forever.

## Limits

- One NVLink domain. The client compares `ClusterUUID` with each store and
  refuses to start across domains (FABRIC import is not permitted there).
- FULL MHA KV pools, TP1 / PP1 only. Overlap scheduling has only been tested off.
- No capacity borrowing: when the write store is full the page is not written.
- Lookups and dedup cost one RPC per configured store; fine for a few nodes,
  a directory or key-hash routing is needed for a full 18-node domain.
- Page-granular LRU can evict the head of a prefix and leave its tail stranded.
- A restarted service is not re-imported by running Engines; restart them too.
