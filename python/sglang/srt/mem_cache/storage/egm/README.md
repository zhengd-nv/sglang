# EGM store: prefix KV in Grace memory, shared over NVLink (experimental)

A small KV store for GB200 / GB300 NVL72 that keeps radix-cache pages in Grace
CPU memory (EGM: CUDA `HOST_NUMA` allocations exported as FABRIC handles) and
lets any SGLang Engine in the same NVLink domain read and write them directly
with its own GPU. The copies go through `kv_peer_read`
(`python/sglang/srt/mem_cache/kv_peer_read.py`: CUDA FABRIC import, page-row
JIT kernel, grouped completion); otherwise it needs only an IMEX channel:
no Mooncake, etcd or metadata server. A domain
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
| `kv_peer_read` (outside this package) | The data plane: the service allocates the slab with `cuda_vmm_utils.create_host_numa_allocation` and hands each Engine a writable lease (`KvPeerExporter`); each Engine imports every slab once through one `KvPeerReader` and reads / writes pages with it. |

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
| `master` | Domain master address (`tcp://host:port`). Lookups and pre-write dedup become one master RPC, the Engine connects to every store the master lists, and pages that do not fit the write store go to a store the master places them on. Falls back to asking every store when the master is down. |
| `refresh_s` | With `master`: how often to poll store membership and re-import restarted stores (default 1.0 s). |
| `secret_file` | Job secret for TCP stores. |
| `namespace` | Part of every key; must differ between models or weight versions. Required. |
| `read_only` | Restore only, never offload. |
| `broadcast` | Replicated pools (MLA / DSA) under TP: rank 0 reads and broadcasts each layer to the other ranks, so the store's entry GPU sends one copy per Engine. |
| `lease_ttl_ms` | Read pin deadline; a load that outlives it is abandoned and the request is recomputed. |
| `audit_path` | Append per-lookup / load / offload timing as JSON lines. |
| `num_blocks` | CTA budget of the copy kernel (default 64; 64 already reaches the ~210 GB/s C2C limit of an EGM source on GB200). |
| `layers_per_group` | Layers per read group, or `"auto"` (default): a load up to `single_group_bytes` (128 MiB) reads every layer in one group, a larger one 4 layers per group. A layer completes when its group lands; fewer groups cut the Python-side host cost a small, CPU-bound prefill feels. |
| `single_group_bytes` | Threshold for `layers_per_group: "auto"` (default 128 MiB). |

## Semantics

- **Write**: `alloc` skips keys any store already holds (domain-wide dedup) and
  reserves slots in the write store (with a master: without evicting first, then
  on the store the master places them on, and only then in the write store with
  LRU eviction); the Engine writes the page objects with
  one `kv_peer_read` write batch and `commit`s. Reservations have a deadline; an expired slot is quarantined
  before reuse so a late copy cannot land in someone else's object.
- **Read**: a lookup asks the master, or without one every store in parallel,
  and takes each page from the first holder (write store first). `acquire` pins the whole batch or nothing;
  the Engine then submits one read batch in groups of `layers_per_group` layers
  and completes each layer as its group lands, so the forward overlaps the remaining layers. Any failure (missing key, store down, expired
  pin) is reported to the Unified Cache wrapper, which recomputes the request.
- **Liveness**: pins and reservations carry deadlines computed by the client
  before the RPC; the service reaps expired ones, so a stuck or vanished client
  cannot hold memory forever.

## Limits

- One NVLink domain. The client compares `ClusterUUID` with each store and
  refuses to start across domains (FABRIC import is not permitted there).
- Pools indexed by KV pages only: FULL MHA, MLA and DSA (including the compact
  DSA indexer layout). TP is supported; PP, attention CP, Mamba hybrids and SWA
  are not. MTP draft layers are not offloaded.
- Without a master, lookups and dedup cost one RPC per configured store, there is
  no capacity borrowing, and a restarted service is not re-imported by running
  Engines. A master fixes all three; one master handles the 36 stores of a full
  NVL72 domain (control plane measured, end to end only on two nodes).
- Eviction is page-granular LRU and happens only in the writer's own store: it
  can evict the head of a prefix and leave its tail stranded, and a store filled
  by spill-over is never refreshed.
- A failed load is recomputed except under chunked prefill and PD prefill, where
  the request still fails with 503.
