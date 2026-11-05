# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Direct linker over standalone EGM store services (FULL MHA, MLA, DSA; TP, no PP / CP).

Loaded through the ``dynamic`` external-linker backend::

    --enable-unified-cache-external-linker --unified-cache-external-linker-backend dynamic
    --unified-cache-external-linker-config '{"linker": "EGMDirectLinker",
        "linker_module_path": "sglang.srt.mem_cache.storage.egm.linker",
        "linker_extra_config": {"stores": ["/tmp/egm-numa0.sock", "tcp://10.0.0.2:29600"],
                                "secret_file": "/shared/job/egm.secret", "namespace": "<model+weights>",
                                "master": "tcp://10.0.0.1:29700"}}'

The Engine never owns EGM: the services (``service.py``) do, so the Engine can
run CUDA Graph and read any NUMA's slab, on this node or another node of the same
NVLink domain. One page is one object holding every pool entry of the page, in
entry order: K of every layer then V (MHA), or latent KV of every layer then
indexer K (DSA). A head-sharded pool (MHA) keys its objects per TP rank and
every rank writes its own; a replicated pool (MLA / DSA) shares one object across
ranks and only rank 0 writes it. Every rank reads by default; with
``"broadcast": true`` rank 0 reads and broadcasts each layer to the other ranks.

Placement and discovery:

* With ``master`` (``master.py``): a lookup and the pre-write dedup are one
  master RPC each; the Engine connects to every store the master lists and
  re-imports a store's slab when it restarts. Offloads go to ``write_store``
  without evicting; pages that do not fit go to the store the master places them
  on, and only then to ``write_store`` with LRU eviction. If the master is
  unreachable, the Engine falls back to the mode below.
* Without a master: offloads go to ``write_store`` (by default the first
  configured store, normally this GPU's NUMA); a lookup asks every connected
  store in parallel and loads each page from the first store that holds it.

Loads: one lease per store, then one gather launch per layer on a private
stream; layer ``l`` completes as soon as its copy lands, so the forward overlaps
the remaining layers. Offloads run on their own thread: ``alloc`` (skips keys
already stored), one gather launch, ``commit``.
"""

import hashlib
import json
import logging
import os
import socket
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)

from .service import EXISTS, FULL, StoreClient, cluster_uuid

logger = logging.getLogger(__name__)

_DIRECTION_TO_REGION, _DIRECTION_FROM_REGION = 2, 3
_METHODS = {"batchapi": 1, "kernel": 2}
# Arbitrary bound on remembered key -> store routes between lookup and load.
_ROUTE_CACHE_KEYS = 1 << 20
# Arbitrary; one thread per store asked in parallel, an NVL72 has at most 36 stores.
_FANOUT_THREADS = 36
_MASTER_ERRORS = (ConnectionError, RuntimeError, PermissionError)


def _ms(start, end):
    return round((end - start) * 1e3, 3)


class _Layout:
    """Vectorized range metadata for one object per page over every pool entry.

    The object holds each entry's buffers in entry order, e.g. K of every layer
    then V for MHA, or latent KV of every layer then indexer K for DSA. ``rows``
    is uint64 [entries, pages]: an entry's row of a page (its first token for a
    token-row pool, the page itself for a page-row pool such as the indexer).
    """

    def __init__(self, entries, num_layers):
        base, stride, size, owner = [], [], [], []
        per_layer = [[] for _ in range(num_layers)]
        for e, entry in enumerate(entries):
            if not entry.packed or len(entry.components) != 1:
                raise ValueError(
                    f"EGM linker needs one packed component group per pool ({entry.name})"
                )
            first = len(base)
            for b, s, n in entry.buffer_meta[0]:
                base.append(b)
                stride.append(s)
                size.append(n)
                owner.append(e)
            for layer in range(num_layers):
                mapped = entry.layer_mapping.get(layer)
                if mapped is None:
                    continue
                mapped = [mapped] if isinstance(mapped, int) else list(mapped)
                per_layer[layer] += [first + i for i in mapped]
        if len({len(x) for x in per_layer}) != 1:
            raise ValueError("every layer must map to the same number of buffers")
        self.all_base = np.array(base, dtype=np.uint64)
        self.all_stride = np.array(stride, dtype=np.uint64)
        self.all_size = np.array(size, dtype=np.uint64)
        self.all_off = np.concatenate([[0], np.cumsum(size)[:-1]]).astype(np.uint64)
        self.all_owner = np.array(owner, dtype=np.int64)
        self.object_bytes = int(sum(size))
        index = np.array(per_layer, dtype=np.int64)  # [layers, items]
        self.layer_base, self.layer_stride = (
            self.all_base[index],
            self.all_stride[index],
        )
        self.layer_size, self.layer_off = self.all_size[index], self.all_off[index]
        self.layer_owner = self.all_owner[index]

    def load_ranges(self, rows, objects):
        """objects: uint64 [pages]. Returns layer-major flat arrays and per-layer group ends."""
        layers, items = self.layer_base.shape
        # [layers, items, pages] -> [layers, pages, items]: each item reads its own entry's row.
        item_rows = rows[self.layer_owner].transpose(0, 2, 1)
        ptr = self.layer_base[:, None, :] + item_rows * self.layer_stride[:, None, :]
        off = objects[None, :, None] + self.layer_off[:, None, :]
        size = np.broadcast_to(self.layer_size[:, None, :], ptr.shape)
        per_layer = rows.shape[1] * items
        ends = np.arange(1, layers + 1, dtype=np.uint64) * per_layer
        return (
            off.reshape(-1),
            ptr.reshape(-1),
            np.ascontiguousarray(size).reshape(-1),
            ends,
        )

    def store_ranges(self, rows, objects):
        item_rows = rows[self.all_owner].T  # [pages, buffers]
        ptr = self.all_base[None, :] + item_rows * self.all_stride[None, :]
        off = objects[:, None] + self.all_off[None, :]
        size = np.broadcast_to(self.all_size[None, :], ptr.shape)
        return off.reshape(-1), ptr.reshape(-1), np.ascontiguousarray(size).reshape(-1)


class _TimedCounter(LayerWiseLoadCounter):
    """Records how long the forward thread actually blocks on each load batch (M8 "exposed wait")."""

    def __init__(self, num_layers, on_layer_ready=None):
        super().__init__(num_layers, on_layer_ready)
        self.exposed = {}

    def wait_until(self, threshold):
        index = self.consumer_index
        start = time.perf_counter()
        super().wait_until(threshold)
        if index in self.futures:
            self.exposed[index] = (
                self.exposed.get(index, 0.0) + time.perf_counter() - start
            )


class _Stores:
    """Connected stores by store id; a replaced client is closed once no copy still uses it.

    A store that restarts owns a new slab; its old import must stay mapped until
    every in-flight gather through it has drained.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._clients = {}  # store_id -> StoreClient
        self._busy = Counter()  # id(client) -> holders
        self._retired = {}  # id(client) -> client still held

    def ids(self):
        with self._lock:
            return list(self._clients)

    def get(self, store_id):
        with self._lock:
            return self._clients.get(store_id)

    def install(self, client):
        with self._lock:
            old = self._clients.get(client.store_id)
            self._clients[client.store_id] = client
            if old is not None:
                self._retired[id(old)] = old
                self._close_idle(old)

    def hold(self, store_id):
        """The current client of ``store_id``, kept open until ``unhold``; None if unknown."""
        with self._lock:
            client = self._clients.get(store_id)
            if client is not None:
                self._busy[id(client)] += 1
            return client

    def unhold(self, client):
        with self._lock:
            self._busy[id(client)] -= 1
            self._close_idle(client)

    def _close_idle(self, client):
        if id(client) in self._retired and self._busy[id(client)] <= 0:
            del self._retired[id(client)]
            self._busy.pop(id(client), None)
            client.region.close()


class EGMDirectLinker(UnifiedCacheLinker):
    def __init__(
        self, server_args, params, *, components, extra_config=None, forward_stream=None
    ):
        config = dict(extra_config or {})
        allowed = {
            "store_path",
            "stores",
            "write_store",
            "secret_file",
            "lease_ttl_ms",
            "namespace",
            "copy_mode",
            "audit_path",
            "fault_dir",
            "read_only",
            "master",
            "refresh_s",
            "broadcast",
        }
        if set(config) - allowed:
            raise ValueError(
                f"Unknown EGM linker settings: {sorted(set(config) - allowed)}"
            )
        addresses = list(
            config.get("stores")
            or ([config["store_path"]] if config.get("store_path") else [])
        )
        if not addresses or not config.get("namespace"):
            raise ValueError("EGM linker needs stores (or store_path) and namespace")
        if params.pp_size != 1 or params.attn_cp_size != 1:
            raise ValueError("EGM linker supports PP1 / attention CP1 only")
        self.forward_stream = forward_stream
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.entries = self.pool_group.entries
        # One object per page holds every entry, so every entry must be indexed by KV pages.
        if any(e.indices_from_pool != PoolName.KV for e in self.entries):
            raise ValueError(
                "EGM linker supports pools indexed by KV pages only (FULL MHA, MLA, DSA), "
                f"got {sorted((e.name.value, e.indices_from_pool.value) for e in self.entries)}"
            )
        self.num_layers = self.pool_group.num_layers
        self.layout = _Layout(self.entries, self.num_layers)
        self.device = self.entries[0].kv_buffer[0].device.index
        self._init_tp(params)
        self.prefix = self._key_prefix(config["namespace"])
        self.method = _METHODS[config.get("copy_mode", "kernel")]
        self.audit_path = config.get("audit_path")
        if self.audit_path and self.tp_size > 1:
            stem, dot, suffix = self.audit_path.rpartition(".")
            self.audit_path = (
                f"{stem}.tp{self.tp_rank}.{suffix}"
                if dot
                else f"{self.audit_path}.tp{self.tp_rank}"
            )
        self.fault_dir = config.get("fault_dir")
        # A consumer-only Engine: restores from the store but never offloads to it.
        self.read_only = bool(config.get("read_only", False))
        self.lease_ttl_ms = config.get("lease_ttl_ms")
        self.secret = (
            Path(config["secret_file"]).read_text().strip()
            if config.get("secret_file")
            else None
        )
        self.master = None
        if config.get("master"):
            from .master import MasterClient

            self.master = MasterClient(config["master"], secret=self.secret)
        start = time.perf_counter()
        self.domain = (
            cluster_uuid()
            if self.master is not None or any(a.startswith("tcp://") for a in addresses)
            else None
        )
        self.stores = _Stores()
        self._addresses = {}  # store_id -> configured address (Unix socket for local stores)
        configured = [self._connect(a) for a in addresses]
        for client, address in zip(configured, addresses):
            self.stores.install(client)
            self._addresses[client.store_id] = address
        write_client = configured[int(config.get("write_store", 0))]
        self.write_id, self.numa = write_client.store_id, write_client.numa
        self.host = socket.gethostname()
        self._routes = OrderedDict()  # key -> id of the store that held it at lookup
        self._fanout_pool = ThreadPoolExecutor(
            max_workers=_FANOUT_THREADS, thread_name_prefix="egm-lookup"
        )
        connect_import_ms = (time.perf_counter() - start) * 1e3

        self.broadcaster = None
        self.broadcast_loads = {}
        if config.get("broadcast") and self.rank_replicated and self.tp_size > 1:
            self.broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
        self.layer_done_counter = _TimedCounter(
            self.num_layers,
            self._broadcast_loaded_layers if self.broadcaster is not None else None,
        )
        self.pending_loads = {}
        self.load_queue, self.offload_queue, self.offload_results = (
            Queue(),
            Queue(),
            Queue(),
        )
        self.stats = Counter(lookup=0, load=0, offload=0)
        self._stopping = threading.Event()
        self.load_thread = threading.Thread(
            target=self._load_loop, daemon=True, name="egm-load"
        )
        self.offload_thread = threading.Thread(
            target=self._offload_loop, daemon=True, name="egm-offload"
        )
        self.load_thread.start()
        self.offload_thread.start()
        self.membership_thread = None
        if self.master is not None:
            self.refresh_s = float(config.get("refresh_s", 1.0))
            self._sync_members()
            self.membership_thread = threading.Thread(
                target=self._membership_loop, daemon=True, name="egm-members"
            )
            self.membership_thread.start()
        # The slab is imported once here, so per-load "lease" is the acquire RPC only.
        self._audit(
            "init",
            device=self.device,
            numa=self.numa,
            prefix=self.prefix,
            object_bytes=self.layout.object_bytes,
            pools=[
                dict(name=e.name.value, buffers=len(e.kv_buffer)) for e in self.entries
            ],
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            rank_replicated=self.rank_replicated,
            offload_owner=self.offload_owner,
            broadcast=self.broadcaster is not None,
            method=self.method,
            stores=[
                dict(store_id=c.store_id, address=a, host=c.host, numa=c.numa)
                for c, a in zip(configured, addresses)
            ],
            write_store=self.write_id,
            master=config.get("master"),
            cluster_uuid=self.domain,
            connect_import_ms=round(connect_import_ms, 3),
        )
        logger.info(
            "EGM direct linker: stores=%s write_store=%s master=%s device=%d object=%d B "
            "pools=%s tp=%d/%d replicated=%s broadcast=%s",
            addresses,
            self.write_id,
            config.get("master"),
            self.device,
            self.layout.object_bytes,
            [e.name.value for e in self.entries],
            self.tp_rank,
            self.tp_size,
            self.rank_replicated,
            self.broadcaster is not None,
        )

    def _init_tp(self, params):
        group = params.attn_tp_cache_group or params.tp_cache_group
        self.tp_group = group
        self.tp_rank, self.tp_size = 0, 1
        if torch.distributed.is_initialized():
            self.tp_rank = torch.distributed.get_rank(group=group)
            self.tp_size = torch.distributed.get_world_size(group=group)
        # MLA / DSA KV is identical on every rank: rank 0 writes it, every rank may read it.
        self.rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not self.rank_replicated or self.tp_rank == 0

    def _key_prefix(self, namespace):
        schema = [
            namespace,
            self.page_size,
            [
                [
                    entry.name.value,
                    sorted(entry.layer_mapping.items()),
                    [(str(t.dtype), list(t.shape[1:])) for t in entry.kv_buffer],
                ]
                for entry in self.entries
            ],
        ]
        if not self.rank_replicated:
            schema.append(f"tp{self.tp_rank}/{self.tp_size}")  # A head-sharded pool
        return hashlib.sha256(json.dumps(schema).encode()).hexdigest()[:32] + ":"

    def _audit(self, op, **fields):
        if self.audit_path:
            with Path(self.audit_path).open("a") as stream:
                stream.write(
                    json.dumps(dict(op=op, time_ns=time.time_ns(), **fields)) + "\n"
                )

    def _keys(self, keys):
        return [self.prefix + key for key in keys]

    def _connect(self, address):
        return StoreClient(
            address,
            device=self.device,
            secret=self.secret,
            local_cluster_uuid=self.domain,
        )

    # ---- membership (master only) ----

    def _membership_loop(self):
        while not self._stopping.wait(self.refresh_s):
            self._sync_members()

    def _sync_members(self):
        """Import every listed store not imported yet, or imported under an older epoch."""
        try:
            members = self.master.rpc("members")
        except _MASTER_ERRORS as error:
            self._audit("members_failed", error=str(error))
            return
        for member in members:
            current = self.stores.get(member["store_id"])
            if current is not None and current.epoch == member["epoch"]:
                continue
            address = self._addresses.get(member["store_id"], member["address"])
            if not address.startswith("tcp://") and member["host"] != self.host:
                continue  # A Unix-only store on another node is out of reach.
            try:
                client = self._connect(address)
            except (
                ConnectionError,
                RuntimeError,
                ValueError,
                PermissionError,
            ) as error:
                self._audit("import_failed", store=member["store_id"], error=str(error))
                continue
            if client.epoch != member["epoch"]:
                client.region.close()  # Restarted again in between; the next round catches up.
                continue
            self.stores.install(client)
            self.stats["imports"] += 1
            self._audit(
                "import",
                store=client.store_id,
                epoch=client.epoch,
                replaced=current is not None,
            )

    # ---- lookup / load ----

    def lookup(self, rid, transfers):
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        if kv is None or not kv.keys:
            return []
        start = time.perf_counter()
        keys = self._keys(kv.keys)
        holders, source = self._locate(rid, keys)
        lookup_ms = (time.perf_counter() - start) * 1e3
        for key, holder in zip(keys, holders):
            self._remember(key, holder)
        self.stats["lookup"] += 1
        self._audit(
            "lookup",
            rid=rid,
            pages=len(kv.keys),
            hit_pages=len(holders),
            stores_hit=sorted(set(holders)),
            source=source,
            lookup_ms=round(lookup_ms, 3),
        )
        return list(range(1, len(holders) + 1))

    def _locate(self, rid, keys):
        """Store id per page of the longest held prefix, and where the answer came from."""
        if self.master is not None:
            try:
                got = self.master.rpc(
                    "lookup", keys=keys, host=self.host, numa=self.numa
                )
            except _MASTER_ERRORS as error:
                self._audit("master_failed", rid=rid, call="lookup", error=str(error))
            else:
                known = set(self.stores.ids())
                holders = []
                for page in got["pages"]:
                    if got["stores"][page] not in known:
                        break  # Not imported yet: a miss until the next membership round.
                    holders.append(got["stores"][page])
                return holders, "master"
        return self._locate_fanout(rid, keys), "fanout"

    def _store_order(self):
        return [self.write_id] + [s for s in self.stores.ids() if s != self.write_id]

    def _exists_on(self, rid, store_ids, keys):
        """One exists RPC per store, in parallel; None for a store that is down (a miss)."""

        def ask(store_id):
            client = self.stores.get(store_id)
            try:
                return client.exists_many(keys)
            except (ConnectionError, RuntimeError) as error:
                self._audit("lookup_failed", rid=rid, store=store_id, error=str(error))
                return None

        if len(store_ids) == 1:
            return [ask(store_ids[0])]
        return list(self._fanout_pool.map(ask, store_ids))

    def _locate_fanout(self, rid, keys):
        order = self._store_order()
        per_store = self._exists_on(rid, order, keys)
        holders = []
        for i in range(len(keys)):
            # Prefer the store this Engine writes to (local NUMA), then the rest.
            holder = next(
                (
                    s
                    for s, flags in zip(order, per_store)
                    if flags is not None and flags[i]
                ),
                None,
            )
            if holder is None:
                break
            holders.append(holder)
        return holders

    def _held_elsewhere(self, keys):
        """Pages some store other than ``write_store`` holds, so this Engine skips writing them."""
        if self.master is not None:
            try:
                got = self.master.rpc(
                    "lookup", keys=keys, host=self.host, numa=self.numa, prefix=False
                )
            except _MASTER_ERRORS as error:
                self._audit("master_failed", call="dedup", error=str(error))
            else:
                stores = got["stores"]
                return np.array(
                    [p >= 0 and stores[p] != self.write_id for p in got["pages"]],
                    dtype=bool,
                )
        held = np.zeros(len(keys), dtype=bool)
        others = [s for s in self.stores.ids() if s != self.write_id]
        for flags in self._exists_on(None, others, keys) if others else ():
            if flags is not None:
                held |= np.asarray(flags, dtype=bool)
        return held

    def _remember(self, key, holder):
        self._routes[key] = holder
        self._routes.move_to_end(key)
        while len(self._routes) > _ROUTE_CACHE_KEYS:
            self._routes.popitem(last=False)

    def load(self, rid, transfers):
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        by_name = self._by_entry(expanded)
        if by_name is None:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"EGM load for rid={rid} is already queued")
        self.pending_loads[rid] = by_name
        return True

    def _by_entry(self, expanded):
        """{pool name: transfer} when every entry of the page object is present, else None."""
        by_name = {t.name: t for t in expanded if t.keys}
        if set(by_name) != set(self.pools):
            # A partial set cannot fill a page object; the caller recomputes (load) or skips (offload).
            if PoolName.KV in by_name:
                logger.warning(
                    "EGM linker skips a transfer without every pool: %s",
                    sorted(n.value for n in by_name),
                )
            return None
        return by_name

    def _entry_rows(self, by_name):
        """uint64 [entries, pages]; runs on a linker thread (moves the indices to the host)."""
        return np.stack(
            [
                np.asarray(
                    entry.prepare_locations(by_name[entry.name].host_indices),
                    dtype=np.uint64,
                )
                for entry in self.entries
            ]
        )

    def cancel_queued_load(self, rid):
        return self.pending_loads.pop(rid, None) is not None

    def start_layer_wise_loading(self):
        broadcaster = self._agreed_broadcaster()
        if not self.pending_loads:
            return -1
        pending, self.pending_loads = self.pending_loads, {}
        index = self.layer_done_counter.update_producer()
        ready = self.record_transfer_ready()
        if broadcaster is not None:
            prepared = broadcaster.prepare_broadcast(
                {
                    name: torch.cat([t[name].host_indices for t in pending.values()])
                    for name in self.pools
                },
                torch.cuda.current_stream(),
            )
            self.broadcast_loads[index] = (0, prepared, ready)
            if not broadcaster.is_src:
                # The data arrives through the source's per-layer broadcast on the forward stream.
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(index, layer)
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put((index, pending, ready, time.perf_counter()))
        self.stats["load"] += len(pending)
        return index

    def _agreed_broadcaster(self):
        """The broadcaster when every rank queued the same logical loads, else None (all ranks read)."""
        if self.broadcaster is None:
            return None
        # Collective on every rank, empty batches included, before deciding.
        plan = [
            (rid, [(name.value, list(t.keys)) for name, t in by_name.items()])
            for rid, by_name in self.pending_loads.items()
        ]
        digest = hashlib.sha256(repr(plan).encode()).digest()
        digests = [None] * self.tp_size
        torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
        if any(other != digest for other in digests):
            logger.warning("EGM load plans differ across ranks; every rank reads.")
            return None
        return self.broadcaster

    def _broadcast_loaded_layers(self, index, threshold):
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, ready = batch
        if first == 0:
            ready.wait(torch.cuda.current_stream())
        # The forward may wait on a layer more than once, or skip layers; broadcast each once, in order.
        for layer in range(first, threshold + 1):
            self.broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (max(first, threshold + 1), prepared, ready)

    def finish_layer_wise_loading(self, counter_index):
        landed = self.layer_done_counter.finish(counter_index)
        exposed = self.layer_done_counter.exposed.pop(counter_index, 0.0)
        self._audit(
            "load_finish",
            index=counter_index,
            landed=landed,
            exposed_wait_ms=round(exposed * 1e3, 3),
        )
        return landed

    def _load_loop(self):
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                index, pending, ready, queued = task
                try:
                    self._load_batch(index, pending, ready, queued)
                except BaseException as error:
                    self.layer_done_counter.fail(index, error)
                    logger.exception("EGM layer-wise load batch failed")
            finally:
                self.load_queue.task_done()

    def _take_fault(self, name):
        if not self.fault_dir:
            return None
        trigger = os.path.join(self.fault_dir, name)
        try:
            with open(trigger) as f:
                spec = json.load(f)
            os.unlink(trigger)
        except FileNotFoundError:
            return None
        with open(os.path.join(self.fault_dir, "injected.jsonl"), "a") as f:
            f.write(
                json.dumps(dict(time=time.time(), pid=os.getpid(), fault=name, **spec))
                + "\n"
            )
        return spec

    def _load_batch(self, index, pending, ready, queued):
        t0 = time.perf_counter()
        ready.synchronize()
        transfers = list(pending.values())
        keys = [k for t in transfers for k in self._keys(t[PoolName.KV].keys)]
        rows = np.concatenate([self._entry_rows(t) for t in transfers], axis=1)
        fault = self._take_fault("fail-next-load")
        mode = None if fault is None else fault.get("mode", "copy")
        if mode == "evict":
            self.stores.get(self._routes.get(keys[0], self.write_id)).drop(keys)
        t1 = time.perf_counter()
        # [(client, page positions, token, offsets, deadline)], each client held until released.
        groups = self._acquire_all(keys)
        t2 = time.perf_counter()
        batches = []
        try:
            ranges = [
                (
                    client,
                    self.layout.load_ranges(
                        rows[:, pos], np.asarray(offsets, dtype=np.uint64)
                    ),
                )
                for client, pos, _, offsets, _ in groups
            ]
            t3 = time.perf_counter()
            fail_layer = int(fault["layer"]) if mode == "copy" else None
            for client, (off, ptr, size, ends) in ranges:
                batches.append(
                    client.region.submit(
                        off, ptr, size, _DIRECTION_FROM_REGION, ends, method=self.method
                    )
                )
            t4 = time.perf_counter()
            if mode == "stall":
                time.sleep(
                    float(fault.get("ms", 0)) / 1e3
                )  # A stuck reader: its lease must expire.
            deadline = min(g[4] for g in groups)
            first = None
            for layer in range(self.num_layers):
                for batch in batches:
                    batch.wait(layer)
                if fail_layer is not None and layer >= fail_layer:
                    raise RuntimeError(f"injected EGM load failure at layer {layer}")
                if time.monotonic() > deadline:
                    # The store may already have unpinned and reused these slots.
                    raise RuntimeError(f"EGM lease expired before layer {layer} landed")
                self.layer_done_counter.complete(index, layer)
                first = first or time.perf_counter()
            t5 = time.perf_counter()
        finally:
            for batch in batches:
                batch.free()  # Drains every group, including those after a failure.
            for client, _, token, _, _ in groups:
                self._release_quietly(client, token)
                self.stores.unhold(client)
        t6 = time.perf_counter()
        # M8 segments: lease_ms / read_ms (build + submit + every layer landed) / finalize_ms;
        # lookup_ms is in the lookup event, exposed_wait_ms in load_finish (joined by index).
        self._audit(
            "load",
            index=index,
            reqs=len(transfers),
            pages=len(keys),
            stores=[dict(store=g[0].store_id, pages=len(g[1])) for g in groups],
            queue_ms=_ms(queued, t0),
            ready_ms=_ms(t0, t1),
            lease_ms=_ms(t1, t2),
            read_ms=_ms(t2, t5),
            build_ms=_ms(t2, t3),
            submit_ms=_ms(t3, t4),
            first_layer_ms=_ms(t4, first),
            copy_ms=_ms(t4, t5),
            finalize_ms=_ms(t5, t6),
            total_ms=_ms(queued, t5),
        )

    def _acquire_all(self, keys):
        """Pin every page on the store the lookup found it in; all-or-nothing."""
        by_store = OrderedDict()
        for i, key in enumerate(keys):
            by_store.setdefault(self._routes.get(key, self.write_id), []).append(i)
        groups = []
        try:
            for store_id, pos in by_store.items():
                client = self.stores.hold(store_id)
                if client is None:
                    raise RuntimeError(f"EGM store {store_id} is no longer connected")
                try:
                    lease = client.acquire([keys[i] for i in pos], self.lease_ttl_ms)
                except BaseException:
                    self.stores.unhold(client)
                    raise
                if lease is None:
                    self.stores.unhold(client)
                    raise RuntimeError(
                        f"EGM objects evicted before load ({len(pos)} pages on {store_id})"
                    )
                token, offsets, deadline = lease
                groups.append(
                    (client, np.asarray(pos, dtype=np.int64), token, offsets, deadline)
                )
        except BaseException:
            for client, _, token, _, _ in groups:
                self._release_quietly(client, token)
                self.stores.unhold(client)
            raise
        return groups

    def _release_quietly(self, client, token):
        try:
            client.release(token)
        except (ConnectionError, RuntimeError) as error:
            # The lease still expires on the service side.
            self._audit("release_failed", store=client.store_id, error=str(error))

    # ---- offload ----

    def offload(self, transfers, session_id=None):
        if self.read_only:
            return False
        by_name = self._by_entry(
            self.pool_group.resolve_transfers(transfers, allow_partial=True)
        )
        if by_name is None:
            return False
        if not self.offload_owner:
            # Rank 0 writes the replicated pages; every rank reports so the per-rank counts match.
            self.offload_results.put(True)
            return True
        self.offload_queue.put(
            (by_name, self.record_transfer_ready(), time.perf_counter())
        )
        return True

    def _offload_loop(self):
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                try:
                    success = self._offload_one(*task)
                    self.stats["offload"] += success
                except BaseException:
                    logger.exception("EGM offload failed")
                    success = False
                self.offload_results.put(success)
            finally:
                self.offload_queue.task_done()

    def _offload_one(self, by_name, ready, queued):
        t0 = time.perf_counter()
        ready.synchronize()
        keys = self._keys(by_name[PoolName.KV].keys)
        rows = self._entry_rows(by_name)
        t1 = time.perf_counter()
        pending = ~self._held_elsewhere(keys)
        dedup_remote = int((~pending).sum())
        t2 = time.perf_counter()
        steps = []
        for store_id, evict in self._write_plan():
            if not pending.any():
                break
            if store_id is None:
                store_id = self._place(int(pending.sum()))
                if store_id is None:
                    continue
            pending, step = self._write_step(
                store_id=store_id, keys=keys, rows=rows, pending=pending, evict=evict
            )
            steps.append(step)
        self._audit(
            "offload",
            pages=len(keys),
            new=sum(s["new"] for s in steps),
            dedup=sum(s["dedup"] for s in steps),
            dedup_remote=dedup_remote,
            full=int(pending.sum()),
            steps=steps,
            queue_ms=_ms(queued, t0),
            ready_ms=_ms(t0, t1),
            dedup_ms=_ms(t1, t2),
            total_ms=_ms(queued, time.perf_counter()),
        )
        return not pending.any()

    def _write_plan(self):
        """(store id, evict) attempts in order; None means "ask the master where"."""
        if self.master is None:
            return [(self.write_id, True)]
        # Fill free space anywhere in the domain before evicting the local store.
        return [(self.write_id, False), (None, False), (self.write_id, True)]

    def _place(self, pages):
        try:
            store_id = self.master.rpc(
                "place",
                nbytes=pages * self.layout.object_bytes,
                exclude=[self.write_id],
                host=self.host,
                numa=self.numa,
            )
        except _MASTER_ERRORS as error:
            self._audit("master_failed", call="place", error=str(error))
            return None
        return store_id if store_id in self.stores.ids() else None

    def _write_step(self, *, store_id, keys, rows, pending, evict):
        """Write the pending pages to one store; returns (pages still pending, audit record)."""
        client = self.stores.hold(store_id)
        if client is None:
            return pending, dict(store=store_id, evict=evict, new=0, dedup=0, full=0)
        try:
            return self._write_to(client, keys, rows, pending, evict)
        finally:
            self.stores.unhold(client)

    def _write_to(self, client, keys, rows, pending, evict):
        t0 = time.perf_counter()
        idx = np.flatnonzero(pending)
        token, offsets, deadline = client.alloc(
            [keys[i] for i in idx], self.layout.object_bytes, evict=evict
        )
        offsets = np.asarray(offsets, dtype=np.int64)
        new = offsets >= 0
        t1 = time.perf_counter()
        fault = self._take_fault("stall-next-offload")
        if fault is not None:
            time.sleep(
                float(fault.get("ms", 0)) / 1e3
            )  # A stuck writer: its reservation must expire.
        if new.any():
            try:
                if time.monotonic() > deadline:
                    # Too late to start: the service may already have quarantined these slots.
                    raise RuntimeError(
                        "EGM write reservation expired before the copy started"
                    )
                off, ptr, size = self.layout.store_ranges(
                    rows[:, idx[new]], offsets[new].astype(np.uint64)
                )
                batch = client.region.submit(
                    off,
                    ptr,
                    size,
                    _DIRECTION_TO_REGION,
                    [len(off)],
                    method=self.method,
                    writable=True,
                )
                try:
                    batch.wait(0)
                finally:
                    batch.free()
            except BaseException:
                client.abort(token)
                raise
            if time.monotonic() > deadline:
                # The service has quarantined these slots; committing would publish nothing useful.
                client.abort(token)
                raise RuntimeError(
                    "EGM write reservation expired before the copy landed"
                )
            # A reservation with no new slot is never recorded, so only commit when one was taken.
            client.commit(token)
        t2 = time.perf_counter()
        full = offsets == FULL
        remaining = pending.copy()
        remaining[idx[~full]] = False
        return remaining, dict(
            store=client.store_id,
            evict=evict,
            new=int(new.sum()),
            dedup=int((offsets == EXISTS).sum()),
            full=int(full.sum()),
            alloc_ms=_ms(t0, t1),
            copy_commit_ms=_ms(t1, t2),
        )

    def num_completed_offloads(self):
        return self.offload_results.qsize()

    def pop_completed_offload(self):
        return self.offload_results.get_nowait()

    # ---- lifecycle ----

    def reset(self):
        self.pending_loads.clear()
        self.broadcast_loads.clear()
        self.load_queue.join()
        self.offload_queue.join()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self):
        self.reset()
        self._stopping.set()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        if self.membership_thread is not None:
            self.membership_thread.join()
        if self.broadcaster is not None:
            self.broadcaster.destroy()
        logger.info("EGM direct linker stats: %s", dict(self.stats))
