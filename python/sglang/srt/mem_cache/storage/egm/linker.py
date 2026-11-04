# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Direct linker over a standalone EGM store service (FULL MHA, TP1).

Loaded through the ``dynamic`` external-linker backend::

    --enable-unified-cache-external-linker --unified-cache-external-linker-backend dynamic
    --unified-cache-external-linker-config '{"linker": "EGMDirectLinker",
        "linker_module_path": "sglang.srt.mem_cache.storage.egm.linker",
        "linker_extra_config": {"stores": ["/tmp/egm-numa0.sock", "tcp://10.0.0.2:29600"],
                                "secret_file": "/shared/job/egm.secret", "namespace": "<model+weights>"}}'

The Engine never owns EGM: the services (``service.py``) do, so the Engine can
run CUDA Graph and read any NUMA's slab, on this node or another node of the same
NVLink domain. One page is one object (K of every layer, then V), the layout of
the MHA device pool entry.

Placement and discovery need no directory: offloads go to ``write_store`` (by
default the first configured store, normally this GPU's NUMA); a lookup asks every
store in parallel and loads each page from the first store that holds it.

Loads: one lease for the whole batch, then one gather launch per layer on a
private stream; layer ``l`` completes as soon as its copy lands, so the forward
overlaps the remaining layers. Offloads run on their own thread: ``alloc``
(skips keys already stored), one gather launch, ``commit``.
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
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


def _ms(start, end):
    return round((end - start) * 1e3, 3)


class _Layout:
    """Vectorized range metadata for one packed device pool entry."""

    def __init__(self, entry, num_layers):
        if not entry.packed or len(entry.components) != 1:
            raise ValueError("EGM linker needs one packed component group")
        meta = entry.buffer_meta[0]
        offsets = entry._component_offsets[0]
        self.object_bytes = sum(size for _, _, size in meta)
        # Whole object: every buffer, in object order (offload).
        self.all_base = np.array([b for b, _, _ in meta], dtype=np.uint64)
        self.all_stride = np.array([s for _, s, _ in meta], dtype=np.uint64)
        self.all_size = np.array([n for _, _, n in meta], dtype=np.uint64)
        self.all_off = np.array(offsets, dtype=np.uint64)
        # Per layer: the buffers that layer maps to, e.g. (K_l, V_l) for MHA.
        per_layer = []
        for layer in range(num_layers):
            mapped = entry.layer_mapping[layer]
            per_layer.append([mapped] if isinstance(mapped, int) else list(mapped))
        if len({len(x) for x in per_layer}) != 1:
            raise ValueError("layers must map to the same number of buffers")
        index = np.array(per_layer, dtype=np.int64)  # [layers, items]
        self.layer_base, self.layer_stride = (
            self.all_base[index],
            self.all_stride[index],
        )
        self.layer_size, self.layer_off = self.all_size[index], self.all_off[index]

    def load_ranges(self, rows, objects):
        """rows / objects: uint64 [pages]. Returns layer-major flat arrays and per-layer group ends."""
        layers, items = self.layer_base.shape
        ptr = (
            self.layer_base[:, None, :]
            + rows[None, :, None] * self.layer_stride[:, None, :]
        )
        off = objects[None, :, None] + self.layer_off[:, None, :]
        size = np.broadcast_to(self.layer_size[:, None, :], ptr.shape)
        per_layer = len(rows) * items
        ends = np.arange(1, layers + 1, dtype=np.uint64) * per_layer
        return (
            off.reshape(-1),
            ptr.reshape(-1),
            np.ascontiguousarray(size).reshape(-1),
            ends,
        )

    def store_ranges(self, rows, objects):
        ptr = self.all_base[None, :] + rows[:, None] * self.all_stride[None, :]
        off = objects[:, None] + self.all_off[None, :]
        size = np.broadcast_to(self.all_size[None, :], ptr.shape)
        return off.reshape(-1), ptr.reshape(-1), np.ascontiguousarray(size).reshape(-1)


class _TimedCounter(LayerWiseLoadCounter):
    """Records how long the forward thread actually blocks on each load batch (M8 "exposed wait")."""

    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.exposed = {}

    def wait_until(self, threshold):
        index = self.consumer_index
        start = time.perf_counter()
        super().wait_until(threshold)
        if index in self.futures:
            self.exposed[index] = (
                self.exposed.get(index, 0.0) + time.perf_counter() - start
            )


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
        if server_args.tp_size != 1 or server_args.pp_size != 1:
            raise ValueError("EGM linker supports TP1 / PP1 only")
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
        if set(self.pools) != {PoolName.KV}:
            raise ValueError("EGM linker supports a single FULL KV pool")
        self.entry = self.pools[PoolName.KV]
        self.num_layers = self.pool_group.num_layers
        self.layout = _Layout(self.entry, self.num_layers)
        self.device = self.entry.kv_buffer[0].device.index
        schema = [
            config["namespace"],
            self.page_size,
            sorted(self.entry.layer_mapping.items()),
            [(str(t.dtype), list(t.shape[1:])) for t in self.entry.kv_buffer],
        ]
        self.prefix = hashlib.sha256(json.dumps(schema).encode()).hexdigest()[:32] + ":"
        self.method = _METHODS[config.get("copy_mode", "kernel")]
        self.audit_path = config.get("audit_path")
        self.fault_dir = config.get("fault_dir")
        # A consumer-only Engine: restores from the store but never offloads to it.
        self.read_only = bool(config.get("read_only", False))
        self.lease_ttl_ms = config.get("lease_ttl_ms")
        secret = (
            Path(config["secret_file"]).read_text().strip()
            if config.get("secret_file")
            else None
        )
        start = time.perf_counter()
        domain = (
            cluster_uuid() if any(a.startswith("tcp://") for a in addresses) else None
        )
        self.stores = [
            StoreClient(a, device=self.device, secret=secret, local_cluster_uuid=domain)
            for a in addresses
        ]
        self.write_index = int(config.get("write_store", 0))
        self.store = self.stores[self.write_index]
        self._routes = OrderedDict()  # key -> index of the store that held it at lookup
        self._lookup_pool = ThreadPoolExecutor(
            max_workers=len(self.stores), thread_name_prefix="egm-lookup"
        )
        connect_import_ms = (time.perf_counter() - start) * 1e3

        self.layer_done_counter = _TimedCounter(self.num_layers)
        self.pending_loads = {}
        self.load_queue, self.offload_queue, self.offload_results = (
            Queue(),
            Queue(),
            Queue(),
        )
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self._load_loop, daemon=True, name="egm-load"
        )
        self.offload_thread = threading.Thread(
            target=self._offload_loop, daemon=True, name="egm-offload"
        )
        self.load_thread.start()
        self.offload_thread.start()
        # The slab is imported once here, so per-load "lease" is the acquire RPC only.
        self._audit(
            "init",
            device=self.device,
            numa=self.store.numa,
            prefix=self.prefix,
            object_bytes=self.layout.object_bytes,
            method=self.method,
            stores=[
                dict(address=c.address, host=c.host, numa=c.numa) for c in self.stores
            ],
            write_store=self.write_index,
            cluster_uuid=domain,
            connect_import_ms=round(connect_import_ms, 3),
        )
        logger.info(
            "EGM direct linker: stores=%s write_store=%d device=%d object=%d B",
            addresses,
            self.write_index,
            self.device,
            self.layout.object_bytes,
        )

    def _audit(self, op, **fields):
        if self.audit_path:
            with Path(self.audit_path).open("a") as stream:
                stream.write(
                    json.dumps(dict(op=op, time_ns=time.time_ns(), **fields)) + "\n"
                )

    def _keys(self, keys):
        return [self.prefix + key for key in keys]

    # ---- lookup / load ----

    def lookup(self, rid, transfers):
        kv = next((t for t in transfers if t.name == PoolName.KV), None)
        if kv is None or not kv.keys:
            return []
        start = time.perf_counter()
        keys = self._keys(kv.keys)
        per_store = self._exists_everywhere(rid, keys)
        lookup_ms = (time.perf_counter() - start) * 1e3
        hits, used = 0, set()
        for i, key in enumerate(keys):
            # Prefer the store this Engine writes to (local NUMA), then configuration order.
            holder = next(
                (
                    j
                    for j in self._store_order()
                    if per_store[j] is not None and per_store[j][i]
                ),
                None,
            )
            if holder is None:
                break
            self._remember(key, holder)
            used.add(holder)
            hits += 1
        self.stats["lookup"] += 1
        self._audit(
            "lookup",
            rid=rid,
            pages=len(kv.keys),
            hit_pages=hits,
            stores_hit=sorted(used),
            lookup_ms=round(lookup_ms, 3),
        )
        return list(range(1, hits + 1))

    def _store_order(self):
        return [self.write_index] + [
            j for j in range(len(self.stores)) if j != self.write_index
        ]

    def _exists_everywhere(self, rid, keys):
        """One exists RPC per store, in parallel; None for a store that is down (treated as a miss)."""

        def ask(j):
            try:
                return self.stores[j].exists_many(keys)
            except ConnectionError as error:
                self._audit("lookup_failed", rid=rid, store=j, error=str(error))
                return None

        if len(self.stores) == 1:
            return [ask(0)]
        return list(self._lookup_pool.map(ask, range(len(self.stores))))

    def _held_elsewhere(self, keys):
        held = np.zeros(len(keys), dtype=bool)
        if len(self.stores) == 1:
            return held

        def ask(j):
            try:
                return self.stores[j].exists_many(keys)
            except ConnectionError:
                return None

        others = [j for j in range(len(self.stores)) if j != self.write_index]
        for flags in self._lookup_pool.map(ask, others):
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
        kv = [t for t in expanded if t.name == PoolName.KV and t.keys]
        if not kv:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"EGM load for rid={rid} is already queued")
        self.pending_loads[rid] = kv[0]
        return True

    def cancel_queued_load(self, rid):
        return self.pending_loads.pop(rid, None) is not None

    def start_layer_wise_loading(self):
        if not self.pending_loads:
            return -1
        pending, self.pending_loads = self.pending_loads, {}
        index = self.layer_done_counter.update_producer()
        self.load_queue.put(
            (index, pending, self.record_transfer_ready(), time.perf_counter())
        )
        self.stats["load"] += len(pending)
        return index

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
        keys = [k for t in transfers for k in self._keys(t.keys)]
        rows = np.concatenate(
            [
                np.asarray(
                    self.entry.prepare_locations(t.host_indices), dtype=np.uint64
                )
                for t in transfers
            ]
        )
        fault = self._take_fault("fail-next-load")
        mode = None if fault is None else fault.get("mode", "copy")
        if mode == "evict":
            self.stores[self._routes.get(keys[0], self.write_index)].drop(keys)
        t1 = time.perf_counter()
        groups = self._acquire_all(
            keys
        )  # [(store index, page positions, token, offsets, deadline)]
        t2 = time.perf_counter()
        batches = []
        try:
            ranges = [
                (
                    j,
                    self.layout.load_ranges(
                        rows[pos], np.asarray(offsets, dtype=np.uint64)
                    ),
                )
                for j, pos, _, offsets, _ in groups
            ]
            t3 = time.perf_counter()
            fail_layer = int(fault["layer"]) if mode == "copy" else None
            for j, (off, ptr, size, ends) in ranges:
                batches.append(
                    self.stores[j].region.submit(
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
            for j, _, token, _, _ in groups:
                self._release_quietly(j, token)
        t6 = time.perf_counter()
        # M8 segments: lease_ms / read_ms (build + submit + every layer landed) / finalize_ms;
        # lookup_ms is in the lookup event, exposed_wait_ms in load_finish (joined by index).
        self._audit(
            "load",
            index=index,
            reqs=len(transfers),
            pages=len(keys),
            stores=[dict(store=g[0], pages=len(g[1])) for g in groups],
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
            by_store.setdefault(self._routes.get(key, self.write_index), []).append(i)
        groups = []
        try:
            for j, pos in by_store.items():
                lease = self.stores[j].acquire(
                    [keys[i] for i in pos], self.lease_ttl_ms
                )
                if lease is None:
                    raise RuntimeError(
                        f"EGM objects evicted before load ({len(pos)} pages on store {j})"
                    )
                token, offsets, deadline = lease
                groups.append(
                    (j, np.asarray(pos, dtype=np.int64), token, offsets, deadline)
                )
        except BaseException:
            for j, _, token, _, _ in groups:
                self._release_quietly(j, token)
            raise
        return groups

    def _release_quietly(self, j, token):
        try:
            self.stores[j].release(token)
        except (ConnectionError, RuntimeError) as error:
            # The lease still expires on the service side.
            self._audit("release_failed", store=j, error=str(error))

    # ---- offload ----

    def offload(self, transfers, session_id=None):
        if self.read_only:
            return False
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        kv = [t for t in expanded if t.name == PoolName.KV and t.keys]
        if not kv:
            return False
        self.offload_queue.put(
            (kv[0], self.record_transfer_ready(), time.perf_counter())
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

    def _offload_one(self, transfer, ready, queued):
        t0 = time.perf_counter()
        ready.synchronize()
        all_keys = self._keys(transfer.keys)
        t1 = time.perf_counter()
        # Domain-wide dedup without a directory: skip pages another store already holds.
        elsewhere = self._held_elsewhere(all_keys)
        keep = ~elsewhere
        keys = [k for k, held in zip(all_keys, elsewhere) if not held]
        if not keys:
            self._audit(
                "offload",
                pages=len(all_keys),
                new=0,
                dedup=0,
                dedup_remote=len(all_keys),
                full=0,
                total_ms=round((time.perf_counter() - queued) * 1e3, 3),
            )
            return True
        token, offsets, deadline = self.store.alloc(keys, self.layout.object_bytes)
        t2 = time.perf_counter()
        offsets = np.asarray(offsets, dtype=np.int64)
        new = offsets >= 0
        copied = 0
        fault = self._take_fault("stall-next-offload")
        if fault is not None:
            time.sleep(
                float(fault.get("ms", 0)) / 1e3
            )  # A stuck writer: its reservation must expire.
        try:
            if new.any() and time.monotonic() > deadline:
                # Too late to start: the service may already have quarantined these slots.
                raise RuntimeError(
                    "EGM write reservation expired before the copy started"
                )
            if new.any():
                rows = np.asarray(
                    self.entry.prepare_locations(transfer.host_indices), dtype=np.uint64
                )[keep]
                off, ptr, size = self.layout.store_ranges(
                    rows[new], offsets[new].astype(np.uint64)
                )
                batch = self.store.region.submit(
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
                copied = int(new.sum())
        except BaseException:
            self.store.abort(token)
            raise
        t3 = time.perf_counter()
        if time.monotonic() > deadline:
            # The service has quarantined these slots; committing would publish nothing useful.
            self.store.abort(token)
            raise RuntimeError("EGM write reservation expired before the copy landed")
        self.store.commit(token)
        t4 = time.perf_counter()
        full = int((offsets == FULL).sum())
        self._audit(
            "offload",
            pages=len(all_keys),
            new=copied,
            dedup=int((offsets == EXISTS).sum()),
            dedup_remote=int(elsewhere.sum()),
            full=full,
            queue_ms=_ms(queued, t0),
            ready_ms=_ms(t0, t1),
            alloc_ms=_ms(t1, t2),
            copy_ms=_ms(t2, t3),
            commit_ms=_ms(t3, t4),
            total_ms=_ms(queued, t4),
        )
        return full == 0

    def num_completed_offloads(self):
        return self.offload_results.qsize()

    def pop_completed_offload(self):
        return self.offload_results.get_nowait()

    # ---- lifecycle ----

    def reset(self):
        self.pending_loads.clear()
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
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("EGM direct linker stats: %s", self.stats)
