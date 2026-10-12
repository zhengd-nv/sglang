# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""FULL/TP1 synchronous Direct Linker reference implementation.

It deliberately finishes copies before returning from start/offload. This has
no transfer/compute overlap, but cannot expose a submitted copy as ready.
Any load failure poisons the linker: the current tree's failure rollback is
not assumed safe. Restart the owning Engine rather than continuing forward.
"""
from collections import deque
from dataclasses import replace
import hashlib
import json

import torch
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker


class LayerCounter:
    def __init__(self, layers):
        self.layers = layers
        self.reset()

    def reset(self):
        self.producer_index = self.consumer_index = -1
        self.ready_index = -1

    def set_consumer(self, index):
        self.consumer_index = index

    def wait_until(self, threshold):
        if not 0 <= threshold < self.layers:
            raise ValueError("invalid layer")
        if self.consumer_index < 0:
            return
        if self.consumer_index > self.ready_index:
            raise RuntimeError("layer not ready")
        # Both K and V accessors may wait on the same layer, including the last.
        # All batches complete synchronously, so a high-water mark is sufficient.


class EGMDirectLinker(UnifiedCacheLinker):
    def __init__(self, entry, num_layers, store, namespace, tp_size=1):
        if tp_size != 1 or str(entry.name) != "kv" or not entry.packed:
            raise ValueError("FULL packed TP1 only")
        if set(entry.layer_mapping) != set(range(num_layers)):
            raise ValueError("all logical layers must be present")
        if any(not t.is_contiguous() or t.device.type != "cuda" or t.device.index != 0
               for t in entry.kv_buffer):
            raise ValueError("contiguous cuda:0 pool required")
        self.entry, self.store = entry, store
        self.num_layers = num_layers
        if not namespace:
            raise ValueError("model/weight namespace required")
        schema = [namespace, entry.page_size, entry.layer_mapping, entry._row_span,
                  [(str(t.dtype), list(t.shape[1:])) for t in entry.kv_buffer],
                  [[n for _, _, n in group] for group in entry.buffer_meta]]
        self.prefix = hashlib.sha256(json.dumps(schema).encode()).hexdigest() + ":"
        self.layer_done_counter = LayerCounter(num_layers)
        self.pending = {}
        self.loads, self.offloads = deque(), deque()
        self.failed = self.closed = False

    def _open(self):
        if self.failed or self.closed:
            raise RuntimeError("linker closed or poisoned; restart Engine")

    def _transfer(self, transfers, indices=True):
        self._open()
        if len(transfers) != 1 or str(transfers[0].name) != "kv":
            raise ValueError("exactly one FULL transfer required")
        t = transfers[0]
        if getattr(t.hit_policy, "value", t.hit_policy) != "all_pages":
            raise ValueError("FULL/all_pages required")
        if not t.keys:
            raise ValueError("empty transfer")
        if indices:
            if t.device_indices is None:
                raise ValueError("device indices required")
            if len(self.entry.prepare_locations(t.device_indices)) != len(t.keys):
                raise ValueError("key/page count mismatch")
        return t

    def lookup(self, rid, transfers):
        t = self._transfer(transfers, indices=False)
        hit = 0
        for key in t.keys:
            if not self.store.exists(self.prefix + key):
                break
            hit += 1
        return list(range(1, hit + 1))

    def load(self, rid, transfers):
        t = self._transfer(transfers)
        if rid in self.pending:
            raise ValueError("duplicate queued request")
        self.pending[rid] = replace(t, keys=list(t.keys), device_indices=t.device_indices.detach().cpu().clone())
        return True

    def cancel_queued_load(self, rid):
        self._open()
        return self.pending.pop(rid, None) is not None

    def start_layer_wise_loading(self):
        self._open()
        counter = self.layer_done_counter
        counter.producer_index += 1
        index = counter.producer_index
        tasks, self.pending = self.pending, {}
        torch.cuda.synchronize(0)
        try:
            keys = list(dict.fromkeys(self.prefix + k for t in tasks.values() for k in t.keys))
            if keys:
                with self.store.reader(keys) as reader:
                    for t in tasks.values():
                        locations = self.entry.prepare_locations(t.device_indices)
                        for layer in range(self.num_layers):
                            ptrs, sizes, offsets = self.entry.get_prepared_layer_range_meta(locations, layer)
                            for key, ps, ns, os in zip(t.keys, ptrs, sizes, offsets):
                                if not reader.get(self.prefix + key, list(zip(ps, ns, os)), device=True):
                                    raise RuntimeError("object disappeared after lookup")
            counter.ready_index = index
            if tasks:
                self.loads.append(list(tasks))
            return index
        except BaseException as error:
            self.failed = True
            raise RuntimeError("EGM restore failed; restart Engine") from error

    def num_completed_loads(self):
        return len(self.loads)

    def pop_completed_load(self):
        return self.loads.popleft()

    def offload(self, transfers):
        t = self._transfer(transfers)
        if not hasattr(self.store, "put"):
            return False
        torch.cuda.synchronize(0)
        ptrs, sizes = self.entry.get_page_buffer_meta(t.device_indices)
        width = len(ptrs) // len(t.keys)
        try:
            results = [self.store.put(self.prefix + key,
                       list(zip(ptrs[i*width:(i+1)*width], sizes[i*width:(i+1)*width])), device=True)
                       for i, key in enumerate(t.keys)]
            self.offloads.append(all(results))
            return True
        except BaseException:
            self.failed = True
            raise

    def num_completed_offloads(self):
        return len(self.offloads)

    def pop_completed_offload(self):
        return self.offloads.popleft()

    def reset(self):
        self._open()  # A failed load cannot be made safe by clearing queues.
        self.pending.clear()
        self.loads.clear()
        self.offloads.clear()
        self.layer_done_counter.reset()

    def close(self):
        # Store ownership is external: multiple adapters may share it.
        self.pending.clear()
        self.loads.clear()
        self.offloads.clear()
        self.closed = True
