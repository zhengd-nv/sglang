# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""SGLang dynamic HiCache backend: FULL MHA, TP/PP/CP=1, no sidecars."""
import hashlib
import json
import logging
from pathlib import Path

import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, PoolTransferResult

from .peer import PeerClient, PeerServer
from .store import EGMStore

logger = logging.getLogger(__name__)


class EGMHiCacheStorage(HiCacheStorage):
    def __init__(self, storage_config, kwargs=None):
        cfg = storage_config
        if any(getattr(cfg, x, 1) != 1 for x in ("tp_size", "pp_size", "attn_cp_size")):
            raise ValueError("EGM prototype supports TP/PP/CP=1 only")
        if cfg.is_mla_model or getattr(cfg, "should_split_heads", False):
            raise ValueError("EGM prototype supports unsplit FULL MHA only")
        extra = cfg.extra_config or {}
        self.config = cfg
        self.pool = None
        self.server = None
        self.closed = False
        self.remote = bool(extra.get("peer_path"))
        self.store = PeerClient(extra["peer_path"]) if self.remote else EGMStore(
            int(extra.get("capacity_bytes", 64 << 20)), int(extra.get("numa", 0)))
        self.namespace = str(extra.get("namespace", "egm-v1"))
        self.audit_path = extra.get("audit_path")
        try:
            if extra.get("listen_path"):
                if self.remote:
                    raise ValueError("peer clients cannot serve")
                self.server = PeerServer(self.store, extra["listen_path"])
        except BaseException:
            if not self.remote:
                self.store.close()
            raise
        logger.info("Independent EGM HiCache initialized: HOST_NUMA/FABRIC, remote=%s", self.remote)
        self._audit("init", 0, 0)

    def _audit(self, op, pages, success):
        if self.audit_path:
            with Path(self.audit_path).open("a") as out:
                out.write(json.dumps(dict(op=op, pages=pages, success=success)) + "\n")

    def register_mem_pool_host(self, pool):
        if self.pool is not None and self.pool is not pool:
            raise ValueError("cannot replace registered pool")
        if pool.layout not in ("layer_first", "page_first", "page_first_direct"):
            raise ValueError("unsupported host layout")
        if not all(hasattr(pool, x) for x in ("head_num", "head_dim", "layer_num", "kv_buffer")):
            raise ValueError("only standard MHA HostPool is supported")
        if not isinstance(pool.kv_buffer, torch.Tensor) or pool.kv_buffer.device.type != "cpu":
            raise ValueError("expected CPU MHA tensor")
        if not pool.kv_buffer.is_contiguous():
            raise ValueError("noncontiguous HostPool unsupported")
        schema = [self.namespace, self.config.model_name, pool.layout, str(pool.dtype),
                  pool.page_size, pool.layer_num, pool.head_num, pool.head_dim]
        self.prefix = hashlib.sha256(json.dumps(schema).encode()).hexdigest()
        self.pool = self.mem_pool_host = pool

    def register_mem_host_pool_v2(self, pool, name):
        if str(name) != "kv":
            raise ValueError("FULL KV only; no sidecars")
        self.register_mem_pool_host(pool)

    def _key(self, key):
        if self.closed or self.pool is None:
            raise RuntimeError("backend closed or pool unregistered")
        return self.prefix + ":" + key

    def exists(self, key):
        return self.store.exists(self._key(key))

    def _pages(self, keys, indices):
        if self.pool is None:
            raise RuntimeError("pool unregistered")
        page = self.pool.page_size
        rows = indices.detach().cpu().to(torch.int64).flatten()
        if rows.numel() != len(keys) * page:
            raise ValueError("key/index length mismatch")
        if not keys:
            return []
        pages = rows.reshape(-1, page)
        starts = pages[:, 0]
        if (torch.any(starts % page) or torch.any(starts < 0) or
            torch.any(starts + page > self.pool.size) or
            not torch.equal(pages, starts[:, None] + torch.arange(page))):
            raise ValueError("expected aligned contiguous pages in pool bounds")
        ptrs, sizes = self.pool.get_page_buffer_meta(rows)
        if isinstance(sizes, int):
            sizes = [sizes] * len(ptrs)
        if len(ptrs) != len(sizes) or len(ptrs) % len(keys):
            raise ValueError("invalid page geometry")
        width = len(ptrs) // len(keys)
        return [list(zip(ptrs[i:i+width], sizes[i:i+width])) for i in range(0, len(ptrs), width)]

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        pages = self._pages(keys, host_indices)
        if self.remote:
            return [False] * len(keys)  # read-only consumer, no fake persistence
        results = [self.store.put(self._key(k), ranges) for k, ranges in zip(keys, pages)]
        logger.info("Independent EGM put pages=%d success=%d", len(results), sum(results))
        self._audit("put", len(results), sum(results))
        return results

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        pages = self._pages(keys, host_indices)
        results = []
        for key, spans in zip(keys, pages):
            offset = 0
            ranges = []
            for ptr, n in spans:
                ranges.append((ptr, n, offset)); offset += n
            results.append(self.store.get(self._key(key), ranges))
        logger.info("Independent EGM get pages=%d success=%d", len(results), sum(results))
        self._audit("get", len(results), sum(results))
        return results

    def _full_only(self, transfers):
        for t in transfers:
            if str(t.name) != "kv" or str(getattr(t.hit_policy, "value", t.hit_policy)) != "all_pages":
                raise ValueError("FULL/all_pages only")
        if len(transfers) > 1:
            raise ValueError("duplicate FULL transfers")

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        self._full_only(pool_transfers or [])
        hit = self.batch_exists(keys, extra_info)
        return PoolTransferResult(hit, {}, list(range(1, hit + 1)))

    def batch_get_v2(self, transfers, extra_info=None):
        self._full_only(transfers)
        return {str(t.name): self.batch_get_v1(t.keys, t.host_indices, extra_info) for t in transfers}

    def batch_set_v2(self, transfers, extra_info=None):
        self._full_only(transfers)
        return {str(t.name): self.batch_set_v1(t.keys, t.host_indices, extra_info) for t in transfers}

    # Legacy tensor API deliberately supported rather than empty abstract stubs.
    def get(self, key, target_location=None, target_sizes=None):
        t = target_location
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or not t.is_contiguous():
            raise ValueError("get requires contiguous CPU target tensor")
        return t if self.store.get(self._key(key), [(t.data_ptr(), t.nbytes, 0)]) else None

    def set(self, key, value=None, target_location=None, target_sizes=None):
        if self.remote:
            return False
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError("set requires contiguous CPU tensor")
        return self.store.put(self._key(key), [(value.data_ptr(), value.nbytes)])

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        if target_locations is None or len(keys) != len(target_locations):
            raise ValueError("target count mismatch")
        return [self.get(k, t) for k, t in zip(keys, target_locations)]

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        if values is None or len(keys) != len(values):
            raise ValueError("value count mismatch")
        results = [self.set(k, t) for k, t in zip(keys, values)]
        return all(results)

    def clear(self):
        if self.remote:
            raise ValueError("remote client cannot clear provider")
        self.store.clear()

    def close(self):
        if self.closed:
            return
        # Keep the control endpoint alive if close refuses outstanding pins.
        if not self.remote:
            self.store.close()
        if self.server:
            self.server.close()
        self.closed = True
