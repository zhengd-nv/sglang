# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Experimental FULL/TP1 Engine adapter for synchronous Fabric prefix reuse."""

import json
import time
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

from .linker import EGMDirectLinker
from .peer import PeerClient, PeerServer
from .store import EGMStore


class EGMEngineLinker(EGMDirectLinker):
    """Provider owns a Store; consumer reads a fixed provider without HostPool.

    ``namespace`` must identify identical weights and KV-affecting execution
    settings, not just a model name. The existing linker adds the pool schema.
    Unknown transfer failures are fatal; automatic remote failover is not part
    of this initial, synchronous integration.
    """

    def __init__(self, server_args, params, *, components):
        if components != {ComponentType.FULL}:
            raise ValueError("EGM Engine linker supports FULL only")
        for name in ("tp_size", "pp_size", "dp_size", "dcp_size", "attn_cp_size"):
            if getattr(server_args, name, 1) != 1:
                raise ValueError(f"EGM Engine linker requires {name}=1")
        if not server_args.disable_cuda_graph or not server_args.disable_overlap_schedule:
            raise ValueError("EGM Engine linker requires disabled CUDA Graph and overlap schedule")
        if server_args.speculative_algorithm or server_args.enable_lora:
            raise ValueError("EGM Engine linker does not support speculative decoding or LoRA")
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        if type(kvcache).__name__ != "MHATokenToKVPool":
            raise ValueError("EGM Engine linker requires the MHA device pool")
        config = json.loads(server_args.hicache_storage_backend_extra_config or "{}")
        allowed = {"namespace", "capacity_bytes", "hbm_capacity_bytes", "write_tier",
                   "numa", "listen_path", "peer_path", "peer_secret", "audit_path"}
        if set(config) - allowed:
            raise ValueError(f"Unknown EGM linker settings: {set(config) - allowed}")
        if not config.get("namespace"):
            raise ValueError("EGM requires an explicit model/weight namespace")
        if bool(config.get("listen_path")) == bool(config.get("peer_path")):
            raise ValueError("Specify exactly one of listen_path (provider) and peer_path (consumer)")
        if kvcache.kv_cache_layout not in ("nhd", "hnd") or kvcache.start_layer != 0:
            raise ValueError("EGM requires a complete NHD/HND MHA layer stack")
        if any(t.dtype not in (torch.bfloat16, torch.float16)
               for t in [*kvcache.k_buffer, *kvcache.v_buffer]):
            raise ValueError("EGM Engine linker requires unquantized BF16/FP16 KV")
        # The generic hybrid assembler does not implement _PlainKvStrategy's
        # direct path. Expose the actual Engine buffers, without a HostPool.
        entry = DevicePoolEntry(
            name=PoolName.KV, indices_from_pool=PoolName.KV, device_pool=kvcache,
            components=[kvcache.k_buffer, kvcache.v_buffer],
            layer_mapping={i: i for i in range(kvcache.layer_num)},
            page_size=params.page_size, rows_are_pages=kvcache.use_hnd, packed=True,
        )
        device = entry.kv_buffer[0].device.index
        self.audit_path = config.get("audit_path")
        self.server = None
        self.provider = bool(config.get("listen_path"))
        secret = config.get("peer_secret")
        if self.provider:
            store = EGMStore(
                capacity=config.get("capacity_bytes", 64 << 20),
                hbm_capacity=config.get("hbm_capacity_bytes", 0),
                default_tier=config.get("write_tier", "dram"),
                numa=config.get("numa", 0), device=device,
            )
        else:
            store = PeerClient(config["peer_path"], device=device, secret=secret)
        try:
            super().__init__(entry, kvcache.layer_num, store, config["namespace"])
            if self.provider:
                self.server = PeerServer(store, config["listen_path"], secret=secret)
        except BaseException:
            if self.provider:
                store.close()
            raise
        self._audit("init", provider=self.provider, device=device, schema=self.prefix,
                    endpoint=self.server.path if self.server else config["peer_path"])

    def _audit(self, op, **fields):
        if self.audit_path:
            with Path(self.audit_path).open("a") as stream:
                stream.write(json.dumps(dict(op=op, time_ns=time.time_ns(), **fields)) + "\n")

    def lookup(self, rid, transfers):
        hits = super().lookup(rid, transfers)
        self._audit("lookup", rid=rid, hit_pages=max(hits, default=0))
        return hits

    def load_private(self, rid, transfers):
        pages = len(transfers[0].keys)
        self._audit("private_load_begin", rid=rid, pages=pages)
        try:
            super().load_private(rid, transfers)
        except BaseException:
            self.failed = True
            self._audit("private_load_failed", rid=rid, pages=pages,
                        quarantined_sessions=len(getattr(self.store, "quarantined", ())))
            raise
        self._audit("private_load_drained", rid=rid, pages=pages,
                    tokens=pages * self.entry.page_size)

    def offload(self, transfers):
        queued = super().offload(transfers)
        if queued:
            self._audit("offload", pages=len(transfers[0].keys), success=self.offloads[-1])
        return queued

    def close(self):
        if self.closed:
            return
        # A poisoned backend must retain mappings and allocations until process
        # teardown; close/reset are not proofs of GPU quiescence.
        self._open()
        if self.provider:
            self.store.close()  # Refuses shutdown while any remote reader pins it.
        if self.server:
            self.server.close()
        super().close()
