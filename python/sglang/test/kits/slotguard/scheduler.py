"""SlotGuard integration fixtures: Scheduler method bodies and a serving composition.

`serving_method` executes checked-in Scheduler methods with CPU boundary fixtures;
`ServingFixture` composes a real guard, Tree and source service over CPU fakes.
"""

import ast
import copy
import json
import logging
import os
import tempfile
import time
from array import array
from collections import deque
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.mem_cache.allocator.slotguard_paged import (
    SlotGuardPagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TreeSlot
from sglang.srt.mem_cache.l1_transfer import config, serving_runtime
from sglang.srt.mem_cache.l1_transfer.control_profile import ControlProfile
from sglang.srt.mem_cache.l1_transfer.directory import (
    EngineDirectory,
    EngineDirectoryEntry,
)
from sglang.srt.mem_cache.l1_transfer.protocol import EngineIdentity
from sglang.srt.mem_cache.l1_transfer.slot_guard import SlotGuard
from sglang.srt.mem_cache.l1_transfer.transport.sglang_adapter import SGLangPoolAdapter
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.kits.slotguard.target import NvlinkEngine, pool
from sglang.test.kits.slotguard.wire import BorrowDriver, reference_native_hash

ROOT = Path(__file__).resolve().parents[5]


def serving_method(relative_path, name):
    path = ROOT / relative_path
    parsed = ast.parse(path.read_text())
    method = copy.deepcopy(
        next(
            n
            for n in ast.walk(parsed)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "deque": deque,
        "partial": partial,
        "time": time,
        "get_parallel": lambda: NS(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
        "PauseGenerationReqInput": PauseGenerationReqInput,
        "get_memory": lambda: NS(enable_flexkv=False),
        "get_disagg": lambda: NS(enable_pd_role_switch=False),
        "NextBatchPlan": lambda **kwargs: NS(**kwargs),
        "set_schedule_time_batch": lambda batch: None,
        "ListKind": __import__(
            "sglang.srt.managers.scheduler_components.request_list",
            fromlist=["ListKind"],
        ).ListKind,
        "current_platform": NS(empty_cache=lambda: None),
        "is_cp_sharded_padded": lambda: False,
        "logger": logging.getLogger(__name__),
        "CheckPhase": NS(FLUSH="flush", BUSY="busy"),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 -- execute checked-in method bodies without optional GPU imports
    return namespace[name]


def serving_identity(name, rank=0):
    return EngineIdentity(
        name,
        name + "-boot",
        "fabric",
        "instance",
        "model",
        "layout",
        name + "-tp",
        4,
        rank,
        f"shard-{rank}",
    )


def cache_fixture(name="source"):
    guard = SlotGuard(
        1,
        serving_identity(name).incarnation,
        max_remote_pinned_pages=32,
        max_active_leases=32,
        max_pending_reclaim_pages=32,
        max_pages=32,
        shared_headroom_credit=True,
    )
    allocator = SlotGuardPagedTokenToKVPoolAllocator(
        2048, 64, torch.float16, "cpu", None, False, slot_guard=guard
    )
    rows = ReqToTokenPool(4, 4096, "cpu", False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            False,
            rows,
            allocator,
            64,
            tree_slots=(TreeSlot(component="FULL", family=TREE_TOKEN),),
            tree_core_backend="python",
        )
    )
    return cache, allocator


class ServingFixture:
    """Source serving runtime over a guarded Tree and CPU transport fakes.

    Call setUp, then close (also on failure); attach_source starts the source side.
    """

    def setUp(self):
        self.hash_patch = patch(
            "sglang.srt.mem_cache.utils.get_native_hash", reference_native_hash
        )
        self.hash_patch.start()
        try:
            self.cache, self.allocator = cache_fixture()
            self.owner = object.__new__(serving_runtime.SlotGuardServingRuntime)
            self.owner.slotguard_mode = True
            self.owner.closed = False
            self.owner.observer = None
            self.owner.engine = NvlinkEngine(0, _driver=BorrowDriver())
            self.owner.adapter = SGLangPoolAdapter(pool(0), self.owner.engine)
            self.ctx = NS(
                params=NS(token_to_kv_pool_allocator=self.allocator, page_size=64),
                server_args=NS(enable_mixed_chunk=False),
                tp_size=4,
                is_dsa=True,
                tp_worker=NS(model_runner=NS(forward_stream=object())),
            )
            self.config = config.ServingConfig(
                EngineDirectory(
                    serving_identity("source"),
                    (
                        EngineDirectoryEntry(
                            "target",
                            serving_identity("target"),
                            "http://127.0.0.1:1234",
                        ),
                    ),
                ),
                ("127.0.0.1", 0),
                "127.0.0.1:50051",
                "test-only-credential",
                control_profile=ControlProfile.SLOTGUARD_HASH_V3,
            )
        except BaseException:
            self.hash_patch.stop()
            raise

    def close(self):
        try:
            self._close_owner()
        finally:
            self.hash_patch.stop()

    def _close_owner(self):
        for name in (
            "gateway",
            "backend",
            "client",
            "source_service",
            "publication_hook",
            "publication_worker",
            "arbiter",
            "adapter",
            "engine",
            "channel",
        ):
            resource = getattr(self.owner, name, None)
            if resource is not None:
                deadline = time.monotonic() + 2
                while True:
                    try:
                        resource.close()
                        break
                    except RuntimeError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.001)

    def attach_source(self):
        self.owner._initialize_slotguard_source(
            self.ctx, self.cache, self.config, NS(device="cpu")
        )


def manifest_data():
    def group(name, port):
        return {
            "reporter_id": name,
            "ranks": [
                {
                    "identity": asdict(serving_identity(name, rank)),
                    "control_endpoint": f"http://127.0.0.1:{port + rank}",
                }
                for rank in range(4)
            ],
        }

    return {
        "local_group": group("source", 31000),
        "peer_groups": [group("target", 32000)],
        "bind_host": "127.0.0.1",
        "kvcm_endpoint": "127.0.0.1:50051",
        "trusted_private_domain": True,
        "control_profile": "slotguard-hash-v3",
        "peer_control_profiles": {"target": "slotguard-hash-v3"},
    }


def load_manifest(data, rank=0):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(data))
        with patch.dict(
            os.environ,
            {
                "SGLANG_NVLINK_CONFIG": str(path),
                "SGLANG_NVLINK_CONTROL_TOKEN": "test-only-credential",
            },
        ):
            return config.ServingConfig.from_env(rank)


def insert_computed(cache, group, tokens):
    def body(_decision):
        alloc = cache.token_to_kv_pool_allocator
        indices = alloc.alloc(tokens)
        span = alloc.last_allocation_spans[0]
        return cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(tokens))),
                value=indices,
                page_span=span,
                inserted_len=tokens,
            )
        )

    return group.execute("test-computed-prefix", body).value
