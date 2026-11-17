"""Execute checked-in Scheduler methods with CPU boundary fixtures."""

import ast
import copy
import logging
import time
from collections import deque
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from sglang.srt.managers.io_struct import PauseGenerationReqInput

ROOT = Path(__file__).resolve().parents[4]


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
