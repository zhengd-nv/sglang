"""Transfer correctness: remote-restored greedy output must equal the local reference.

Port of ``--case transfer`` in test/manual/l1_transfer/test-glm52_nvfp4-sharing.py:709-825.
For (source, target) in ((0, 1), (1, 0)) and length in (257, 1025):
fresh nonce prompt (cold in KVCM for both reporters) -> greedy 8 tokens on the
source P (local reference) -> KVCM shows source >= H/64 pages, target 0 ->
same prompt on the target P (restores [0, H) from source, H = (len-1)//64*64)
-> outputs must be identical -> KVCM shows target >= H/64 pages.

Byte verification is enabled at launch. The outer gate checks trace files;
this client never uses debug/admin HTTP endpoints.
"""

import hashlib
import json
import time
from pathlib import Path

from . import records
from .kvcm import KVCMError, wait_hosts
from .pd import GroupFailed, Request, dispatch_group
from .workload import new_rid, transfer_prompt, transfer_sampling

PAGE = 64
DIRECTIONS = ((0, 1), (1, 0))


def _generate(dispatch, dep, engine, tokens, sampling, timeout, stream_mode, tag):
    request = Request(
        new_rid("l1-transfer"), engine, sampling, input_ids=tuple(tokens), tag=tag
    )
    result = dispatch(
        [request], dep.prefills, dep.decode, timeout=timeout, stream_mode=stream_mode
    )
    return result["samples"][0]


def run_transfer_check(
    dep,
    output,
    *,
    tokenizer,
    directory=None,
    lengths=(257, 1025),
    max_new_tokens=8,
    timeout=300.0,
    kvcm_timeout=30.0,
    stream_mode="auto",
    dispatch=dispatch_group,
    log=print,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reporters = dep.reporters()
    use_kvcm = directory is not None and bool(reporters)
    sampling = transfer_sampling(max_new_tokens)
    report = dict(
        deployment=dep.describe(),
        lengths=list(lengths),
        kvcm_checked=use_kvcm,
        trace_collected=False,
        trials=[],
        created_unix=time.time(),
    )
    first_pages = set()
    try:
        for source, target in DIRECTIONS:
            for length in lengths:
                trial = _run_trial(
                    dep,
                    output,
                    tokenizer,
                    source,
                    target,
                    length,
                    trials=report["trials"],
                    first_pages=first_pages,
                    sampling=sampling,
                    directory=directory if use_kvcm else None,
                    reporters=reporters,
                    dispatch=dispatch,
                    timeout=timeout,
                    kvcm_timeout=kvcm_timeout,
                    stream_mode=stream_mode,
                )
                log(f"[transfer] {source}->{target} len={length}: {trial['status']}")
    finally:
        trials = report["trials"]
        report["status"] = (
            "PASS"
            if len(trials) == len(DIRECTIONS) * len(lengths)
            and all(t["status"] == "PASS" for t in trials)
            else "FAIL"
        )
        records.write_json(output / "summary.json", report)
    return report


def _run_trial(
    dep,
    output,
    tokenizer,
    source,
    target,
    length,
    *,
    trials,
    first_pages,
    sampling,
    directory,
    reporters,
    dispatch,
    timeout,
    kvcm_timeout,
    stream_mode,
):
    tokens = transfer_prompt(tokenizer, length)
    if tuple(tokens[:PAGE]) in first_pages:
        raise RuntimeError("nonce did not distinguish the first token page")
    first_pages.add(tuple(tokens[:PAGE]))
    hit = (length - 1) // PAGE * PAGE
    prefix = tokens[:hit]
    trial = dict(
        source=source,
        target=target,
        prompt_tokens=len(tokens),
        hit_tokens=hit,
        prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        status="STARTED",
        checks={},
    )
    trials.append(trial)
    tag = dict(source=source, target=target, length=length)
    try:
        if directory is not None:
            before = directory.hosts(prefix)
            trial["kvcm_initial"] = before
            trial["checks"]["cold"] = all(before.get(r, 0) == 0 for r in reporters)
        local = _generate(
            dispatch,
            dep,
            source,
            tokens,
            sampling,
            timeout,
            stream_mode,
            dict(tag, role="local_reference"),
        )
        trial["local"] = records.request_record(local)
        if directory is not None:
            trial["kvcm_after_local"] = wait_hosts(
                directory,
                prefix,
                lambda h: (
                    h.get(reporters[source], 0) >= hit // PAGE
                    and h.get(reporters[target], 0) == 0
                ),
                timeout=kvcm_timeout,
            )
        remote = _generate(
            dispatch,
            dep,
            target,
            tokens,
            sampling,
            timeout,
            stream_mode,
            dict(tag, role="remote"),
        )
        trial["remote"] = records.request_record(remote)
        trial["checks"]["output_match"] = remote["output_ids"] == local["output_ids"]
        if directory is not None:
            trial["kvcm_after_remote"] = wait_hosts(
                directory,
                prefix,
                lambda h: h.get(reporters[target], 0) >= hit // PAGE,
                timeout=kvcm_timeout,
            )
        trial["status"] = "PASS" if all(trial["checks"].values()) else "FAIL"
    except GroupFailed as exc:
        trial.update(
            status="ERROR",
            error=str(exc),
            samples=[records.request_record(s) for s in exc.result["samples"]],
        )
        raise
    except (KVCMError, AssertionError) as exc:
        trial.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    return trial
