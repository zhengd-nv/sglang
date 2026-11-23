#!/usr/bin/env python3
"""Real two-engine TP1/MHA smoke; run on a host with SGLang + grpc installed.

Uses synthetic token IDs, /generate streaming, /flush_cache and authenticated
control /debug/l1-trace. Never toggles production routing or restarts engines.
The directory is the external tiny-model manifest (identities, ips, reporters,
control_port, expected_layout). All fault cases remain explicitly NOT_RUN.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path


def validate_manifest(manifest, serving_sha, instance_id):
    """Require externally measured provenance; never synthesize missing facts."""
    evidence = manifest.get("evidence", {})
    if type(evidence.get("schema")) is not int or evidence["schema"] != 1:
        raise AssertionError("manifest evidence schema 1 required")
    revisions = evidence.get("revisions", {})
    for name in (
        "source_base",
        "transport_reference",
        "kvcm_adapter_reference",
        "serving",
    ):
        if not isinstance(revisions.get(name), str) or not re.fullmatch(
            r"[0-9a-f]{40}", revisions[name]
        ):
            raise AssertionError("missing or invalid revision: " + name)
    if revisions["serving"] != serving_sha:
        raise AssertionError("serving revision differs from CLI")
    checkpoint = evidence.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
        raise AssertionError("checkpoint SHA256 required")
    identities, engines = manifest.get("identities", []), evidence.get("engines", [])
    if not (
        len(identities)
        == len(engines)
        == len(manifest.get("ips", []))
        == len(manifest.get("reporters", []))
        == 2
    ):
        raise AssertionError("exactly two engine provenance records required")
    for identity, engine in zip(identities, engines):
        for name in (
            "engine_id",
            "pod_uid",
            "physical_host",
            "gpu_uuid",
            "fabric_cluster_uuid",
        ):
            if not isinstance(engine.get(name), str) or not engine[name].strip():
                raise AssertionError("missing engine evidence: " + name)
        if (
            engine["engine_id"] != identity["engine_id"]
            or engine["fabric_cluster_uuid"] != identity["domain_id"]
            or engine.get("fabric_state") != "Completed"
            or engine.get("fabric_status") != "Success"
            or engine.get("serving_sha") != serving_sha
            or engine.get("checkpoint_sha256") != checkpoint
        ):
            raise AssertionError("engine provenance disagrees with deployment identity")
        if (
            not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", engine["gpu_uuid"])
            or str(uuid.UUID(engine["gpu_uuid"][4:])) != engine["gpu_uuid"][4:].lower()
            or str(uuid.UUID(engine["fabric_cluster_uuid"]))
            != engine["fabric_cluster_uuid"]
        ):
            raise AssertionError("invalid GPU/Fabric UUID")
        if identity["instance_id"] != instance_id or identity["tp_size"] != 1:
            raise AssertionError("instance/TP contract mismatch")
    for name in ("engine_id", "pod_uid", "physical_host", "gpu_uuid"):
        if engines[0][name].lower() == engines[1][name].lower():
            raise AssertionError("engines must have distinct " + name)
    if engines[0]["fabric_cluster_uuid"] != engines[1]["fabric_cluster_uuid"]:
        raise AssertionError("engines must share a Fabric domain")
    for name in ("model_id", "layout_digest"):
        if identities[0][name] != identities[1][name]:
            raise AssertionError("engine " + name + " mismatch")
    if not all(
        b["layout"].startswith("NHD:") for b in manifest["expected_layout"]["buffers"]
    ):
        raise AssertionError("smoke requires NHD layout")


def http(url, payload=None, *, token=None, raw=False):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode(),
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return body.decode() if raw else json.loads(body)


def consume_generate_stream(lines, rid, max_new_tokens, start, evidence):
    """Validate native cumulative /generate SSE through its explicit DONE marker."""
    output_ids, pending = [], None
    first, finished, done = None, False, False
    final_meta = None
    evidence.update(rid=rid, requested_tokens=max_new_tokens, frames=[], done=False)
    for line in lines:
        if len(line) > 65536 or not line.endswith(b"\n"):
            raise AssertionError("oversized or truncated SSE line")
        line = line.rstrip(b"\r\n")
        if line:
            if not line.startswith(b"data: ") or pending is not None:
                raise AssertionError("unexpected native SSE framing")
            pending = line[6:]
            continue
        if pending is None:
            continue
        payload, pending = pending, None
        if done:
            raise AssertionError("extra SSE frame after DONE")
        if payload == b"[DONE]":
            done = evidence["done"] = True
            if not finished:
                raise AssertionError("DONE without a complete terminal result")
            continue
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise TypeError("SSE data must be an object")
        # Save the received protocol evidence before validating it, including on
        # failure. Never retain request headers or arbitrary server-info values.
        frame = {
            key: result[key]
            for key in ("output_ids", "meta_info", "error")
            if key in result
        }
        evidence["frames"].append(frame)
        if len(evidence["frames"]) > 4 * max_new_tokens + 16:
            raise AssertionError("too many SSE frames")
        if result.get("error") is not None:
            raise AssertionError("generation returned SSE error")
        if finished:
            raise AssertionError("extra result after terminal SSE frame")
        meta, current = result.get("meta_info"), result.get("output_ids")
        if not isinstance(meta, dict) or meta.get("id") != rid:
            raise AssertionError("SSE request ID mismatch")
        if (
            not isinstance(current, list)
            or any(type(token) is not int or token < 0 for token in current)
            or type(meta.get("completion_tokens")) is not int
            or meta["completion_tokens"] != len(current)
            or len(current) > max_new_tokens
        ):
            raise AssertionError("SSE cumulative token count mismatch")
        if current[: len(output_ids)] != output_ids:
            raise AssertionError("SSE cumulative output changed or became a delta")
        output_ids = current
        if first is None and output_ids:
            first = time.monotonic_ns()
        finish = meta.get("finish_reason")
        if finish is not None:
            if (
                not isinstance(finish, dict)
                or finish.get("type") != "length"
                or type(finish.get("length")) is not int
                or finish["length"] != max_new_tokens
                or len(output_ids) != max_new_tokens
            ):
                raise AssertionError(
                    "SSE terminal reason or requested token count mismatch"
                )
            finished, final_meta = True, meta
    if pending is not None or not done or not finished or first is None:
        raise AssertionError("incomplete SSE stream")
    return {
        "rid": rid,
        "output_ids": output_ids,
        "ttft_ns": first - start,
        "total_ns": time.monotonic_ns() - start,
        "cached_tokens": final_meta.get("cached_tokens"),
        "completion_tokens": final_meta["completion_tokens"],
        "finish_reason": final_meta["finish_reason"],
        "done": True,
    }


def require_cumulative_mode(server_info):
    if server_info.get("incremental_streaming_output") is not False:
        raise AssertionError(
            "smoke requires explicit incremental_streaming_output=False"
        )


def generate(
    url, tokens, rid, max_new_tokens, *, incremental_streaming_output, evidence
):
    require_cumulative_mode(
        {"incremental_streaming_output": incremental_streaming_output}
    )
    payload = {
        "input_ids": tokens,
        "rid": rid,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise AssertionError("expected native SSE content type")
            result = consume_generate_stream(
                response, rid, max_new_tokens, start, evidence
            )
        evidence["status"] = "PASS"
        return result
    except Exception as exc:
        evidence.update(status="FAIL", error_type=type(exc).__name__)
        raise


def validate_trace(events, rid, expected_bytes, expected_buffers):
    events = [e for e in events if e.get("rid") == rid]
    required = [
        "lookup",
        "reservation",
        "pin_start",
        "pin_reply",
        "import_done",
        "copy_submitted",
        "copy_done",
        "verify_done",
        "imports_released",
        "unpin_ack",
        "backend_ready",
        "private_ready",
        "prefill_forward",
        "adopted",
        "checkpoint",
    ]
    ready_states = [
        e
        for e in events
        if e["phase"] == "backend_state" and e.get("status") == "READY"
    ]
    # Use a distinct logical phase without mutating source trace rows.
    phases = ["backend_ready" if e in ready_states else e["phase"] for e in events]
    for phase in required:
        if phase not in phases:
            raise AssertionError("missing transfer evidence: " + phase)
        if phase != "prefill_forward" and phases.count(phase) != 1:
            raise AssertionError("duplicate transfer phase: " + phase)
    if [phases.index(p) for p in required] != sorted(phases.index(p) for p in required):
        raise AssertionError("transfer evidence out of order")
    if len({e["attempt"] for e in events}) != 1:
        raise AssertionError("multiple transfer attempts")
    if len(ready_states) != 1 or any(
        e.get("status") in ("QUARANTINED", "FAILED_SAFE")
        for e in events
        if e["phase"] == "backend_state"
    ):
        raise AssertionError("missing READY or failed transfer state")
    for phase, allowed in (
        ("pin_reply", {"OK"}),
        ("unpin_ack", {"RELEASED", "ALREADY_RELEASED"}),
    ):
        rows = [e for e in events if e["phase"] == phase]
        if len(rows) != 1 or rows[0].get("status") not in allowed:
            raise AssertionError("invalid control response: " + phase)
    if (
        next(e for e in events if e["phase"] == "verify_done").get("verified")
        is not True
    ):
        raise AssertionError("verification completion is not successful")
    done = next(e for e in events if e["phase"] == "copy_done")
    if (
        done["status"] != "SUCCESS"
        or done["requested_bytes"] != expected_bytes
        or done["completed_bytes"] != expected_bytes
    ):
        raise AssertionError("native byte count or completion mismatch")
    if (
        next(e for e in events if e["phase"] == "copy_submitted").get("requested_bytes")
        != expected_bytes
    ):
        raise AssertionError("submitted byte count mismatch")
    verified = [e for e in events if e["phase"] == "verify_buffer"]
    if (
        len(verified) != expected_buffers
        or [e["buffers"] for e in verified] != list(range(1, expected_buffers + 1))
        or sum(e["requested_bytes"] for e in verified) != expected_bytes
        or not all(e["verified"] for e in verified)
    ):
        raise AssertionError("incomplete per-buffer byte oracle")
    if any(
        not phases.index("copy_done") < index < phases.index("verify_done")
        for index, event in enumerate(events)
        if event["phase"] == "verify_buffer"
    ):
        raise AssertionError("byte oracle outside copy/unmap interval")
    ready = next(e for e in events if e["phase"] == "private_ready")
    if (
        sum(e["prefill_tokens"] for e in events if e["phase"] == "prefill_forward")
        != ready["remaining_prefill_tokens"]
    ):
        raise AssertionError("actual prefill tokens do not match restored boundary")
    return events


def validate_source_trace(events, rid, transfer):
    rows = [e for e in events if e.get("rid") == rid]
    attempt = transfer[0]["attempt"]
    if not rows or any(e["attempt"] != attempt for e in rows):
        raise AssertionError("source attempt correlation mismatch")
    pins = [e for e in rows if e["phase"] == "source_pin"]
    releases = [e for e in rows if e["phase"] == "source_release"]
    if (
        len(pins) != 1
        or len(releases) != 1
        or rows.index(pins[0]) >= rows.index(releases[0])
        or releases[0].get("pinned_tokens") != 0
    ):
        raise AssertionError("source pin/release evidence invalid")
    reservation = next(e for e in transfer if e["phase"] == "reservation")
    lookup = next(e for e in transfer if e["phase"] == "lookup")
    if (
        pins[0].get("hit_tokens") != reservation.get("hit_tokens")
        or pins[0].get("hit_tokens") != lookup.get("hit_tokens")
        or pins[0].get("local_tokens") != reservation.get("local_tokens")
        or pins[0].get("pinned_tokens") != reservation.get("hit_tokens")
    ):
        raise AssertionError("source/target prefix boundary mismatch")
    return rows


class Evidence:
    def __init__(self, manifest, token):
        self.controls = [
            f"http://{ip}:{manifest['control_port']}" for ip in manifest["ips"]
        ]
        self.token, self.cursors = token, [0, 0]
        self.engine_ids = [identity["engine_id"] for identity in manifest["identities"]]
        self.events, self.status = [[], []], []

    def collect(self, *, initial=False):
        self.status = []
        for index, url in enumerate(self.controls):
            snapshot = http(url + "/debug/l1-trace", token=self.token)
            if snapshot.get("engine") != self.engine_ids[index] or any(
                e.get("engine") != self.engine_ids[index] for e in snapshot["events"]
            ):
                raise AssertionError("trace engine identity mismatch")
            if snapshot["dropped"] or snapshot["errors"]:
                raise AssertionError("trace sink dropped events or failed writing")
            fresh = [e for e in snapshot["events"] if e["seq"] > self.cursors[index]]
            if not initial and (
                snapshot["seq"] < self.cursors[index]
                or [e["seq"] for e in fresh]
                != list(range(self.cursors[index] + 1, snapshot["seq"] + 1))
            ):
                raise AssertionError("trace ring overrun, sequence gap, or reset")
            self.cursors[index] = snapshot["seq"]
            if not initial:
                self.events[index].extend(fresh)
            self.status.append({k: v for k, v in snapshot.items() if k != "events"})


def run(args, report):
    manifest = json.loads(Path(args.directory).read_text())
    validate_manifest(manifest, args.serving_sha, args.instance_id)
    import grpc

    from sglang.srt.mem_cache.l1_transfer.generated import (
        tair_kvcache_meta_service_pb2 as pb,
    )
    from sglang.srt.mem_cache.l1_transfer.generated import (
        tair_kvcache_meta_service_pb2_grpc as rpc,
    )
    from sglang.srt.mem_cache.l1_transfer.metadata import root_event_keys

    identities, layout = manifest["identities"], manifest["expected_layout"]
    page = layout["page_size"]
    assert args.prompt_tokens > page and args.prompt_tokens <= 2048
    # Deterministic non-special synthetic IDs, valid for the tiny vocab=256.
    tokens = [3 + ((i * 17 + 11) % 200) for i in range(args.prompt_tokens)]
    keys = root_event_keys(tokens, page)
    channel = grpc.insecure_channel(args.kvcm_endpoint)
    stub = rpc.MetaServiceStub(channel)
    query = pb.GetHostCacheStateRequest(
        instance_id=args.instance_id,
        trace_id="l1-smoke",
        query_type=pb.QT_PREFIX_MATCH,
        block_cache_keys=keys,
        medium=["hbm"],
        p2p_host_count=0,
    )

    def hosts():
        response = stub.GetHostCacheState(query, timeout=3)
        if response.header.status.code != pb.OK:
            raise AssertionError("KVCM status not OK")
        return {h.host_ip_port: h.local for h in response.hosts}

    def wait_hosts(predicate):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            value = hosts()
            if predicate(value):
                return value
            time.sleep(0.1)
        raise AssertionError("KVCM visibility deadline")

    evidence = Evidence(manifest, os.environ[args.bearer_env])
    evidence.collect(initial=True)
    urls = [args.source_url.rstrip("/"), args.target_url.rstrip("/")]
    # Preserve only known non-secret serving parameters, never raw launch_command
    # or arbitrary server_info values (which can carry credentials).
    safe_fields = (
        "model_path",
        "dtype",
        "kv_cache_dtype",
        "tp_size",
        "pp_size",
        "dp_size",
        "page_size",
        "max_total_tokens",
        "context_length",
        "max_running_requests",
        "chunked_prefill_size",
        "disable_cuda_graph",
        "enable_unified_cache_external_linker",
        "unified_cache_external_linker_backend",
        "incremental_streaming_output",
    )
    report["server_args"] = []
    for index, url in enumerate(urls):
        info = http(url + "/server_info")
        selected = {key: info[key] for key in safe_fields if key in info}
        report["server_args"].append(selected)
        require_cumulative_mode(selected)
        if selected.get("model_path") != identities[index]["model_id"]:
            raise AssertionError("live model path differs from directory")
    report.update(
        directory=manifest,
        directions=[],
        generations=[],
        fault_cases={
            name: {"status": "NOT_RUN", "cpu_test_file": file}
            for name, file in [
                ("evict_before_pin", "test_lease_manager.py"),
                ("bidirectional_busy", "test_lease_manager.py"),
                ("pin_timeout_late_reply", "test_nvlink_backend.py"),
                ("abort", "test_prefetch_admission.py"),
                ("unpin_ack_loss", "test_nvlink_backend.py"),
                ("lease_expiry", "test_lease_manager.py"),
            ]
        },
        router={"status": "NOT_RUN"},
        scope="TP1 synthetic MHA mechanism smoke; oracle overhead included; not performance or accuracy acceptance",
        baseline_scope="local compute with empty peer directory state; sharing feature remains enabled",
        publication_scope="KVCM absent before shared request, checkpoint completion traced, KVCM present afterward; no continuous KVCM absence proof",
    )
    reporters = manifest["reporters"]

    def checked_generate(url, rid):
        stream_evidence = {"rid": rid, "status": "STARTED"}
        report["generations"].append(stream_evidence)
        return generate(
            url,
            tokens,
            rid,
            args.max_new_tokens,
            incremental_streaming_output=False,
            evidence=stream_evidence,
        )

    try:
        for source, target in ((0, 1), (1, 0)):
            for url in urls:
                http(url + "/flush_cache", {}, raw=True)
            wait_hosts(lambda h: all(h.get(r, 0) == 0 for r in reporters))
            prefix = "l1-smoke-" + uuid.uuid4().hex
            baseline = checked_generate(urls[target], prefix + "-baseline")
            evidence.collect()
            if any(
                e.get("rid") == baseline["rid"] and e["phase"] == "copy_submitted"
                for e in evidence.events[target]
            ):
                raise AssertionError("baseline unexpectedly transferred")
            http(urls[target] + "/flush_cache", {}, raw=True)
            wait_hosts(lambda h: all(h.get(r, 0) == 0 for r in reporters))
            warmup = checked_generate(urls[source], prefix + "-warmup")
            before = wait_hosts(
                lambda h, source=source, target=target: (
                    h.get(reporters[source], 0) >= len(keys)
                    and h.get(reporters[target], 0) == 0
                )
            )
            evidence.collect()
            shared = checked_generate(urls[target], prefix + "-shared")
            evidence.collect()
            if (
                baseline["output_ids"] != shared["output_ids"]
                or warmup["output_ids"] != shared["output_ids"]
            ):
                raise AssertionError("local and shared greedy output mismatch")
            restored = (len(tokens) - 1) // page * page
            expected = (
                restored // page * sum(b["page_bytes"] for b in layout["buffers"])
            )
            transfer = validate_trace(
                evidence.events[target], shared["rid"], expected, len(layout["buffers"])
            )
            source_events = validate_source_trace(
                evidence.events[source], shared["rid"], transfer
            )
            after = wait_hosts(
                lambda h, target=target: h.get(reporters[target], 0) >= len(keys)
            )
            report["directions"].append(
                {
                    "source": source,
                    "target": target,
                    "status": "PASS",
                    "baseline": baseline,
                    "warmup": warmup,
                    "shared": shared,
                    "kvcm_before": before,
                    "kvcm_after": after,
                    "transfer": transfer,
                    "source_events": source_events,
                }
            )
        if args.router_url:
            # Independent routing stage after direct restore has been proved.
            http(urls[1] + "/flush_cache", {}, raw=True)
            wait_hosts(
                lambda h: (
                    h.get(manifest["reporters"][0], 0) >= len(keys)
                    and h.get(manifest["reporters"][1], 0) == 0
                )
            )
            routed = checked_generate(
                args.router_url.rstrip("/"),
                "l1-router-" + uuid.uuid4().hex,
            )
            evidence.collect()
            seen = [
                [
                    e
                    for e in rows
                    if e.get("rid") == routed["rid"] and e["phase"] == "discovery_start"
                ]
                for rows in evidence.events
            ]
            if len(seen[0]) != 1 or seen[1]:
                raise AssertionError("router did not select sole warm engine")
            report["router"] = {
                "status": "PASS",
                "result": routed,
                "selected_engine": identities[0]["engine_id"],
            }
        # Allow background writer to catch up; independent snapshot detects loss.
        deadline = time.monotonic() + 5
        while True:
            evidence.collect()
            if all(s["written"] == s["seq"] for s in evidence.status):
                break
            if time.monotonic() >= deadline:
                raise AssertionError("trace writer did not drain")
            time.sleep(0.05)
        report["status"] = "MAIN_PATH_PASS_TASK5_INCOMPLETE"
        if any(
            not any(e["phase"] == "scheduler_tick" for e in rows)
            for rows in evidence.events
        ):
            report["status"] = "FAIL"
            raise AssertionError("Scheduler sampling evidence absent")
    finally:
        report["trace_events"], report["trace_status"] = (
            evidence.events,
            evidence.status,
        )
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "source-url",
        "target-url",
        "kvcm-endpoint",
        "instance-id",
        "directory",
        "output",
    ):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--router-url")
    parser.add_argument("--bearer-env", default="SGLANG_NVLINK_CONTROL_TOKEN")
    parser.add_argument("--prompt-tokens", type=int, default=257)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--serving-sha",
        required=True,
        help="exact deployed source SHA, externally verified",
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 32 or not 0 < args.timeout <= 120:
        parser.error("max-new-tokens must be 1..32 and timeout in (0,120]")
    report = {
        "status": "FAIL",
        "args": vars(args),
        "started_unix_ns": time.time_ns(),
        "harness_repository_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        ).strip(),
        "harness_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    try:
        run(args, report)
    except Exception as exc:
        # Error type only: HTTP exceptions can include remote private payloads.
        report["error_type"] = type(exc).__name__
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
