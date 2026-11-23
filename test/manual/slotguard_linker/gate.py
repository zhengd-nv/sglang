"""Production-only GLM gate: real discovery, remote restore and cache reset."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_transfer(report):
    trials = report["trials"]
    if len(trials) != 4:
        raise AssertionError("expected four directional restore trials")
    for trial in trials:
        remote = trial.get("remote", {})
        assert remote.get("status") == "COMPLETE", trial
        assert remote.get("output_ids"), trial
        assert remote.get("p_meta", {}).get("cached_tokens") == trial["hit_tokens"], (
            trial
        )
        assert trial.get("checks", {}).get("cold") is True, trial
        assert "kvcm_after_remote" in trial, trial


def validate_diag(rows):
    assert len(rows) == 6, "expected six deterministic comparisons"
    for row in rows:
        hit = (row["length"] - 1) // 64 * 64
        for name in ("p0_cold", "p0_hit", "p1_remote", "p1_hit"):
            sample = row[name]
            assert sample["status"] == "COMPLETE" and sample["out"], row
        assert row["p0_cold"]["cached"] == 0, row
        assert all(
            row[name]["cached"] == hit for name in ("p0_hit", "p1_remote", "p1_hit")
        ), row
        # Restored bytes are checked by validate_bytes. Outputs are compared only
        # within P1: two identical instances of main can decode differently.
        assert row["remote_eq_target_hit"], row


def validate_bytes(run, expected_rids):
    verified = set()
    rows = []
    for path in run.glob("p*-*/trace/nvlink.rank*"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("phase") == "verify_done":
                rows.append(event)
                assert event.get("verified") is True, event
                verified.add((event["rid"], event["tp_rank"]))
    missing = [
        (rid, rank)
        for rid in expected_rids
        for rank in range(4)
        if (rid, rank) not in verified
    ]
    assert not missing, f"missing byte evidence: {missing}"
    return dict(verified_events=len(rows), requests=len(expected_rids), missing=missing)


def flush(dep, output, tag):
    receipts = []
    for server in [*dep.prefills, *dep.decodes]:
        deadline = time.monotonic() + 90
        while True:
            response = requests.post(
                server.url + "/flush_cache?timeout=10",
                headers=dep.admin_headers,
                timeout=30,
            )
            receipts.append(
                dict(url=server.url, status=response.status_code, body=response.text)
            )
            write(output / f"flush-{tag}.json", receipts)
            if response.status_code == 200:
                break
            if response.status_code != 400 or time.monotonic() >= deadline:
                response.raise_for_status()
            time.sleep(1)


def main(run, config_path):
    # In-tree HTTP/KVCM helpers only. Keep the raw cold-versus-hit verdict.
    from gate_client import config
    from gate_client.kvcm import KVCMDirectory, wait_hosts
    from gate_client.pd import Request, dispatch_group
    from gate_client.transfer import run_transfer_check
    from gate_client.workload import (
        load_tokenizer,
        new_rid,
        transfer_prompt,
        transfer_sampling,
    )

    output = run / "driver"
    output.mkdir(exist_ok=True)
    dep = config.load(str(config_path))
    if len(dep.prefills) != 2 or len(dep.decodes) != 1:
        raise ValueError("This gate requires exactly two prefills and one decode")
    tokenizer = load_tokenizer(dep.model)
    info = dict(
        model=dep.model,
        vocab_size=len(tokenizer),
        template_sha256=hashlib.sha256(
            str(tokenizer.chat_template).encode()
        ).hexdigest(),
    )
    directory = KVCMDirectory(dep.kvcm_endpoint, dep.kvcm_instance)
    write(output / "deployment.json", dict(deployment=dep.describe(), tokenizer=info))
    expected_rids = []
    rows = []
    try:
        flush(dep, output, "before-transfer")
        transfer = run_transfer_check(
            dep,
            output / "transfer-e",
            tokenizer=tokenizer,
            directory=directory,
            kvcm_timeout=60,
        )
        validate_transfer(transfer)
        expected_rids.extend(t["remote"]["rid"] for t in transfer["trials"])
        paused = []
        receipts = []
        try:
            for server in dep.prefills:
                response = requests.post(
                    server.url + "/pause_generation",
                    json={"mode": "in_place"},
                    headers=dep.admin_headers,
                    timeout=60,
                )
                response.raise_for_status()
                paused.append(server)
                receipts.append(
                    dict(url=server.url, action="pause", status=response.status_code)
                )
            flush(dep, output, "while-paused")
        finally:
            for server in paused:
                response = requests.post(
                    server.url + "/continue_generation",
                    json={"torch_empty_cache": False},
                    headers=dep.admin_headers,
                    timeout=60,
                )
                receipts.append(
                    dict(url=server.url, action="continue", status=response.status_code)
                )
                write(output / "pause-continue.json", receipts)
                response.raise_for_status()

        def generate(engine, tokens, tag):
            request = Request(
                new_rid("i4-diag"),
                engine,
                transfer_sampling(8),
                input_ids=tuple(tokens),
                tag=dict(role=tag),
            )
            sample = dispatch_group([request], dep.prefills, dep.decode, timeout=300)[
                "samples"
            ][0]
            return dict(
                rid=sample["rid"],
                out=list(sample.get("output_ids") or []),
                status=sample.get("status"),
                cached=(sample.get("p_meta") or {}).get("cached_tokens"),
            )

        for trial in range(3):
            for length in (257, 1025):
                tokens = transfer_prompt(tokenizer, length)
                hit = (length - 1) // 64 * 64
                row = dict(trial=trial, length=length)
                rows.append(row)
                row["p0_cold"] = generate(0, tokens, "p0_cold")
                row["kvcm"] = wait_hosts(
                    directory,
                    tokens[:hit],
                    lambda h: h.get(dep.reporters()[0], 0) >= hit // 64,
                    timeout=60,
                )
                for name, engine in (("p1_remote", 1), ("p1_hit", 1), ("p0_hit", 0)):
                    row[name] = generate(engine, tokens, name)
                expected_rids.append(row["p1_remote"]["rid"])
                row["remote_eq_target_hit"] = (
                    row["p1_remote"]["out"] == row["p1_hit"]["out"]
                )
                row["remote_eq_source_hit"] = (
                    row["p1_remote"]["out"] == row["p0_hit"]["out"]
                )
                row["target_hit_eq_source_hit"] = (
                    row["p1_hit"]["out"] == row["p0_hit"]["out"]
                )
                row["remote_eq_cold"] = row["p1_remote"]["out"] == row["p0_cold"]["out"]
                write(
                    output / "diag-determinism.json",
                    dict(B_sharing_on_verify_bytes=rows),
                )
                print(
                    "diag",
                    trial,
                    length,
                    row["remote_eq_target_hit"],
                    row["remote_eq_source_hit"],
                    flush=True,
                )
        validate_diag(rows)
        flush(dep, output, "after-gate")
        byte_evidence = validate_bytes(run, expected_rids)
        write(
            output / "gate.json",
            dict(
                status="PASS",
                restored=4,
                comparisons=6,
                raw_transfer_status=transfer["status"],
                bytes=byte_evidence,
            ),
        )
        (run / "complete.txt").write_text("complete\n")
    except BaseException as exc:
        write(output / "failure.json", dict(error=repr(exc), diag=rows))
        (run / "failed.txt").write_text(repr(exc) + "\n")
        raise
    finally:
        directory.close()
        (run / "shared/stop").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Fresh shared run directory containing p*/trace/nvlink.rank* logs",
    )
    parser.add_argument("--config", required=True, type=Path, help="Deployment JSON")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "shared").mkdir(exist_ok=True)
    if (args.run_dir / "driver").exists():
        parser.error("run-dir already contains driver output; choose a fresh run")
    main(args.run_dir, args.config)
