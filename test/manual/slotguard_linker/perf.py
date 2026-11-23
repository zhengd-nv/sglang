"""GLM 2P1D TTFT driver: remote L1 restore versus the target's own prefill.

Run by launch.sh with DRIVER=perf, which starts the servers without byte
verification or tracing. Each round flushes every server, then sends C fresh
prompts per lane, all lanes on the same prompt set except local_off:

  source_cold  set X on P0 (publishes the prefix)
  remote       set X on P1 after KVCM lists P0 for it (L1 restore)
  local_off    a fresh set Y on P1 (no shared prefix: the target's own prefill)
  target_hit   set X on P1 again (local radix hit)
  source_hit   set X on P0 again; its outputs are compared with remote
"""

import argparse
import json
import os
import statistics
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

DEFAULT_PLAN = "128000x8x3 256000x2x3 512000x2x3"
FILLER = "Explain addition using small integers. "
LANES = ("source_cold", "remote", "local_off", "target_hit", "source_hit")


def write(path, value):
    path.write_text(json.dumps(value, indent=1) + "\n")


def parse_plan(text):
    plan = []
    for item in text.split():
        length, concurrency, rounds = (int(v) for v in item.split("x"))
        plan.append((length, concurrency, rounds))
    return plan


def prompt(tokenizer, length):
    # The nonce leads the prompt, so no two prompts share their first page.
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": uuid.uuid4().hex + " " + FILLER * (length // 6 + 64)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    if len(value) < length:
        raise AssertionError(f"prompt underflow: {len(value)} < {length}")
    return value[:length]


def summarize(samples):
    ttft = [s["first_token"] - s["send"] for s in samples]
    e2e = [s["completion"] - s["send"] for s in samples]
    return dict(
        n=len(samples),
        ttft_mean=statistics.fmean(ttft),
        ttft_p50=statistics.median(ttft),
        ttft_max=max(ttft),
        e2e_mean=statistics.fmean(e2e),
        cached=[(s.get("p_meta") or {}).get("cached_tokens") for s in samples],
    )


def main(run, config_path, plan):
    from gate_client import config
    from gate_client.kvcm import KVCMDirectory, wait_hosts
    from gate_client.pd import Request, dispatch_group
    from gate_client.workload import load_tokenizer, new_rid, transfer_sampling

    from gate import flush

    output = run / "driver"
    raw = output / "perf"
    raw.mkdir(parents=True)
    dep = config.load(str(config_path))
    tokenizer = load_tokenizer(dep.model)
    directory = KVCMDirectory(dep.kvcm_endpoint, dep.kvcm_instance)
    reporter = dep.reporters()[0]
    sampling = transfer_sampling(int(os.environ.get("PERF_NEW_TOKENS", "32")))
    write(output / "deployment.json", dict(deployment=dep.describe(), plan=plan))

    def send(engine, prompts, lane):
        requests_ = [
            Request(new_rid(f"perf-{lane}"), engine, sampling, input_ids=tuple(p), tag=dict(lane=lane))
            for p in prompts
        ]
        return dispatch_group(requests_, dep.prefills, dep.decode, timeout=3600)["samples"]

    summary = []
    try:
        for length, concurrency, rounds in plan:
            hit = (length - 1) // 64 * 64
            for r in range(rounds):
                tag = f"{length // 1000}k-c{concurrency}-r{r}"
                flush(dep, raw, tag)
                xs = [prompt(tokenizer, length) for _ in range(concurrency)]
                ys = [prompt(tokenizer, length) for _ in range(concurrency)]
                lanes = dict(source_cold=send(0, xs, "source_cold"))
                placement = time.monotonic()
                for tokens in xs:
                    wait_hosts(
                        directory,
                        tokens[:hit],
                        lambda h: h.get(reporter, 0) >= hit // 64,
                        timeout=300,
                    )
                placement_s = time.monotonic() - placement
                lanes["remote"] = send(1, xs, "remote")
                lanes["local_off"] = send(1, ys, "local_off")
                lanes["target_hit"] = send(1, xs, "target_hit")
                lanes["source_hit"] = send(0, xs, "source_hit")
                write(raw / f"{tag}.json", lanes)
                row = dict(
                    length=length,
                    concurrency=concurrency,
                    round=r,
                    placement_wait_s=placement_s,
                    lanes={k: summarize(lanes[k]) for k in LANES},
                    remote_eq_target_hit=[
                        a["output_ids"] == b["output_ids"]
                        for a, b in zip(lanes["remote"], lanes["target_hit"])
                    ],
                    remote_eq_source_hit=[
                        a["output_ids"] == b["output_ids"]
                        for a, b in zip(lanes["remote"], lanes["source_hit"])
                    ],
                )
                summary.append(row)
                write(output / "perf.json", summary)
                print(tag, {k: round(v["ttft_mean"], 3) for k, v in row["lanes"].items()}, flush=True)
        flush(dep, output, "after-perf")
        (run / "complete.txt").write_text("complete\n")
    except BaseException as exc:
        write(output / "failure.json", dict(error=repr(exc)))
        (run / "failed.txt").write_text(repr(exc) + "\n")
        raise
    finally:
        directory.close()
        (run / "shared/stop").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Deployment JSON")
    parser.add_argument("--plan", default=os.environ.get("PERF_PLAN", DEFAULT_PLAN),
                        help="Space-separated <tokens>x<concurrency>x<rounds>")
    args = parser.parse_args()
    if (args.run_dir / "driver").exists():
        parser.error("run-dir already contains driver output; choose a fresh run")
    main(args.run_dir, args.config, parse_plan(args.plan))
