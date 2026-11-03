"""Show one server reusing a prefix that another server computed.

1. The writer prefills a long prompt; write-through offloads its KV pages to
   the shared EGM pool.
2. The writer answers the same prompt again from its own GPU cache: this is the
   reference output.
3. The reader, whose GPU cache is flushed, answers the same prompt. It should
   find the prefix in the pool, load it over NVLink instead of recomputing,
   and produce the reference output token for token.

Only the standard library is needed, so this runs from any host that can
reach both servers.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

PAGE = 64


def post(url, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw.strip().startswith(b"{") else raw.decode()


def flush(url):
    # The server refuses to flush while a previous request is still finishing.
    for _ in range(40):
        try:
            post(url, "/flush_cache")
            return
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
            time.sleep(0.05)
    raise RuntimeError(f"{url}: flush_cache kept returning 400")


def document(num_tokens_hint, seed):
    # Natural-looking text keeps greedy decoding away from near-tied logits,
    # which random token ids would hit.
    rng = random.Random(seed)
    colors = ["red", "blue", "green", "amber", "violet", "silver", "black", "white"]
    things = [
        "lantern",
        "compass",
        "teapot",
        "bicycle",
        "violin",
        "kettle",
        "anchor",
        "atlas",
    ]
    cities = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Hanoi", "Perth"]
    lines = []
    for i in range(num_tokens_hint // 14 + 1):
        lines.append(
            f"Entry {i}: a {rng.choice(colors)} {rng.choice(things)} was shipped "
            f"from {rng.choice(cities)} weighing {rng.randint(1, 999)} grams."
        )
    return "Inventory log.\n" + "\n".join(lines)


def generate(url, input_ids, new_tokens):
    start = time.perf_counter()
    out = post(
        url,
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": new_tokens,
                "ignore_eos": True,
            },
        },
    )
    return {
        "latency_ms": round((time.perf_counter() - start) * 1e3, 1),
        "output_ids": out["output_ids"],
        "cached_tokens": out["meta_info"].get("cached_tokens"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--writer", required=True, help="URL of the server that computes the prefix"
    )
    parser.add_argument(
        "--reader", required=True, help="URL of the server that reuses it"
    )
    parser.add_argument("--prefix-tokens", type=int, default=16384)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="untimed rounds with other documents before the measured one",
    )
    parser.add_argument(
        "--offload-wait",
        type=float,
        default=3.0,
        help="seconds to let write-through finish before the reader asks",
    )
    args = parser.parse_args()

    prefix_len = args.prefix_tokens // PAGE * PAGE
    question = post(
        args.writer,
        "/tokenize",
        {"prompt": "\nQuestion: which city shipped the most items? Answer:"},
    )["tokens"]

    def prompt(seed):
        tokens = post(args.writer, "/tokenize", {"prompt": document(prefix_len, seed)})[
            "tokens"
        ]
        if len(tokens) < prefix_len:
            sys.exit(f"document too short: {len(tokens)} < {prefix_len} tokens")
        return tokens[:prefix_len] + question

    def share(input_ids):
        for url in (args.writer, args.reader):
            flush(url)
        cold = generate(args.writer, input_ids, args.new_tokens)
        time.sleep(args.offload_wait)
        reference = generate(args.writer, input_ids, args.new_tokens)
        flush(args.reader)
        return cold, reference, generate(args.reader, input_ids, args.new_tokens)

    # The first request on each server pays for kernel compilation and, on the
    # reader, for mapping the provider's memory; keep that out of the numbers.
    for seed in range(1, args.warmup_rounds + 1):
        share(prompt(seed))
    cold, reference, shared = share(prompt(0))

    checks = {
        "reader hit the shared prefix": (shared["cached_tokens"] or 0)
        >= prefix_len - PAGE,
        "reader output matches the writer's cached output": shared["output_ids"]
        == reference["output_ids"],
    }
    summary = {
        "prefix_tokens": prefix_len,
        "writer_cold": cold,
        "writer_cached": reference,
        "reader_from_pool": shared,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(summary, indent=1))
    print(
        f"DEMO_RESULT {'PASS' if summary['passed'] else 'FAIL'} prefix={prefix_len} "
        f"writer_cold={cold['latency_ms']}ms reader_from_pool={shared['latency_ms']}ms "
        f"reader_cached_tokens={shared['cached_tokens']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
