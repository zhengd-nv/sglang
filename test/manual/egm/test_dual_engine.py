# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Two real Engines: fixed peer, private synchronous restore, no HostPool.

Run on an allocated node with >=3 GPUs and EGM_NATIVE_LIB, EGM_RUNROOT set.
Uses one saved random-weight model shared by all Engines, not per-process dummy
weights. A metadata proxy injects a bad second-page range to test fail-stop
after a partial (synchronously drained) copy, without faking a CUDA failure.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import socketserver
import threading
import time
import traceback


def engine_worker(connection, kwargs):
    import sglang

    engine = None
    try:
        engine = sglang.Engine(**kwargs)
        connection.send({"ready": True})
        while True:
            command = connection.recv()
            if command is None:
                break
            connection.send(engine.generate(**command))
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        if engine:
            engine.shutdown()


class Worker:
    def __init__(self, kwargs):
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=engine_worker, args=(child, kwargs))
        self.process.start()
        child.close()
        result = self.receive(600)
        assert result.get("ready"), result

    def receive(self, timeout=180):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"Engine {self.process.pid} did not respond")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def generate(self, ids):
        self.connection.send(dict(
            input_ids=ids,
            sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            return_logprob=True,
        ))
        return self.receive()

    def close(self):
        if self.process.is_alive():
            self.connection.send(None)
            self.process.join(60)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(15)
        self.connection.close()


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class FaultProxy:
    def __init__(self, endpoint, secret):
        from sglang.srt.mem_cache.storage.egm.peer import PeerClient, receive

        client = PeerClient(endpoint, secret=secret)
        self.acquired = self.released = False
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                req = receive(self.rfile)
                assert req.pop("auth") == secret
                op = req.pop("op")
                value = client.rpc(op, **req)
                if op == "acquire":
                    owner.acquired = True
                    objects = list(value["objects"].values())
                    assert len(objects) > 1
                    # First page is copied; second page fails range validation.
                    objects[1]["offset"] = 1 << 60
                if op == "release":
                    owner.released = True
                self.wfile.write(json.dumps(dict(ok=True, value=value)).encode() + b"\n")

        self.server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.endpoint = self.server.server_address
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("hbm", "dram"), required=True)
    args = parser.parse_args()
    root = Path(os.environ["EGM_RUNROOT"])
    model = root / "shared-model"
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(20261012)
    config = LlamaConfig(
        vocab_size=128, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512,
    )
    LlamaForCausalLM(config).save_pretrained(model)
    secret = secrets.token_hex(32)
    common = dict(
        model_path=str(model), skip_tokenizer_init=True, tp_size=1,
        dtype="bfloat16", context_length=512, max_total_tokens=1024,
        mem_fraction_static=0.15, disable_cuda_graph=True,
        disable_overlap_schedule=True, page_size=16, random_seed=42,
        chunked_prefill_size=-1,
    )
    namespace = "dual-engine-" + secrets.token_hex(12)
    provider_audit, consumer_audit = root / "provider.jsonl", root / "consumer.jsonl"
    assert not provider_audit.exists() and not consumer_audit.exists(), "fresh run required"

    def linker_kwargs(device, audit, **extra):
        return dict(common, base_gpu_id=device,
                    enable_unified_cache_external_linker=True,
                    unified_cache_external_linker_backend="egm",
                    hicache_storage_backend_extra_config=json.dumps(dict(
                        namespace=namespace, audit_path=str(audit), peer_secret=secret, **extra)))

    workers = []
    proxy = None
    try:
        provider = Worker(linker_kwargs(
            0, provider_audit, listen_path=["127.0.0.1", 0],
            capacity_bytes=64 << 20, hbm_capacity_bytes=64 << 20,
            write_tier=args.tier,
        ))
        workers.append(provider)
        endpoint = next(e["endpoint"] for e in events(provider_audit) if e["op"] == "init")
        prefix = [4 + i % 80 for i in range(320)]
        prompt = prefix + [90 + i % 20 for i in range(32)]
        warm = provider.generate(prefix)
        deadline = time.monotonic() + 30
        while not any(e["op"] == "offload" and e["success"] for e in events(provider_audit)):
            if time.monotonic() > deadline:
                raise AssertionError("provider never offloaded KV")
            time.sleep(0.2)

        baseline = Worker(dict(common, base_gpu_id=2))
        workers.append(baseline)
        expected = baseline.generate(prompt)
        assert expected["meta_info"]["cached_tokens"] == 0
        baseline.close()
        workers.remove(baseline)

        consumer = Worker(linker_kwargs(1, consumer_audit, peer_path=endpoint))
        workers.append(consumer)
        miss = consumer.generate([110 + i % 10 for i in range(320)])
        assert miss["meta_info"]["cached_tokens"] == 0
        assert not any(e["op"] == "private_load_begin" for e in events(consumer_audit))
        restored = consumer.generate(prompt)
        assert restored["meta_info"]["cached_tokens"] == 320, restored
        details = restored["meta_info"]["cached_tokens_details"]
        assert [details[k] for k in ("device", "host", "storage")] == [0, 0, 320], restored
        assert restored["output_ids"] == expected["output_ids"], (restored, expected)
        expected_lp = [p[0] for p in expected["meta_info"]["output_token_logprobs"]]
        actual_lp = [p[0] for p in restored["meta_info"]["output_token_logprobs"]]
        assert max(abs(a - b) for a, b in zip(expected_lp, actual_lp)) < 0.02
        before = sum(e["op"] == "private_load_drained" for e in events(consumer_audit))
        local = consumer.generate(prompt)
        assert local["output_ids"] == expected["output_ids"]
        assert local["meta_info"]["cached_tokens"] >= 320
        assert sum(e["op"] == "private_load_drained" for e in events(consumer_audit)) == before
        init = [next(e for e in events(p) if e["op"] == "init") for p in (provider_audit, consumer_audit)]
        assert init[0]["schema"] == init[1]["schema"]
        consumer.close()
        workers.remove(consumer)

        # A metadata range fault after the first synchronous page copy must
        # terminate the consumer, not return generated output or a ready event.
        proxy = FaultProxy(endpoint, secret)
        failed_audit = root / "failed-consumer.jsonl"
        failed = Worker(linker_kwargs(1, failed_audit, peer_path=proxy.endpoint))
        workers.append(failed)
        try:
            failed.generate(prompt)
        except (EOFError, RuntimeError, ConnectionError):
            pass
        else:
            raise AssertionError("failed restore returned a successful response")
        failed.process.join(30)
        assert not failed.process.is_alive(), "failed Engine did not stop"
        failure_events = events(failed_audit)
        assert any(e["op"] == "private_load_failed" and e["quarantined_sessions"] == 1 for e in failure_events)
        assert not any(e["op"] == "private_load_drained" for e in failure_events)
        assert proxy.acquired and not proxy.released, "failed reader released its pin"
        result = dict(status="PASS", tier=args.tier, warm=warm, baseline=expected,
                      miss=miss, restored=restored, local=local,
                      failure=dict(exitcode=failed.process.exitcode, source_pin_retained=True),
                      provider_events=events(provider_audit), consumer_events=events(consumer_audit))
        (root / "dual-engine.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(status="PASS", tier=args.tier, restored_tokens=320)), flush=True)
    finally:
        if proxy:
            proxy.close()
        for worker in reversed(workers):
            worker.close()


if __name__ == "__main__":
    main()
