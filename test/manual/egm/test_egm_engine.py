# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import _bootstrap  # Optional installed-runtime overlay for manual tests.

"""Actual Engine smoke: dynamic EGM backend + small locally generated model."""
import argparse
import json
import os
from pathlib import Path
import time

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, PreTrainedTokenizerFast
import sglang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("dram", "hbm"), default="dram")
    args = parser.parse_args()
    root = Path(os.environ["EGM_RUNROOT"])
    audit = root / "engine-storage-audit.jsonl"
    if audit.exists():
        raise FileExistsError("use a fresh run directory for Engine tests")
    model = root / "tiny-model"
    model.mkdir(exist_ok=True)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({f"token{i}": i for i in range(4, 128)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                           unk_token="<unk>", pad_token="<pad>").save_pretrained(model)
    cfg = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    cfg.architectures = ["LlamaForCausalLM"]
    cfg.save_pretrained(model)
    engine = sglang.Engine(model_path=str(model), load_format="dummy", tp_size=1,
        dtype="bfloat16", context_length=512, max_total_tokens=1024, mem_fraction_static=0.15,
        disable_cuda_graph=True, page_size=16, enable_hierarchical_cache=True,
        hicache_ratio=2, hicache_mem_layout="page_first", hicache_io_backend="kernel",
        hicache_write_policy="write_through", hicache_storage_backend="dynamic",
        hicache_storage_backend_extra_config=json.dumps({
            "backend_name": "egm", "module_path": "sglang.srt.mem_cache.storage.egm.hicache", "class_name": "EGMHiCacheStorage",
            "interface_v1": True, "capacity_bytes": 64 << 20,
            "hbm_capacity_bytes": 64 << 20 if args.tier == "hbm" else 0, "write_tier": args.tier,
            "namespace": "tiny-engine-smoke", "numa": 0,
            "audit_path": str(audit),
        }))
    try:
        prompt = " ".join(["token4", "token5", "token6", "token7"] * 80)
        responses = []
        for iteration in range(2):
            response = engine.generate(prompt, sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True})
            assert response["meta_info"]["completion_tokens"] == 8, response
            responses.append(response)
            time.sleep(2)  # Let asynchronous HiCache write-through finish.
            if iteration == 0:
                for _ in range(30):
                    events = [json.loads(line) for line in audit.read_text().splitlines()]
                    if any(e["op"] == "put" and e["success"] > 0 for e in events):
                        break
                    time.sleep(1)
                else:
                    raise AssertionError("Engine did not write any pages to EGM")
                assert engine.flush_cache(), "flush refused"
        assert responses[0]["output_ids"] == responses[1]["output_ids"]
        events = [json.loads(line) for line in audit.read_text().splitlines()]
        assert any(e["op"] == "get" and e["success"] > 0 for e in events), events
        (root / "engine-egm.json").write_text(json.dumps({"status": "PASS", "tier": args.tier, "responses": responses, "storage_events": events}, indent=2))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
