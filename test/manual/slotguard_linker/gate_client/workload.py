"""GLM transfer prompt from the existing L1 harness; no admin/perf imports."""

import uuid
from collections.abc import Mapping

from .pd import Sampling


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model, trust_remote_code=True, local_files_only=True
    )


def new_rid(prefix):
    return f"{prefix}-{uuid.uuid4().hex}"


def transfer_sampling(max_new_tokens=8):
    return Sampling(
        max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, ignore_eos=True
    )


def transfer_prompt(tokenizer, length, nonce=None):
    nonce = nonce or uuid.uuid4().hex
    value = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": nonce
                + " "
                + "Explain addition using small integers. " * length,
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        require(len(value) == 1, "unexpected tokenizer batch")
        value = value[0]
    require(
        isinstance(value, list) and len(value) >= length, "prompt tokenizer underflow"
    )
    tokens = value[:length]
    require(
        all(type(t) is int and 0 <= t < len(tokenizer) for t in tokens),
        "invalid tokenizer IDs",
    )
    return tokens
