"""Durable result files: fsynced JSONL appends and atomic JSON writes."""

import gzip
import json
import os
from pathlib import Path


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=list)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(_dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=list
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json_gz(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(_dumps(value))
    return Path(path).name


def request_record(sample: dict, **context) -> dict:
    """One JSONL row per request; derived TTFT/TPOT use the upstream definitions."""
    row = dict(context, **sample)
    send, first, done = (
        sample.get("send"),
        sample.get("first_token"),
        sample.get("completion"),
    )
    count = len(sample.get("output_ids") or [])
    row["ttft_s"] = first - send if None not in (send, first) else None
    row["tpot_s"] = (
        (done - first) / (count - 1)
        if None not in (first, done) and count > 1
        else None
    )
    row["e2e_s"] = done - send if None not in (send, done) else None
    return row
