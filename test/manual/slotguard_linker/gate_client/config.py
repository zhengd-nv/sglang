"""Deployment JSON for the two-prefill/one-decode production gate."""

import json
from dataclasses import dataclass
from pathlib import Path

from .pd import Server


@dataclass
class Deployment:
    prefills: list
    decodes: list
    engine_configs: list
    kvcm_endpoint: str
    kvcm_instance: str
    model: str
    # Bearer header for admin routes; prefill servers set SGLANG_ADMIN_API_KEY from this file.
    admin_headers: dict

    @property
    def decode(self):
        return self.decodes[0]

    def reporters(self):
        return [c["local_group"]["reporter_id"] for c in self.engine_configs]

    def describe(self):
        return dict(
            prefill=[s.url for s in self.prefills],
            decode=[s.url for s in self.decodes],
            reporters=self.reporters(),
            kvcm_endpoint=self.kvcm_endpoint,
            kvcm_instance=self.kvcm_instance,
            model=self.model,
        )


def load(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())

    def manifest(entry):
        value = Path(entry["nvlink_config"])
        if not value.is_absolute():
            value = path.parent / value
        return json.loads(value.read_text())

    return Deployment(
        prefills=[
            Server(e["url"], e.get("bootstrap_host"), e.get("bootstrap_port", 8998))
            for e in data["prefill"]
        ],
        decodes=[Server(e["url"]) for e in data["decode"]],
        engine_configs=[manifest(e) for e in data["prefill"]],
        kvcm_endpoint=data["kvcm"]["endpoint"],
        kvcm_instance=data["kvcm"]["instance_id"],
        model=data["model"],
        admin_headers=admin_headers(path.parent, data.get("admin_key_file")),
    )


def admin_headers(base, key_file):
    if not key_file:
        return {}
    value = Path(key_file)
    if not value.is_absolute():
        value = base / value
    return {"Authorization": f"Bearer {value.read_text().strip()}"}
