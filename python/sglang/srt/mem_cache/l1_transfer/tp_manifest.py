"""Generate a credential-free GLM TP4 manifest; serving verifies actual layout.

This describes the one supported geometry, not a capacity or readiness claim.
Run once per engine, reversing local/peer arguments for the opposite direction.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import SlotGuardLimits, layout_digest
from .directory import EngineDirectoryEntry, EngineTPGroup
from .protocol import EngineIdentity
from .transport.layout import BufferLayout, PoolLayout


def glm_layout(rank, shard):
    buffers = tuple(
        BufferLayout(
            f"layer{layer}.{kind}",
            size,
            f"GLM-DSA-v1:{ordering}:tp=4:rank={rank}:shard={shard}",
        )
        for kind, size, ordering in (
            ("mla", 36864, "raw-fp8-e4m3:token[nope512,rope64]"),
            ("dsa_index", 8448, "page[key-fp8[64,128],scale-fp32[64]]"),
        )
        for layer in range(78)
    )
    return PoolLayout("GLM-MLA:fp8_e4m3;DSA:fp8_e4m3+fp32", 64, buffers)


def make_group(
    engine, incarnation, host, reporter, *, domain, instance, model, base_port
):
    ranks = tuple(
        EngineDirectoryEntry(
            reporter,
            EngineIdentity(
                engine,
                incarnation,
                domain,
                instance,
                model,
                layout_digest(glm_layout(rank, f"tp{rank}")),
                f"{engine}-tp4",
                4,
                rank,
                f"tp{rank}",
            ),
            f"http://{host}:{base_port + rank}",
        )
        for rank in range(4)
    )
    group = EngineTPGroup(reporter, ranks)
    return {
        "reporter_id": reporter,
        "ranks": [
            {
                "identity": asdict(entry.identity),
                "control_endpoint": entry.control_endpoint,
            }
            for entry in group.ranks
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-unpin",
        action="store_true",
        help="Wait for the Unpin ACK before READY (SlotGuard default is async)",
    )
    parser.add_argument("--max-pin-pages", type=int, default=4096)
    parser.add_argument("--max-remote-pinned-pages", type=int, default=4096)
    for role in ("local", "peer"):
        for field in ("engine", "incarnation", "host", "reporter"):
            parser.add_argument(f"--{role}-{field}", required=True)
        parser.add_argument(f"--{role}-base-port", type=int, required=True)
    for field in (
        "domain",
        "instance",
        "model",
        "bind-host",
        "kvcm-endpoint",
        "output",
    ):
        parser.add_argument(f"--{field}", required=True)
    args = parser.parse_args(argv)
    try:
        limits = SlotGuardLimits(
            max_pin_pages=args.max_pin_pages,
            max_remote_pinned_pages=args.max_remote_pinned_pages,
        )
    except ValueError as error:
        parser.error(str(error))
    groups = {}
    for role in ("local", "peer"):
        groups[role] = make_group(
            *(
                getattr(args, f"{role}_{name}")
                for name in ("engine", "incarnation", "host", "reporter")
            ),
            domain=args.domain,
            instance=args.instance,
            model=args.model,
            base_port=getattr(args, f"{role}_base_port"),
        )
    data = {
        "local_group": groups["local"],
        "peer_groups": [groups["peer"]],
        "bind_host": args.bind_host,
        "kvcm_endpoint": args.kvcm_endpoint,
        "trusted_private_domain": True,
        "bearer_env": "SGLANG_NVLINK_CONTROL_TOKEN",
        "min_headroom_tokens": 256,
    }
    profile = "slotguard-hash-v3"
    data.update(
        control_profile=profile,
        peer_control_profiles={args.peer_engine: profile},
        min_headroom_tokens=8192,
        prefetch_pending_capacity=32,
        prefetch_pipeline_depth=1,
        publication_context_entries=65536,
        publication_context_bytes=64 * 1024 * 1024,
        slotguard_limits=asdict(limits),
    )
    if args.sync_unpin:
        data["async_unpin"] = False
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
