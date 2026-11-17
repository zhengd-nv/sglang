"""Start a single-node Tair KVCache Manager for serving and provision the SlotGuard instance.

Reuses the provisioning code of sgl-kv-indexer's live-test setup script but overrides what
serving needs: fixed ports reachable from the peer Prefill, block_size 64 (page 64), and a
liveness timeout well above the indexer's 30 s heartbeat (the test's 2 s would reclaim
placements between heartbeats and silently zero remote hits).
Usage: kvcm-serve.py --workdir DIR --binary BIN --sg-src SG_SRC --instance ID [--rpc-port 40017] [--reprovision]
Writes DIR/kvcm-state.json; the manager keeps running in its own session after exit.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

# Upstream #45414 moved experimental/sgl-router to the top level; sglang2 trees forked before it.
SETUP = "sgl-router/sgl-kv-indexer/tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py"


def load_setup(sg_src):
    path = next(
        (
            p
            for p in (
                os.path.join(sg_src, SETUP),
                os.path.join(sg_src, "experimental", SETUP),
            )
            if os.path.exists(p)
        ),
        os.path.join(sg_src, SETUP),
    )
    spec = importlib.util.spec_from_file_location("kvcm_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--sg-src", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--rpc-port", type=int, default=40017)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--reprovision", action="store_true")
    args = parser.parse_args()

    setup = load_setup(args.sg_src)
    setup.INSTANCE_ID = args.instance
    setup.INSTANCE_GROUP = f"{args.instance}_group"
    setup.STORAGE_L1P5 = f"{args.instance}_event_report_l1p5"
    setup.STORAGE_L2 = f"{args.instance}_event_report_l2"
    setup.BLOCK_SIZE = 64
    setup.MODEL_NAME = "GLM-5.2-NVFP4"
    # Indexer heartbeats every 30 s by default; give 3x headroom before a reporter is reclaimed.
    # snapshot_min_interval_ms: KVCM rate-limits empty snapshots (the indexer's CLEAR) per reporter,
    # 30 s by default; the 128K protocol resets both engines between back-to-back cases.
    setup.LIVENESS = {
        "heartbeat_timeout_ms": 90000,
        "cleanup_grace_ms": 30000,
        "liveness_check_interval_ms": 1000,
        "snapshot_min_interval_ms": 100,
    }
    if args.reprovision:  # apply the storage spec to an already running manager
        state = json.load(open(os.path.join(args.workdir, "kvcm-state.json")))
        setup.provision(state)
        print(json.dumps({"reprovisioned": True, "liveness": setup.LIVENESS}))
        return 0

    os.makedirs(args.workdir, exist_ok=True)
    ports = dict(
        rpc_port=args.rpc_port,
        http_port=args.rpc_port + 1,
        admin_rpc_port=args.rpc_port + 2,
        admin_http_port=args.rpc_port + 3,
    )
    registry = os.path.join(args.workdir, "registry.json")
    command = (
        [args.binary]
        + [f"--env=kvcm.service.{k}={v}" for k, v in ports.items()]
        + [
            f"--env=kvcm.registry_storage.uri=local://{registry}",
            "--env=kvcm.service.enable_debug_service=false",
            "--env=kvcm.logger.log_level=4",
        ]
    )
    with (
        open(os.path.join(args.workdir, "stdout.log"), "ab") as out,
        open(os.path.join(args.workdir, "stderr.log"), "ab") as err,
    ):
        process = subprocess.Popen(
            command, cwd=args.workdir, stdout=out, stderr=err, start_new_session=True
        )
    state = dict(
        ports,
        pid=process.pid,
        binary=args.binary,
        command=command,
        instance_id=setup.INSTANCE_ID,
        instance_group=setup.INSTANCE_GROUP,
        block_size=setup.BLOCK_SIZE,
        liveness=setup.LIVENESS,
    )
    with open(os.path.join(args.workdir, "kvcm-state.json"), "w") as f:
        json.dump(state, f, indent=1)
    setup.wait_ready(ports["http_port"], args.startup_timeout)
    setup.provision(state)
    print(
        json.dumps(
            {"ready": True, **{k: state[k] for k in ("pid", "rpc_port", "instance_id")}}
        )
    )


if __name__ == "__main__":
    sys.exit(main())
