"""Donate one NUMA node's Grace DRAM (EGM) to the Mooncake store pool.

Run one provider per NUMA node; each registers a segment named <host>:<port>.
The process holds the memory until it receives SIGTERM or SIGINT.
"""

import argparse
import signal
import socket
import threading

from mooncake.store import MooncakeDistributedStore

GIB = 1 << 30


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, help="host:port of mooncake_master")
    parser.add_argument("--metadata", required=True, help="HTTP metadata URL")
    parser.add_argument("--host", default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--numa", type=int, required=True)
    parser.add_argument("--pool-gib", type=int, default=64)
    args = parser.parse_args()

    store = MooncakeDistributedStore()
    rc = store.setup(
        {
            "local_hostname": f"{args.host}:{args.port}",
            "metadata_server": args.metadata,
            "master_server_addr": args.master,
            "protocol": "nvlink",
            "global_segment_size": str(args.pool_gib * GIB),
            "local_buffer_size": "0",
            "enable_egm_store_pool": "true",
            "egm_numa_nodes": str(args.numa),
        }
    )
    if rc != 0:
        print(f"EGM provider setup failed: rc={rc}", flush=True)
        return 1

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print(
        f"EGM provider ready: segment={args.host}:{args.port} numa={args.numa} "
        f"pool={args.pool_gib} GiB",
        flush=True,
    )
    stop.wait()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
