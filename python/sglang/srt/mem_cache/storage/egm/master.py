# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Domain master for EGM store services: membership, prefix directory and placement.

One process per NVLink domain, metadata only. Stores keep owning slots, pins and
leases, and readers still ``acquire`` by key at the store, which returns the slot
offsets under a pin. A stale directory entry therefore costs a miss (the reader
recomputes), never a read of the wrong slot, so the directory carries no slot
offsets or allocation generations.

* Membership: a store registers with its slab epoch and reports at least once
  per heartbeat. A store silent for ``suspect_s`` is no longer offered for
  placement and is used for lookups only when no other holder exists; after
  ``offline_s`` it is dropped with every key it held. Registering with a new
  epoch (a restarted store) drops the old keys as well.
* Directory: stores report the keys they commit and evict, and send a full
  snapshot whenever they (re)register, so a restarted master is rebuilt by its
  stores within one heartbeat.
* Lookup: the longest prefix of the given page keys held by live stores, one
  holder per page, preferring the caller's node and NUMA.
* Placement: the live store with the most free bytes, excluding the stores the
  caller already tried. Free bytes come from the last report, minus what was
  placed since, so concurrent writers spread out between heartbeats.

Control is line-delimited JSON over TCP with the job secret, as in
``service.py``. Usage::

    python -m sglang.srt.mem_cache.storage.egm.master --tcp 0.0.0.0:29700 \\
        --secret-file /shared/job/egm.secret
"""

import argparse
import json
import os
import signal
import socket
import socketserver
import threading
import time
from collections import Counter

HEARTBEAT_S = 1.0
# Same two-stage liveness as the Mooncake master (10 s suspect, 30 s offline).
SUSPECT_S = 10.0
OFFLINE_S = 30.0

ACTIVE, SUSPECT = "active", "suspect"


class StaleStore(Exception):
    """The store is unknown to the master or reports an old epoch; it must register again."""


class _Member:
    __slots__ = (
        "store_id",
        "address",
        "host",
        "numa",
        "cluster_uuid",
        "epoch",
        "capacity",
        "free",
        "objects",
        "seen",
        "keys",
    )

    def __init__(
        self, *, store_id, address, host, numa, cluster_uuid, epoch, capacity, now
    ):
        self.store_id, self.address, self.host, self.numa = (
            store_id,
            address,
            host,
            numa,
        )
        self.cluster_uuid, self.epoch, self.capacity = cluster_uuid, epoch, capacity
        self.free, self.objects, self.seen = capacity, 0, now
        self.keys = set()

    def describe(self, state):
        return dict(
            store_id=self.store_id,
            address=self.address,
            host=self.host,
            numa=self.numa,
            epoch=self.epoch,
            capacity=self.capacity,
            free=self.free,
            objects=self.objects,
            state=state,
        )


class Directory:
    """Thread-safe master state; the socket server is a thin shell around it."""

    def __init__(
        self, *, suspect_s=SUSPECT_S, offline_s=OFFLINE_S, clock=time.monotonic
    ):
        self.suspect_s, self.offline_s, self.clock = suspect_s, offline_s, clock
        self.lock = threading.Lock()
        self.members = {}  # store_id -> _Member
        self.holders = {}  # key -> [store_id], in report order
        self.cluster_uuid = None  # NVLink domain of the first store with one
        self.stats = Counter()

    # ---- membership ----

    def register(self, *, store_id, address, host, numa, cluster_uuid, epoch, capacity):
        with self.lock:
            if cluster_uuid and self.cluster_uuid and cluster_uuid != self.cluster_uuid:
                self.stats["register_rejected_domain"] += 1
                raise ValueError(
                    f"store {store_id} is in NVLink domain {cluster_uuid}, this master serves "
                    f"{self.cluster_uuid}: its slab cannot be imported by the other stores' clients"
                )
            self.cluster_uuid = self.cluster_uuid or cluster_uuid
            old = self.members.get(store_id)
            if old is not None:
                self._purge(old)
                self.stats["reregistered" if old.epoch == epoch else "restarted"] += 1
            self.members[store_id] = _Member(
                store_id=store_id,
                address=address,
                host=host,
                numa=numa,
                cluster_uuid=cluster_uuid,
                epoch=epoch,
                capacity=capacity,
                now=self.clock(),
            )
            self.stats["registered"] += 1

    def report(self, *, store_id, epoch, add=(), delete=(), snapshot=None, status=None):
        """Apply one batch of store events; ``snapshot`` replaces every key the store held."""
        with self.lock:
            member = self.members.get(store_id)
            if member is None or member.epoch != epoch:
                self.stats["report_stale"] += 1
                raise StaleStore(f"store {store_id} epoch {epoch} is not registered")
            member.seen = self.clock()
            if snapshot is not None:
                self._purge(member)
                self._add(member, snapshot)
                self.stats["snapshots"] += 1
            self._add(member, add)
            self._delete(member, delete)
            if status:
                member.free, member.objects = (
                    int(status["free"]),
                    int(status["objects"]),
                )
            self.stats["reports"] += 1
            self.stats["report_keys"] += len(add) + len(delete)

    def _add(self, member, keys):
        for key in keys:
            if key not in member.keys:
                member.keys.add(key)
                self.holders.setdefault(key, []).append(member.store_id)

    def _delete(self, member, keys):
        for key in keys:
            if key in member.keys:
                member.keys.discard(key)
                self._unhold(key, member.store_id)

    def _unhold(self, key, store_id):
        holders = self.holders.get(key)
        if holders is None:
            return
        holders.remove(store_id)
        if not holders:
            del self.holders[key]

    def _purge(self, member):
        for key in member.keys:
            self._unhold(key, member.store_id)
        member.keys = set()

    def _state(self, member, now):
        age = now - member.seen
        return ACTIVE if age < self.suspect_s else SUSPECT

    def sweep(self):
        """Drop stores silent past ``offline_s``, with every key they held."""
        now = self.clock()
        with self.lock:
            gone = [m for m in self.members.values() if now - m.seen >= self.offline_s]
            for member in gone:
                self._purge(member)
                del self.members[member.store_id]
            self.stats["offline"] += len(gone)
        return [m.store_id for m in gone]

    def describe(self):
        now = self.clock()
        with self.lock:
            return [
                m.describe(self._state(m, now))
                for m in self.members.values()
                if now - m.seen < self.offline_s
            ]

    # ---- lookup / placement ----

    def lookup(self, keys, *, host=None, numa=None, prefix=True):
        """One holder per key (None for a miss); with ``prefix`` stop at the first miss.

        Returns (store_ids, pages): ``pages[i]`` indexes ``store_ids`` or is -1.
        """
        now = self.clock()
        with self.lock:
            rank = {}
            for member in self.members.values():
                age = now - member.seen
                if age >= self.offline_s:
                    continue
                # Lower is better: live before suspect, then same NUMA, then same node.
                rank[member.store_id] = (
                    age >= self.suspect_s,
                    not (member.host == host and member.numa == numa),
                    member.host != host,
                )
            store_ids, index, pages = [], {}, []
            for key in keys:
                live = [s for s in self.holders.get(key, ()) if s in rank]
                if not live:
                    if prefix:
                        break
                    pages.append(-1)
                    continue
                best = min(live, key=rank.__getitem__)
                if best not in index:
                    index[best] = len(store_ids)
                    store_ids.append(best)
                pages.append(index[best])
            self.stats["lookups"] += 1
            self.stats["lookup_keys"] += len(keys)
            self.stats["lookup_hits"] += sum(p >= 0 for p in pages)
        return store_ids, pages

    def place(self, *, nbytes, exclude=(), host=None, numa=None):
        """The live store with the most free bytes that fits ``nbytes``, or None."""
        now = self.clock()
        with self.lock:
            candidates = [
                m
                for m in self.members.values()
                if m.store_id not in exclude
                and now - m.seen < self.suspect_s
                and m.free >= nbytes
            ]
            if not candidates:
                self.stats["place_none"] += 1
                return None
            # Most free first; ties go to the caller's node, so a near store wins when equal.
            best = max(
                candidates, key=lambda m: (m.free, m.host == host, m.numa == numa)
            )
            best.free -= nbytes
            self.stats["placed"] += 1
            return best.store_id

    def snapshot_stats(self):
        with self.lock:
            return dict(
                stores=len(self.members),
                keys=len(self.holders),
                replicas=sum(len(m.keys) for m in self.members.values()),
                cluster_uuid=self.cluster_uuid,
                **self.stats,
            )


class _Handler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        with self.server.connections_lock:
            self.server.connections.add(self.connection)

    def finish(self):
        with self.server.connections_lock:
            self.server.connections.discard(self.connection)
        super().finish()

    def handle(self):
        directory, secret = self.server.directory, self.server.secret
        authenticated = secret is None
        for line in self.rfile:
            req = json.loads(line)
            op = req.pop("op", None)
            try:
                if op == "hello":
                    if not authenticated and req.get("secret") != secret:
                        directory.stats["auth_rejected"] += 1
                        raise PermissionError("bad or missing secret")
                    authenticated = True
                    value = dict(pid=os.getpid(), host=socket.gethostname())
                elif not authenticated:
                    raise PermissionError("hello with the job secret first")
                else:
                    value = self._dispatch(directory, op, req)
                response = dict(ok=True, value=value)
            except StaleStore as error:
                response = dict(ok=False, stale=True, error=str(error))
            except Exception as error:  # noqa: BLE001 - reported to the client
                response = dict(ok=False, error=f"{type(error).__name__}: {error}")
            self.wfile.write(json.dumps(response).encode() + b"\n")
            self.wfile.flush()
            if not authenticated:
                return

    @staticmethod
    def _dispatch(directory, op, req):
        if op == "register":
            directory.register(**req)
            return True
        if op == "report":
            directory.report(**req)
            return True
        if op == "lookup":
            store_ids, pages = directory.lookup(
                req["keys"],
                host=req.get("host"),
                numa=req.get("numa"),
                prefix=req.get("prefix", True),
            )
            return dict(stores=store_ids, pages=pages)
        if op == "place":
            return directory.place(
                nbytes=int(req["nbytes"]),
                exclude=set(req.get("exclude", ())),
                host=req.get("host"),
                numa=req.get("numa"),
            )
        if op == "members":
            return directory.describe()
        if op == "stats":
            return directory.snapshot_stats()
        raise ValueError(f"unknown op {op!r}")


class _TcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.connections, self.connections_lock = set(), threading.Lock()
        super().__init__(*args, **kwargs)

    def stop(self):
        """Stop serving and drop every client connection, as a process exit would."""
        self.shutdown()
        with self.connections_lock:
            for connection in list(self.connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.server_close()


class MasterClient:
    """One persistent connection per thread; raises ConnectionError when the master is unreachable."""

    def __init__(self, address, *, secret=None, timeout=5.0):
        if not address.startswith("tcp://"):
            raise ValueError(f"master address must be tcp://host:port, got {address!r}")
        host, port = address[len("tcp://") :].rsplit(":", 1)
        self.address, self._target, self.secret = address, (host, int(port)), secret
        self.timeout = timeout
        self._local = threading.local()

    def _stream(self):
        stream = getattr(self._local, "stream", None)
        if stream is None:
            sock = socket.create_connection(self._target, timeout=self.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            stream = sock.makefile("rwb")
            reply = self._call(stream, dict(op="hello", secret=self.secret))
            if not reply["ok"]:
                sock.close()
                raise PermissionError(reply["error"])
            self._local.stream = stream
        return stream

    @staticmethod
    def _call(stream, request):
        stream.write(json.dumps(request).encode() + b"\n")
        stream.flush()
        line = stream.readline()
        if not line:
            raise ConnectionError("EGM master closed the connection")
        return json.loads(line)

    def rpc(self, op, **args):
        # Every op is safe to replay (key sets, re-register plus snapshot, a placement estimate),
        # so a cached connection that died with a restarted master is retried once on a fresh one.
        for retry in (False, True):
            reused = getattr(self._local, "stream", None) is not None
            try:
                reply = self._call(self._stream(), dict(op=op, **args))
                break
            except PermissionError:
                raise  # A wrong secret is not an outage; it subclasses OSError.
            except (OSError, ValueError) as error:
                self._local.stream = None
                if retry or not reused:
                    raise ConnectionError(
                        f"EGM master {self.address} unreachable: {error}"
                    ) from error
        if not reply["ok"]:
            raise (StaleStore if reply.get("stale") else RuntimeError)(reply["error"])
        return reply["value"]

    def close(self):
        stream = getattr(self._local, "stream", None)
        if stream is not None:
            stream.close()
            self._local.stream = None


def serve(*, bind, secret, suspect_s=SUSPECT_S, offline_s=OFFLINE_S):
    """Start a master in background threads; returns (server, directory). Used by main() and tests."""
    host, port = bind.rsplit(":", 1)
    server = _TcpServer((host, int(port)), _Handler)
    directory = Directory(suspect_s=suspect_s, offline_s=offline_s)
    server.directory, server.secret = directory, secret

    def sweeper():
        while True:
            time.sleep(min(1.0, suspect_s / 4))
            directory.sweep()

    threading.Thread(target=sweeper, daemon=True, name="egm-master-sweep").start()
    threading.Thread(
        target=server.serve_forever, daemon=True, name="egm-master"
    ).start()
    return server, directory


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tcp", required=True, help="HOST:PORT to serve on")
    p.add_argument(
        "--secret-file", required=True, help="job secret every client must present"
    )
    p.add_argument("--suspect-s", type=float, default=SUSPECT_S)
    p.add_argument("--offline-s", type=float, default=OFFLINE_S)
    p.add_argument("--ready", help="write a JSON file once listening")
    p.add_argument(
        "--stats-out", help="write final stats JSON here on SIGTERM / SIGINT"
    )
    p.add_argument(
        "--stats-interval",
        type=float,
        default=0,
        help="also append stats lines every N s",
    )
    args = p.parse_args()
    secret = open(args.secret_file).read().strip()
    server, directory = serve(
        bind=args.tcp, secret=secret, suspect_s=args.suspect_s, offline_s=args.offline_s
    )

    def stop(*_):
        if args.stats_out:
            with open(args.stats_out, "w") as out:
                json.dump(dict(time=time.time(), **directory.snapshot_stats()), out)
        os._exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.ready:
        with open(args.ready, "w") as out:
            json.dump(
                dict(tcp=args.tcp, pid=os.getpid(), host=socket.gethostname()), out
            )
    while True:
        time.sleep(args.stats_interval or 3600)
        if args.stats_interval and args.stats_out:
            with open(args.stats_out + ".jsonl", "a") as out:
                out.write(
                    json.dumps(dict(time=time.time(), **directory.snapshot_stats()))
                    + "\n"
                )


if __name__ == "__main__":
    main()
