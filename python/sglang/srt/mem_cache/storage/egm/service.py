# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Standalone EGM store service: one process owns one NUMA's EGM slab; Engines are clients.

The service never touches KV payloads. It allocates slots, publishes objects and
pins them for readers; clients import the slab once over CUDA Fabric and copy
directly between their own device pool and the slab (reads and writes). Clients
may run on other nodes of the same NVLink domain.

* Writes: ``alloc`` reserves one slot per new key (existing or in-flight keys are
  reported so the writer skips them, which is the cross-Engine dedup), the client
  copies, then ``commit`` publishes. ``abort`` frees the slots.
* Reads: ``acquire`` pins every key of a load until ``release`` or the lease
  deadline; a missing key fails the whole acquire before any copy starts, so the
  caller can recompute. The client checks the deadline after its copy lands.
* Eviction: LRU over committed, unpinned objects of the requested slot size, only
  when the slab has no free slot. Pins and in-flight writes are never evicted.
* Directory (optional): with ``--master`` the service registers with the domain
  master (``master.py``) and reports the keys it commits and evicts, so Engines
  find pages with one master lookup instead of asking every store.
* Liveness: leases and reservations carry deadlines, so a dead or stuck client
  (local or on another node) never holds a pin or a slot past its TTL. A
  reservation that expires, or whose client disconnects, is quarantined before
  reuse: a late copy from a slow writer must not land in a slot that already
  holds someone else's object.

Control is line-delimited JSON over a Unix socket and/or TCP (one persistent
connection per client thread). TCP clients must present the job secret in
``hello``. Usage::

    EGM_NATIVE_LIB=libegm.so python -m sglang.srt.mem_cache.storage.egm.service \\
        --numa 0 --capacity-gib 64 --listen /tmp/egm-numa0.sock \\
        --tcp 0.0.0.0:29600 --secret-file /shared/job/egm.secret \\
        [--master tcp://10.0.0.1:29700 --advertise 10.0.0.2]
"""

import argparse
import json
import logging
import os
import signal
import socket
import socketserver
import subprocess
import threading
import time
import uuid
from collections import Counter, OrderedDict

logger = logging.getLogger(__name__)

ALIGN = 64
EXISTS, FULL = -1, -2
LEASE_TTL_MS = 10_000
WRITE_TTL_MS = 30_000
# Arbitrary; must exceed the longest copy a late writer could still have in flight.
QUARANTINE_MS = 60_000


def cluster_uuid():
    """NVLink domain id of this node (one per NVL72), or None when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "ClusterUUID" in line:
            value = line.split(":", 1)[1].strip()
            return value if value and value != "N/A" else None
    return None


class SlotStore:
    def __init__(
        self,
        region,
        capacity,
        lease_ttl_ms=LEASE_TTL_MS,
        write_ttl_ms=WRITE_TTL_MS,
        quarantine_ms=QUARANTINE_MS,
        clock=time.monotonic,
    ):
        self.region = region
        self.capacity = capacity
        self.lease_ttl, self.write_ttl, self.quarantine_s = (
            lease_ttl_ms / 1e3,
            write_ttl_ms / 1e3,
            quarantine_ms / 1e3,
        )
        self.clock = clock
        self.lock = threading.Lock()
        self.cursor = 0
        self.free = {}  # slot size -> [offset]
        self.objects = OrderedDict()  # key -> (offset, size); oldest first
        self.pins = Counter()  # key -> active reader leases
        self.leases = {}  # token -> (keys, deadline)
        self.writes = {}  # token -> ([(key, offset, size)], deadline)
        self.reserved = set()  # keys being written
        self.quarantine = []  # (reusable_at, size, offset)
        self.stats = Counter()
        # key -> held now; the latest change per key, kept only while a master is attached.
        self._changes = None

    # ---- allocation ----

    def _changed(self, key, held):
        if self._changes is not None:
            self._changes[key] = held

    def _take(self, size, evict):
        slots = self.free.get(size)
        if slots:
            return slots.pop()
        if self.capacity - self.cursor >= size:
            offset, self.cursor = self.cursor, self.cursor + size
            return offset
        if not evict:
            return None
        for key, (offset, osize) in self.objects.items():
            if osize == size and not self.pins[key]:
                del self.objects[key]
                self.pins.pop(key, None)
                self._changed(key, False)
                self.stats["evicted"] += 1
                return offset
        return None

    def alloc(self, keys, size, ttl_ms=None, evict=True):
        """``evict=False`` reports FULL instead of evicting, so the writer can place the page elsewhere."""
        size = (size + ALIGN - 1) // ALIGN * ALIGN
        token, offsets, entries = uuid.uuid4().hex, [], []
        ttl = self.write_ttl if ttl_ms is None else ttl_ms / 1e3
        with self.lock:
            for key in keys:
                if key in self.objects or key in self.reserved:
                    if key in self.objects:
                        self.objects.move_to_end(key)
                    self.stats["put_dedup"] += 1
                    offsets.append(EXISTS)
                    continue
                offset = self._take(size, evict)
                if offset is None:
                    self.stats["put_full"] += 1
                    offsets.append(FULL)
                    continue
                self.reserved.add(key)
                entries.append((key, offset, size))
                offsets.append(offset)
            self.stats["put_new"] += len(entries)
            if entries:
                self.writes[token] = (entries, self.clock() + ttl)
        return token, offsets, int(ttl * 1e3)

    def commit(self, token):
        with self.lock:
            entries, _ = self.writes.pop(token, (None, None))
            if entries is None:
                self.stats["commit_expired"] += 1
                raise KeyError("reservation expired or unknown")
            for key, offset, size in entries:
                self.reserved.discard(key)
                self.objects[key] = (offset, size)
                self._changed(key, True)
                self.stats["committed"] += 1

    def abort(self, token, quarantine=False):
        """quarantine=True when the client may still have a copy in flight (expiry, lost connection)."""
        with self.lock:
            entries, _ = self.writes.pop(token, ((), None))
            now = self.clock()
            for key, offset, size in entries:
                self.reserved.discard(key)
                if quarantine:
                    self.quarantine.append((now + self.quarantine_s, size, offset))
                    self.stats["quarantined"] += 1
                else:
                    self.free.setdefault(size, []).append(offset)
                    self.stats["aborted"] += 1

    # ---- reads ----

    def exists(self, keys):
        with self.lock:
            self.stats["exists"] += len(keys)
            return [key in self.objects for key in keys]

    def acquire(self, keys, ttl_ms=None):
        ttl = self.lease_ttl if ttl_ms is None else ttl_ms / 1e3
        with self.lock:
            missing = sum(key not in self.objects for key in keys)
            if missing:
                self.stats["acquire_missing"] += 1
                return None
            token = uuid.uuid4().hex
            unique = list(dict.fromkeys(keys))
            for key in unique:
                self.pins[key] += 1
                self.objects.move_to_end(key)
            self.leases[token] = (unique, self.clock() + ttl)
            self.stats["acquired_keys"] += len(unique)
            return token, [self.objects[key][0] for key in keys], int(ttl * 1e3)

    def renew(self, token, ttl_ms=None):
        ttl = self.lease_ttl if ttl_ms is None else ttl_ms / 1e3
        with self.lock:
            if token not in self.leases:
                return False
            keys, _ = self.leases[token]
            self.leases[token] = (keys, self.clock() + ttl)
            return True

    def _unpin(self, keys):
        for key in keys:
            self.pins[key] -= 1
            if self.pins[key] <= 0:
                del self.pins[key]

    def release(self, token):
        with self.lock:
            keys, _ = self.leases.pop(token, ((), None))
            self._unpin(keys)

    def reap(self):
        """Expire leases and reservations past their deadline; return quarantined slots that cooled down."""
        now = self.clock()
        with self.lock:
            for token in [
                t for t, (_, deadline) in self.leases.items() if deadline <= now
            ]:
                keys, _ = self.leases.pop(token)
                self._unpin(keys)
                self.stats["lease_expired"] += 1
            expired = [t for t, (_, deadline) in self.writes.items() if deadline <= now]
        for token in expired:
            self.abort(token, quarantine=True)
            self.stats["write_expired"] += 1
        with self.lock:
            cooled = [q for q in self.quarantine if q[0] <= now]
            self.quarantine = [q for q in self.quarantine if q[0] > now]
            for _, size, offset in cooled:
                self.free.setdefault(size, []).append(offset)
            self.stats["unquarantined"] += len(cooled)

    def drop(self, keys):
        """Evict now (fault injection: object disappears between lookup and load)."""
        dropped = 0
        with self.lock:
            for key in keys:
                if key in self.objects and not self.pins[key]:
                    offset, size = self.objects.pop(key)
                    self.free.setdefault(size, []).append(offset)
                    self._changed(key, False)
                    dropped += 1
            self.stats["dropped"] += dropped
        return dropped

    def track_changes(self):
        """Start a change log and return every key held now (the snapshot for a master)."""
        with self.lock:
            self._changes = {}
            return list(self.objects)

    def take_changes(self):
        """(added, removed) since the last call or snapshot."""
        with self.lock:
            changes, self._changes = self._changes or {}, {}
        added = [key for key, held in changes.items() if held]
        return added, [key for key, held in changes.items() if not held]

    def status(self):
        """Bytes allocatable without evicting (quarantined slots excluded), for master placement."""
        with self.lock:
            free = self.capacity - self.cursor
            free += sum(size * len(slots) for size, slots in self.free.items())
            return dict(free=free, objects=len(self.objects))

    def snapshot(self):
        with self.lock:
            used = sum(size for _, size in self.objects.values())
            return dict(
                objects=len(self.objects),
                used_bytes=used,
                capacity_bytes=self.capacity,
                bump_bytes=self.cursor,
                pinned_keys=len(self.pins),
                leases=len(self.leases),
                writes=len(self.writes),
                quarantine=len(self.quarantine),
                **self.stats,
            )


class _Handler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        if self.connection.family != socket.AF_UNIX:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def handle(self):
        store, writes = self.server.store, set()
        # Unix-socket peers are already restricted by the socket file mode; TCP peers must prove the secret.
        authenticated = (
            self.server.secret is None or self.connection.family == socket.AF_UNIX
        )
        try:
            for line in self.rfile:
                req = json.loads(line)
                op = req.get("op")
                try:
                    if op == "hello":
                        if (
                            not authenticated
                            and req.get("secret") != self.server.secret
                        ):
                            store.stats["auth_rejected"] += 1
                            raise PermissionError("bad or missing secret")
                        authenticated = True
                        value = dict(
                            store_id=self.server.store_id,
                            descriptor=self.server.descriptor,
                            capacity=store.capacity,
                            numa=self.server.numa,
                            pid=os.getpid(),
                            host=socket.gethostname(),
                            cluster_uuid=self.server.cluster_uuid,
                            lease_ttl_ms=int(store.lease_ttl * 1e3),
                        )
                    elif not authenticated:
                        raise PermissionError("hello with the job secret first")
                    elif op == "exists":
                        value = store.exists(req["keys"])
                    elif op == "acquire":
                        got = store.acquire(req["keys"], req.get("ttl_ms"))
                        value = (
                            None
                            if got is None
                            else dict(token=got[0], offsets=got[1], ttl_ms=got[2])
                        )
                    elif op == "renew":
                        value = store.renew(req["token"], req.get("ttl_ms"))
                    elif op == "release":
                        store.release(req["token"])
                        value = True
                    elif op == "alloc":
                        token, offsets, ttl_ms = store.alloc(
                            req["keys"],
                            int(req["size"]),
                            req.get("ttl_ms"),
                            evict=req.get("evict", True),
                        )
                        writes.add(token)
                        value = dict(token=token, offsets=offsets, ttl_ms=ttl_ms)
                    elif op == "commit":
                        writes.discard(req["token"])
                        store.commit(req["token"])
                        value = True
                    elif op == "abort":
                        writes.discard(req["token"])
                        store.abort(req["token"])
                        value = True
                    elif op == "drop":
                        value = store.drop(req["keys"])
                    elif op == "stats":
                        value = store.snapshot()
                    else:
                        raise ValueError(f"unknown op {op!r}")
                    response = dict(ok=True, value=value)
                except Exception as error:  # noqa: BLE001 - reported to the client
                    response = dict(ok=False, error=f"{type(error).__name__}: {error}")
                self.wfile.write(json.dumps(response).encode() + b"\n")
                self.wfile.flush()
                if not authenticated:
                    return
        finally:
            # Pins stay until their deadline: a peer that lost its connection may still be copying out.
            # Reservations are quarantined for the same reason; a late write must not hit a reused slot.
            for token in writes:
                store.abort(token, quarantine=True)
            if writes:
                store.stats["orphan_writes"] += len(writes)


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _TcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_address(address):
    """'tcp://host:port' -> ('tcp', (host, port)); 'unix:/path' or '/path' -> ('unix', path)."""
    if address.startswith("tcp://"):
        host, port = address[len("tcp://") :].rsplit(":", 1)
        return "tcp", (host, int(port))
    return "unix", address[len("unix:") :] if address.startswith("unix:") else address


class StoreClient:
    """Engine-side client: one persistent control connection per thread, one imported slab per process."""

    def __init__(self, address, device, secret=None, local_cluster_uuid=None):
        self.address, self.device, self.secret = address, device, secret
        self._kind, self._target = parse_address(address)
        self._local = threading.local()
        self.descriptor = None
        hello = self.rpc("hello")
        self.descriptor, self.capacity, self.numa = (
            hello["descriptor"],
            hello["capacity"],
            hello["numa"],
        )
        self.host, self.cluster_uuid = hello["host"], hello["cluster_uuid"]
        self.store_id, self.epoch = hello["store_id"], self.descriptor["epoch"]
        if (
            local_cluster_uuid
            and self.cluster_uuid
            and local_cluster_uuid != self.cluster_uuid
        ):
            raise ValueError(
                f"EGM store {address} on {self.host} is in NVLink domain {self.cluster_uuid}, this "
                f"Engine in {local_cluster_uuid}: a Fabric slab cannot be imported across domains"
            )
        from .region import Region

        try:
            self.region = Region(descriptor=self.descriptor, device=device)
        except RuntimeError as error:
            raise RuntimeError(
                f"importing the EGM slab of {address} ({self.host}) failed; both ends must share "
                f"an NVLink domain and an IMEX channel: {error}"
            ) from error

    def _connect(self):
        if self._kind == "tcp":
            sock = socket.create_connection(self._target, timeout=30)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self._target)
        return sock

    def _stream(self):
        stream = getattr(self._local, "stream", None)
        if stream is None:
            sock = self._connect()
            stream = sock.makefile("rwb")
            hello = self._call(
                stream, "hello", {"secret": self.secret} if self.secret else {}
            )
            if not hello["ok"]:
                sock.close()
                raise PermissionError(hello["error"])
            # A restarted service owns a new slab; offsets from it must never index the old import.
            if (
                self.descriptor is not None
                and hello["value"]["descriptor"]["epoch"] != self.descriptor["epoch"]
            ):
                sock.close()
                raise ConnectionError("EGM store service restarted with a new slab")
            self._local.sock, self._local.stream = sock, stream
            self._local.hello = hello["value"]
        return stream

    @staticmethod
    def _call(stream, op, args):
        stream.write(json.dumps(dict(op=op, **args)).encode() + b"\n")
        stream.flush()
        line = stream.readline()
        if not line:
            raise ConnectionError("EGM store service closed the connection")
        return json.loads(line)

    def rpc(self, op, **args):
        try:
            stream = self._stream()
            if op == "hello":
                return self._local.hello
            result = self._call(stream, op, args)
        except (OSError, ValueError) as error:
            self._local.stream = (
                None  # Reconnect (and re-check the slab epoch) next time.
            )
            raise ConnectionError(
                f"EGM store service {self.address} unreachable: {error}"
            ) from error
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["value"]

    def exists_many(self, keys):
        return self.rpc("exists", keys=list(keys))

    def acquire(self, keys, ttl_ms=None):
        """(token, offsets, deadline) with every key pinned, or None when any key is missing.

        ``deadline`` is on this host's monotonic clock, taken before the request is sent,
        so it never outlasts the server-side lease.
        """
        sent = time.monotonic()
        got = self.rpc(
            "acquire", keys=list(keys), **({} if ttl_ms is None else {"ttl_ms": ttl_ms})
        )
        return (
            None
            if got is None
            else (got["token"], got["offsets"], sent + got["ttl_ms"] / 1e3)
        )

    def release(self, token):
        self.rpc("release", token=token)

    def alloc(self, keys, size, ttl_ms=None, evict=True):
        """(token, offsets, deadline); commit after ``deadline`` is rejected by the service."""
        sent = time.monotonic()
        got = self.rpc(
            "alloc",
            keys=list(keys),
            size=size,
            evict=evict,
            **({} if ttl_ms is None else {"ttl_ms": ttl_ms}),
        )
        return got["token"], got["offsets"], sent + got["ttl_ms"] / 1e3

    def commit(self, token):
        self.rpc("commit", token=token)

    def abort(self, token):
        self.rpc("abort", token=token)

    def drop(self, keys):
        return self.rpc("drop", keys=list(keys))

    def stats(self):
        return self.rpc("stats")


class _Reporter:
    """Keeps this store registered with the domain master and streams its key changes.

    Any failure (master restart, network, unknown epoch) re-registers with a full
    snapshot, so a dropped batch of changes never leaves the master out of date.
    """

    def __init__(self, *, store, master, identity, heartbeat_s, flush_s=0.02):
        self.store, self.master, self.identity = store, master, identity
        self.heartbeat_s, self.flush_s = heartbeat_s, flush_s
        self._stopping = threading.Event()

    def stop(self):
        self._stopping.set()

    def run(self):
        from .master import StaleStore

        registered, last = False, 0.0
        while not self._stopping.is_set():
            try:
                if not registered:
                    self._register()
                    registered, last = True, time.monotonic()
                elif self._flush(force=time.monotonic() - last >= self.heartbeat_s):
                    last = time.monotonic()
            except (
                StaleStore,
                ConnectionError,
                RuntimeError,
                PermissionError,
            ) as error:
                if registered:
                    logger.warning(
                        "EGM master report failed, re-registering: %s", error
                    )
                registered = False
                self.store.stats["master_errors"] += 1
                time.sleep(self.heartbeat_s)
            time.sleep(self.flush_s)

    def _register(self):
        self.master.rpc("register", **self.identity)
        keys = self.store.track_changes()
        self.master.rpc(
            "report",
            store_id=self.identity["store_id"],
            epoch=self.identity["epoch"],
            snapshot=keys,
            status=self.store.status(),
        )
        self.store.stats["master_registered"] += 1

    def _flush(self, *, force):
        added, removed = self.store.take_changes()
        if not (added or removed or force):
            return False
        self.master.rpc(
            "report",
            store_id=self.identity["store_id"],
            epoch=self.identity["epoch"],
            add=added,
            delete=removed,
            status=self.store.status(),
        )
        return True


def _store_identity(args, *, numa, epoch, capacity, domain):
    host = socket.gethostname()
    if args.tcp:
        advertise = args.advertise or socket.gethostbyname(host)
        address = f"tcp://{advertise}:{args.tcp.rsplit(':', 1)[1]}"
        store_id = address
    else:
        # Reachable from this node only.
        address, store_id = args.listen, f"unix:{args.listen}@{host}"
    return dict(
        store_id=store_id,
        address=address,
        host=host,
        numa=numa,
        cluster_uuid=domain,
        epoch=epoch,
        capacity=capacity,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--numa", type=int, required=True)
    p.add_argument(
        "--device",
        type=int,
        help="CUDA device that owns the slab; defaults to the NUMA's first GPU",
    )
    p.add_argument("--capacity-gib", type=float, required=True)
    p.add_argument("--listen", help="Unix socket path")
    p.add_argument(
        "--tcp", help="HOST:PORT to also serve on TCP (clients on other nodes)"
    )
    p.add_argument(
        "--secret-file", help="job secret TCP clients must present; required with --tcp"
    )
    p.add_argument("--master", help="tcp://HOST:PORT of the domain master (master.py)")
    p.add_argument(
        "--advertise",
        help="host or IP other nodes use to reach --tcp; defaults to this host's address",
    )
    p.add_argument("--heartbeat-s", type=float, default=1.0)
    p.add_argument("--lease-ttl-ms", type=int, default=LEASE_TTL_MS)
    p.add_argument("--write-ttl-ms", type=int, default=WRITE_TTL_MS)
    p.add_argument("--quarantine-ms", type=int, default=QUARANTINE_MS)
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
    if not args.listen and not args.tcp:
        p.error("need --listen and/or --tcp")
    if (args.tcp or args.master) and not args.secret_file:
        p.error("--tcp and --master need --secret-file")
    from .region import Region

    device = args.device if args.device is not None else 2 * args.numa
    capacity = int(args.capacity_gib * (1 << 30)) // ALIGN * ALIGN
    region = Region(capacity, args.numa, tier="dram", device=device)
    store = SlotStore(
        region, capacity, args.lease_ttl_ms, args.write_ttl_ms, args.quarantine_ms
    )
    secret = open(args.secret_file).read().strip() if args.secret_file else None
    servers = []
    if args.listen:
        if os.path.exists(args.listen):
            raise FileExistsError(args.listen)
        servers.append(_UnixServer(args.listen, _Handler))
        os.chmod(args.listen, 0o600)
    if args.tcp:
        host, port = args.tcp.rsplit(":", 1)
        servers.append(_TcpServer((host, int(port)), _Handler))
    domain = cluster_uuid()
    descriptor = region.descriptor()
    identity = _store_identity(
        args,
        numa=args.numa,
        epoch=descriptor["epoch"],
        capacity=capacity,
        domain=domain,
    )
    for server in servers:
        server.store, server.descriptor, server.numa = store, descriptor, args.numa
        server.secret, server.cluster_uuid = secret, domain
        server.store_id = identity["store_id"]
    if args.master:
        from .master import MasterClient

        reporter = _Reporter(
            store=store,
            master=MasterClient(args.master, secret=secret),
            identity=identity,
            heartbeat_s=args.heartbeat_s,
        )
        threading.Thread(target=reporter.run, daemon=True, name="egm-report").start()

    def stop(*_):
        if args.stats_out:
            with open(args.stats_out, "w") as out:
                json.dump(dict(time=time.time(), **store.snapshot()), out)
        if args.listen:
            os.unlink(args.listen)
        os._exit(
            0
        )  # The slab dies with the process; importers keep their own mapping alive.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def reaper():
        while True:
            time.sleep(0.2)
            store.reap()

    threading.Thread(target=reaper, daemon=True).start()
    if args.stats_interval and args.stats_out:

        def sample():
            while True:
                time.sleep(args.stats_interval)
                with open(args.stats_out + ".jsonl", "a") as out:
                    out.write(
                        json.dumps(dict(time=time.time(), **store.snapshot())) + "\n"
                    )

        threading.Thread(target=sample, daemon=True).start()
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    if args.ready:
        with open(args.ready, "w") as out:
            json.dump(
                dict(
                    store_id=identity["store_id"],
                    listen=args.listen,
                    tcp=args.tcp,
                    numa=args.numa,
                    device=device,
                    capacity=capacity,
                    pid=os.getpid(),
                    host=socket.gethostname(),
                    cluster_uuid=domain,
                ),
                out,
            )
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
