# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Unix or job-private authenticated TCP metadata; CUDA Fabric carries payloads.

No timeout-based lease reclamation: a lost reader quarantines its pin until
explicit recovery/process restart. This favors safety over availability.
"""
import hmac
import json
import os
import socket
import socketserver
import threading
from contextlib import contextmanager

from .region import Region
from .store import validate_ranges

MAX_MESSAGE = 1 << 20


def receive(stream, stats=None):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line.endswith(b"\n") or len(line) > MAX_MESSAGE:
        raise ValueError("invalid/oversized control message")
    if stats is not None:
        stats["received"] += len(line)
    return json.loads(line)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        try:
            req = receive(self.rfile)
            if self.server.secret is not None:
                supplied = req.pop("auth", None)
                if not isinstance(supplied, str) or not hmac.compare_digest(supplied, self.server.secret):
                    raise PermissionError("control authentication failed")
            store = self.server.store
            if req["op"] == "exists":
                value = [store.exists(k) for k in req["keys"]]
            elif req["op"] == "acquire":
                value = store.acquire(req["keys"])
            elif req["op"] == "release":
                store.release(req["token"], req["epoch"])
                value = True
            else:
                raise ValueError("unknown operation")
            response = dict(ok=True, value=value)
        except Exception as e:
            response = dict(ok=False, error=str(e))
        self.wfile.write(json.dumps(response).encode() + b"\n")


class PeerServer:
    def __init__(self, store, path, secret=None):
        # The caller owns a private directory; never unlink an existing endpoint.
        tcp = isinstance(path, (tuple, list))
        if tcp and (not isinstance(secret, str) or len(secret) < 32):
            raise ValueError("TCP requires a private job-scoped secret (>=32 chars)")
        if not tcp and os.path.exists(path):
            raise FileExistsError(path)
        self.path = path
        self.server = (socketserver.TCPServer(tuple(path), _Handler) if tcp
                       else socketserver.UnixStreamServer(path, _Handler))
        self.path = self.server.server_address
        self.tcp = tcp
        self.server.secret = secret
        if not tcp:
            os.chmod(path, 0o600)
        self.server.store = store
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        if not self.tcp:
            os.unlink(self.path)


class PeerClient:
    def __init__(self, path, device=0, secret=None):
        self.path = tuple(path) if isinstance(path, (tuple, list)) else path
        self.device, self.secret = device, secret
        self.quarantined = []
        self.control_bytes = {"sent": 0, "received": 0}

    def rpc(self, op, **args):
        payload = json.dumps(dict(op=op, auth=self.secret, **args)).encode() + b"\n"
        if len(payload) > MAX_MESSAGE:
            raise ValueError("control message too large")
        with socket.socket(socket.AF_INET if isinstance(self.path, tuple) else socket.AF_UNIX) as sock:
            sock.settimeout(30)
            sock.connect(self.path)
            sock.sendall(payload)
            self.control_bytes["sent"] += len(payload)
            with sock.makefile("rb") as stream:
                result = receive(stream, self.control_bytes)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["value"]

    def exists(self, key):
        return self.rpc("exists", keys=[key])[0]

    def get(self, key, ranges, device=False):
        if not self.exists(key):
            return False
        with self.reader([key]) as reader:
            return reader.get(key, ranges, device)

    @contextmanager
    def reader(self, keys):
        lease = self.rpc("acquire", keys=list(dict.fromkeys(keys)))
        regions = {}
        started = False

        def close_mappings():
            for region in regions.values():
                region.close()

        try:
            if lease.get("version") != 2:
                raise ValueError("unsupported Store lease protocol")
            for tier, descriptor in lease["regions"].items():
                if descriptor.get("tier") != tier:
                    raise ValueError("lease tier/descriptor mismatch")
                regions[tier] = Region(descriptor=descriptor, device=self.device)
            class Reader:
                def get(self, key, ranges, device=False):
                    nonlocal started
                    obj = lease["objects"][key]
                    validate_ranges(ranges, obj["size"])
                    region = regions[obj["tier"]]
                    for ptr, n, off in ranges:
                        started = True
                        region.copy(obj["offset"] + off, ptr, n, 3 if device else 1)
                    return True
            yield Reader()
            close_mappings()
        except BaseException:
            if started:
                self.quarantined.append((lease, regions))
            else:
                try:
                    close_mappings()
                    self.rpc("release", token=lease["token"], epoch=lease["epoch"])
                except BaseException:
                    self.quarantined.append((lease, regions))
            raise
        try:
            self.rpc("release", token=lease["token"], epoch=lease["epoch"])
        except BaseException:
            # Lost ACK is not evidence that the owner released the pin.
            self.quarantined.append((lease, regions))
            raise
