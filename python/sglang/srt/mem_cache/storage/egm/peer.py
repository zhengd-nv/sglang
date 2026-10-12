# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Same-user Unix control plane; payloads travel via CUDA Fabric mappings.

No timeout-based lease reclamation: a lost reader quarantines its pin until
explicit recovery/process restart. This favors safety over availability.
"""
import json
import os
import socket
import socketserver
import threading
from contextlib import contextmanager

from .region import Region
from .store import validate_ranges

MAX_MESSAGE = 1 << 20


def receive(stream):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line.endswith(b"\n") or len(line) > MAX_MESSAGE:
        raise ValueError("invalid/oversized control message")
    return json.loads(line)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        try:
            req = receive(self.rfile)
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
    def __init__(self, store, path):
        # The caller owns a private directory; never unlink an existing endpoint.
        if os.path.exists(path):
            raise FileExistsError(path)
        self.path = path
        self.server = socketserver.UnixStreamServer(path, _Handler)
        os.chmod(path, 0o600)
        self.server.store = store
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        os.unlink(self.path)


class PeerClient:
    def __init__(self, path):
        self.path = path
        self.quarantined = []

    def rpc(self, op, **args):
        payload = json.dumps(dict(op=op, **args)).encode() + b"\n"
        if len(payload) > MAX_MESSAGE:
            raise ValueError("control message too large")
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(30)
            sock.connect(self.path)
            sock.sendall(payload)
            with sock.makefile("rb") as stream:
                result = receive(stream)
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
        region = None
        started = False
        try:
            region = Region(descriptor=lease["region"])
            class Reader:
                def get(self, key, ranges, device=False):
                    nonlocal started
                    offset, size = lease["objects"][key]
                    validate_ranges(ranges, size)
                    for ptr, n, off in ranges:
                        started = True
                        region.copy(offset + off, ptr, n, 3 if device else 1)
                    return True
            yield Reader()
            region.close()
            region = None
        except BaseException:
            if started:
                # Unknown completion: never ACK or let an allocation be reused.
                self.quarantined.append((lease, region))
            else:
                if region is not None:
                    region.close()
                self.rpc("release", token=lease["token"], epoch=lease["region"]["epoch"])
            raise
        self.rpc("release", token=lease["token"], epoch=lease["region"]["epoch"])
