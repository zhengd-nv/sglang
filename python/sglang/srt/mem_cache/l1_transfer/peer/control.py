# SPDX-License-Identifier: Apache-2.0
"""Versioned, bounded JSON RPC for fixed trusted peers.

Serve dispatch on the cache owner's thread. No network worker mutates the cache.
This protocol is unauthenticated: only use inside an isolated trusted job. Lost
responses are not permission to release leases. No automatic reconnect/retry.
"""
import hashlib
import json
import socket
import struct
import uuid
from dataclasses import asdict, dataclass

MAX_FRAME = 8 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ModelIdentity:
    weights_revision: str
    model_config_digest: str
    position_config_digest: str
    adapter_revision: str
    context_digest: str
    tp_size: int = 1
    tp_rank: int = 0

    def validate(self):
        if type(self.tp_size) is not int or type(self.tp_rank) is not int or (self.tp_size, self.tp_rank) != (1, 0):
            raise ValueError('only TP1 supported')
        for name in ('weights_revision', 'adapter_revision'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be explicit')
        for name in ('model_config_digest', 'position_config_digest', 'context_digest'):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
                raise ValueError(f'{name} must be a SHA256 digest')


def contract(identity, layout):
    identity.validate(); layout.validate()
    return {'model': asdict(identity), 'layout': asdict(layout)}


def wire_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def recv_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size - len(chunks))
        if not block:
            raise ConnectionError('peer closed control connection')
        chunks.extend(block)
    return bytes(chunks)


def receive(sock):
    size, = struct.unpack('!I', recv_exact(sock, 4))
    if not 0 < size <= MAX_FRAME:
        raise ValueError('invalid control frame size')
    value = json.loads(recv_exact(sock, size))
    if not isinstance(value, dict):
        raise ValueError('control message must be an object')
    return value


def send(sock, value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if not 0 < len(data) <= MAX_FRAME:
        raise ValueError('control frame too large')
    sock.sendall(struct.pack('!I', len(data)) + data)


class RemoteRejected(RuntimeError):
    """A request was rejected before creating new source ownership."""


class SocketRPC:
    def __init__(self, sock):
        self.sock = sock
        self.failed = False

    def call(self, operation, **payload):
        if self.failed:
            raise RuntimeError('control channel quarantined')
        request = {'version': 1, 'id': uuid.uuid4().hex, 'operation': operation, **payload}
        try:
            send(self.sock, request)
            response = receive(self.sock)
            if response.get('version') != 1 or response.get('id') != request['id']:
                raise ValueError('stale or mismatched response')
            if response.get('ok') is not True:
                if response.get('safe_rejection') is True:
                    raise RemoteRejected(response.get('error', 'rejected'))
                raise RuntimeError(response.get('error', 'uncertain remote failure'))
            return response['result']
        except RemoteRejected:
            raise
        except BaseException:
            self.failed = True
            raise


def serve_once(sock, endpoint):
    request = receive(sock)
    response = {'version': 1, 'id': request.get('id'), 'ok': False}
    try:
        response['result'] = endpoint.dispatch(request)
        response['ok'] = True
    except RemoteRejected as exc:
        response.update(error=str(exc), safe_rejection=True)
    except Exception as exc:
        response.update(error=str(exc), safe_rejection=False)
    send(sock, response)
