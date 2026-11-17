"""Opt-in HTTP control RPC for a trusted private domain, not a tenant boundary.

Bearer credentials are supplied by the owner (never logged). TLS may be supplied
by the caller or a private proxy. One shared credential authenticates the domain,
not an individual peer; exact identity allowlists fence routing/incarnations.
No serving factory is installed here. The gateway submits directly to the
independent SlotGuardService. This module owns bounded networking only.
Closing networking never releases leases: retain the selected source service
and its owners for reconciliation of unknown Pins.
"""

import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import queue
import socket
import ssl
import threading
import time
from concurrent.futures import (
    Future,
    InvalidStateError,
    ThreadPoolExecutor,
    TimeoutError,
)
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

from .control_profile import ControlProfile, PinRequest, UnpinRequest, require_profile
from .directory import EngineDirectory
from .protocol import (
    EngineIdentity,
    _integer,
    _typed,
    validate_peer,
)
from .slotguard_control_wire import MAX_BODY_BYTES, WireError


def _encode(profile, message, *, max_body_bytes):
    return profile.codec.encode(message, max_body_bytes=max_body_bytes)


class ControlHTTPError(RuntimeError):
    """Sanitized transport failure; never implies source cleanup."""


def _timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return value


def _credential(token, trusted_private_domain):
    if trusted_private_domain is not True:
        raise ValueError("trusted private domain must be explicitly enabled")
    if (
        type(token) is not str
        or len(token) < 16
        or len(token) > 4096
        or any(ord(ch) < 33 or ord(ch) > 126 for ch in token)
    ):
        raise ValueError("bearer token must contain 16..4096 visible ASCII characters")
    return "Bearer " + token


def _peer_key(identity):
    return hashlib.sha256(
        json.dumps(asdict(identity), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _interrupt_socket(connection):
    if connection is not None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()


@dataclass
class _Call:
    future: Future
    deadline: float
    connection: socket.socket | None = None
    finished: bool = False
    work_done: bool = False
    delivered: bool = False


class HTTPPrefixControlClient:
    """Exact directory endpoints, fixed workers, bounded running + queued calls.

    pin/unpin perform no encoding or network I/O on the submitting worker.
    Cancellation/timeouts only affect observation; they cannot cancel source work.
    Future callbacks must be nonblocking; close() belongs to the lifecycle owner,
    never a callback. Fixed delivery workers isolate callbacks from the watchdog
    and socket workers. Even a blocked callback cannot delay socket interruption.
    """

    def __init__(
        self,
        directory: EngineDirectory,
        *,
        bearer_token: str,
        trusted_private_domain: bool,
        workers: int = 4,
        queue_capacity: int = 16,
        connect_timeout: float = 2,
        read_timeout: float = 30,
        rpc_timeout: float | None = None,
        max_body_bytes: int = MAX_BODY_BYTES,
        tls_context: ssl.SSLContext | None = None,
        control_profile: ControlProfile,
    ):
        _typed("directory", directory, EngineDirectory)
        self._profile = require_profile(control_profile)
        for name, value in (
            ("workers", workers),
            ("queue_capacity", queue_capacity),
            ("max_body_bytes", max_body_bytes),
        ):
            _integer(name, value, 1)
        self._authorization = _credential(bearer_token, trusted_private_domain)
        self._directory = directory
        self._peer = _peer_key(directory.target)
        self._connect_timeout = _timeout(connect_timeout)
        self._read_timeout = _timeout(read_timeout)
        self._rpc_timeout = _timeout(
            read_timeout if rpc_timeout is None else rpc_timeout
        )
        self._max_body = max_body_bytes
        self._tls = tls_context
        self._capacity = threading.BoundedSemaphore(workers + queue_capacity)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="l1-http-client"
        )
        self._lock = threading.Lock()
        self._closed = False
        self._calls = {}
        # Resolve the immutable directory once, before accepting RPCs. No worker
        # enters the platform's non-interruptible getaddrinfo during an RPC or
        # shutdown. The original hostname still supplies HTTP Host and TLS SNI.
        # Select one address; do not retry/fall back to another resolved address.
        self._addresses = {}
        for entry in directory.entries:
            url = urlsplit(entry.control_endpoint)
            self._addresses[entry.control_endpoint] = socket.getaddrinfo(
                url.hostname, url.port, type=socket.SOCK_STREAM
            )[0]
        self._stop = threading.Event()
        self._deliveries = queue.Queue(maxsize=workers + queue_capacity)
        self._delivery_stop = threading.Event()
        self._delivery_workers = [
            threading.Thread(target=self._deliver, name=f"l1-http-client-delivery-{i}")
            for i in range(workers)
        ]
        self._watchdog = threading.Thread(
            target=self._watch, name="l1-http-client-deadlines"
        )
        for thread in (*self._delivery_workers, self._watchdog):
            thread.start()

    def pin(self, endpoint: str, request: PinRequest) -> Future:
        pin, response, _, _ = self._profile.types
        return self._submit(endpoint, request, pin, response, self._profile.paths[0])

    def unpin(self, endpoint: str, request: UnpinRequest) -> Future:
        _, _, unpin, response = self._profile.types
        return self._submit(endpoint, request, unpin, response, self._profile.paths[1])

    def _submit(self, endpoint, request, request_type, response_type, path):
        # Failed local admission is represented by the same Future API.
        failed = Future()
        try:
            if type(request) is not request_type:
                raise ControlHTTPError("invalid control request")
            source, target, transfer_id = self._profile.endpoints(request)
            if not any(
                entry.identity == source and entry.control_endpoint == endpoint
                for entry in self._directory.entries
            ):
                raise ControlHTTPError("control endpoint identity mismatch")
            if target is not None and target != self._directory.target:
                raise ControlHTTPError("control target mismatch")
            if transfer_id.target_incarnation != self._directory.target.incarnation:
                raise ControlHTTPError("control target incarnation mismatch")
            with self._lock:
                if self._closed:
                    raise ControlHTTPError("control client closed")
                if not self._capacity.acquire(blocking=False):
                    raise ControlHTTPError("control client capacity exhausted")
                call = _Call(failed, time.monotonic() + self._rpc_timeout)
                self._calls[failed] = call
                try:
                    task = self._executor.submit(
                        self._run, call, endpoint, request, response_type, path
                    )
                except BaseException:
                    self._calls.pop(failed)
                    self._capacity.release()
                    raise

                # A cancelled public Future must not release the permit while
                # its work item still occupies the executor's physical queue.
                def completed(task):
                    with self._lock:
                        call.work_done = True
                        self._retire_locked(call)

            # A fast task may already be done; its callback must not reacquire
            # the client lock while submit still holds it.
            task.add_done_callback(completed)
            return failed
        except Exception:  # noqa: BLE001 - keep transport errors payload-free
            failed.set_exception(ControlHTTPError("control request rejected locally"))
            return failed

    def _run(self, call, endpoint, request, response_type, path):
        with self._lock:
            if call.finished:
                return
            if not call.future.set_running_or_notify_cancel():
                call.finished = call.delivered = True
                return
        try:
            result = self._call(call, endpoint, request, response_type, path)
        except ControlHTTPError as error:
            self._complete(call, error=error)
        else:
            self._complete(call, result=result)

    def _complete(self, call, *, result=None, error=None):
        with self._lock:
            if call.finished:
                return
            call.finished = True
            if time.monotonic() >= call.deadline:
                error = ControlHTTPError(
                    "control RPC deadline exceeded; source lease state unknown"
                )
        self._deliveries.put_nowait((call, result, error))

    def _retire_locked(self, call):
        # The one admission permit reserves both a physical executor item and
        # one delivery. Callback stalls therefore cannot grow either queue.
        if call.work_done and call.delivered:
            self._calls.pop(call.future)
            self._capacity.release()

    def _deliver(self):
        while not self._delivery_stop.is_set() or not self._deliveries.empty():
            try:
                call, result, error = self._deliveries.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._resolve(call, result, error)
            finally:
                with self._lock:
                    call.delivered = True
                    self._retire_locked(call)

    @staticmethod
    def _resolve(call, result=None, error=None):
        # Only fixed delivery workers invoke callbacks, never the watchdog.
        try:
            if error is None:
                call.future.set_result(result)
            else:
                call.future.set_exception(error)
        except InvalidStateError:
            pass  # Cancellation is observation only, never Unpin permission.

    def _expire(self, *, closing=False):
        expired = []
        with self._lock:
            for call in self._calls.values():
                if not call.finished and (closing or time.monotonic() >= call.deadline):
                    call.finished = True
                    expired.append(call)
        for call in expired:
            _interrupt_socket(call.connection)
        # Interrupt every expired socket before publishing any completion. The
        # watchdog never invokes user code, even if all delivery workers stall.
        for call in expired:
            self._deliveries.put_nowait(
                (
                    call,
                    None,
                    ControlHTTPError("control RPC ended; source lease state unknown"),
                )
            )

    def _watch(self):
        while not self._stop.wait(0.01):
            self._expire()

    def _track_socket(self, call, connection):
        with self._lock:
            if self._closed or call.finished or time.monotonic() >= call.deadline:
                connection.close()
                raise ControlHTTPError("control RPC deadline exceeded")
            call.connection = connection
        connection.settimeout(
            min(self._connect_timeout, call.deadline - time.monotonic())
        )

    def _call(self, call, endpoint, request, response_type, path):
        connection = None
        try:
            body = _encode(self._profile, request, max_body_bytes=self._max_body)
            url = urlsplit(endpoint)
            cls = (
                http.client.HTTPSConnection
                if url.scheme == "https"
                else http.client.HTTPConnection
            )
            kwargs = {"timeout": self._connect_timeout}
            if url.scheme == "https":
                kwargs["context"] = self._tls
            connection = cls(url.hostname, url.port, **kwargs)
            family, socktype, proto, _, address = self._addresses[endpoint]
            connection.sock = socket.socket(family, socktype, proto)
            self._track_socket(call, connection.sock)
            connection.sock.connect(address)
            if url.scheme == "https":
                context = self._tls or ssl.create_default_context()
                connection.sock = context.wrap_socket(
                    connection.sock,
                    server_hostname=url.hostname,
                    do_handshake_on_connect=False,
                )
                self._track_socket(call, connection.sock)
                connection.sock.do_handshake()
            connection.sock.settimeout(self._read_timeout)
            connection.request(
                "POST",
                path,
                body=body,
                headers={
                    "Authorization": self._authorization,
                    "X-L1-Peer": self._peer,
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            # http.client does not follow redirects. No retry/fallback endpoint.
            lengths = response.headers.get_all("Content-Length", [])
            if (
                response.status != 200
                or len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
                or int(lengths[0]) > self._max_body
                or response.headers.get("Transfer-Encoding") is not None
                or response.headers.get("Content-Type") != "application/json"
            ):
                raise ControlHTTPError("invalid control HTTP response")
            length = int(lengths[0])
            payload = response.read(length + 1)
            if len(payload) != length:
                raise ControlHTTPError("incomplete control response")
            result = self._profile.decode(
                payload,
                response_type,
                max_body_bytes=self._max_body,
                request_context=request,
            )
            matches = (
                self._profile.pin_matches(result, request)
                if type(request) is self._profile.types[0]
                else result.request == request
            )
            if not matches:
                raise ControlHTTPError("control response correlation mismatch")
            return result
        except Exception:  # noqa: BLE001 - no network exception details on the Future
            raise ControlHTTPError(
                "control RPC failed; source lease state unknown"
            ) from None
        finally:
            with self._lock:
                call.connection = None
            if connection is not None:
                connection.close()

    def close(self):
        if threading.current_thread() in self._delivery_workers:
            raise RuntimeError("close must run outside control Future callbacks")
        with self._lock:
            self._closed = True
        self._expire(closing=True)
        self._stop.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._watchdog.join()
        self._delivery_stop.set()
        for thread in self._delivery_workers:
            thread.join()


class _HTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        with self.gateway._lock:
            self.gateway._deadlines[request] = (
                time.monotonic() + self.gateway._socket_timeout
            )
        try:
            self.gateway._sockets.put_nowait((request, client_address))
        except queue.Full:
            with self.gateway._lock:
                self.gateway._deadlines.pop(request, None)
            self.shutdown_request(request)

    def service_actions(self):
        # Listener-owned deadline enforcement never executes Future callbacks.
        self.gateway._expire_sockets()

    def handle_error(self, request, client_address):
        # BaseServer normally prints tracebacks (possibly including payloads).
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        self._reply(code, b'{"error":"control request rejected"}')

    def _reply(self, code, body):
        self.close_connection = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        import json

        gateway = self.server.gateway
        authorization = self.headers.get_all("Authorization", [])
        if len(authorization) != 1 or not hmac.compare_digest(
            authorization[0].encode("latin-1"), gateway._authorization.encode("ascii")
        ):
            self.send_error(401)
            return
        if self.path != "/debug/l1-trace" or gateway.observer is None:
            self.send_error(404)
            return
        self._reply(200, json.dumps(gateway.observer.snapshot()).encode())

    def do_POST(self):
        gateway = self.server.gateway
        try:
            authorization = self.headers.get_all("Authorization", [])
            peers = self.headers.get_all("X-L1-Peer", [])
            if (
                len(authorization) != 1
                or len(peers) != 1
                or not hmac.compare_digest(
                    authorization[0].encode("latin-1"),
                    gateway._authorization.encode("ascii"),
                )
                or peers[0] not in gateway._peers
            ):
                self.send_error(401)
                return
            expected = dict(
                zip(
                    gateway._profile.paths,
                    (gateway._profile.types[0], gateway._profile.types[2]),
                )
            ).get(self.path)
            if expected is None:
                self.send_error(404)
                return
            lengths = self.headers.get_all("Content-Length", [])
            if (
                len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
                or self.headers.get("Transfer-Encoding") is not None
                or self.headers.get_all("Content-Type", []) != ["application/json"]
            ):
                self.send_error(400)
                return
            length = int(lengths[0])
            if length > gateway._max_body:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            if len(body) != length:
                self.send_error(400)
                return
            observer = getattr(gateway._submitter, "observer", None)
            decode_start = time.monotonic_ns() if observer is not None else 0
            request = gateway._profile.decode(
                body, expected, max_body_bytes=gateway._max_body
            )
            peer = gateway._peers[peers[0]]
            source, target, transfer_id = gateway._profile.endpoints(request)
            if (
                source != gateway.source
                or transfer_id.target_incarnation != peer.incarnation
                or (target is not None and target != peer)
            ):
                self.send_error(403)
                return
            if observer is not None and expected is gateway._profile.types[0]:
                from .observation import emit

                emit(
                    observer,
                    "source_pin_decode",
                    transfer_id,
                    duration_ns=time.monotonic_ns() - decode_start,
                )
            gateway._response_deadline(self.connection)
            future = gateway._submitter.submit(request)
            deadline = time.monotonic() + gateway._response_timeout
            while not gateway._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.send_error(504)
                    return
                try:
                    result = future.result(timeout=min(remaining, 0.05))
                except TimeoutError:
                    continue
                self._reply(
                    200,
                    _encode(gateway._profile, result, max_body_bytes=gateway._max_body),
                )
                return
            # No cancellation or unlock. Scheduler still owns submitted work.
        except (WireError, ValueError):
            self.send_error(400)
        except Exception:  # noqa: BLE001 - sanitize every control boundary failure
            # Includes disconnect, timeout and sanitized Scheduler errors.
            try:
                self.send_error(503)
            except OSError:
                pass


class HTTPControlGateway:
    """Explicit control profile; start/close and fixed bounded socket workers.

    close stops networking but never closes the SlotGuardService or its leases;
    the service remains alive for authoritative reconciliation.
    """

    def __init__(
        self,
        source: EngineIdentity,
        peers: tuple[EngineIdentity, ...],
        *,
        bind_address: tuple[str, int],
        bearer_token: str,
        trusted_private_domain: bool,
        workers: int = 4,
        queue_capacity: int = 16,
        socket_timeout: float = 5,
        response_timeout: float = 25,
        max_body_bytes: int = MAX_BODY_BYTES,
        tls_context: ssl.SSLContext | None = None,
        observer=None,
        control_profile: ControlProfile,
        slotguard_service,
    ):
        self.observer = observer
        self._profile = require_profile(control_profile)
        from .slotguard_service import SlotGuardService

        if type(slotguard_service) is not SlotGuardService:
            raise ValueError(
                "gateway requires an explicit independent SlotGuardService"
            )
        if (
            slotguard_service.local_identity != source
            or slotguard_service.exact_peers != frozenset(peers)
        ):
            raise ValueError("service/gateway exact peers mismatch")
        self._submitter = slotguard_service
        _typed("source", source, EngineIdentity)
        self._authorization = _credential(bearer_token, trusted_private_domain)
        if type(bind_address) is not tuple or len(bind_address) != 2:
            raise ValueError("explicit private bind address required")
        address = ipaddress.ip_address(bind_address[0])
        if (
            address.is_unspecified
            or address.is_multicast
            or not (address.is_private or address.is_loopback)
        ):
            raise ValueError("explicit private bind address required")
        _integer("bind port", bind_address[1])
        if bind_address[1] > 65535:
            raise ValueError("invalid bind port")
        for name, value in (
            ("workers", workers),
            ("queue_capacity", queue_capacity),
            ("max_body_bytes", max_body_bytes),
        ):
            _integer(name, value, 1)
        if type(peers) is not tuple or not peers:
            raise ValueError("nonempty immutable peer allowlist required")
        for peer in peers:
            validate_peer(source, peer)
        if len({peer.incarnation for peer in peers}) != len(peers):
            raise ValueError("peer incarnations must be unique")
        self._peers = {_peer_key(peer): peer for peer in peers}
        self.source = source
        self._socket_timeout = _timeout(socket_timeout)
        self._response_timeout = _timeout(response_timeout)
        self._max_body, self._tls = max_body_bytes, tls_context
        self._stop = threading.Event()
        self._sockets = queue.Queue(maxsize=queue_capacity)
        self._active = set()
        self._deadlines = {}
        self._lock = threading.Lock()
        self._started = self._closed = False
        server_cls = type(
            "_BoundHTTPServer",
            (_HTTPServer,),
            {
                "address_family": (
                    socket.AF_INET6 if address.version == 6 else socket.AF_INET
                ),
            },
        )
        self._server = server_cls(bind_address, _Handler)
        self._server.gateway = self
        self._workers = [
            threading.Thread(target=self._work, name=f"l1-http-server-{i}")
            for i in range(workers)
        ]
        self._listener = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="l1-http-listener",
        )

    @property
    def address(self):
        return self._server.server_address[:2]

    def start(self):
        if self._closed or self._started:
            raise RuntimeError("gateway cannot be started twice or after close")
        self._started = True
        for thread in self._threads():
            thread.start()
        return self

    def _threads(self):
        return (
            *self._workers,
            self._listener,
        )

    def _work(self):
        while not self._stop.is_set():
            try:
                connection, address = self._sockets.get(timeout=0.05)
            except queue.Empty:
                continue
            with self._lock:
                self._active.add(connection)
            try:
                with self._lock:
                    deadline = self._deadlines.get(connection, 0)
                if self._stop.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("control request deadline exceeded")
                connection.settimeout(self._socket_timeout)
                if self._tls is not None:
                    # Do not block the listener on a TLS handshake.
                    original = connection
                    connection = self._tls.wrap_socket(
                        connection, server_side=True, do_handshake_on_connect=False
                    )
                    with self._lock:
                        self._active.discard(original)
                        self._active.add(connection)
                        # Expiration can race TLS socket replacement. Missing
                        # deadlines remain expired; never revive a timed-out socket.
                        deadline = self._deadlines.pop(original, 0)
                        self._deadlines[connection] = deadline
                    if time.monotonic() >= deadline:
                        raise TimeoutError("control request deadline exceeded")
                    connection.do_handshake()
                self._server.finish_request(connection, address)
            except Exception:  # noqa: BLE001, S110 - never log request/descriptor data
                pass
            finally:
                with self._lock:
                    self._active.discard(connection)
                    self._deadlines.pop(connection, None)
                self._server.shutdown_request(connection)

    def _response_deadline(self, connection):
        with self._lock:
            now = time.monotonic()
            if self._stop.is_set() or now >= self._deadlines.get(connection, 0):
                raise TimeoutError("control request deadline exceeded")
            # Header/body/queue budget cannot be reset by incoming bytes. Once
            # validated, a separate absolute budget covers preparation, Scheduler
            # waiting, response serialization and writes.
            self._deadlines[connection] = now + self._response_timeout

    def _expire_sockets(self):
        with self._lock:
            expired = [
                connection
                for connection, deadline in self._deadlines.items()
                if time.monotonic() >= deadline
            ]
            for connection in expired:
                self._deadlines.pop(connection)
        for connection in expired:
            _interrupt_socket(connection)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # start() can fail after only some worker threads were launched. Never
        # wait for serve_forever's shutdown event when its listener did not
        # start, and never join an unstarted Thread. ident is retained even if
        # a successfully launched thread already exited.
        if self._listener.ident is not None:
            self._server.shutdown()
        self._server.server_close()
        with self._lock:
            connections = tuple(self._deadlines)
            self._deadlines.clear()
        for connection in connections:
            _interrupt_socket(connection)
        while True:
            try:
                connection, _ = self._sockets.get_nowait()
            except queue.Empty:
                break
            self._server.shutdown_request(connection)
        for thread in self._threads():
            if thread.ident is not None:
                thread.join()
