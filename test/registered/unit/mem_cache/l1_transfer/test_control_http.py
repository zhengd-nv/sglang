"""Hermetic actual loopback control RPC over a real SlotGuard source service."""

import http.client
import json
import socket
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from unittest.mock import patch

from sglang.srt.mem_cache.l1_transfer import control_http as h
from sglang.srt.mem_cache.l1_transfer import directory as d
from sglang.srt.mem_cache.l1_transfer import hash_pin as hp
from sglang.srt.mem_cache.l1_transfer import slot_guard as s
from sglang.srt.mem_cache.l1_transfer import slotguard_control_wire as w
from sglang.srt.mem_cache.l1_transfer import slotguard_protocol as v
from sglang.srt.mem_cache.l1_transfer import slotguard_service as svc
from sglang.srt.mem_cache.l1_transfer.control_profile import ControlProfile
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.slotguard.source import ServiceFixture
from sglang.test.kits.slotguard.wire import identity
from sglang.test.test_utils import CustomTestCase

V3 = ControlProfile.SLOTGUARD_HASH_V3
v, w = v, w
TOKEN = "test-only-private-domain-token"

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def request(source=None, attempt=0):
    """Compact hash Pin for the fixture's published prefix, optionally re-sourced."""
    result = ServiceFixture().lookup(attempt)
    if source is not None:
        result = replace(result, source=source)
    lease = s.LeaseIdentity(
        result.source.incarnation,
        result.query.target.incarnation,
        result.transfer_id,
        f"nonce-{attempt}",
    )
    return v.PinReq(hp.HashLookupResult.from_lookup(result), lease)


class DrippingPeer:
    """Actual socket peer that never finishes its HTTP response."""

    def __init__(self, mode):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.mode = mode
        self.stop = threading.Event()
        self.started = threading.Event()
        self.connection = None
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    @property
    def endpoint(self):
        host, port = self.listener.getsockname()
        return f"http://{host}:{port}"

    def run(self):
        try:
            connection, _ = self.listener.accept()
            self.connection = connection
            connection.settimeout(1)
            connection.recv(65536)
            if self.mode == "headers":
                connection.sendall(b"HTTP/1.1 200 OK\r\nX-Progress: ")
            elif self.mode == "body":
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n"
                    b"Content-Type: application/json\r\nConnection: close\r\n\r\n"
                )
            payload = b"HTTP/1.1 100 Continue\r\n\r\n" if self.mode == "1xx" else b"."
            while not self.stop.is_set():
                connection.sendall(payload)
                self.started.set()
                self.stop.wait(0.02)
        except OSError:
            pass

    def close(self):
        self.stop.set()
        h._interrupt_socket(self.connection)
        self.listener.close()
        self.thread.join(timeout=3)


class AbsoluteDeadlineTest(CustomTestCase):
    def client(self, peer, **options):
        directory = d.EngineDirectory(
            identity("target"),
            (d.EngineDirectoryEntry("source", identity("source"), peer.endpoint),),
        )
        return h.HTTPPrefixControlClient(
            directory,
            bearer_token=TOKEN,
            trusted_private_domain=True,
            workers=1,
            queue_capacity=1,
            control_profile=V3,
            **options,
        )

    def test_1xx_headers_and_body_drips_cannot_extend_rpc_deadline(self):
        for mode in ("1xx", "headers", "body"):
            with self.subTest(mode=mode):
                peer = DrippingPeer(mode)
                client = self.client(peer, read_timeout=0.12)
                try:
                    start = time.monotonic()
                    future = client.pin(peer.endpoint, request())
                    self.assertTrue(peer.started.wait(1))
                    with self.assertRaises(h.ControlHTTPError):
                        future.result(timeout=0.8)
                    self.assertLess(time.monotonic() - start, 0.8)
                finally:
                    client.close()
                    peer.close()

    def test_close_interrupts_active_1xx_even_with_long_deadline(self):
        peer = DrippingPeer("1xx")
        client = self.client(peer, read_timeout=20, rpc_timeout=30)
        finished = threading.Event()
        closer = threading.Thread(target=lambda: (client.close(), finished.set()))
        try:
            future = client.pin(peer.endpoint, request())
            self.assertTrue(peer.started.wait(1))
            closer.start()
            self.assertTrue(
                finished.wait(0.8), "close waited on endless interim responses"
            )
            with self.assertRaises(h.ControlHTTPError):
                future.result(timeout=0)
            self.assertFalse(client._watchdog.is_alive())
        finally:
            peer.close()
            if closer.ident is not None:
                closer.join(2)
            client.close()

    def test_queue_wait_is_included_and_expired_work_never_connects(self):
        peer = DrippingPeer("1xx")
        client = self.client(peer, read_timeout=10, rpc_timeout=0.12)
        started, unblock = threading.Event(), threading.Event()
        calls = []

        def blocked(*args):
            calls.append(1)
            started.set()
            unblock.wait(2)
            raise h.ControlHTTPError("blocked worker")

        try:
            with patch.object(client, "_call", side_effect=blocked):
                first = client.pin(peer.endpoint, request())
                self.assertTrue(started.wait(1))
                second = client.pin(peer.endpoint, request())
                with self.assertRaises(h.ControlHTTPError):
                    second.result(timeout=0.8)
                self.assertTrue(first.done())
                self.assertEqual(len(calls), 1)
                unblock.set()
                client.close()
                self.assertEqual(len(calls), 1)
        finally:
            unblock.set()
            client.close()
            peer.close()

    def test_blocked_callback_does_not_delay_another_rpc_deadline(self):
        peers = [DrippingPeer("1xx"), DrippingPeer("1xx")]
        sources = [identity("source"), identity("source-two")]
        directory = d.EngineDirectory(
            identity("target"),
            tuple(
                d.EngineDirectoryEntry(str(index), source, peer.endpoint)
                for index, (source, peer) in enumerate(zip(sources, peers))
            ),
        )
        client = h.HTTPPrefixControlClient(
            directory,
            bearer_token=TOKEN,
            trusted_private_domain=True,
            workers=2,
            queue_capacity=2,
            rpc_timeout=0.12,
            read_timeout=10,
            control_profile=V3,
        )
        blocked, unblock = threading.Event(), threading.Event()
        callback_threads = []

        def block_callback(_):
            callback_threads.append(threading.current_thread().name)
            blocked.set()
            unblock.wait(3)

        try:
            first = client.pin(peers[0].endpoint, request())
            first.add_done_callback(block_callback)
            second_req = request(source=sources[1])
            second = client.pin(peers[1].endpoint, second_req)
            self.assertTrue(all(peer.started.wait(1) for peer in peers))
            self.assertTrue(blocked.wait(0.8))
            with self.assertRaises(h.ControlHTTPError):
                second.result(timeout=0.35)
            self.assertFalse(unblock.is_set())
            self.assertTrue(client._watchdog.is_alive())
            self.assertEqual(len(client._delivery_workers), 2)
            self.assertTrue(
                all(
                    name.startswith("l1-http-client-delivery-")
                    for name in callback_threads
                )
            )
            for peer in peers:
                peer.thread.join(0.5)
                self.assertFalse(
                    peer.thread.is_alive(), "deadline did not interrupt socket"
                )
        finally:
            unblock.set()
            client.close()
            for peer in peers:
                peer.close()

    def test_saturated_delivery_remains_bounded_and_all_sockets_are_interrupted(self):
        peers = [DrippingPeer("1xx") for _ in range(4)]
        sources = [identity(f"source-{index}") for index in range(4)]
        directory = d.EngineDirectory(
            identity("target"),
            tuple(
                d.EngineDirectoryEntry(str(index), source, peer.endpoint)
                for index, (source, peer) in enumerate(zip(sources, peers))
            ),
        )
        client = h.HTTPPrefixControlClient(
            directory,
            bearer_token=TOKEN,
            trusted_private_domain=True,
            workers=2,
            queue_capacity=2,
            rpc_timeout=0.2,
            read_timeout=10,
            control_profile=V3,
        )
        unblock = threading.Event()
        registered = threading.Event()
        saturated = threading.Event()
        callback_lock = threading.Lock()
        blocked_count = 0
        resolve = client._resolve

        def registered_resolve(call, result=None, error=None):
            # A Future must not resolve before all four calls and callbacks are
            # registered. Otherwise add_done_callback may execute on this test
            # thread, or a fast unblocked delivery may release a permit early.
            registered.wait(3)
            resolve(call, result, error)

        def callback(_):
            nonlocal blocked_count
            with callback_lock:
                blocked_count += 1
                if blocked_count == 2:
                    saturated.set()
            unblock.wait(3)

        delivery_barrier = patch.object(client, "_resolve", registered_resolve)
        delivery_barrier.start()
        try:
            futures = []
            for index in range(4):
                req = request(source=sources[index])
                future = client.pin(peers[index].endpoint, req)
                futures.append(future)
                # Block whichever two deliveries finish first, not specifically
                # request 0/1: completion order is deliberately unspecified.
                future.add_done_callback(callback)
            registered.set()
            self.assertTrue(saturated.wait(0.8))
            # Two network workers are free after deadline interruption, but the
            # four admission permits stay held by blocked/pending deliveries.
            with self.assertRaises(h.ControlHTTPError):
                client.pin(
                    peers[0].endpoint,
                    request(source=sources[0]),
                ).result(timeout=0.2)
            self.assertLessEqual(client._deliveries.qsize(), 2)
            self.assertEqual(len(client._calls), 4)
            self.assertTrue(client._watchdog.is_alive())
            for peer in peers[:2]:
                peer.thread.join(0.5)
                self.assertFalse(peer.thread.is_alive())
        finally:
            registered.set()
            unblock.set()
            client.close()
            delivery_barrier.stop()
            for peer in peers:
                peer.close()


class HTTPControlTest(CustomTestCase):
    def setUp(self):
        self.base = ServiceFixture()
        self.base.setUp()
        self.service = self.base.service
        self.gateway = h.HTTPControlGateway(
            identity("source"),
            (identity("target"),),
            bind_address=("127.0.0.1", 0),
            bearer_token=TOKEN,
            trusted_private_domain=True,
            workers=2,
            queue_capacity=2,
            response_timeout=2,
            control_profile=V3,
            slotguard_service=self.service,
        ).start()
        host, port = self.gateway.address
        self.endpoint = f"http://{host}:{port}"
        self.directory = d.EngineDirectory(
            identity("target"),
            (
                d.EngineDirectoryEntry(
                    "source-reporter", identity("source"), self.endpoint
                ),
            ),
        )
        self.client = self.make_client(workers=2, queue_capacity=2, read_timeout=3)

    def make_client(self, **options):
        return h.HTTPPrefixControlClient(
            self.directory,
            bearer_token=TOKEN,
            trusted_private_domain=True,
            control_profile=V3,
            **options,
        )

    def tearDown(self):
        self.gateway.close()
        self.client.close()
        self.base.tearDown()

    def release(self, req):
        """Reconcile a lease whose reply the target never accepted."""
        state = self.service.query(req.identity)
        self.service.submit(
            v.UnpinReq(req.reference, state.lease_id, True, True)
        ).result(2)

    def raw(self, body, *, token=TOKEN, path=V3.paths[0], peer=None):
        conn = http.client.HTTPConnection(*self.gateway.address, timeout=2)
        try:
            conn.request(
                "POST",
                path,
                body,
                headers={
                    "Authorization": "Bearer " + token,
                    "X-L1-Peer": h._peer_key(identity("target"))
                    if peer is None
                    else peer,
                    "Content-Type": "application/json",
                },
            )
            response = conn.getresponse()
            response.read()
            return response.status
        finally:
            conn.close()

    def test_rejected_before_submission(self):
        req = request()
        unknown = request(source=identity("unknown"))
        root = json.loads(w.encode(req))
        root["payload"]["fingerprint"] = "forged"
        cases = [
            (w.encode(req), {"token": "wrong"}, 401),
            (w.encode(req), {"peer": "unknown"}, 401),
            (w.encode(req), {"path": "/admin"}, 404),
            (w.encode(unknown), {}, 403),
            (json.dumps(root).encode(), {}, 400),
            (b"invalid", {}, 400),
        ]
        with patch.object(self.service, "submit", wraps=self.service.submit) as submit:
            for body, options, status in cases:
                self.assertEqual(self.raw(body, **options), status)
        self.assertEqual(submit.call_count, 0)

    def test_client_bounds_endpoints_close_and_timeout(self):
        self.client.close()
        self.client = self.make_client(workers=1, queue_capacity=1, read_timeout=0.1)
        # The source accepts each request but never answers within the RPC budget.
        with patch.object(
            self.service, "submit", side_effect=lambda req: Future()
        ) as submit:
            first = self.client.pin(self.endpoint, request())
            second = self.client.pin(self.endpoint, request())
            third = self.client.pin(self.endpoint, request())
            with self.assertRaises(h.ControlHTTPError):
                third.result()
            with self.assertRaises(h.ControlHTTPError):
                self.client.pin(self.endpoint + "/", request()).result()
            with self.assertRaises(h.ControlHTTPError):
                first.result(timeout=2)
            with self.assertRaises(h.ControlHTTPError):
                second.result(timeout=2)
            self.assertGreaterEqual(submit.call_count, 1)
            self.client.close()
            with self.assertRaises(h.ControlHTTPError):
                self.client.pin(self.endpoint, request()).result()

    def test_cancelled_queue_items_keep_admission_permits(self):
        self.client.close()
        self.client = self.make_client(workers=1, queue_capacity=1)
        started, unblock = threading.Event(), threading.Event()

        def blocked_call(*args):
            started.set()
            unblock.wait(3)
            raise h.ControlHTTPError("test blocked call")

        try:
            with patch.object(self.client, "_call", side_effect=blocked_call):
                first = self.client.pin(self.endpoint, request())
                self.assertTrue(started.wait(1))
                second = self.client.pin(self.endpoint, request())
                self.assertTrue(second.cancel())
                for _ in range(32):
                    with self.assertRaises(h.ControlHTTPError):
                        self.client.pin(self.endpoint, request()).result()
                self.assertEqual(self.client._executor._work_queue.qsize(), 1)
                unblock.set()
                with self.assertRaises(h.ControlHTTPError):
                    first.result(timeout=2)
        finally:
            unblock.set()

    def test_submit_failure_after_enqueue_never_sends_the_rpc(self):
        """A Pin the client rejected locally must never reach the source."""
        ran = threading.Event()

        def run_then_signal(function, args):
            try:
                function(*args)
            finally:
                ran.set()

        def queued_then_failed(function, *args):
            # The item is queued, then thread start fails; a worker runs it later.
            threading.Thread(target=run_then_signal, args=(function, args)).start()
            raise RuntimeError("thread start failed after enqueue")

        class WorkerFirst(Future):
            def set_exception(self, exception):
                ran.wait(2)  # The queued item runs before the caller sees rejection.
                super().set_exception(exception)

        with (
            patch.object(h, "Future", WorkerFirst),
            patch.object(
                self.client._executor, "submit", side_effect=queued_then_failed
            ),
            patch.object(self.client, "_call") as call,
        ):
            future = self.client.pin(self.endpoint, request())
            self.assertTrue(ran.wait(2))
        with self.assertRaises(h.ControlHTTPError):
            future.result(0)
        call.assert_not_called()
        self.assertEqual(self.client._calls, {})

    def test_partial_thread_start_failure_stops_started_client_threads(self):
        """A failed client construction must not leave its threads polling."""
        original_start = threading.Thread.start
        started = []

        def start(thread):
            if thread.name == "l1-http-client-deadlines":
                raise RuntimeError("thread-start-failure")
            original_start(thread)
            started.append(thread)

        try:
            with (
                patch.object(threading.Thread, "start", start),
                self.assertRaisesRegex(RuntimeError, "thread-start-failure"),
            ):
                self.make_client(workers=2)
            self.assertEqual(len(started), 2)
            self.assertFalse(any(thread.is_alive() for thread in started))
        finally:
            # Rescue a regressed constructor so its threads cannot hang the run.
            for thread in started:
                if thread.is_alive():
                    thread._target.__self__._delivery_stop.set()
                    thread.join(2)

    def test_body_limit_below_the_wire_maximum_is_rejected(self):
        """A smaller limit would drop a PREPARED reply whose lease the source keeps."""
        with self.assertRaises(ValueError):
            self.make_client(max_body_bytes=w.MAX_BODY_BYTES - 1)

    def test_fixed_server_workers_and_socket_queue_bounds(self):
        sockets = []
        try:
            for _ in range(4):
                sockets.append(socket.create_connection(self.gateway.address))
                time.sleep(0.02)
            self.assertEqual(len(self.gateway._workers), 2)
            self.assertEqual(len(self.gateway._active), 2)
            self.assertEqual(self.gateway._sockets.qsize(), 2)
            extra = socket.create_connection(self.gateway.address)
            sockets.append(extra)
            extra.settimeout(1)
            self.assertEqual(extra.recv(1), b"")
            self.gateway.close()
            self.assertTrue(
                all(not worker.is_alive() for worker in self.gateway._workers)
            )
        finally:
            for connection in sockets:
                connection.close()

    def test_server_header_and_body_slow_drips_have_absolute_deadline(self):
        self.gateway._socket_timeout = 0.12
        for mode in ("headers", "body"):
            with self.subTest(mode=mode):
                connection = socket.create_connection(self.gateway.address)
                connection.settimeout(0.8)
                stopped, sender_finished = threading.Event(), threading.Event()
                if mode == "headers":
                    connection.sendall(
                        (
                            f"POST {V3.paths[0]} HTTP/1.1\r\nHost: localhost\r\nX-Drip: "
                        ).encode()
                    )
                else:
                    connection.sendall(
                        (
                            f"POST {V3.paths[0]} HTTP/1.1\r\nHost: localhost\r\n"
                            f"Authorization: Bearer {TOKEN}\r\n"
                            f"X-L1-Peer: {h._peer_key(identity('target'))}\r\n"
                            "Content-Type: application/json\r\nContent-Length: 10000\r\n\r\n"
                        ).encode()
                    )

                def drip(
                    connection=connection,
                    stopped=stopped,
                    sender_finished=sender_finished,
                ):
                    try:
                        while not stopped.wait(0.02):
                            connection.sendall(b".")
                    except OSError:
                        pass
                    finally:
                        sender_finished.set()

                sender = threading.Thread(target=drip)
                sender.start()
                try:
                    with patch.object(self.service, "submit") as submit:
                        start = time.monotonic()
                        self.assertEqual(connection.recv(1), b"")
                        self.assertLess(time.monotonic() - start, 0.8)
                    self.assertEqual(submit.call_count, 0)
                finally:
                    stopped.set()
                    connection.close()
                    sender.join(1)
                self.assertTrue(sender_finished.is_set())

    def test_directory_resolution_is_not_repeated_by_rpc_worker(self):
        with patch.object(
            socket, "getaddrinfo", side_effect=AssertionError("RPC DNS lookup")
        ):
            response = self.client.pin(self.endpoint, request()).result(3)
            self.assertEqual(response.status, v.PinStatus.PREPARED)
            ack = self.client.unpin(self.endpoint, V3.make_unpin(response)).result(3)
        self.assertTrue(V3.released(ack, V3.make_unpin(response)))

    def test_redirect_response_is_never_followed(self):
        original = h._Handler._reply

        def redirect(handler, code, body):
            if code == 200:
                handler.close_connection = True
                handler.send_response(302)
                handler.send_header("Location", self.endpoint + "/admin")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
            else:
                original(handler, code, body)

        req = request()
        with (
            patch.object(h._Handler, "_reply", redirect),
            self.assertRaises(h.ControlHTTPError),
        ):
            self.client.pin(self.endpoint, req).result(3)
        # The source prepared exactly one lease that the target never saw.
        self.assertEqual(self.base.g.owner_counts().active_leases, 1)
        self.release(req)

    def test_source_admission_failure_is_503_without_a_lease(self):
        with patch.object(self.service, "submit", side_effect=svc.ServiceBusy("full")):
            self.assertEqual(self.raw(w.encode(request())), 503)
        self.assertEqual(self.base.g.owner_counts().active_leases, 0)

    def test_invalid_interval_rejected_before_submission(self):
        root = json.loads(w.encode(request()))
        result = root["payload"]["result"]
        result["local_hit_tokens"] = result["hit_tokens"]
        with patch.object(self.service, "submit") as submit:
            self.assertEqual(self.raw(json.dumps(root).encode()), 400)
        self.assertEqual(submit.call_count, 0)


class GatewayConfigurationTest(CustomTestCase):
    def setUp(self):
        self.base = ServiceFixture()
        self.base.setUp()

    def tearDown(self):
        self.base.tearDown()

    def test_requires_opt_in_private_explicit_bind_and_credential(self):
        args = {
            "source": identity("source"),
            "peers": (identity("target"),),
            "bind_address": ("127.0.0.1", 0),
            "bearer_token": TOKEN,
            "trusted_private_domain": True,
            "control_profile": V3,
            "slotguard_service": self.base.service,
        }
        h.HTTPControlGateway(**args).close()
        for changes in (
            {"bind_address": ("0.0.0.0", 0)},
            {"bind_address": ("8.8.8.8", 0)},
            {"trusted_private_domain": False},
            {"bearer_token": ""},
            {"socket_timeout": float("inf")},
            {"peers": ()},
            {"slotguard_service": None},
            {"max_body_bytes": w.MAX_BODY_BYTES - 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                h.HTTPControlGateway(**(args | changes))


if __name__ == "__main__":
    unittest.main()
