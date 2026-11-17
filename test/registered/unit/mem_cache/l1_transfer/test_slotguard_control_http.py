"""Real loopback v2 RPC over the independent service and fake CPU engine."""

import http.client
import importlib
import json
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_slotguard_service as fixture
from test_protocol import PACKAGE, identity, register_cpu_ci

from sglang.test.test_utils import CustomTestCase

h = importlib.import_module(f"{PACKAGE}.control_http")
d = importlib.import_module(f"{PACKAGE}.directory")
profiles = importlib.import_module(f"{PACKAGE}.control_profile")
v, w = fixture.v, fixture.w
TOKEN = "slotguard-loopback-test-credential"
V3 = profiles.ControlProfile.SLOTGUARD_HASH_V3
register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class GatewayPartialStartupTest(CustomTestCase):
    def test_partial_thread_start_failure_closes_actual_gateway_without_hanging(self):
        for failed_index in (0, 1, 2):
            with self.subTest(failed_index=failed_index):
                base = fixture.ServiceTest()
                base.setUp()
                gateway = h.HTTPControlGateway(
                    identity("source"),
                    (identity(),),
                    bind_address=("127.0.0.1", 0),
                    bearer_token=TOKEN,
                    trusted_private_domain=True,
                    control_profile=V3,
                    slotguard_service=base.service,
                    workers=2,
                )
                failed = gateway._threads()[failed_index]
                completed, errors = threading.Event(), []

                def close(gateway=gateway, errors=errors, completed=completed):
                    try:
                        gateway.close()
                    except BaseException as exc:  # noqa: BLE001 -- collect thread failure
                        errors.append(type(exc).__name__)
                    finally:
                        completed.set()

                try:
                    with (
                        patch.object(
                            failed,
                            "start",
                            side_effect=RuntimeError("thread-start-failure"),
                        ),
                        self.assertRaisesRegex(RuntimeError, "thread-start-failure"),
                    ):
                        gateway.start()
                    lifecycle = threading.Thread(target=close, daemon=True)
                    lifecycle.start()
                    bounded = completed.wait(0.5)
                    if not bounded:
                        # Rescue the old regression without leaving a hanging
                        # test process. The assertion still records RED.
                        gateway._server._BaseServer__is_shut_down.set()
                    lifecycle.join(1)
                    self.assertTrue(
                        bounded, "close waited for a listener that never started"
                    )
                    self.assertEqual(errors, [])
                    self.assertFalse(
                        any(thread.is_alive() for thread in gateway._threads())
                    )
                finally:
                    base.tearDown()


class SlotGuardHTTPTest(CustomTestCase):
    def setUp(self):
        self.base = fixture.ServiceTest()
        self.base.setUp()
        self.gateway = h.HTTPControlGateway(
            identity("source"),
            (identity(),),
            bind_address=("127.0.0.1", 0),
            bearer_token=TOKEN,
            trusted_private_domain=True,
            control_profile=V3,
            slotguard_service=self.base.service,
            workers=2,
            response_timeout=0.5,
        ).start()
        host, port = self.gateway.address
        self.endpoint = f"http://{host}:{port}"
        directory = d.EngineDirectory(
            identity(),
            (d.EngineDirectoryEntry("source", identity("source"), self.endpoint),),
        )
        self.client = h.HTTPPrefixControlClient(
            directory,
            bearer_token=TOKEN,
            trusted_private_domain=True,
            control_profile=V3,
            workers=2,
            read_timeout=1,
        )

    def tearDown(self):
        self.client.close()
        self.gateway.close()
        self.base.tearDown()

    def test_success_exact_replay_and_fence(self):
        req = self.base.pin()
        response = self.client.pin(self.endpoint, req).result(2)
        self.assertIs(response.request, req)
        self.assertEqual(response.status, v.PinStatus.PREPARED)
        self.assertEqual(self.client.pin(self.endpoint, req).result(2), response)
        self.base.service.fence_pins(1)
        self.assertEqual(
            self.client.pin(self.endpoint, self.base.pin(1)).result(2).status,
            v.PinStatus.BUSY,
        )
        proof = self.base.unpin(response)
        self.assertEqual(
            self.client.unpin(self.endpoint, proof).result(2).status,
            v.UnpinStatus.RELEASED,
        )
        self.assertEqual(
            self.client.unpin(self.endpoint, proof).result(2).status,
            v.UnpinStatus.ALREADY_RELEASED,
        )

    def test_maintenance_off_pin_and_exact_replay_never_consult_guard(self):
        response = self.client.pin(self.endpoint, self.base.pin()).result(2)
        proof = self.base.unpin(response)
        self.assertEqual(
            self.client.unpin(self.endpoint, proof).result(2).status,
            v.UnpinStatus.RELEASED,
        )
        self.base.service.pause_for_maintenance(1)
        with (
            patch.object(
                self.base.g,
                "owner_counts",
                side_effect=AssertionError("inactive guard"),
            ),
            patch.object(
                self.base.g, "try_pin", side_effect=AssertionError("inactive Pin")
            ),
            patch.object(
                self.base.g, "release", side_effect=AssertionError("old Unpin release")
            ),
        ):
            self.assertEqual(
                self.client.pin(self.endpoint, self.base.pin(1)).result(2).status,
                v.PinStatus.BUSY,
            )
            self.assertEqual(
                self.client.unpin(self.endpoint, proof).result(2).status,
                v.UnpinStatus.ALREADY_RELEASED,
            )
            self.assertEqual(self.base.service.external_snapshot().owners, 0)

    def test_ack_lost_after_release_retries_same_compact_proof(self):
        response = self.client.pin(self.endpoint, self.base.pin()).result(2)
        proof = self.base.unpin(response)
        reply = h._Handler._reply
        lost = threading.Event()

        def drop(handler, status, body):
            if b'"type":"UnpinResp"' in body and not lost.is_set():
                lost.set()
                h._interrupt_socket(handler.connection)
                return
            return reply(handler, status, body)

        with patch.object(h._Handler, "_reply", drop):
            with self.assertRaises(h.ControlHTTPError):
                self.client.unpin(self.endpoint, proof).result(2)
            self.assertTrue(lost.is_set())
            self.assertEqual(
                self.client.unpin(self.endpoint, proof).result(2).status,
                v.UnpinStatus.ALREADY_RELEASED,
            )

    def test_timeout_disconnect_and_network_close_retain_source_lease(self):
        entered, proceed = threading.Event(), threading.Event()
        export = self.base.engine.export_memory

        def blocked(memory):
            entered.set()
            proceed.wait(2)
            return export(memory)

        self.base.engine.export_memory = blocked
        self.client._rpc_timeout = 0.05
        req = self.base.pin()
        try:
            pending = self.client.pin(self.endpoint, req)
            self.assertTrue(entered.wait(1))
            with self.assertRaises(h.ControlHTTPError):
                pending.result(1)
            self.assertEqual(self.base.g.owner_counts().active_leases, 1)
            self.gateway.close()
        finally:
            proceed.set()
        deadline = time.monotonic() + 2
        while (
            self.base.service.query(req.identity).state == "PREPARING"
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        self.assertEqual(self.base.service.query(req.identity).state, "PREPARED")
        self.assertEqual(self.base.g.owner_counts().active_leases, 1)
        lease = self.base.service.query(req.identity).lease_id
        # Gateway close leaves the independent owner usable for reconciliation.
        self.assertEqual(
            self.base.service.submit(
                v.UnpinReq(
                    req.reference,
                    lease,
                    True,
                    True,
                    req.protocol_version,
                    req.capability,
                )
            )
            .result(2)
            .status,
            v.UnpinStatus.RELEASED,
        )

    def test_wrong_compact_target_is_rejected_and_legacy_path_never_falls_back(self):
        response = self.client.pin(self.endpoint, self.base.pin()).result(2)
        proof = self.base.unpin(response)
        ref = proof.reference
        target = replace(ref.target, engine_id="other-target")
        changed = replace(proof, reference=replace(ref, target=target))
        with self.assertRaises(h.ControlHTTPError):
            self.client.unpin(self.endpoint, changed).result(2)
        connection = http.client.HTTPConnection(*self.gateway.address, timeout=1)
        try:
            connection.request(
                "POST",
                "/v1/pin",
                body=b"{}",
                headers={
                    "Authorization": "Bearer " + TOKEN,
                    "X-L1-Peer": self.client._peer,
                    "Content-Type": "application/json",
                },
            )
            self.assertEqual(connection.getresponse().status, 404)
        finally:
            connection.close()
        self.assertEqual(self.base.g.owner_counts().active_leases, 1)
        self.client.unpin(self.endpoint, proof).result(2)

    def test_partial_export_failure_proves_cleanup_before_busy(self):
        export = self.base.engine.export_memory
        calls = []

        def partial(memory):
            if calls:
                raise RuntimeError("injected export failure")
            calls.append(memory)
            return export(memory)

        self.base.engine.export_memory = partial
        self.assertEqual(
            self.client.pin(self.endpoint, self.base.pin()).result(2).status,
            v.PinStatus.BUSY,
        )
        self.assertEqual(self.base.g.owner_counts().active_leases, 0)
        self.assertFalse(self.base.engine._exports)

    def test_wrong_reply_binding_is_unknown_and_preserves_original_lease(self):
        req = self.base.pin()
        original = h._encode

        def forged(profile, message, *, max_body_bytes):
            body = original(profile, message, max_body_bytes=max_body_bytes)
            if type(message) is v.PinPreparedResp:
                value = json.loads(body)
                value["payload"]["reference"]["content_digest"] = "0" * 64
                return json.dumps(value).encode()
            return body

        with patch.object(h, "_encode", forged), self.assertRaises(h.ControlHTTPError):
            self.client.pin(self.endpoint, req).result(2)
        state = self.base.service.query(req.identity)
        self.assertEqual(state.state, "PREPARED")
        self.assertEqual(self.base.g.owner_counts().active_leases, 1)
        self.client.unpin(
            self.endpoint,
            v.UnpinReq(
                req.reference,
                state.lease_id,
                True,
                True,
                req.protocol_version,
                req.capability,
            ),
        ).result(2)


if __name__ == "__main__":
    unittest.main()
