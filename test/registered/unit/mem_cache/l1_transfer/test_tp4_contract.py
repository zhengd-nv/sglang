"""TP4 policy contracts, independent of CUDA and request arrival timing."""

import importlib
import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_slotguard_backend as backend_fixture
from test_protocol import PACKAGE, register_cpu_ci

from sglang.test.test_utils import CustomTestCase

g = importlib.import_module(f"{PACKAGE}.tp_group")
register_cpu_ci(est_time=45, suite="base-a-test-cpu")


class TP4ContractTest(CustomTestCase):
    def test_enqueue_consensus_precedes_every_pin_rpc(self):
        self.assertIsNone(
            g.worker_decision([("ENQUEUED", True)] * 3 + [("WAIT", False)])
        )
        self.assertEqual(g.worker_decision([("ENQUEUED", True)] * 4), "PIN")
        self.assertEqual(
            g.worker_decision([("ENQUEUED", True)] * 3 + [("ENQUEUED", False)]), "DROP"
        )
        self.assertEqual(
            g.worker_decision([("ENQUEUED", True)] * 4, cancelled=True), "DROP"
        )

    def test_different_lookup_lengths_choose_common_source_and_minimum(self):
        hints = [("group-a", n) for n in (1024, 960, 896, 1024)]
        self.assertEqual(g.common_hint(hints), ("group-a", 896))
        self.assertIsNone(g.common_hint(hints[:3] + [("group-b", 1024)]))
        self.assertIsNone(g.common_hint(hints[:3] + [None]))

    def test_pin_barrier_never_releases_dma_early_or_on_one_rejection(self):
        self.assertIsNone(g.worker_decision([("PIN", True)] * 3 + [("WAIT", False)]))
        self.assertEqual(g.worker_decision([("PIN", True)] * 4), "COPY")
        self.assertEqual(
            g.worker_decision([("PIN", True)] * 3 + [("PIN", False)]), "ABORT"
        )

    def test_unpin_requires_all_drained_even_after_copy_failure(self):
        self.assertIsNone(
            g.worker_decision([("DRAINED", True)] * 3 + [("COPY", False)])
        )
        self.assertEqual(
            g.worker_decision([("DRAINED", True)] * 3 + [("DRAINED", False)]), "RELEASE"
        )
        self.assertEqual(
            g.worker_decision([("DRAINED", True)] * 3 + [("UNKNOWN", False)]),
            "QUARANTINE",
        )

    def test_worker_gate_has_no_collective_and_duplicate_permit_is_idempotent(self):
        gate = g.WorkerRendezvous()
        got = []
        worker = threading.Thread(target=lambda: got.append(gate.arrive("PIN", True)))
        worker.start()
        self.assertTrue(gate.waiting.wait(1))
        self.assertEqual(gate.snapshot(), ("PIN", True))
        self.assertEqual(got, [])
        gate.permit("COPY")
        gate.permit("COPY")
        worker.join(1)
        self.assertEqual(got, ["COPY"])
        got.clear()
        worker = threading.Thread(
            target=lambda: got.append(gate.arrive("DRAINED", True))
        )
        worker.start()
        self.assertTrue(gate.waiting.wait(1))
        gate.permit("COPY")  # delayed decision based on the preceding snapshot
        self.assertEqual(got, [])
        gate.permit("RELEASE")
        worker.join(1)
        self.assertEqual(got, ["RELEASE"])

    def test_collective_is_scheduler_thread_only(self):
        collective = g.SchedulerCollective(lambda value: [value] * 4, 0)
        self.assertEqual(collective.gather("idle"), ["idle"] * 4)
        errors = []
        worker = threading.Thread(target=lambda: self._wrong_thread(collective, errors))
        worker.start()
        worker.join()
        self.assertEqual(len(errors), 1)

    @staticmethod
    def _wrong_thread(collective, errors):
        try:
            collective.gather("worker")
        except RuntimeError as exc:
            errors.append(exc)


class GroupDirectoryTest(CustomTestCase):
    def test_generated_config_uses_runtime_rank_and_identical_full_manifest_digest(
        self,
    ):
        manifest = importlib.import_module(f"{PACKAGE}.tp_manifest")
        config = importlib.import_module(f"{PACKAGE}.config")
        groups = [
            manifest.make_group(
                name,
                name + "-boot",
                "127.0.0.1",
                name + "-tp0",
                domain="fabric",
                instance="isolated",
                model="/exact/model",
                base_port=12000 + index * 10,
            )
            for index, name in enumerate(("local", "peer"))
        ]
        data = {
            "local_group": groups[0],
            "peer_groups": [groups[1]],
            "bind_host": "127.0.0.1",
            "kvcm_endpoint": "127.0.0.1:50051",
            "trusted_private_domain": True,
            "prefetch_pipeline_depth": 2,
            "control_profile": "slotguard-hash-v3",
            "peer_control_profiles": {"peer": "slotguard-hash-v3"},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "group.json"
            path.write_text(json.dumps(data))
            with patch.dict(
                os.environ,
                {
                    "SGLANG_NVLINK_CONFIG": str(path),
                    "SGLANG_NVLINK_CONTROL_TOKEN": "test-only-credential",
                },
            ):
                ranks = [config.ServingConfig.from_env(rank) for rank in range(4)]
        self.assertEqual([r.directory.target.tp_rank for r in ranks], [0, 1, 2, 3])
        self.assertEqual(
            [r.bind_address[1] for r in ranks], [12000, 12001, 12002, 12003]
        )
        self.assertEqual(len({r.group_manifest_digest for r in ranks}), 1)
        self.assertEqual([r.prefetch_pipeline_depth for r in ranks], [2] * 4)
        self.assertTrue(all("test-only-credential" not in repr(r) for r in ranks))

    def test_generated_manifest_layout_matches_actual_adapter_geometry(self):
        from test_glm_fabric_adapter import glm_pool
        from test_sglang_adapter import SGLangPoolAdapter

        manifest = importlib.import_module(f"{PACKAGE}.tp_manifest")
        for rank in range(4):
            _, layout, _ = SGLangPoolAdapter._prepare_glm(
                glm_pool(256), 0, rank, 4, f"tp{rank}"
            )
            self.assertEqual(manifest.glm_layout(rank, f"tp{rank}"), layout)

    def test_exact_rank_map_and_boot_binding(self):
        from test_protocol import identity

        directory = importlib.import_module(f"{PACKAGE}.directory")

        def group(name):
            return directory.EngineTPGroup(
                name + "-reporter",
                tuple(
                    directory.EngineDirectoryEntry(
                        name + "-reporter",
                        replace(
                            identity(name),
                            tp_size=4,
                            tp_rank=rank,
                            shard_id=f"shard{rank}",
                            layout_digest=f"layout{rank}",
                        ),
                        f"http://{name}:{12000 + rank}",
                    )
                    for rank in range(4)
                ),
            )

        target, source = group("target"), group("source")
        for rank in range(4):
            resolved = target.directory((source,), rank)
            self.assertEqual(resolved.entries[0].identity, source.ranks[rank].identity)
            self.assertEqual(resolved.entries[0].reporter_id, "source-reporter")
        with self.assertRaises(ValueError):
            replace(source, ranks=source.ranks[:-1])
        with self.assertRaises(ValueError):
            replace(
                source,
                ranks=source.ranks[:3]
                + (
                    replace(
                        source.ranks[3],
                        identity=replace(
                            source.ranks[3].identity, incarnation="other-boot"
                        ),
                    ),
                ),
            )
        with self.assertRaises(ValueError):
            target.directory(
                (
                    replace(
                        source,
                        ranks=source.ranks[:3]
                        + (
                            replace(
                                source.ranks[3],
                                identity=replace(
                                    source.ranks[3].identity, shard_id="other-shard"
                                ),
                            ),
                        ),
                    ),
                ),
                0,
            )


class GroupCoordinatorContractTest(CustomTestCase):
    def patch_attribute(self, target, name, value):
        self.enterContext(patch.object(target, name, value))

    def test_actual_sglang_group_coordinator_unwraps_cpu_process_group(self):
        import torch

        from sglang.srt.distributed.parallel_state import GroupCoordinator
        from sglang.srt.mem_cache.l1_transfer.tp_group import SchedulerCollective

        cpu, cache = object(), object()
        # Use the real runtime wrapper class without allocating model communicators.
        wrapper = object.__new__(GroupCoordinator)
        wrapper.cpu_group, wrapper.ranks = cpu, [4, 5, 6, 7]
        wrapper.world_size, wrapper.rank_in_group = 4, 1
        seen = []

        def members(group):
            assert group in (cpu, cache)
            seen.append(group)
            return [4, 5, 6, 7]

        self.patch_attribute(torch.distributed, "get_process_group_ranks", members)
        self.patch_attribute(torch.distributed, "get_world_size", lambda group: 4)
        self.patch_attribute(torch.distributed, "get_rank", lambda group: 1)
        self.patch_attribute(torch.distributed, "get_backend", lambda group: "gloo")
        collective = SchedulerCollective.from_runtime_tp_group(wrapper, cache, 1)
        assert collective.rank == 1
        assert seen == [cpu, cache]


a = importlib.import_module(f"{PACKAGE}.transport_role_arbiter")


s = importlib.import_module(f"{PACKAGE}.slot_guard")


class ReservedBackendTest(CustomTestCase):
    def setUp(self):
        self.cohort = backend_fixture.SlotGuardBackendTest()
        self.cohort.setUp()
        self.ranks = self.cohort.ranks
        self.identities = []
        for rank in self.ranks:
            arbiter = a.TransportRoleArbiter()
            identity = s.LeaseIdentity(
                rank.result.source.incarnation,
                rank.result.query.target.incarnation,
                rank.result.transfer_id,
                f"preallocated-{rank.result.source.tp_rank}",
            )
            self.assertTrue(arbiter.acquire(a.TransportRole.INBOUND, identity))
            rank.backend.transport_arbiter = arbiter
            self.identities.append(identity)

    def tearDown(self):
        for rank in self.ranks:
            if rank.result.transfer_id not in rank.backend._records:
                rank.backend.close()
                rank.adapter.close()
                rank.engine.close()
                rank.base.tearDown()
            else:
                rank.close()

    def start(self):
        for rank, identity in zip(self.ranks, self.identities):
            rank.backend.prefetch(rank.result, rank.slots, identity=identity)

    def test_missing_or_conflicting_binding_rejects_before_worker_and_dma(self):
        rank = self.ranks[0]
        with self.assertRaisesRegex(ValueError, "reservation required"):
            rank.backend.prefetch(rank.result, rank.slots)
        wrong = replace(self.identities[0], nonce="conflicting")
        with self.assertRaisesRegex(ValueError, "reservation required"):
            rank.backend.prefetch(rank.result, rank.slots, identity=wrong)
        self.assertIsNone(rank.backend._live)
        self.assertEqual(rank.driver.submits, 0)

    def test_exact_identity_and_role_survive_copy_unmap_and_delayed_unpin_ack(self):
        self.ranks[-1].client.hold_ack = True
        self.start()
        self.cohort.until(lambda: self.ranks[-1].client.held_ack is not None)
        for rank, identity in zip(self.ranks, self.identities):
            self.assertIs(rank.client.calls[0][1].identity, identity)
            self.assertTrue(
                rank.backend.transport_arbiter.owns(a.TransportRole.INBOUND, identity)
            )
            record = rank.backend._records[rank.result.transfer_id]
            self.assertFalse(record.imports)
            self.assertIsNone(record.batch)
            self.assertNotEqual(rank.status(), backend_fixture.p.PrefetchStatus.READY)
        future, ack = self.ranks[-1].client.held_ack
        future.set_result(ack)
        self.cohort.until(
            lambda: all(
                r.status() is backend_fixture.p.PrefetchStatus.READY for r in self.ranks
            )
        )
        self.assertEqual(self.cohort.decisions, ["PIN", "COPY", "RELEASE"])
        for rank, identity in zip(self.ranks, self.identities):
            # The backend does not invent target private/receipt retirement;
            # admission keeps this reservation until its all-rank safe handoff.
            self.assertTrue(
                rank.backend.transport_arbiter.owns(a.TransportRole.INBOUND, identity)
            )

    def test_unknown_pin_reply_retains_every_target_role_without_dma(self):
        self.ranks[-1].client.hold_pin = True
        self.start()
        self.cohort.until(
            lambda: any(
                r.status() is backend_fixture.p.PrefetchStatus.QUARANTINED
                for r in self.ranks
            )
        )
        for rank, identity in zip(self.ranks, self.identities):
            self.assertTrue(
                rank.backend.transport_arbiter.owns(a.TransportRole.INBOUND, identity)
            )
            self.assertIsNotNone(rank.backend._live)
            self.assertEqual(rank.driver.submits, 0)
            self.assertFalse(any(kind == "unpin" for kind, _ in rank.client.calls))


if __name__ == "__main__":
    unittest.main()
