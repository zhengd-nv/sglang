"""Experimental synchronous NVLink PD backend, TP1 / NHD / early allocation.

PD owns page lifetimes. Destination exports live until sender acknowledges that
all writes drained and imported mappings were closed. An ambiguous failure is
fail-stop: never let the scheduler recycle remotely writable pages.
"""
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver, CommonKVSender
from sglang.srt.disaggregation.common.utils import AuxDataCodec
from sglang.srt.mem_cache.l1_transfer import NvlinkEngine, SGLangPoolAdapter
from sglang.srt.mem_cache.l1_transfer.transport.api import Opcode, RegionDescriptor, TransferState
from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan, bind_regions, plan_transfer

logger = logging.getLogger(__name__)


def fail_stop(reason):
    logger.critical("NVLINK_PD_FAIL_STOP: %s", reason)
    # SystemExit in a receiver thread does not stop the scheduler.
    os._exit(70)


class NvlinkKVManager(CommonKVManager):
    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend=False):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        if (is_mla_backend or args.state_types or self.defer_decode_allocation
            or any(n != 1 for n in (self.attn_tp_size, self.attn_cp_size, self.pp_size, self.system_dp_size))
            or server_args.max_running_requests != 1
            or not server_args.disable_radix_cache):
            raise ValueError("nvlink PD requires MHA TP/CP/PP/DP1, early allocation, max-running-requests=1, disable-radix-cache")
        self.engine = NvlinkEngine(args.gpu_id)
        self.adapter = SGLangPoolAdapter(args.nvlink_pool, self.engine)
        self.inbox = {}
        self.inbox_lock = threading.Lock()
        self.active_room = None
        threading.Thread(target=self._receive, daemon=True).start()
        logger.info("NVLINK_PD_READY layout=%s capacity=%d", self.adapter.layout, self.adapter.capacity)

    def _receive(self):
        try:
            while True:
                frames = self.server_socket.recv_multipart()
                header = json.loads(frames[0])
                key = (header["kind"], header["room"])
                with self.inbox_lock:
                    if key in self.inbox:
                        raise ValueError("duplicate NVLink control message")
                    self.inbox[key] = (header, frames[1:])
        except BaseException as exc:
            fail_stop(f"control receiver: {exc}")

    def message(self, kind, room, pop=False):
        with self.inbox_lock:
            return self.inbox.pop((kind, room), None) if pop else self.inbox.get((kind, room))

    def send_message(self, endpoint, header, frames=()):
        self._send_multipart_locked(endpoint, [json.dumps(header).encode(), *frames])


class NvlinkKVSender(CommonKVSender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._imports = []
        self._remote_bindings = None

    def poll(self):
        if self.conclude_state is not None:
            return self.conclude_state
        if self._imports:
            return KVPoll.Transferring
        if self.kv_mgr.message("target", self.bootstrap_room):
            return KVPoll.WaitingForInput
        return KVPoll.Bootstrapping

    def poll_pp_consensus(self):
        return self.poll()

    def send(self, kv_indices, state_indices=None, num_kv_tokens=None):
        mgr, adapter = self.kv_mgr, self.kv_mgr.adapter
        try:
            if state_indices:
                raise ValueError("state transfer is unsupported")
            indices, index_slice, last, skip = self._prepare_send_indices(kv_indices, state_indices)
            assert not skip
            header, _ = mgr.message("target", self.bootstrap_room)
            # JSON roundtrip canonicalizes dataclass tuples to lists.
            if header["layout"] != json.loads(json.dumps(asdict(adapter.layout))):
                raise ValueError("PD pool layouts differ")
            if len(header["pages"]) != self.num_kv_indices:
                raise ValueError("PD source/destination page counts differ")
            if self._remote_bindings is None:
                handles = {}
                for descriptor in header["regions"]:
                    remote = mgr.engine.import_memory(RegionDescriptor.from_dict(descriptor))
                    self._imports.append(remote)
                    handles[descriptor["region_id"]] = remote
                self._remote_bindings = {
                    name: tuple(RegionSpan(s[0], s[1], handles[s[2]], s[3]) for s in spans)
                    for name, spans in header["bindings"].items()
                }
            plan = plan_transfer(adapter.layout, adapter.layout, indices.tolist(), header["pages"][index_slice],
                                 source_capacity=adapter.capacity, target_capacity=header["capacity"])
            requests = bind_regions(plan, adapter.bindings, self._remote_bindings, Opcode.WRITE)
            torch.cuda.synchronize(mgr.kv_args.gpu_id)
            if requests:
                batch = mgr.engine.submit_transfer(requests)
                status = mgr.engine.wait(batch, timeout=60)
                if status.state != TransferState.SUCCESS:
                    fail_stop(f"transfer incomplete: {status}")
                mgr.engine.release_batch(batch)
            self._record_transfer_indices(indices, state_indices)
            logger.info("NVLINK_PD_WRITE room=%s pages=%d ranges=%d bytes=%d last=%s",
                        self.bootstrap_room, len(indices), len(requests), sum(r.length for r in requests), last)
            if last:
                for remote in reversed(self._imports):
                    mgr.engine.unregister_memory(remote)
                self._imports.clear()
                self._remote_bindings = None
                args = mgr.kv_args
                aux = [AuxDataCodec.serialize_data_from_buffer(ptr + size * self.aux_index, size)
                       for ptr, size in zip(args.aux_data_ptrs, args.aux_item_lens)]
                mgr.send_message(header["endpoint"], {"kind": "done", "room": self.bootstrap_room, "nonce": header["nonce"]}, aux)
                mgr.message("target", self.bootstrap_room, pop=True)
                self.conclude_state = KVPoll.Success
        except BaseException as exc:
            fail_stop(f"sender room={self.bootstrap_room}: {exc}")

    def abort(self):
        if self._imports:
            fail_stop("sender abort with live imports")
        super().abort()

    def clear(self):
        if self._imports:
            fail_stop("sender clear with live imports")
        super().clear()

    def failure_exception(self):
        raise RuntimeError("NVLink PD sender failed")


class NvlinkKVReceiver(CommonKVReceiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exports = []
        self.nonce = uuid.uuid4().hex

    def _register_kv_args(self):
        return True  # Each room carries its own scoped descriptors.

    def send_metadata(self, kv_indices, aux_index=None, state_indices=None, decode_prefix_len=0, **kwargs):
        mgr, adapter = self.kv_mgr, self.kv_mgr.adapter
        try:
            if state_indices or decode_prefix_len or len(self.bootstrap_infos) != 1 or mgr.active_room is not None:
                raise ValueError("nvlink PD supports one outstanding destination, no prefix/state transfer")
            mgr.active_room = self.bootstrap_room
            self.aux_index = aux_index
            torch.cuda.synchronize(mgr.kv_args.gpu_id)
            self.exports = [mgr.engine.export_memory(m, access="write") for m in adapter.memories]
            header = {"kind": "target", "room": self.bootstrap_room, "nonce": self.nonce,
                      "endpoint": f"tcp://{mgr.local_ip}:{mgr.rank_port}",
                      "pages": kv_indices.tolist(), "capacity": adapter.capacity,
                      "layout": asdict(adapter.layout), "regions": [d.to_dict() for d in self.exports],
                      "bindings": {name: [[s.buffer_offset, s.length, s.memory.region_id, s.region_offset] for s in spans]
                                   for name, spans in adapter.bindings.items()}}
            sock, lock = self._connect_to_bootstrap_server(self.bootstrap_infos[0])
            with lock:
                sock.send_multipart([json.dumps(header).encode()])
            self.init_time = time.monotonic()
            mgr.update_status(self.bootstrap_room, KVPoll.Transferring)
        except BaseException as exc:
            fail_stop(f"publish destination: {exc}")

    def poll(self):
        if self.conclude_state is not None:
            return self.conclude_state
        message = self.kv_mgr.message("done", self.bootstrap_room, pop=True)
        if message:
            header, frames = message
            args = self.kv_mgr.kv_args
            if header["nonce"] != self.nonce or [len(f) for f in frames] != args.aux_item_lens:
                fail_stop("invalid completion identity or auxiliary data lengths")
            for i, frame in enumerate(frames):
                AuxDataCodec.deserialize_data_to_buffer(args, i, self.aux_index, frame)
            for descriptor in self.exports:
                self.kv_mgr.engine.release_export(descriptor.lease_id)
            self.exports.clear()
            self.kv_mgr.active_room = None
            self.conclude_state = KVPoll.Success
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
            logger.info("NVLINK_PD_COMPLETE room=%s exports=0 imports=0", self.bootstrap_room)
        elif self.exports and time.monotonic() - self.init_time > 120:
            fail_stop("destination completion timeout; remote quiescence unknown")
        return self.conclude_state or self.kv_mgr.check_status(self.bootstrap_room)

    def abort(self):
        if self.exports:
            fail_stop("abort with published destination")
        self.conclude_state = KVPoll.Failed

    def clear(self):
        if self.exports:
            fail_stop("clear with published destination")
        super().clear()

    def failure_exception(self):
        raise RuntimeError("NVLink PD receiver failed")
