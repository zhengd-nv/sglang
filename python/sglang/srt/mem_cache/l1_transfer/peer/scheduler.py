"""Explicit fixed-peer restore operations on an idle TP1 scheduler thread."""
import json
import logging
import os
import socket
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import torch

from .control import ModelIdentity, SocketRPC, serve_once, RemoteRejected
from .nonblocking import AsyncRPC, SourceService
from .peer import PeerReceiver, SourceEndpoint
from .restore import RestoreState
from ..transport import NvlinkEngine
from ..transport.sglang_adapter import SGLangPoolAdapter

logger = logging.getLogger(__name__)


class SchedulerPeer:
    def __init__(self, scheduler, identity):
        args = scheduler.server_args
        if (args.tp_size != 1 or args.pp_size != 1 or args.dp_size != 1
                or args.disaggregation_mode != 'null' or args.speculative_algorithm is not None
                or args.enable_lora or args.max_running_requests != 1):
            raise ValueError('Peer restore requires ordinary TP/PP/DP1, no speculation/LoRA, max_running_requests=1')
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        _, pool = self.cache.source_leases.validate_cache()
        self.engine = NvlinkEngine(scheduler.tp_worker.model_runner.gpu_id)
        self.adapter = SGLangPoolAdapter(pool, self.engine)
        self.identity = ModelIdentity(**identity)
        self.identity.validate()
        self.source = SourceEndpoint(self.cache, self.adapter, self.identity, torch.cuda.synchronize)
        self.receiver = None
        self.listener = None
        self.service = None
        self.request = None
        self.phase = None
        self.completed = None
        self.cancel_reason = None
        self.deadline = None
        self.required_capacity = 0
        self.poll_interval = float(os.environ.get('SGLANG_L1_POLL_INTERVAL', '0.001'))
        if not math.isfinite(self.poll_interval) or not 0 <= self.poll_interval <= 0.1:
            raise ValueError('SGLANG_L1_POLL_INTERVAL must be finite and in [0, 0.1] seconds')
        # One worker per Engine; connections and receipts remain per transaction.
        self.rpc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='l1-rpc')
        from .timing import install
        install(self)

    def stats(self):
        allocator = self.cache.token_to_kv_pool_allocator
        return dict(wait_state=getattr(self, "phase", None), source_leases=len(self.source.active), target_active=self.receiver is not None,
                    exports=len(self.engine._exports), imports=sum(m.public.remote for m in self.engine._memories.values()),
                    batches=len(self.engine._batches), available_tokens=allocator.available_size(),
                    pinned_pages=sum(getattr(allocator, '_l1_pinned_pages', {}).values()))

    def listen(self, host='127.0.0.1', port=0):
        if self.listener is not None:
            raise RuntimeError('listener already open')
        self.listener = socket.socket()
        self.listener.bind((host, port))
        self.listener.listen(1)
        self.listener.settimeout(120)
        return dict(host=host, port=self.listener.getsockname()[1])

    def serve(self):
        if self.listener is None:
            raise RuntimeError('call listen first')
        # This RPC blocks the source scheduler: producers and eviction cannot
        # race the exported source pool. Only one acquire/release exchange.
        try:
            conn, _ = self.listener.accept()
            with conn:
                conn.settimeout(120)
                serve_once(conn, self.source)
                if self.source.active:
                    serve_once(conn, self.source)
                if self.source.active:
                    raise RuntimeError('source release not confirmed')
            return self.stats()
        except BaseException:
            if self.source.active:
                logger.exception('L1_PEER source ownership uncertain; stopping scheduler')
                os._exit(70)
            raise
        finally:
            self.listener.close()
            self.listener = None

    def restore(self, tokens, host=None, port=None):
        if self.receiver is not None:
            raise RuntimeError('finish previous consumer before restoring')
        class LazyRPC:
            connection = None
            rpc = None
            def call(inner, operation, **kwargs):
                if inner.rpc is None:
                    inner.connection = socket.create_connection((host, port), timeout=120)
                    inner.rpc = SocketRPC(inner.connection)
                return inner.rpc.call(operation, **kwargs)
        rpc = LazyRPC()
        receiver = PeerReceiver(self.cache, self.adapter, self.identity, rpc)
        self.receiver = receiver
        try:
            session = receiver.prepare(tokens)
            reservation, tx = session['reservation'], session['transaction']
            copied = 0
            if tx is not None:
                tx.start()
                if tx.wait(60) != RestoreState.PUBLISHED:
                    raise RuntimeError('restore did not publish')
                copied = sum(r.length for r in tx.requests)
            result = dict(restored_tokens=len(reservation.indices), local_tokens=reservation.prefix_len,
                          transferred_bytes=copied, **self.stats())
            logger.info('L1_PEER_RESTORED %s', result)
            return result
        except BaseException:
            if receiver.pending is not None:
                logger.exception('L1_PEER restore ownership uncertain; stopping scheduler')
                os._exit(70)
            self.receiver = None  # explicit safe rejection before ownership escaped
            raise
        finally:
            if rpc.connection is not None:
                rpc.connection.close()

    def finish(self):
        if self.receiver is not None:
            # A terminal response can precede deferred device work completing.
            # Keep consumer pins until all preceding local GPU work is done.
            torch.cuda.synchronize(self.engine.device)
            self.receiver.finish_consumer()
            self.receiver = None
        return self.stats()


    @property
    def blocks_compute(self):
        return bool(self.source.active) or self.phase in ('WAIT_ACQUIRE', 'WAIT_TRANSFER', 'WAIT_RELEASE')

    def allows_request(self, req):
        # Capacity waiting owns no pages. Once reserved, no unrelated request
        # may consume the full-request headroom before our consumer finishes.
        return self.phase in (None, 'WAIT_CAPACITY') or req is self.request

    def start_service(self, host='127.0.0.1', port=0):
        if not self.scheduler.server_args.disable_overlap_schedule:
            raise ValueError('nonblocking peer requires disable_overlap_schedule')
        address = self.listen(host, port)
        self.service = SourceService(self.listener)
        return address

    def start_request(self, req, options):
        from .request import restore_payload
        if not self.scheduler.server_args.disable_overlap_schedule:
            raise ValueError('nonblocking peer requires disable_overlap_schedule')
        if self.request is not None or self.receiver is not None or self.source.active:
            raise ValueError('nonblocking peer permits one pending shared request')
        payload = restore_payload(dict(input_ids=list(req.origin_input_ids)), options,
                                  self.scheduler.server_args.page_size)
        output = req.sampling_params.max_new_tokens
        if type(output) is not int or output < 0:
            raise ValueError('serialized sharing requires a bounded max_new_tokens')
        page = self.scheduler.server_args.page_size
        required = (len(req.origin_input_ids) + output + page - 1) // page * page
        self.request = req
        self.payload = payload
        self.completed = None
        self.cancel_reason = None
        self.restore_id = uuid.uuid4().hex
        self.required_capacity = required
        self.deadline = time.monotonic() + options.get('wait_timeout_ms', 30000) / 1000
        self.phase = 'WAIT_CAPACITY'
        self.poll_counts = dict(WAIT_CAPACITY=0, WAIT_ACQUIRE=0, WAIT_TRANSFER=0, WAIT_RELEASE=0)
        if required > self.cache.token_to_kv_pool_allocator.size:
            self._fail_request('CAPACITY_EXCEEDED')

    def _try_capacity(self):
        if self.scheduler._engine_paused or not self.scheduler.is_fully_idle(ignore_l1=True):
            return
        payload = self.payload
        self.rpc = AsyncRPC(payload['host'], payload['port'], executor=self.rpc_executor)
        self.receiver = PeerReceiver(self.cache, self.adapter, self.identity, self.rpc)
        self.steps = self.receiver.prepare_steps(payload['tokens'], required_capacity=self.required_capacity)
        try:
            operation, data = next(self.steps)
        except MemoryError:
            # TargetReservation rolls back its temporary lock before raising.
            # A partial preparation is never treated as a capacity miss.
            if self.receiver.pending is not None:
                raise
            self.rpc.close()
            self.receiver = None
            return
        except StopIteration as done:
            self._prepared(done.value)
        else:
            data['id'] = self.restore_id
            self.phase = 'WAIT_ACQUIRE'
            self.acquire_future = self.rpc.submit(operation, data)

    def _prepared(self, session):
        self.session = session
        tx = session['transaction']
        if tx is None:
            if self.cancel_reason:
                self._fail_request(self.cancel_reason)
            else:
                self._ready()
        else:
            if self.cancel_reason:
                tx.cancel()
                self.phase = 'WAIT_RELEASE'
            else:
                tx.start()
                self.phase = 'WAIT_TRANSFER'

    def _ready(self):
        reservation, tx = self.session['reservation'], self.session['transaction']
        self.restored = dict(restored_tokens=len(reservation.indices), local_tokens=reservation.prefix_len,
                             transferred_bytes=sum(r.length for r in tx.requests) if tx else 0,
                             wait_polls=dict(self.poll_counts), **self.stats())
        self.restored.update(restore_id=self.restore_id, state='READY', reason=None)
        self.rpc.close()
        self.phase = 'READY'
        self.deadline = None
        self.request._l1_consumer_release = self.finish_request
        self.scheduler._add_request_to_queue(self.request)

    def cancel_request(self, reason='CANCELLED'):
        """No ownership is released merely because cancellation was requested."""
        if self.request is None:
            return
        if self.cancel_reason is None:
            self.cancel_reason = reason
        if self.phase == 'WAIT_CAPACITY':
            self._fail_request(self.cancel_reason)
        elif self.phase in ('WAIT_TRANSFER', 'WAIT_RELEASE'):
            self.session['transaction'].cancel()
        # ACQUIRE: retain the future and reservation, then clean a valid late
        # offer without DMA. Unknown reply still reaches pump's fail-stop.
        # READY/RUNNING: ordinary abort owns queue removal and GPU-safe cleanup.

    def _fail_request(self, reason):
        """Report a terminal failure only after all acquired owners are safe."""
        req = self.request
        if self.receiver is not None:
            self.finish()
            self.rpc.close()
        result = dict(restore_id=self.restore_id,
                      state='CANCELLED' if reason == 'CANCELLED' else 'FAILED',
                      reason=reason, consumer_released=True, transferred_bytes=None,
                      wait_polls=dict(self.poll_counts))
        self.completed = (req.rid, result)
        self.request = None
        self.phase = None
        self.required_capacity = 0
        self.deadline = None
        self.scheduler.finish_l1_waiting_request(req, reason)

    def finish_request(self):
        # Called before ordinary request KV/page cleanup on the same owner thread.
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        aborted = isinstance(self.request.finished_reason, FINISH_ABORT) or isinstance(self.request.to_finish, FINISH_ABORT)
        if not self.request.finished() and not aborted:
            raise RuntimeError('L1 consumer retraction is unsupported; stop Engine')
        rid = self.request.rid
        released = self.finish()
        self.completed = (rid, dict(self.restored, consumer_released=(
            not released['target_active'] and released['pinned_pages'] == 0),
            state=('CANCELLED' if self.cancel_reason == 'CANCELLED' else
                   'FAILED' if aborted or self.cancel_reason else 'FINISHED'),
            reason=self.cancel_reason or ('GENERATION_ABORTED' if aborted else None)))
        self.request._l1_consumer_release = None
        self.request = None
        self.phase = None
        self.required_capacity = 0
        self.deadline = None

    def request_result(self, rid):
        if self.completed is None or self.completed[0] != rid:
            raise RuntimeError('no completed L1 consumer for request')
        return self.completed[1]

    def pump(self):
        try:
            self._pump()
        except BaseException:
            logger.exception('L1_PEER ownership uncertain; stopping scheduler')
            os._exit(70)

    def _pump(self):
        if self.service is not None:
            # Bound each turn; network threads never touch cache or CUDA.
            for _ in range(4):
                try:
                    kind, data, future = self.service.events.get_nowait()
                except Empty:
                    break
                if kind != 'request':
                    if self.source.active:
                        raise RuntimeError('source connection ended with outstanding lease')
                    continue
                try:
                    if data.get('operation') == 'acquire' and (
                            self.source.active or self.receiver is not None or not self.scheduler.is_fully_idle()):
                        raise RemoteRejected('source Engine is busy')
                    answer = self.source.dispatch(data)
                except RemoteRejected as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(answer)
        if self.request is not None and self.deadline is not None and time.monotonic() >= self.deadline:
            self.cancel_request('WAIT_TIMEOUT')
        if self.phase == 'WAIT_CAPACITY':
            self._try_capacity()
        if self.request is not None and self.phase in self.poll_counts:
            self.poll_counts[self.phase] += 1
        if self.phase == 'WAIT_ACQUIRE' and self.acquire_future.done():
            try:
                answer = self.acquire_future.result()
            except RemoteRejected as exc:
                try:
                    self.steps.throw(exc)
                except RemoteRejected:
                    pass
                self._fail_request(self.cancel_reason or 'SOURCE_REJECTED')
                return
            try:
                self.steps.send(answer)
            except StopIteration as done:
                self._prepared(done.value)
        if self.phase in ('WAIT_TRANSFER', 'WAIT_RELEASE'):
            state = self.session['transaction'].poll()
            if state == RestoreState.PUBLISHED:
                if self.cancel_reason:
                    self._fail_request(self.cancel_reason)
                else:
                    self._ready()
            elif state == RestoreState.DISCARDED:
                self._fail_request(self.cancel_reason or 'TRANSFER_FAILED_DRAINED')
            elif state == RestoreState.RELEASING:
                self.phase = 'WAIT_RELEASE'
            elif state != RestoreState.IN_FLIGHT:
                raise RuntimeError(f'restore did not publish: {state}')


def scheduler_peer_rpc(scheduler, action, payload):
    # Observations remain available while the scheduler progresses a restore.
    if action not in ('stats', 'request_result') and not scheduler.is_fully_idle():
        raise RuntimeError('Peer control requires an idle Engine')
    options = json.loads(payload)
    if action == 'init':
        if hasattr(scheduler, '_l1_peer'):
            raise RuntimeError('Peer already initialized')
        scheduler._l1_peer = SchedulerPeer(scheduler, options['identity'])
        result = scheduler._l1_peer.stats()
    else:
        if action not in ('listen', 'serve', 'restore', 'finish', 'stats', 'start_service', 'request_result'):
            raise ValueError('unknown Peer action')
        peer = scheduler._l1_peer
        if action in ('serve', 'restore', 'finish') and (peer.service is not None or peer.request is not None):
            raise RuntimeError('cannot mix explicit ownership operations with nonblocking service/request')
        result = getattr(peer, action)(**options)
    return json.dumps(result)
