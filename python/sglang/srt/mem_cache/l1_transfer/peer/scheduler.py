"""Explicit fixed-peer restore operations on an idle TP1 scheduler thread."""
import contextlib
import json
import logging
import os
import socket
import math
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

import torch

from .control import ModelIdentity, SocketRPC, digest, serve_once, RemoteRejected
from .nonblocking import AsyncRPC, SourceService
from .peer import PeerReceiver, SourceEndpoint
from .restore import RestoreState
from ..transport import NvlinkEngine
from ..transport.sglang_adapter import SGLangPoolAdapter

logger = logging.getLogger(__name__)

# Terminal reasons that end the request; every other safe failure recomputes it.
ABORT_REASONS = ('CANCELLED', 'CAPACITY_EXCEEDED')


def _mark(req, phase):
    """First wall-clock time of ``phase`` in this restore's timeline, if it has one."""
    timeline = getattr(req, '_l1_timeline', None)
    if isinstance(timeline, dict):
        timeline.setdefault(phase, time.time())


def _mapping_peer_device(device):
    value = os.environ.get('SGLANG_L1_MAPPING_PEER_DEVICE')
    if value is None:
        return None
    try:
        peer = int(value)
    except ValueError:
        raise ValueError('SGLANG_L1_MAPPING_PEER_DEVICE must be a local CUDA ordinal') from None
    if peer < 0 or peer == device:
        raise ValueError('SGLANG_L1_MAPPING_PEER_DEVICE must be nonnegative and differ from the receiving device')
    return peer


class SchedulerPeer:
    def __init__(self, scheduler, identity, *, concurrent=False, queue_limit=16,
                 source_max_leases=4, source_max_lease_fraction=0.5, source_offthread=True,
                 source_protection='tree', retain_imports=False, target_slots=1, source_export_once=False):
        args = scheduler.server_args
        # TP>1 is a source-only prototype: c' locks ride the step broadcast (V-G).
        tp_source = (args.tp_size > 1 and concurrent and source_protection == 'step' and source_offthread)
        if ((args.tp_size != 1 and not tp_source) or args.pp_size != 1 or args.dp_size != 1
                or args.disaggregation_mode != 'null' or args.speculative_algorithm is not None
                or args.enable_lora or (not concurrent and args.max_running_requests != 1)):
            raise ValueError('Peer restore requires ordinary TP/PP/DP1 (TP>1 only as a step-protected concurrent source), '
                             'no speculation/LoRA, max_running_requests=1')
        from sglang.srt.runtime_context import get_parallel
        self.tp_size = args.tp_size
        self.tp_rank = get_parallel().tp_rank if args.tp_size > 1 else 0
        if self.tp_size > 1 and scheduler.request_receiver.recv_skipper is not None:
            raise ValueError('TP>1 step commands need every step to broadcast its input')
        if type(concurrent) is not bool or type(source_offthread) is not bool:
            raise ValueError('concurrent and source_offthread must be boolean')
        if source_protection not in ('tree', 'step'):
            raise ValueError('source_protection must be tree or step')
        if source_protection == 'step' and not (concurrent and source_offthread):
            raise ValueError('step source protection requires a concurrent off-thread source')
        self.source_protection = source_protection
        if type(retain_imports) is not bool or (retain_imports and not concurrent):
            raise ValueError('retain_imports must be boolean and requires concurrent mode')
        if type(target_slots) is not int or target_slots < 1 or (target_slots > 1 and not concurrent):
            raise ValueError('target_slots must be a positive integer; more than one requires concurrent mode')
        if type(queue_limit) is not int or queue_limit < 0:
            raise ValueError('queue_limit must be a nonnegative integer')
        if type(source_max_leases) is not int or source_max_leases < 1:
            raise ValueError('source_max_leases must be a positive integer')
        if (type(source_max_lease_fraction) not in (int, float)
                or not 0 < source_max_lease_fraction <= 1):
            raise ValueError('source_max_lease_fraction must be in (0, 1]')
        if concurrent and not scheduler.is_generation:
            raise ValueError('concurrent L1 requires a generation Engine')
        if concurrent and (not args.disable_overlap_schedule or scheduler.enable_priority_preemption):
            raise ValueError('concurrent L1 requires normal scheduler without priority preemption')
        if concurrent and os.environ.get('SGLANG_RADIX_FORCE_MISS', '0') not in ('0', '', 'false', 'False'):
            raise ValueError('concurrent L1 cannot force radix misses')
        self.concurrent = concurrent
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        # A HiCache Engine serves its L2 (EGM host pool) as a source only (route A).
        self.l2_source = self.cache.cache_controller is not None
        if self.l2_source and not concurrent:
            raise ValueError('an L2 (HiCache) source requires concurrent mode')
        self.cache.source_leases.allow_host_pool = self.l2_source
        _, pool = self.cache.source_leases.validate_cache()
        from .capacity import RequestCapacity
        self.capacity = RequestCapacity(self.cache) if concurrent else None
        self.overlap_forwards = dict(WAIT_ACQUIRE=0, WAIT_TRANSFER=0, WAIT_RELEASE=0)
        self.forward_count = 0
        device = scheduler.tp_worker.model_runner.gpu_id
        self.mapping_peer_device = _mapping_peer_device(device)
        # A concurrent Engine may copy into its pool while peers read leased pages of it.
        # retain_imports (D6): one mapping per peer region, reused by every later lease of it.
        # Several slots copy into disjoint private pages of the same pool regions at once.
        self.engine = NvlinkEngine(device, mapping_peer_device=self.mapping_peer_device,
                                   shared_read_exports=concurrent, retain_imports=retain_imports,
                                   concurrent_reads=target_slots > 1)
        self.adapter = SGLangPoolAdapter(pool, self.engine)
        self.identity = ModelIdentity(**identity)
        self.identity.validate()
        # Off-thread source (D1): network workers lease under the tree access lock
        # instead of queuing for a scheduler turn.
        self.tree_lock = None
        if concurrent and source_offthread:
            from sglang.srt.mem_cache.tree_access_lock import install_tree_access_lock
            # Without overlap, forwards that wrote tree KV ran on schedule_stream.
            self.cache.source_leases.enable_key_index(ready_stream=scheduler.schedule_stream)
            self.tree_lock = install_tree_access_lock(self.cache)
        self.host_adapter = None
        if self.l2_source:
            from sglang.srt.mem_cache.hicache_storage import PoolName
            from ..egm_host import EGMHostPoolAdapter
            host_pool = self.cache.host_pool_group.entry_map[PoolName.KV].host_pool
            self.host_adapter = EGMHostPoolAdapter(host_pool, self.engine, self.adapter.layout)
        if concurrent:
            # Online source: lock and record the producer event, offer once it completes.
            allocator = self.cache.token_to_kv_pool_allocator
            self.source = SourceEndpoint(self.cache, self.adapter, self.identity, None,
                max_leases=source_max_leases,
                max_lease_pages=int(allocator.size // allocator.page_size * source_max_lease_fraction),
                host_adapter=self.host_adapter, protection=source_protection,
                export_once=source_export_once)
        else:
            self.source = SourceEndpoint(self.cache, self.adapter, self.identity, torch.cuda.synchronize)
        if self.tp_size > 1:
            self.cache.source_leases.broadcast_steps = True
        self.source_waiters = {}  # request_id -> acquire Future awaiting the producer event
        # Wall-clock lease timeline (enqueued, begin, offered, released) for online sources.
        self.source_timing_open = {}
        self.source_timing = deque(maxlen=8192)
        self.receiver = None
        self.listener = None
        self.service = None
        self.service_address = None
        self.request = None
        self.phase = None
        self.completed = None
        self.completed_results = {}
        self.cancel_reason = None
        self.deadline = None
        self.required_capacity = 0
        self.queue_limit = queue_limit
        # (req, candidates, required_capacity, deadline); holds no ownership.
        self.queue = deque()
        # rid -> (req, unallocated headroom) for READY requests not yet admitted.
        self.unadmitted = {}
        # rid -> cancelled flag for detached READY consumers until their release.
        self.consumers = {}
        self.candidates = ()
        self.candidate_index = 0
        self.attempts = []
        self.poll_interval = float(os.environ.get('SGLANG_L1_POLL_INTERVAL', '0.001'))
        if not math.isfinite(self.poll_interval) or not 0 <= self.poll_interval <= 0.1:
            raise ValueError('SGLANG_L1_POLL_INTERVAL must be finite and in [0, 0.1] seconds')
        # One worker per Engine; connections and receipts remain per transaction.
        self.rpc_executor = ThreadPoolExecutor(max_workers=target_slots, thread_name_prefix='l1-rpc')
        # D5: restores in flight at once; slot 0 is the peer itself.
        self.slots = [self] + [_RestoreSlot(self) for _ in range(target_slots - 1)]
        from .timing import install
        install(self)

    def stats(self):
        allocator = self.cache.token_to_kv_pool_allocator
        result = dict(wait_state=getattr(self, "phase", None), source_leases=len(self.source.active),
                    source_pending=len(self.source.pending), target_active=any(slot.receiver is not None for slot in self._slots()),
                    forward_count=self.forward_count,
                    exports=len(self.engine._exports), imports=sum(m.public.remote for m in self.engine._memories.values()),
                    batches=len(self.engine._batches), available_tokens=allocator.available_size(),
                    pinned_pages=sum(getattr(allocator, '_l1_pinned_pages', {}).values()),
                    l2_source=getattr(self, 'l2_source', False),
                    mapping_peer_device=getattr(self, 'mapping_peer_device', None))
        if getattr(self, 'concurrent', False):
            result.update(admitted_requests=len(self.capacity.active),
                          remaining_growth=self.capacity.remaining(),
                          queued_requests=len(self.queue),
                          unadmitted_requests=len(self.unadmitted),
                          ready_consumers=len(self.consumers),
                          restore_headroom=self.reserved_headroom(),
                          overlap_forwards=dict(self.overlap_forwards))
        return result

    def listen(self, host='127.0.0.1', port=0):
        if self.listener is not None:
            raise RuntimeError('listener already open')
        self.listener = socket.socket()
        self.listener.bind((host, port))
        self.listener.listen(16)
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
            if not getattr(self, 'concurrent', False):
                torch.cuda.synchronize(self.engine.device)
            # Concurrent private pages are either never consumed (copy already
            # drained), or adopted by the ordinary request row. No extra GPU
            # owner remains to release; normal scheduler cleanup orders reuse.
            self.receiver.finish_consumer()
            self.receiver = None
        return self.stats()


    @property
    def blocks_compute(self):
        # Leased pages are complete tree pages; concurrent forwards only write other pages.
        if getattr(self, 'concurrent', False):
            return False
        return bool(self.source.active) or self.phase in ('WAIT_ACQUIRE', 'WAIT_TRANSFER', 'WAIT_RELEASE')

    def allows_request(self, req):
        # Capacity waiting owns no pages. Once reserved, no unrelated request
        # may consume the full-request headroom before our consumer finishes.
        if not getattr(self, 'concurrent', False):
            return self.phase in (None, 'WAIT_CAPACITY') or req is self.request
        slots = self._slots()
        if any(req is slot.request for slot in slots) or req.rid in self.capacity.active or req.rid in self.unadmitted:
            return True
        # Drain existing consumers instead of starving a waiting restore with
        # an unbounded stream of newly admitted ordinary requests.
        if any(slot.phase == 'WAIT_CAPACITY' for slot in slots):
            return False
        return self.capacity.can_admit(req, self.reserved_headroom())

    def reserved_headroom(self):
        return self._unadmitted_headroom() + sum(slot._outstanding() for slot in self._slots())

    def _outstanding(self):
        """Growth this slot's restore still needs beyond the pages it has reserved."""
        if self.request is None or self.request.rid in self.capacity.active:
            return 0
        if self.receiver is not None and self.receiver.pending is not None:
            return max(0, self.required_capacity - len(self.receiver.pending['reservation'].indices))
        return self.required_capacity

    def _slots(self):
        return getattr(self, 'slots', None) or [self]

    def _unadmitted_headroom(self):
        return sum(headroom for _, headroom in self.unadmitted.values())

    def note_admitted(self, req):
        _mark(req, 'admitted')
        if getattr(self, 'concurrent', False):
            self.capacity.admit(req)
            self.unadmitted.pop(req.rid, None)

    def note_forward(self, batch):
        self.forward_count += 1
        if getattr(self, 'tp_size', 1) > 1 and self.forward_count % 100 == 0:
            # Compared offline across ranks: identical at every step unless the trees diverged.
            leases = self.cache.source_leases
            logger.info('L1_TREE step=%d evictable=%d protected=%d held=%d', self.forward_count,
                        self.cache.evictable_size(), self.cache.protected_size(), len(leases._held))
        if getattr(self, 'concurrent', False):
            for slot in self._slots():
                if slot.phase in slot.overlap_forwards:
                    slot.overlap_forwards[slot.phase] += 1

    def start_service(self, host='127.0.0.1', port=0):
        if not self.scheduler.server_args.disable_overlap_schedule:
            raise ValueError('nonblocking peer requires disable_overlap_schedule')
        address = self.listen(host, port)
        self.service = SourceService(self.listener,
                                     direct=self._serve_direct if self.tree_lock is not None else None)
        self.service_address = address
        return address

    def endpoint_info(self):
        """What a router or peer needs to address and check this source."""
        with getattr(self, 'tree_lock', None) or contextlib.nullcontext():
            step_log = list(self.cache.source_leases.step_log.values())
        return dict(service=self.service_address, contract_digest=digest(self.source.contract),
                    concurrent=self.concurrent, stats=self.stats(), source_timing=list(self.source_timing),
                    source_protection=getattr(self, 'source_protection', 'tree'), source_step_log=step_log)

    def start_request(self, req, options):
        from .request import restore_payload
        if getattr(self, 'tp_size', 1) > 1:
            raise ValueError('a TP>1 L1 peer is a source only')
        if getattr(self, 'l2_source', False):
            raise ValueError('an L2 (HiCache) source Engine cannot restore as a target')
        if not self.scheduler.server_args.disable_overlap_schedule:
            raise ValueError('nonblocking peer requires disable_overlap_schedule')
        # Raw server_args.page_size is None unless set explicitly; the allocator holds the resolved size.
        page = self.cache.token_to_kv_pool_allocator.page_size
        payload = restore_payload(dict(input_ids=list(req.origin_input_ids)), options, page)
        if getattr(self, 'concurrent', False):
            self.capacity.bound(req)  # reject unbudgeted beam expansion before ownership
            if req._compute_max_prefix_len(len(req.origin_input_ids)) < max(
                    len(c['tokens']) for c in payload['candidates']):
                raise ValueError('requested logprobs require recomputing restored prefix')
        output = req.sampling_params.max_new_tokens
        if type(output) is not int or output < 0:
            raise ValueError('serialized sharing requires a bounded max_new_tokens')
        required = (len(req.origin_input_ids) + output + page - 1) // page * page
        if required > self.cache.token_to_kv_pool_allocator.size:
            self._end_unstarted(req, 'CAPACITY_EXCEEDED')
            return
        deadline = time.monotonic() + options.get('wait_timeout_ms', 30000) / 1000
        if len(self.queue) >= self.queue_limit and not self._can_activate():
            self._end_unstarted(req, 'QUEUE_FULL')
            return
        # Wall-clock phase marks of this restore; returned with its result.
        req._l1_timeline = dict(queued=time.time())
        self.queue.append((req, payload['candidates'], required, deadline))
        self._activate_next()

    def _can_activate(self):
        # The serial entry never overlaps target and source ownership.
        return self.request is None and self.receiver is None and (
            getattr(self, 'concurrent', False) or not self.source.active)

    def _activate_next(self):
        if not self.queue or not self._can_activate():
            return
        req, candidates, required, deadline = self.queue.popleft()
        _mark(req, 'activated')
        self.request = req
        self.candidates = candidates
        self.candidate_index = 0
        self.attempts = []
        self.payload = candidates[0]
        self.completed = None
        self.cancel_reason = None
        self.restore_id = uuid.uuid4().hex
        self.required_capacity = required
        self.deadline = deadline
        self.phase = 'WAIT_CAPACITY'
        self.poll_counts = dict(WAIT_CAPACITY=0, WAIT_ACQUIRE=0, WAIT_COPY=0, WAIT_TRANSFER=0, WAIT_RELEASE=0)
        self.overlap_forwards = dict(WAIT_ACQUIRE=0, WAIT_TRANSFER=0, WAIT_RELEASE=0)

    def _expire_queued(self):
        now = time.monotonic()
        if not any(entry[3] <= now for entry in self.queue):
            return
        expired = [entry[0] for entry in self.queue if entry[3] <= now]
        self.queue = deque(entry for entry in self.queue if entry[3] > now)
        for req in expired:
            self._end_unstarted(req, 'WAIT_TIMEOUT')

    def _end_unstarted(self, req, reason):
        """End a request that never owned a reservation or source lease."""
        self._end(req, dict(state=None, reason=reason, consumer_released=True,
                            transferred_bytes=0, attempts=[]))

    def _end(self, req, result):
        """Report a terminal outcome; safe non-cancel failures recompute the request."""
        reason = result['reason']
        if reason in ABORT_REASONS:
            result['state'] = 'CANCELLED' if reason == 'CANCELLED' else 'FAILED'
            self._record_result(req.rid, result)
            self.scheduler.finish_l1_waiting_request(req, reason)
        else:
            result['state'] = 'FALLBACK'
            self._publish_result(req, result)
            self.scheduler._add_request_to_queue(req)

    def abort_matching(self, rid, abort_all):
        """Cancel the active shared requests and drop queued ones; queued own nothing."""
        for slot in self._slots():
            if slot.request is not None and (abort_all or slot.request.rid.startswith(rid)):
                slot.cancel_request()
        for consumer in self.consumers:
            if abort_all or consumer.startswith(rid):
                self.consumers[consumer] = True  # ordinary abort releases it
        matched = [entry[0] for entry in self.queue if abort_all or entry[0].rid.startswith(rid)]
        if not matched:
            return
        self.queue = deque(entry for entry in self.queue
                           if not (abort_all or entry[0].rid.startswith(rid)))
        for req in matched:
            self._end_unstarted(req, 'CANCELLED')

    def _try_capacity(self):
        if self.scheduler._engine_paused:
            return
        concurrent = getattr(self, 'concurrent', False)
        if not concurrent and not self.scheduler.is_fully_idle(ignore_l1=True):
            return
        payload = self.payload
        self.rpc = AsyncRPC(payload['host'], payload['port'], executor=self.rpc_executor)
        self.receiver = PeerReceiver(self.cache, self.adapter, self.identity, self.rpc)
        self.steps = self.receiver.prepare_steps(payload['tokens'], required_capacity=self.required_capacity,
            tier=payload.get('tier', 'hbm'),
            **({'headroom': self.capacity.remaining() + self._unadmitted_headroom()
                            + sum(slot._outstanding() for slot in self._slots() if slot is not self),
                'private': True}
               if concurrent else {}))
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
            _mark(getattr(self, 'request', None), 'reserved')
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
                self.phase = 'WAIT_COPY'
                self._start_copy()

    def _start_copy(self):
        # Without concurrent reads a pool region carries one batch at a time: copies go one by one.
        engine = getattr(self, 'engine', None)
        if (not getattr(engine, 'concurrent_reads', False)
                and any(slot is not self and slot.phase == 'WAIT_TRANSFER' for slot in self._slots())):
            return
        tx = self.session['transaction']
        if getattr(self, 'concurrent', False):
            self._check_roles_disjoint()
        # Order reuse of target slots after prior schedule-stream work.
        # Future ordinary forwards remain independent and use other pages.
        tx.start(**({'producer_stream': self.scheduler.schedule_stream.cuda_stream}
                    if getattr(self, 'concurrent', False) else {}))
        _mark(getattr(self, 'request', None), 'copy_started')
        self.phase = 'WAIT_TRANSFER'

    def _ready(self):
        reservation, tx = self.session['reservation'], self.session['transaction']
        self.restored = dict(restored_tokens=len(reservation.indices), local_tokens=reservation.prefix_len,
                             transferred_bytes=sum(r.length for r in tx.requests) if tx else 0,
                             wait_polls=dict(self.poll_counts), **self.stats())
        _mark(getattr(self, 'request', None), 'ready')
        self.restored.update(restore_id=self.restore_id, state='READY', reason=None,
                             overlap_forwards=dict(getattr(self, "overlap_forwards", {})),
                             timeline=getattr(self.request, '_l1_timeline', None))
        self.restored['attempts'] = list(self.attempts)
        self.rpc.close()
        self.deadline = None
        req = self.request
        if getattr(self, 'concurrent', False):
            self._detach_ready(req, reservation)
        else:
            self.phase = 'READY'
            req._l1_consumer_release = self.finish_request
        self.scheduler._add_request_to_queue(req)

    def _detach_ready(self, req, reservation):
        """Hand a READY private reservation to its request; the slot takes the next restore.

        Private pages and the source ACK are settled, so the consumer owns nothing
        that needs this peer; its unallocated growth stays reserved until admission.
        """
        receiver, restored = self.receiver, self.restored
        req._l1_private_reservation = reservation
        self.unadmitted[req.rid] = (req, max(0, self.required_capacity - len(reservation.indices)))
        self.consumers[req.rid] = False
        req._l1_consumer_release = lambda: self._release_detached(req, receiver, restored)
        self.receiver = None
        self.request = None
        self.phase = None
        self.required_capacity = 0

    def _release_detached(self, req, receiver, restored):
        # Same owner-thread contract as finish_request, scoped to one consumer.
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        aborted = isinstance(req.finished_reason, FINISH_ABORT) or isinstance(req.to_finish, FINISH_ABORT)
        if not req.finished() and not aborted:
            raise RuntimeError('L1 consumer retraction is unsupported; stop Engine')
        receiver.finish_consumer()
        self.unadmitted.pop(req.rid, None)
        cancelled = self.consumers.pop(req.rid)
        self._publish_result(req, dict(restored, consumer_released=receiver.pending is None,
            state='CANCELLED' if cancelled else 'FAILED' if aborted else 'FINISHED',
            reason='CANCELLED' if cancelled else 'GENERATION_ABORTED' if aborted else None))
        req._l1_consumer_release = None
        req._l1_private_reservation = None

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
        elif self.phase == 'WAIT_COPY':
            self.session['transaction'].cancel()  # never submitted: discards and releases the source
            self.phase = 'WAIT_RELEASE'
        # ACQUIRE: retain the future and reservation, then clean a valid late
        # offer without DMA. Unknown reply still reaches pump's fail-stop.
        # READY/RUNNING: ordinary abort owns queue removal and GPU-safe cleanup.

    def _fail_request(self, reason):
        """Report a terminal failure only after all acquired owners are safe."""
        req = self.request
        if self.receiver is not None:
            self.finish()
            self.rpc.close()
        result = dict(restore_id=self.restore_id, state=None, reason=reason,
                      consumer_released=True, transferred_bytes=None,
                      wait_polls=dict(self.poll_counts), attempts=list(self.attempts),
                      timeline=getattr(req, '_l1_timeline', None))
        self.request = None
        self.phase = None
        self.required_capacity = 0
        self.deadline = None
        self._end(req, result)

    def _next_candidate(self, error):
        """Acquire was safely rejected and the reservation discarded; try the next source."""
        if self.receiver.pending is not None:
            raise RuntimeError('rejected acquire retained a target reservation')
        self.attempts.append(dict(host=self.payload['host'], port=self.payload['port'], error=error))
        self.rpc.close()
        self.receiver = None
        self.candidate_index += 1
        if self.cancel_reason or self.candidate_index == len(self.candidates):
            self._fail_request(self.cancel_reason or 'SOURCE_REJECTED')
            return
        self.payload = self.candidates[self.candidate_index]
        self.restore_id = uuid.uuid4().hex
        self.phase = 'WAIT_CAPACITY'

    def finish_request(self):
        # Called before ordinary request KV/page cleanup on the same owner thread.
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        aborted = isinstance(self.request.finished_reason, FINISH_ABORT) or isinstance(self.request.to_finish, FINISH_ABORT)
        if not self.request.finished() and not aborted:
            raise RuntimeError('L1 consumer retraction is unsupported; stop Engine')
        released = self.finish()
        self._publish_result(self.request, dict(self.restored, consumer_released=(
            not released['target_active'] and released['pinned_pages'] == 0),
            state=('CANCELLED' if self.cancel_reason == 'CANCELLED' else
                   'FAILED' if aborted or self.cancel_reason else 'FINISHED'),
            reason=self.cancel_reason or ('GENERATION_ABORTED' if aborted else None)))
        self.request._l1_consumer_release = None
        self.request._l1_private_reservation = None
        self.request = None
        self.phase = None
        self.required_capacity = 0
        self.deadline = None

    def _publish_result(self, req, result):
        # Request-level customized_info reaches meta_info['l1_restore'] of the final output.
        self._record_result(req.rid, result)
        req.customized_info = {**(req.customized_info or {}), 'l1_restore': [result]}

    def _record_result(self, rid, result):
        self.completed = (rid, result)
        if not hasattr(self, 'completed_results'):
            self.completed_results = {}
        self.completed_results[rid] = result
        while len(self.completed_results) > 128:
            del self.completed_results[next(iter(self.completed_results))]

    def reject_request(self, req, reason):
        # An invalid shared request acquires nothing and must not tear down the
        # Engine (or overwrite the active transaction's ownership).
        self._record_result(req.rid, dict(state='FAILED', reason=reason,
            consumer_released=True, transferred_bytes=0))
        self.scheduler.finish_l1_waiting_request(req, reason)

    def request_result(self, rid):
        if rid in getattr(self, 'completed_results', {}):
            return self.completed_results[rid]
        if self.completed is None or self.completed[0] != rid:
            raise RuntimeError('no completed L1 consumer for request')
        return self.completed[1]

    def pump(self):
        try:
            self._pump()
        except BaseException:
            logger.exception('L1_PEER ownership uncertain; stopping scheduler')
            os._exit(70)

    def _record_producer(self):
        # Without overlap, forwards that wrote tree KV ran on schedule_stream.
        event = torch.cuda.Event()
        event.record(self.scheduler.schedule_stream)
        return event

    def _serve_source_events(self):
        concurrent = getattr(self, 'concurrent', False)
        # Bound each turn; network threads never touch cache or CUDA.
        for _ in range(16):
            try:
                kind, data, future = self.service.events.get_nowait()
            except Empty:
                break
            if kind == 'error':
                logger.warning('L1 source connection error: %r', data)
                continue
            if kind == 'closed':
                with getattr(self, 'tree_lock', None) or contextlib.nullcontext():
                    outstanding = data is not None and (data in self.source.active or data in self.source.pending)
                if outstanding:
                    raise RuntimeError('source connection ended with outstanding lease')
                continue
            try:
                if not concurrent:
                    if data.get('operation') == 'acquire' and (
                            self.source.active or self.receiver is not None or not self.scheduler.is_fully_idle()):
                        raise RemoteRejected('source Engine is busy')
                    answer = self.source.dispatch(data)
                else:
                    begin = time.time()
                    answer = self.source.dispatch_deferred(data, record_producer=self._record_producer)
            except RemoteRejected as exc:
                future.set_exception(exc)
            else:
                if answer is None:
                    self.source_waiters[data['id']] = future
                    self.source_timing_open[data['id']] = dict(
                        enqueued=future.enqueued, begin=begin, pages=len(self.source.pending[data['id']][0].page_indices))
                else:
                    if concurrent and data.get('operation') == 'release':
                        self._close_source_timing(data['receipt']['request_id'])
                    future.set_result(answer)
        if getattr(self, 'tree_lock', None) is not None:
            return  # off-thread acquires answer from the network worker
        offers = self.source.poll_ready()
        if offers:
            self._check_roles_disjoint()
        for request_id, offer in offers:
            if request_id in self.source_timing_open:
                self.source_timing_open[request_id]['offered'] = time.time()
            self.source_waiters.pop(request_id).set_result(offer)

    def drain_step_commands(self):
        """Rank 0: lock / unlock commands to ride this step's input broadcast."""
        with getattr(self, 'tree_lock', None) or contextlib.nullcontext():
            return self.cache.source_leases.drain_step_commands()

    def apply_step_commands(self, received):
        """Every rank, after the broadcast and before this step allocates: apply and strip them."""
        from sglang.srt.mem_cache.source_lease import StepCommand
        commands = [r for r in received if isinstance(r, StepCommand)]
        if not commands:
            return received
        with getattr(self, 'tree_lock', None) or contextlib.nullcontext():
            for command in commands:
                self.cache.source_leases.apply_step_command(command)
        return [r for r in received if not isinstance(r, StepCommand)]

    def _serve_direct(self, request):
        """Network-worker dispatch for an off-thread source; never touches the target."""
        torch.cuda.set_device(self.engine.device)
        if request.get('operation') == 'regions':
            # TP>1 readers copy each rank's shard; this rank's pool, exported once.
            with self.tree_lock:
                return self.source.standing_offer()
        if request.get('operation') == 'verify':
            pages, generations = request.get('pages'), request.get('generations')
            if not isinstance(pages, list) or not isinstance(generations, list) or len(pages) != len(generations):
                raise RemoteRejected('pages and generations required')
            return {'valid': self.cache.source_leases.generations.snapshot(pages) == tuple(generations)}
        if request.get('operation') == 'acquire' and getattr(self, 'tp_rank', 0) != 0:
            raise RemoteRejected('acquire only on TP rank 0')
        if request.get('operation') == 'acquire':
            begin = time.time()
            offer = self.source.acquire_offthread(request, self.tree_lock)
            marks = self.source.acquire_spans.get(request['id'], {})
            # Per-phase ms of this acquire: lock wait, lease, ready events, export, offer.
            spans = {f'{a}->{b}': (marks[b] - marks[a]) * 1e3
                     for a, b in zip(('entered', 'locked', 'leased', 'ready', 'exported', 'relocked'),
                                     ('locked', 'leased', 'ready', 'exported', 'relocked', 'offered'))
                     if a in marks and b in marks}
            self.source_timing_open[request['id']] = dict(
                enqueued=begin, begin=begin, offered=time.time(), spans=spans,
                pages=sum(count for _, count in offer['page_runs']))
            return offer
        if request.get('operation') == 'release':
            answer = self.source.release_offthread(request, self.tree_lock)
            receipt = request.get('receipt')
            if isinstance(receipt, dict):
                self._close_source_timing(receipt.get('request_id'))
            return answer
        raise RemoteRejected('unknown operation')

    def _close_source_timing(self, request_id):
        entry = self.source_timing_open.pop(request_id, None)
        if entry is not None:
            entry['released'] = time.time()
            self.source_timing.append(entry)

    def _check_roles_disjoint(self):
        """Shared read exports are safe only if no leased page is a local copy destination.

        Leased pages are locked tree pages and copy destinations are newly allocated
        private pages, so an overlap means the allocator invariant broke: fail-stop.
        """
        targets = {page for slot in self._slots() if slot.receiver is not None and slot.receiver.pending is not None
                   for page in slot.receiver.pending['reservation'].pages}
        if not targets:
            return
        with getattr(self, 'tree_lock', None) or contextlib.nullcontext():
            leases = ([e['lease'] for e in self.source.active.values()]
                      + [p[0] for p in self.source.pending.values()])
        # Host (egm) page IDs index the L2 pool, not this device pool.
        leased = {page for lease in leases if getattr(lease, 'tier', 'hbm') == 'hbm' for page in lease.page_indices}
        if leased & targets:
            raise RuntimeError('source lease overlaps a target copy destination')

    def _pump(self):
        if getattr(self, 'concurrent', False):
            self.capacity.prune()
        if self.service is not None:
            self._serve_source_events()
        if getattr(self, 'source_protection', 'tree') == 'step':
            # The step boundary: after input ingestion, before this step allocates or evicts.
            with self.tree_lock:
                self.cache.source_leases.apply_step_locks()
        self._expire_queued()
        for slot in self._slots():
            slot._pump_slot()

    def _pump_slot(self):
        """Advance one restore slot (the peer itself is slot 0)."""
        if self.request is not None and self.deadline is not None and time.monotonic() >= self.deadline:
            self.cancel_request('WAIT_TIMEOUT')
        self._activate_next()
        if self.phase == 'WAIT_COPY':
            self._start_copy()
        if self.phase == 'WAIT_CAPACITY':
            self._try_capacity()
        if self.request is not None and self.phase in self.poll_counts:
            self.poll_counts[self.phase] += 1
        if self.phase == 'WAIT_ACQUIRE' and self.acquire_future.done():
            _mark(getattr(self, 'request', None), 'offered')
            try:
                answer = self.acquire_future.result()
            except RemoteRejected as exc:
                try:
                    self.steps.throw(exc)
                except RemoteRejected:
                    pass
                self._next_candidate(str(exc))
                return
            try:
                self.steps.send(answer)
            except StopIteration as done:
                self._prepared(done.value)
        if self.phase in ('WAIT_TRANSFER', 'WAIT_RELEASE'):
            state = self.session['transaction'].poll()
            if state != RestoreState.IN_FLIGHT:
                _mark(getattr(self, 'request', None), 'copied')
            if state == RestoreState.PUBLISHED:
                if self.cancel_reason:
                    self._fail_request(self.cancel_reason)
                else:
                    self._ready()
            elif state == RestoreState.DISCARDED:
                stale = self.session['transaction'].stale
                self._fail_request(self.cancel_reason or ('SOURCE_STALE' if stale else 'TRANSFER_FAILED_DRAINED'))
            elif state == RestoreState.RELEASING:
                self.phase = 'WAIT_RELEASE'
            elif state != RestoreState.IN_FLIGHT:
                raise RuntimeError(f'restore did not publish: {state}')


class _RestoreSlot:
    """One more concurrent restore. Holds only per-restore state; everything else,
    including the shared queue, budgets and result records, is the peer's."""

    def __init__(self, peer):
        self._peer = peer
        self.request = self.receiver = self.session = self.phase = None
        self.required_capacity = 0
        self.deadline = self.cancel_reason = None
        self.overlap_forwards = dict(WAIT_ACQUIRE=0, WAIT_TRANSFER=0, WAIT_RELEASE=0)

    def __getattr__(self, name):
        return getattr(self._peer, name)

    # The per-restore state machine, run against this slot's state.
    _pump_slot = SchedulerPeer._pump_slot
    _can_activate = SchedulerPeer._can_activate
    _activate_next = SchedulerPeer._activate_next
    _try_capacity = SchedulerPeer._try_capacity
    _prepared = SchedulerPeer._prepared
    _start_copy = SchedulerPeer._start_copy
    _ready = SchedulerPeer._ready
    _detach_ready = SchedulerPeer._detach_ready
    cancel_request = SchedulerPeer.cancel_request
    _fail_request = SchedulerPeer._fail_request
    _next_candidate = SchedulerPeer._next_candidate
    _outstanding = SchedulerPeer._outstanding
    finish = SchedulerPeer.finish


L1_CONFIG_KEYS = {'identity', 'concurrent', 'service_host', 'service_port', 'queue_limit',
                  'source_max_leases', 'source_max_lease_fraction', 'source_offthread',
                  'source_protection', 'retain_imports', 'target_slots', 'source_export_once'}
_PEER_OPTION_KEYS = ('queue_limit', 'source_max_leases', 'source_max_lease_fraction',
                     'source_offthread', 'source_protection', 'retain_imports', 'target_slots',
                     'source_export_once')


def parse_l1_config(text):
    """--l1-config: a JSON object, or '@' followed by a JSON file path."""
    if text.startswith('@'):
        with open(text[1:]) as f:
            text = f.read()
    config = json.loads(text)
    if not isinstance(config, dict) or 'identity' not in config or set(config) - L1_CONFIG_KEYS:
        raise ValueError(f'--l1-config requires identity and only keys {sorted(L1_CONFIG_KEYS)}')
    return config


def init_scheduler_peer_from_config(scheduler, text):
    """Build the peer on the scheduler thread and start its source service."""
    config = parse_l1_config(text)
    peer = SchedulerPeer(scheduler, config['identity'], concurrent=config.get('concurrent', False),
                         **{k: config[k] for k in _PEER_OPTION_KEYS if k in config})
    port = config.get('service_port', 0)
    # One service per TP rank: rank r listens on service_port + r.
    address = peer.start_service(config.get('service_host', '127.0.0.1'), port + peer.tp_rank if port else 0)
    logger.info('L1_PEER_SERVICE %s', address)
    return peer


def scheduler_peer_rpc(scheduler, action, payload):
    # Observations remain available while the scheduler progresses a restore.
    if action not in ('stats', 'request_result') and not scheduler.is_fully_idle():
        raise RuntimeError('Peer control requires an idle Engine')
    options = json.loads(payload)
    if action == 'init':
        if hasattr(scheduler, '_l1_peer'):
            raise RuntimeError('Peer already initialized')
        scheduler._l1_peer = SchedulerPeer(scheduler, options['identity'], concurrent=options.get('concurrent', False),
                                           **{k: options[k] for k in _PEER_OPTION_KEYS if k in options})
        result = scheduler._l1_peer.stats()
    else:
        if action not in ('listen', 'serve', 'restore', 'finish', 'stats', 'start_service', 'request_result'):
            raise ValueError('unknown Peer action')
        peer = scheduler._l1_peer
        if action in ('serve', 'restore', 'finish') and (peer.service is not None or peer.request is not None or peer.concurrent):
            raise RuntimeError('cannot mix explicit ownership operations with nonblocking service/request')
        result = getattr(peer, action)(**options)
    return json.dumps(result)
