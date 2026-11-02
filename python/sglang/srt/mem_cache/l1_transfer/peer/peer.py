# SPDX-License-Identifier: Apache-2.0
"""Fixed-peer source endpoint and synchronous receiver preparation."""
from dataclasses import asdict
from array import array
import time

from sglang.srt.mem_cache.l1_transfer.transport import BusyError, Opcode, RegionDescriptor
from sglang.srt.mem_cache.l1_transfer.transport.api import MemoryHandle, TransferRequest
from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan, plan_transfer, bind_regions
from sglang.srt.mem_cache.source_index import prefix_page_keys
from sglang.srt.mem_cache.source_lease import page_runs
from .control import PROTOCOL_VERSION, contract, digest, wire_copy, RemoteRejected
from .restore import RestoreTransaction


def plan_remote_regions(descriptors, bindings, plan, target_bindings):
    """Plan using descriptor sizes only; placeholders must never reach transport.

    Full bindings are still validated. Return only descriptors touched by READ,
    in original order, and requests whose remote handles need materialization.
    """
    if len({d.region_id for d in descriptors}) != len(descriptors):
        raise ValueError('duplicate source region identity')
    views = [MemoryHandle(d.region_id, d.size, -1, True, 0) for d in descriptors]
    spans = {}
    for key, rows in bindings.items():
        parsed = []
        for a, n, i, off in rows:
            if type(i) is not int or not 0 <= i < len(views):
                raise ValueError('invalid source region index')
            parsed.append(RegionSpan(a, n, views[i], off))
        spans[key] = tuple(parsed)
    requests = bind_regions(plan, spans, target_bindings, Opcode.READ)
    needed = {r.remote.region_id for r in requests}
    return [d for d in descriptors if d.region_id in needed], requests


class SourceEndpoint:
    """Scheduler-thread source leases.

    With ``ready`` the producer is finished synchronously inside acquire. Without
    it, acquire is split: begin_acquire locks the tree and records the producer
    event, and poll_ready exports and offers only after that event completes.
    """
    def __init__(self, cache, adapter, identity, ready, *, max_leases=None, max_lease_pages=None,
                 host_adapter=None, protection='tree'):
        self.cache, self.adapter, self.ready = cache, adapter, ready
        # tree: hbm leases are radix-locked before the offer. step (c'): locked at the
        # next scheduler step, read optimistically, validated by page generations at release.
        if protection not in ('tree', 'step'):
            raise ValueError('protection must be tree or step')
        self.protection = protection
        if protection == 'step':
            cache.source_leases.enable_step_protection()
        # tier -> pool adapter whose pages a lease of that tier names (egm: the L2 host pool).
        self.adapters = {'hbm': adapter, **({'egm': host_adapter} if host_adapter is not None else {})}
        cache.source_leases.enable_key_index()  # targets name prefixes by page key
        self.contract = wire_copy(contract(identity, adapter.layout))
        self.max_leases = max_leases
        self.max_lease_pages = max_lease_pages
        self.active = {}
        self.pending = {}  # request_id -> (lease, fingerprint, producer event); nothing exported
        # Per-tier constant offer fields, built on first use; regions vary only by lease_id.
        self._static_offers = {}
        self.released = {}  # bounded by endpoint lifetime; no TTL reclamation
        self.released_answers = {}  # request_id -> release answer, replayed on a duplicate release
        self._standing = None  # this pool exported once, for TP>1 readers (standing_offer)

    def dispatch(self, request):
        self.cache.source_leases._check_thread()
        if request.get('version') != PROTOCOL_VERSION:
            raise RemoteRejected('unsupported protocol version')
        operation = request.get('operation')
        if operation == 'acquire':
            return self.acquire(request)
        if operation == 'release':
            return self.release(request)
        raise RemoteRejected('unknown operation')

    def dispatch_deferred(self, request, *, record_producer):
        """Like dispatch, but acquire returns None and its offer comes from poll_ready."""
        self.cache.source_leases._check_thread()
        if request.get('version') != PROTOCOL_VERSION:
            raise RemoteRejected('unsupported protocol version')
        if request.get('operation') == 'acquire':
            self.begin_acquire(request, record_producer=record_producer)
            return None
        return self.dispatch(request)

    def _validate_acquire(self, request):
        if request.get('contract') != self.contract:
            raise RemoteRejected('incompatible model/context/layout')
        key, end, start = request.get('key'), request.get('end'), request.get('start')
        tier = request.get('tier', 'hbm')
        page = self.adapter.layout.page_size
        if (not isinstance(key, str) or len(key) != 64 or type(end) is not int or type(start) is not int
                or not 0 <= start < end or end % page or start % page):
            raise RemoteRejected('page key and page-aligned 0 <= start < end required')
        if tier not in self.adapters:
            raise RemoteRejected(f'tier {tier!r} not served by this source')
        request_id = request.get('id')
        if not isinstance(request_id, str) or len(request_id) != 32:
            raise RemoteRejected('invalid request ID')
        if request_id in self.released:
            raise RemoteRejected('request already released')
        if request_id in self.pending:
            raise RemoteRejected('request ID already pending')
        span = (key, end, start, tier)
        return span, request_id, digest({'contract': self.contract, 'key': key, 'end': end, 'start': start,
                                         'tier': tier})

    def _check_quota(self, span):
        leases = list(self.active.values()) + [{'lease': p[0]} for p in self.pending.values()]
        if self.max_leases is not None and len(leases) >= self.max_leases:
            raise RemoteRejected('source busy: lease quota')
        if self.max_lease_pages is not None:
            pages = (span[1] - span[2]) // self.adapter.layout.page_size
            held = sum(len(item['lease'].page_indices) for item in leases)
            if held + pages > self.max_lease_pages:
                raise RemoteRejected('source busy: leased page quota')

    def _lock(self, span, fingerprint, ready):
        key, end, start, tier = span
        try:
            return self.cache.acquire_source_lease_by_key(
                key, end=end, start=start, fingerprint=fingerprint, ready=ready, tier=tier,
                protection=self.protection if tier == 'hbm' else 'tree')
        except ValueError as exc:
            raise RemoteRejected(str(exc)) from exc

    def acquire(self, request):
        span, request_id, fingerprint = self._validate_acquire(request)
        if request_id in self.active:
            entry = self.active[request_id]
            if entry['receipt']['fingerprint'] != fingerprint:
                raise RemoteRejected('request ID reused with different prefix')
            return entry['offer']
        self._check_quota(span)
        return self._offer(request_id, self._lock(span, fingerprint, self.ready), fingerprint)

    def begin_acquire(self, request, *, record_producer):
        """Lock the prefix and record the producer event; the offer follows in poll_ready."""
        span, request_id, fingerprint = self._validate_acquire(request)
        if request_id in self.active:
            raise RemoteRejected('request ID already acquired')
        self._check_quota(span)
        events = []
        lease = self._lock(span, fingerprint, lambda: events.append(record_producer()))
        self.pending[request_id] = (lease, fingerprint, events[0])
        return request_id

    def poll_ready(self):
        """Return [(request_id, offer)] whose producer work has completed.

        Exports wait while a local copy batch is in flight: regions cannot be
        exported then. That copy completes on its own, so waiting cannot deadlock.
        """
        if not self.pending or self.adapter.engine.any_in_flight(self.adapter.memories):
            return []
        done = [rid for rid, (_, _, event) in self.pending.items() if event.query()]
        offers = []
        for rid in done:
            lease, fingerprint, _ = self.pending.pop(rid)
            offers.append((rid, self._offer(rid, lease, fingerprint)))
        return offers

    def acquire_offthread(self, request, lock, *, export_timeout=30.0):
        """Acquire on a network worker: tree and endpoint state under ``lock``.

        Waiting for the path's producer events and exporting happen outside the
        lock, so the scheduler keeps running cache calls meanwhile.
        """
        with lock:
            span, request_id, fingerprint = self._validate_acquire(request)
            if request_id in self.active:
                entry = self.active[request_id]
                if entry['receipt']['fingerprint'] != fingerprint:
                    raise RemoteRejected('request ID reused with different prefix')
                return entry['offer']
            self._check_quota(span)
            lease = self._lock(span, fingerprint, lambda: None)
            self.pending[request_id] = (lease, fingerprint, None)
        exports = []
        try:
            for event in lease.ready_events:
                event.synchronize()
            exports = self._export_when_idle(self.adapters[lease.tier], export_timeout)
            with lock:
                offer = self._offer_from(request_id, lease, fingerprint, exports)
                del self.pending[request_id]
            return offer
        except BaseException:
            # No descriptor escaped: the offer was never returned.
            with lock:
                self.pending.pop(request_id, None)
                self.active.pop(request_id, None)
                self.adapter.engine.release_exports([d.lease_id for d in exports])
                self.cache.release_source_lease(lease, drained=True)
            raise

    def release_offthread(self, request, lock):
        with lock:
            return self.release(request)

    def _export_when_idle(self, adapter, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                return adapter.engine.export_memories(adapter.memories, 'read')
            except BusyError:
                # A local copy batch into this pool is in flight; it completes on its own.
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.0002)

    def standing_offer(self):
        """The whole device pool, exported once: regions, bindings and per-buffer page bytes.

        A TP>1 reader uses it for ranks other than 0; page safety comes from the c' lease
        acquired on rank 0 and from each rank's own generation check, not from this export.
        """
        if self._standing is None:
            adapter = self.adapter
            exports = self._export_when_idle(adapter, 30.0)
            ids = {h.region_id: i for i, h in enumerate(adapter.memories)}
            self._standing = wire_copy({
                'contract': self.contract, 'capacity': adapter.capacity,
                'page_bytes': {b.name: b.page_bytes for b in adapter.layout.buffers},
                'bindings': {k: [[s.buffer_offset, s.length, ids[s.memory.region_id], s.region_offset]
                                 for s in spans] for k, spans in adapter.bindings.items()},
                'regions': [d.to_dict() for d in exports]})
        return self._standing

    def _offer(self, request_id, lease, fingerprint):
        exports = []
        try:
            adapter = self.adapters[lease.tier]
            exports = adapter.engine.export_memories(adapter.memories, 'read')
            return self._offer_from(request_id, lease, fingerprint, exports)
        except BaseException:
            # No descriptor escaped this synchronous operation.
            self.adapter.engine.release_exports([d.lease_id for d in exports])
            self.cache.release_source_lease(lease, drained=True)
            raise

    def _offer_from(self, request_id, lease, fingerprint, exports):
        if lease.tier not in self._static_offers:
            adapter = self.adapters[lease.tier]
            ids = {h.region_id:i for i,h in enumerate(adapter.memories)}
            # hbm pages are radix-locked (or optimistic under step); egm pages are held by host_lock_ref.
            protection = ('host-tree-pin' if lease.tier == 'egm'
                          else 'optimistic' if lease.protection == 'step' else 'tree-pin')
            self._static_offers[lease.tier] = (wire_copy({
                'contract': self.contract, 'capacity': adapter.capacity, 'tier': lease.tier,
                'protection': protection,
                'bindings': {k:[[s.buffer_offset, s.length, ids[s.memory.region_id], s.region_offset]
                                for s in spans] for k,spans in adapter.bindings.items()}}),
                [d.to_dict() for d in exports])
        static_offer, region_templates = self._static_offers[lease.tier]
        receipt = {'request_id': request_id, 'incarnation': lease.incarnation,
                   'epoch': lease.epoch, 'lease_id': lease.lease_id, 'fingerprint': fingerprint}
        # Every field is already a JSON value (str/int), so no deep copy is needed.
        offer = dict(static_offer, receipt=receipt, page_runs=page_runs(lease.page_indices),
                     regions=[dict(t, lease_id=d.lease_id) for t, d in zip(region_templates, exports)])
        if getattr(lease, 'protection', 'tree') == 'step':
            # TP>1 readers check every rank's own generations against this snapshot.
            offer['generations'] = list(lease.generations)
        self.active[request_id] = {'receipt':receipt, 'offer':offer, 'lease':lease, 'exports':exports}
        return offer

    def release(self, request):
        receipt = request.get('receipt')
        # unmapped: the target closed its import. drained: its DMA finished and it keeps a
        # retained mapping of the region (D6); the lease no longer covers any access.
        if not isinstance(receipt, dict) or not (request.get('unmapped') is True or request.get('drained') is True):
            raise RemoteRejected('exact receipt and unmapped or drained attestation required')
        rid = receipt.get('request_id')
        if rid in self.released:
            if self.released[rid] != receipt:
                raise RemoteRejected('altered released receipt')
            return self.released_answers[rid]
        entry = self.active.get(rid)
        if entry is None or entry['receipt'] != receipt:
            raise RemoteRejected('stale or altered source receipt')
        self.adapter.engine.release_exports([d.lease_id for d in entry['exports']])
        self.cache.release_source_lease(entry['lease'], drained=True)
        # valid: an optimistic read saw no reuse of its pages; a locked lease is always valid.
        answer = {'released': True, 'valid': self.cache.source_leases.lease_valid(entry['lease'])}
        self.released[rid] = receipt
        self.released_answers[rid] = answer
        del self.active[rid]
        return answer


class PeerReceiver:
    """One serialized prepare at a time. Keep this object on every exception.

    Unknown acquire/import/prepare failures retain reservation, mappings and the
    remote lease (if created); the caller must fail-stop both peers. Only explicit
    pre-acquire rejections are automatically discarded. No silent fallback.
    """
    def __init__(self, cache, adapter, identity, rpc):
        self.cache, self.adapter, self.rpc = cache, adapter, rpc
        self.contract = wire_copy(contract(identity, adapter.layout))
        self.pending = None

    def prepare(self, tokens):
        steps = self.prepare_steps(tokens)
        try:
            operation, payload = next(steps)
            try:
                answer = self.rpc.call(operation, **payload)
            except BaseException as exc:
                steps.throw(exc)
                raise
            steps.send(answer)
        except StopIteration as done:
            return done.value
        raise RuntimeError('unexpected preparation step')

    def prepare_steps(self, tokens, *, required_capacity=None, headroom=0, private=False, tier='hbm'):
        """Yield network work; every cache/transport operation stays on the owner."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.target_reservation import TargetReservation
        if self.pending is not None:
            raise RuntimeError('receiver has an outstanding restore')
        if not isinstance(tokens, list) or any(type(x) is not int or not 0 <= x < 2**31 for x in tokens):
            raise ValueError('expected token IDs')
        page = self.adapter.layout.page_size
        complete = tokens[:len(tokens)//page*page]
        if not complete:
            raise ValueError('no complete prefix page')
        reservation = TargetReservation(self.cache, RadixKey(array('q', complete)),
                                        **({'required_capacity': required_capacity, 'headroom': headroom}
                                           if required_capacity is not None else {}))
        self.pending = {'reservation':reservation, 'imports':[], 'offer':None, 'transaction':None}
        publish = reservation.ready_private if private else reservation.publish
        if reservation.prefix_len == len(complete):
            publish()
            return self.pending  # all-local hit: no RPC/data transfer
        try:
            key = prefix_page_keys(complete, page)[-1]
            start, end = reservation.prefix_len, len(complete)
            offer = yield 'acquire', dict(key=key, end=end, start=start, tier=tier, contract=self.contract)
        except RemoteRejected:
            reservation.discard(); self.pending = None
            raise
        self.pending['offer'] = offer
        expected = digest({'contract': self.contract, 'key': key, 'end': end, 'start': start, 'tier': tier})
        if (offer['contract'] != self.contract or offer['receipt']['fingerprint'] != expected
                or offer.get('tier') != tier):
            raise ValueError('offer identity mismatch; fail-stop')
        runs = offer['page_runs']
        first_page = 1 if tier == 'hbm' else 0  # device page 0 is padding; host page 0 is real
        if not isinstance(runs, list) or any(not isinstance(r, list) or len(r) != 2
                or any(type(x) is not int for x in r) or r[0] < first_page or r[1] <= 0 for r in runs):
            raise ValueError('invalid source page runs; fail-stop')
        pages = [first + i for first, count in runs for i in range(count)]
        if len(pages)*page != end - start or len(set(pages)) != len(pages):
            raise ValueError('invalid source pages; fail-stop')
        descriptors = [RegionDescriptor.from_dict(d) for d in offer['regions']]
        if not descriptors or len({d.owner_id for d in descriptors}) != 1 or any(d.access != 'read' for d in descriptors):
            raise ValueError('invalid source region ownership; fail-stop')
        plan = plan_transfer(self.adapter.layout, self.adapter.layout,
            pages, reservation.pages,
            source_capacity=offer['capacity'], target_capacity=self.adapter.capacity)
        needed, planned = plan_remote_regions(descriptors, offer['bindings'], plan, self.adapter.bindings)
        imported = self.pending['imports']
        materialized = {}
        for d in needed:
            memory = self.adapter.engine.import_memory(d)
            imported.append(memory)  # retain every successful import on any later failure
            materialized[d.region_id] = memory
        requests = tuple(TransferRequest(r.opcode, r.local, materialized[r.remote.region_id],
                                         r.local_offset, r.remote_offset, r.length) for r in planned)
        def release():
            answer = self.rpc.call('release', receipt=offer['receipt'],
                                   **({'unmapped': True} if transaction.retained_count == 0 else {'drained': True}))
            if answer is None:  # asynchronous release ACK is still pending
                return False
            if answer.get('released') is not True:
                raise RuntimeError('source release not confirmed')
            return answer
        if offer.get('protection') not in ('tree-pin', 'host-tree-pin', 'optimistic'):
            raise ValueError('unknown source protection; fail-stop')
        transaction = self.pending['transaction'] = RestoreTransaction(self.adapter.engine, requests, imported,
            publish=publish, discard=reservation.discard, release_source=release,
            keepalive=(reservation, self.adapter, offer),
            validate_on_release=offer['protection'] == 'optimistic')
        return self.pending

    def finish_consumer(self):
        if self.pending is None:
            return
        tx = self.pending['transaction']
        if tx is not None and not tx.source_released:
            raise RuntimeError('source not released; fail-stop or finish transfer first')
        reservation = self.pending['reservation']
        if reservation.state == 'READY_PRIVATE':
            reservation.discard()  # cancelled before normal request allocation
        if reservation.state == 'ADOPTED':
            self.pending = None  # normal request lifecycle owns every page
            return
        if reservation.state not in ('PUBLISHED', 'DISCARDED', 'RELEASED'):
            raise RuntimeError('target not safely finished')
        reservation.release()
        self.pending = None
