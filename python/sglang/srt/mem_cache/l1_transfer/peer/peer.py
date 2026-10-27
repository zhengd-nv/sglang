# SPDX-License-Identifier: Apache-2.0
"""Fixed-peer source endpoint and synchronous receiver preparation."""
from dataclasses import asdict
from array import array

from sglang.srt.mem_cache.l1_transfer.transport import Opcode, RegionDescriptor
from sglang.srt.mem_cache.l1_transfer.transport.api import MemoryHandle, TransferRequest
from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan, plan_transfer, bind_regions
from .control import contract, digest, wire_copy, RemoteRejected
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
    def __init__(self, cache, adapter, identity, ready, *, max_leases=None, max_lease_pages=None):
        self.cache, self.adapter, self.ready = cache, adapter, ready
        self.contract = wire_copy(contract(identity, adapter.layout))
        self.max_leases = max_leases
        self.max_lease_pages = max_lease_pages
        self.active = {}
        self.pending = {}  # request_id -> (lease, fingerprint, producer event); nothing exported
        self.released = {}  # bounded by endpoint lifetime; no TTL reclamation

    def dispatch(self, request):
        self.cache.source_leases._check_thread()
        if request.get('version') != 1:
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
        if request.get('version') != 1:
            raise RemoteRejected('unsupported protocol version')
        if request.get('operation') == 'acquire':
            self.begin_acquire(request, record_producer=record_producer)
            return None
        return self.dispatch(request)

    def _validate_acquire(self, request):
        if request.get('contract') != self.contract:
            raise RemoteRejected('incompatible model/context/layout')
        tokens = request.get('tokens')
        page = self.adapter.layout.page_size
        if (not isinstance(tokens, list) or not tokens or len(tokens) % page or
                any(type(x) is not int or not 0 <= x < 2**31 for x in tokens)):
            raise RemoteRejected('complete token pages required')
        request_id = request.get('id')
        if not isinstance(request_id, str) or len(request_id) != 32:
            raise RemoteRejected('invalid request ID')
        if request_id in self.released:
            raise RemoteRejected('request already released')
        if request_id in self.pending:
            raise RemoteRejected('request ID already pending')
        return tokens, request_id, digest({'contract': self.contract, 'tokens': tokens})

    def _check_quota(self, tokens):
        leases = list(self.active.values()) + [{'lease': p[0]} for p in self.pending.values()]
        if self.max_leases is not None and len(leases) >= self.max_leases:
            raise RemoteRejected('source busy: lease quota')
        if self.max_lease_pages is not None:
            pages = len(tokens) // self.adapter.layout.page_size
            held = sum(len(item['lease'].page_indices) for item in leases)
            if held + pages > self.max_lease_pages:
                raise RemoteRejected('source busy: leased page quota')

    def _lock(self, tokens, fingerprint, ready):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        try:
            return self.cache.acquire_source_lease(RadixKey(array('q', tokens)), fingerprint=fingerprint, ready=ready)
        except ValueError as exc:
            raise RemoteRejected(str(exc)) from exc

    def acquire(self, request):
        tokens, request_id, fingerprint = self._validate_acquire(request)
        if request_id in self.active:
            entry = self.active[request_id]
            if entry['receipt']['fingerprint'] != fingerprint:
                raise RemoteRejected('request ID reused with different prefix')
            return entry['offer']
        self._check_quota(tokens)
        return self._offer(request_id, self._lock(tokens, fingerprint, self.ready), fingerprint)

    def begin_acquire(self, request, *, record_producer):
        """Lock the prefix and record the producer event; the offer follows in poll_ready."""
        tokens, request_id, fingerprint = self._validate_acquire(request)
        if request_id in self.active:
            raise RemoteRejected('request ID already acquired')
        self._check_quota(tokens)
        events = []
        lease = self._lock(tokens, fingerprint, lambda: events.append(record_producer()))
        self.pending[request_id] = (lease, fingerprint, events[0])
        return request_id

    def poll_ready(self):
        """Return [(request_id, offer)] whose producer work has completed."""
        done = [rid for rid, (_, _, event) in self.pending.items() if event.query()]
        offers = []
        for rid in done:
            lease, fingerprint, _ = self.pending.pop(rid)
            offers.append((rid, self._offer(rid, lease, fingerprint)))
        return offers

    def _offer(self, request_id, lease, fingerprint):
        exports = []
        try:
            for h in self.adapter.memories:
                exports.append(self.adapter.engine.export_memory(h, 'read'))
            receipt = {'request_id': request_id, 'incarnation': lease.incarnation,
                       'epoch': lease.epoch, 'lease_id': lease.lease_id, 'fingerprint': fingerprint}
            ids = {h.region_id:i for i,h in enumerate(self.adapter.memories)}
            offer = {'receipt':receipt, 'contract':self.contract,
                     'pages': list(lease.page_indices), 'capacity': self.adapter.capacity,
                     'regions':[d.to_dict() for d in exports],
                     'bindings':{k:[[s.buffer_offset, s.length, ids[s.memory.region_id], s.region_offset]
                                    for s in spans] for k,spans in self.adapter.bindings.items()}}
            offer = wire_copy(offer)
            self.active[request_id] = {'receipt':receipt, 'offer':offer, 'lease':lease, 'exports':exports}
            return offer
        except BaseException:
            # No descriptor escaped this synchronous operation.
            for d in exports: self.adapter.engine.release_export(d.lease_id)
            self.cache.release_source_lease(lease, drained=True)
            raise

    def release(self, request):
        receipt = request.get('receipt')
        if not isinstance(receipt, dict) or request.get('unmapped') is not True:
            raise RemoteRejected('exact receipt and unmapped attestation required')
        rid = receipt.get('request_id')
        if rid in self.released:
            if self.released[rid] != receipt:
                raise RemoteRejected('altered released receipt')
            return {'released':True}
        entry = self.active.get(rid)
        if entry is None or entry['receipt'] != receipt:
            raise RemoteRejected('stale or altered source receipt')
        for d in entry['exports']: self.adapter.engine.release_export(d.lease_id)
        self.cache.release_source_lease(entry['lease'], drained=True)
        self.released[rid] = receipt
        del self.active[rid]
        return {'released':True}


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

    def prepare_steps(self, tokens, *, required_capacity=None, headroom=0, private=False):
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
            offer = yield 'acquire', dict(tokens=complete, contract=self.contract)
        except RemoteRejected:
            reservation.discard(); self.pending = None
            raise
        self.pending['offer'] = offer
        expected = digest({'contract':self.contract, 'tokens':complete})
        if offer['contract'] != self.contract or offer['receipt']['fingerprint'] != expected:
            raise ValueError('offer identity mismatch; fail-stop')
        pages = offer['pages']
        if len(pages)*page != len(complete) or len(set(pages)) != len(pages) or any(type(x) is not int or x <= 0 for x in pages):
            raise ValueError('invalid source pages; fail-stop')
        descriptors = [RegionDescriptor.from_dict(d) for d in offer['regions']]
        if not descriptors or len({d.owner_id for d in descriptors}) != 1 or any(d.access != 'read' for d in descriptors):
            raise ValueError('invalid source region ownership; fail-stop')
        plan = plan_transfer(self.adapter.layout, self.adapter.layout,
            pages[reservation.prefix_len//page:], reservation.pages,
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
            answer = self.rpc.call('release', receipt=offer['receipt'], unmapped=True)
            if answer is None:  # asynchronous release ACK is still pending
                return False
            if answer.get('released') is not True:
                raise RuntimeError('source release not confirmed')
        self.pending['transaction'] = RestoreTransaction(self.adapter.engine, requests, imported,
            publish=publish, discard=reservation.discard, release_source=release,
            keepalive=(reservation, self.adapter, offer))
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
