# SPDX-License-Identifier: Apache-2.0
"""Fixed-peer source endpoint and synchronous receiver preparation."""
from dataclasses import asdict
from array import array

from sglang.srt.mem_cache.l1_transfer.transport import Opcode, RegionDescriptor
from sglang.srt.mem_cache.l1_transfer.transport.layout import RegionSpan, plan_transfer, bind_regions
from .control import contract, digest, wire_copy, RemoteRejected
from .restore import RestoreTransaction


class SourceEndpoint:
    def __init__(self, cache, adapter, identity, ready):
        self.cache, self.adapter, self.ready = cache, adapter, ready
        self.contract = wire_copy(contract(identity, adapter.layout))
        self.active = {}
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

    def acquire(self, request):
        from sglang.srt.mem_cache.radix_cache import RadixKey
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
        fingerprint = digest({'contract': self.contract, 'tokens': tokens})
        if request_id in self.active:
            entry = self.active[request_id]
            if entry['receipt']['fingerprint'] != fingerprint:
                raise RemoteRejected('request ID reused with different prefix')
            return entry['offer']
        if request_id in self.released:
            raise RemoteRejected('request already released')
        try:
            lease = self.cache.acquire_source_lease(RadixKey(array('q', tokens)), fingerprint=fingerprint, ready=self.ready)
        except ValueError as exc:
            raise RemoteRejected(str(exc)) from exc
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
        reservation = TargetReservation(self.cache, RadixKey(array('q', complete)))
        self.pending = {'reservation':reservation, 'imports':[], 'offer':None, 'transaction':None}
        if reservation.prefix_len == len(complete):
            reservation.publish()
            return self.pending  # all-local hit: no RPC/data transfer
        try:
            offer = self.rpc.call('acquire', tokens=complete, contract=self.contract)
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
        imported = self.pending['imports']
        for d in descriptors:
            imported.append(self.adapter.engine.import_memory(d))
        bindings = {k:tuple(RegionSpan(a,n,imported[i],off) for a,n,i,off in spans)
                    for k,spans in offer['bindings'].items()}
        plan = plan_transfer(self.adapter.layout, self.adapter.layout,
            pages[reservation.prefix_len//page:], reservation.pages,
            source_capacity=offer['capacity'], target_capacity=self.adapter.capacity)
        requests = bind_regions(plan, bindings, self.adapter.bindings, Opcode.READ)
        def release():
            answer = self.rpc.call('release', receipt=offer['receipt'], unmapped=True)
            if answer.get('released') is not True:
                raise RuntimeError('source release not confirmed')
        self.pending['transaction'] = RestoreTransaction(self.adapter.engine, requests, imported,
            publish=reservation.publish, discard=reservation.discard, release_source=release,
            keepalive=(reservation, self.adapter, offer))
        return self.pending

    def finish_consumer(self):
        if self.pending is None:
            return
        tx = self.pending['transaction']
        if tx is not None and not tx.source_released:
            raise RuntimeError('source not released; fail-stop or finish transfer first')
        reservation = self.pending['reservation']
        if reservation.state not in ('PUBLISHED', 'DISCARDED', 'RELEASED'):
            raise RuntimeError('target not safely finished')
        reservation.release()
        self.pending = None
