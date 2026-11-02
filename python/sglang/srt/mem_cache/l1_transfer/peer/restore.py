# SPDX-License-Identifier: Apache-2.0
"""Serialized, fail-stop READ transaction for an already prepared restore.

The caller acquires a ready source lease, reserves private destination pages and
imports regions before constructing this object. Ownership transfers only after
successful construction. No RPC, prefix matching, model identity or allocation
policy lives here. Retain the transaction until terminal; quarantine requires
stopping the affected Engine, never timeout-based reclamation.
"""
from enum import Enum
import os
import threading

from sglang.srt.mem_cache.l1_transfer.transport.api import Opcode, TransferRequest, TransferState


class RestoreState(str, Enum):
    PREPARED = 'PREPARED'
    IN_FLIGHT = 'IN_FLIGHT'
    RELEASING = 'RELEASING'
    PUBLISHED = 'PUBLISHED'
    DISCARDED = 'DISCARDED'
    QUARANTINED = 'QUARANTINED'


class RestoreTransaction:
    """Own one READ batch, its exclusive imports and private target reservation.

    publish() adopts target pages into the cache; discard() returns private pages.
    Neither may run concurrently with cache mutations. release_source() must ACK
    the exact source receipt and export leases, only after imports are unmapped.
    A callback exception can mean partial side effects: fail-stop, never retry or
    roll back. In particular, a publish exception MUST NOT trigger discard.
    Keepalive retains pool/reservation/lease owners; it does not enforce page pin.
    Source ready and target exclusive ownership remain adapter prerequisites.
    """
    def __init__(self, engine, requests, imports, *, publish, discard,
                 release_source, keepalive=(), validate_on_release=False):
        requests, imports = tuple(requests), tuple(imports)
        if not requests or any(not isinstance(r, TransferRequest) or r.opcode is not Opcode.READ
                               for r in requests):
            raise ValueError('restore requires a nonempty READ plan')
        if len({id(h) for h in imports}) != len(imports):
            raise ValueError('duplicate imported handle')
        if not {id(r.remote) for r in requests} <= {id(h) for h in imports}:
            raise ValueError('imports must cover the READ plan')
        if any(not h.remote for h in imports) or any(r.local.remote for r in requests):
            raise ValueError('expected local target and imported source')
        if not all(callable(f) for f in (publish, discard, release_source)):
            raise TypeError('ownership callbacks must be callable')
        self.engine, self.requests, self.imports = engine, requests, imports
        self._publish, self._discard, self._release_source = publish, discard, release_source
        self.keepalive = tuple(keepalive)
        self.state = RestoreState.PREPARED
        self.batch = None
        self.status = None
        self.error = ''
        self.published = False
        self.discarded = False
        self.source_released = False
        self.cancel_requested = False
        # Optimistic source: publish only after the release ACK says the read was intact.
        self.validate_on_release = validate_on_release
        self.copied = False
        self.stale = False
        self.unmapped_count = 0
        self.retained_count = 0
        self._context = (os.getpid(), threading.get_ident())
        self._busy = False

    def _enter(self):
        if self._context != (os.getpid(), threading.get_ident()) or self._busy:
            raise RuntimeError('restore requires its serialized owner thread; no reentry')
        if self.state == RestoreState.QUARANTINED:
            raise RuntimeError('restore quarantined; affected Engine must stop')
        self._busy = True

    def _quarantine(self, exc):
        self.error = str(exc)
        self.state = RestoreState.QUARANTINED

    def start(self, *, producer_stream=None):
        self._enter()
        try:
            if self.state != RestoreState.PREPARED:
                raise RuntimeError('restore already submitted or finished')
            # An exception may occur after partial native submission. Even a
            # validation exception is conservatively fail-stop with this API.
            try:
                self.batch = self.engine.submit_transfer(self.requests,
                    **({"producer_stream": producer_stream} if producer_stream is not None else {}))
                self.state = RestoreState.IN_FLIGHT
            except BaseException as exc:
                self._quarantine(exc)
                raise
        finally:
            self._busy = False

    def _finish(self, success):
        try:
            if self.batch is not None:
                self.engine.release_batch(self.batch)
            for handle in self.imports:
                if self.engine.retains(handle):
                    self.retained_count += 1  # shared across restores; drained is the proof
                    continue
                self.engine.unregister_memory(handle)
                self.unmapped_count += 1
            self.copied = success
            if not self.validate_on_release:
                self._settle(success)
            self.state = RestoreState.RELEASING
            return self._poll_release()
        except BaseException as exc:
            self._quarantine(exc)
            raise
        return self.state

    def _settle(self, success):
        if success:
            self._publish()
            self.published = True
        else:
            self._discard()
            self.discarded = True

    def _poll_release(self):
        try:
            # False means ACK pending; None preserves the synchronous callback API.
            answer = self._release_source()
            if answer is not False:
                self.source_released = True
                if self.validate_on_release:
                    if type(answer) is not dict or type(answer.get('valid')) is not bool:
                        raise RuntimeError('optimistic source release must report validity')
                    self.stale = self.copied and not answer['valid']
                    self._settle(self.copied and answer['valid'])
                self.state = RestoreState.PUBLISHED if self.published else RestoreState.DISCARDED
        except BaseException as exc:
            self._quarantine(exc)
            raise
        return self.state

    def poll(self):
        """Observe one device event / release ACK without waiting for either."""
        return self.wait(_poll=True)

    def wait(self, timeout=None, *, _poll=False):
        self._enter()
        try:
            if self.state in (RestoreState.PUBLISHED, RestoreState.DISCARDED):
                return self.state  # duplicate completion performs no callbacks
            if self.state == RestoreState.RELEASING:
                return self._poll_release()
            if self.state != RestoreState.IN_FLIGHT:
                raise RuntimeError('restore has not been submitted')
            try:
                self.status = (self.engine.get_transfer_status(self.batch) if _poll
                               else self.engine.wait(self.batch, timeout))
            except TimeoutError:
                raise  # observation only; retry wait, never reclaim
            except BaseException as exc:
                self._quarantine(exc)
                raise
            if self.status.state == TransferState.IN_FLIGHT:
                return self.state
            if not self.status.drained:
                self._quarantine(self.status.error or 'transfer completion is uncertain')
                return self.state
            if self.status.state == TransferState.SUCCESS:
                expected = sum(r.length for r in self.requests)
                if self.status.requested_bytes != expected or self.status.completed_bytes != expected:
                    self._quarantine('inconsistent successful byte count')
                    return self.state
            return self._finish(self.status.state == TransferState.SUCCESS and not self.cancel_requested)
        finally:
            self._busy = False

    def cancel(self):
        """Record intent; submitted copies retain all owners until proven drain.

        Publication cannot be rolled back. A cancellation after publication
        waits for the existing release ACK, then the caller releases its consumer.
        """
        self._enter()
        try:
            self.cancel_requested = True
            if self.state in (RestoreState.DISCARDED, RestoreState.PUBLISHED):
                return self.state
            if self.state == RestoreState.PREPARED:
                return self._finish(False)
            return self.state
        finally:
            self._busy = False
