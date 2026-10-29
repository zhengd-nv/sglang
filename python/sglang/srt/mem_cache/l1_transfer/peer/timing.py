# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostic spans; disabled unless SGLANG_L1_TRACE_DIR is set.

Nested spans are inclusive, not additive. perf_counter clocks are only comparable
inside one host. No CUDA synchronization is introduced by the recorder.
"""
import functools
import json
import os
import threading
import time
from pathlib import Path


def install(peer):
    directory = os.environ.get('SGLANG_L1_TRACE_DIR')
    if not directory:
        return
    from . import peer as peer_module
    from .nonblocking import AsyncRPC
    from .restore import RestoreTransaction
    from ..transport._driver import Driver
    from sglang.srt.mem_cache.target_reservation import TargetReservation

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    output = open(path / f'{os.getpid()}.jsonl', 'a', buffering=65536)
    peer._timing_output = output
    lock = threading.Lock()

    def wrap(obj, name, label, flush=False):
        original = getattr(obj, name)
        if getattr(original, '_l1_timed', False):
            return
        @functools.wraps(original)
        def timed(*args, **kwargs):
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            # Queue rejection may release the consumer inside _ready().
            ready_rid = peer.request.rid if label == 'target._ready' else None
            ready_tx = peer.session['transaction'] if label == 'target._ready' else None
            if label == 'target.start_request':
                peer._trace_request_start = (start, cpu)
            elif label in ('source.acquire', 'source.begin_acquire', 'source.acquire_offthread'):
                # Online sources overlap leases; lease_total then pairs with the latest begin.
                peer._trace_source_start = (start, cpu)
            try:
                return original(*args, **kwargs)
            finally:
                end, used = time.perf_counter_ns(), time.thread_time_ns() - cpu
                row = dict(name=label, start_ns=start, end_ns=end, cpu_ns=used,
                           tid=threading.get_ident())
                if label == 'target.start_request':
                    row['rid'] = args[0].rid
                elif label == 'target._ready':
                    row['rid'] = ready_rid
                    tx = ready_tx
                    row['ranges'] = len(tx.requests) if tx else 0
                    row['regions'] = len(tx.imports) if tx else 0
                    row['bytes'] = sum(r.length for r in tx.requests) if tx else 0
                    row['mapped_bytes'] = sum(h.size for h in tx.imports) if tx else 0
                with lock:
                    output.write(json.dumps(row, separators=(',', ':')) + '\n')
                    if label in ('target._ready', 'source.release', 'source.release_offthread'):
                        before, before_cpu = (peer._trace_request_start if label == 'target._ready'
                                              else peer._trace_source_start)
                        interval = dict(name='target.restore_total' if label == 'target._ready' else 'source.lease_total',
                                        start_ns=before, end_ns=end,
                                        cpu_ns=time.thread_time_ns()-before_cpu, tid=threading.get_ident())
                        if label == 'target._ready':
                            interval['rid'] = ready_rid
                        output.write(json.dumps(interval, separators=(',', ':')) + '\n')
                    if flush:
                        output.flush()
        timed._l1_timed = True
        setattr(obj, name, timed)

    for name in ('start_request', '_ready', 'finish', 'finish_request'):
        wrap(peer, name, 'target.' + name, name == 'finish_request')
    for name in ('acquire', 'release', 'ready', 'begin_acquire', 'poll_ready',
                 'acquire_offthread', 'release_offthread'):
        if getattr(peer.source, name) is not None:  # ready is None for an online source
            wrap(peer.source, name, 'source.' + name, name in ('release', 'release_offthread'))
    wrap(peer.cache.source_leases, 'acquire', 'source.lease')
    wrap(peer.cache.source_leases, 'acquire_by_key', 'source.lease')
    for name in ('__init__', 'publish', 'release', '_set_pins'):
        wrap(TargetReservation, name, 'reservation.' + name)
    for name in ('plan_transfer', 'bind_regions'):
        wrap(peer_module, name, name)
    for name in ('export_memory', 'import_memory', 'unregister_memory', 'submit_transfer', 'release_batch'):
        wrap(peer.engine, name, 'engine.' + name)
    for name in ('submit', 'import_region', 'close'):
        wrap(Driver, name, 'driver.' + name)
    wrap(AsyncRPC, '_call', 'rpc.call')
    wrap(RestoreTransaction, 'start', 'transaction.start')
