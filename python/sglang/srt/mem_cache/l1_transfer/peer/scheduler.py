"""Explicit fixed-peer restore operations on an idle TP1 scheduler thread."""
import json
import logging
import os
import socket

import torch

from .control import ModelIdentity, SocketRPC, serve_once
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

    def stats(self):
        allocator = self.cache.token_to_kv_pool_allocator
        return dict(source_leases=len(self.source.active), target_active=self.receiver is not None,
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
            self.receiver.finish_consumer()
            self.receiver = None
        return self.stats()


def scheduler_peer_rpc(scheduler, action, payload):
    # Requests are intentionally explicit and serialized; callers must finish
    # generate() before issuing a control operation on the same Engine.
    if not scheduler.is_fully_idle():
        raise RuntimeError('Peer control requires an idle Engine')
    options = json.loads(payload)
    if action == 'init':
        if hasattr(scheduler, '_l1_peer'):
            raise RuntimeError('Peer already initialized')
        scheduler._l1_peer = SchedulerPeer(scheduler, options['identity'])
        result = scheduler._l1_peer.stats()
    else:
        if action not in ('listen', 'serve', 'restore', 'finish', 'stats'):
            raise ValueError('unknown Peer action')
        result = getattr(scheduler._l1_peer, action)(**options)
    return json.dumps(result)
