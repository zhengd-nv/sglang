# SPDX-License-Identifier: Apache-2.0
"""Network workers only. Cache, CUDA and ownership stay on the scheduler thread."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
import socket
import threading
import time

from .control import RemoteRejected, SocketRPC, serve_once


class AsyncRPC:
    def __init__(self, host, port, *, executor=None):
        self.address = (host, port)
        self.owns_executor = executor is None
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='l1-rpc')
        self.connection = None
        self.rpc = None
        self.release_future = None

    def _call(self, operation, payload):
        if self.connection is None:
            try:
                self.connection = socket.create_connection(self.address, timeout=120)
            except OSError as exc:
                # Nothing was sent, so the source cannot hold a lease for this request.
                raise RemoteRejected(f'source unreachable: {exc}') from exc
            self.rpc = SocketRPC(self.connection)
        return self.rpc.call(operation, **payload)

    def submit(self, operation, payload):
        return self.executor.submit(self._call, operation, payload)

    def call(self, operation, **payload):
        # PeerReceiver's release callback is polled on the scheduler thread.
        if operation != 'release':
            raise RuntimeError('acquire must be submitted asynchronously')
        if self.release_future is None:
            self.release_future = self.submit(operation, payload)
        if not self.release_future.done():
            return None
        return self.release_future.result()

    def close(self):
        # Only after an acknowledged release or safe acquire rejection.
        if self.connection is not None:
            self.connection.close()
        if self.owns_executor:
            self.executor.shutdown(wait=False)


class SourceService:
    """One thread per connection; each connection carries one acquire/release exchange.

    Each dispatch is handed to the cache owner via a bounded queue. A worker
    timeout is an error report, never permission to reclaim the source lease.
    The daemon is tied to Engine lifetime; no independent service shutdown API.
    """
    def __init__(self, listener):
        self.listener = listener
        self.events = Queue(maxsize=64)
        listener.settimeout(None)
        threading.Thread(target=self._run, name='l1-source', daemon=True).start()

    def _run(self):
        while True:
            conn, _ = self.listener.accept()
            threading.Thread(target=self._serve, args=(conn,), name='l1-source-conn', daemon=True).start()

    def _serve(self, conn):
        service = self

        class Dispatch:
            lease_id = None

            def dispatch(self, request):
                future = Future()
                future.enqueued = time.time()  # wall clock, for source lease timing samples
                service.events.put(('request', request, future))
                result = future.result(timeout=120)
                if request.get('operation') == 'acquire':
                    # The RPC id of an acquire is the lease request id.
                    self.lease_id = request.get('id')
                return result

        endpoint = Dispatch()
        try:
            with conn:
                conn.settimeout(120)
                serve_once(conn, endpoint)
                if endpoint.lease_id is not None:
                    serve_once(conn, endpoint)
        except BaseException as exc:
            self.events.put(('error', exc, None))
        finally:
            # The owner checks that this connection's lease was released.
            self.events.put(('closed', endpoint.lease_id, None))
