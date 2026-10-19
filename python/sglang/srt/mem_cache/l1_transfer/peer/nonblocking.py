# SPDX-License-Identifier: Apache-2.0
"""Network workers only. Cache, CUDA and ownership stay on the scheduler thread."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
import socket
import threading

from .control import SocketRPC, serve_once


class AsyncRPC:
    def __init__(self, host, port):
        self.address = (host, port)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='l1-rpc')
        self.connection = None
        self.rpc = None
        self.release_future = None

    def _call(self, operation, payload):
        if self.connection is None:
            self.connection = socket.create_connection(self.address, timeout=120)
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
        self.executor.shutdown(wait=False)


class SourceService:
    """One connection / lease at a time, persistent across request consumers.

    Each dispatch is handed to the cache owner via a bounded queue. A worker
    timeout is an error report, never permission to reclaim the source lease.
    The daemon is tied to Engine lifetime; no independent service shutdown API.
    """
    def __init__(self, listener):
        self.listener = listener
        self.events = Queue(maxsize=4)
        listener.settimeout(None)
        threading.Thread(target=self._run, name='l1-source', daemon=True).start()

    def _run(self):
        while True:
            conn, _ = self.listener.accept()
            service = self

            class Dispatch:
                acquired = False

                def dispatch(self, request):
                    future = Future()
                    service.events.put(('request', request, future))
                    result = future.result(timeout=120)
                    self.acquired = request.get('operation') == 'acquire'
                    return result

            endpoint = Dispatch()
            try:
                with conn:
                    conn.settimeout(120)
                    serve_once(conn, endpoint)
                    if endpoint.acquired:
                        serve_once(conn, endpoint)
            except BaseException as exc:
                self.events.put(('error', exc, None))
            finally:
                self.events.put(('closed', None, None))
