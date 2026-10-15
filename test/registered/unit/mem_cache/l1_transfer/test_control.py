# SPDX-License-Identifier: Apache-2.0
import socket
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor
from sglang.srt.mem_cache.l1_transfer.peer.control import (ModelIdentity, SocketRPC, RemoteRejected, digest,
                                 receive, send, serve_once, MAX_FRAME)


class ControlTests(unittest.TestCase):
    def test_identity_fields(self):
        identity = ModelIdentity('weights-sha',digest({'model':1}),digest({'rope':1}),'none',digest({'salt':1}))
        identity.validate()
        from dataclasses import replace
        for delta in ({'weights_revision':''},{'context_digest':'label'},{'tp_size':2},{'tp_rank':True}):
            with self.assertRaises(ValueError): replace(identity,**delta).validate()
        self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))

    def test_rpc_roundtrip_and_safe_rejection(self):
        class Endpoint:
            def dispatch(self,r):
                if r['operation']=='reject': raise RemoteRejected('mismatch')
                return {'echo':r['value']}
        a,b=socket.socketpair()
        with a,b,ThreadPoolExecutor(max_workers=1) as ex:
            rpc=SocketRPC(a)
            for op in ('echo','reject','echo'):
                future=ex.submit(serve_once,b,Endpoint())
                if op=='reject':
                    with self.assertRaises(RemoteRejected):rpc.call(op,value=3)
                else:self.assertEqual(rpc.call(op,value=3),{'echo':3})
                future.result()
            self.assertFalse(rpc.failed)

    def test_stale_response_quarantines_channel(self):
        a,b=socket.socketpair()
        def wrong():
            receive(b);send(b,{'version':1,'id':'stale','ok':True,'result':{}})
        with a,b,ThreadPoolExecutor(max_workers=1) as ex:
            future=ex.submit(wrong);rpc=SocketRPC(a)
            with self.assertRaises(ValueError):rpc.call('acquire')
            with self.assertRaises(RuntimeError):rpc.call('acquire')
            future.result()

    def test_disconnect_is_uncertain(self):
        a,b=socket.socketpair();b.close()
        with a:
            rpc=SocketRPC(a)
            with self.assertRaises(OSError):rpc.call('acquire')
            self.assertTrue(rpc.failed)

    def test_frame_limit(self):
        a,b=socket.socketpair()
        with a,b:
            a.sendall(struct.pack('!I',MAX_FRAME+1))
            with self.assertRaises(ValueError):receive(b)
            with self.assertRaises(ValueError):send(a,{'payload':'x'*MAX_FRAME})

    def test_remote_uncertain_error_is_not_safe_rejection(self):
        class Endpoint:
            def dispatch(self,r):raise RuntimeError('partial mutation')
        a,b=socket.socketpair()
        with a,b,ThreadPoolExecutor(max_workers=1) as ex:
            future=ex.submit(serve_once,b,Endpoint());rpc=SocketRPC(a)
            with self.assertRaises(RuntimeError):rpc.call('acquire')
            self.assertTrue(rpc.failed);future.result()


if __name__=='__main__':unittest.main()
