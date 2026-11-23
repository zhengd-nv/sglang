"""KVCM placement queries: which reporter holds how many root-prefix pages.

Port of the harness ``Directory`` (test/manual/l1_transfer/test-glm52_nvfp4-sharing.py:663-696).
The in-tree KVCMClient (mem_cache/l1_transfer/kvcm_client.py) is not reused: it
needs an EngineDirectory/LookupQuery for a specific target engine and returns only
the best candidate, while the protocol needs the full {reporter: pages} map.
Keys are the native event chain (metadata.root_event_keys), so they match what
the indexer/bridge reported; ``local`` is the number of matching root-prefix pages.
"""

from typing import Optional, Sequence

PAGE = 64


class KVCMError(RuntimeError):
    pass


class KVCMDirectory:
    def __init__(
        self,
        endpoint: str,
        instance_id: str,
        *,
        page_size: int = PAGE,
        timeout: float = 3.0,
        channel=None,
    ):
        import grpc

        from sglang.srt.mem_cache.l1_transfer.generated import (
            tair_kvcache_meta_service_pb2 as pb,
        )
        from sglang.srt.mem_cache.l1_transfer.generated import (
            tair_kvcache_meta_service_pb2_grpc as rpc,
        )
        from sglang.srt.mem_cache.l1_transfer.metadata import root_event_keys

        self.pb, self._keys = pb, root_event_keys
        self.endpoint, self.instance_id = endpoint, instance_id
        self.page_size, self.timeout = page_size, timeout
        self.channel = channel or grpc.insecure_channel(endpoint)
        self.stub = rpc.MetaServiceStub(self.channel)

    def keys(self, tokens: Sequence[int]):
        return self._keys(list(tokens), self.page_size)

    def hosts(self, tokens: Sequence[int]) -> dict:
        response = self.stub.GetHostCacheState(
            self.pb.GetHostCacheStateRequest(
                instance_id=self.instance_id,
                trace_id="slotguard-repro-client",
                query_type=self.pb.QT_PREFIX_MATCH,
                block_cache_keys=self.keys(tokens),
                medium=["hbm"],
                p2p_host_count=0,
            ),
            timeout=self.timeout,
        )
        if response.header.status.code != self.pb.OK:
            raise KVCMError(f"KVCM status {response.header.status.code}")
        result = {h.host_ip_port: h.local for h in response.hosts}
        if len(result) != len(response.hosts):
            raise KVCMError("duplicate KVCM reporter")
        return result

    def close(self):
        self.channel.close()


def wait_hosts(
    directory,
    tokens,
    predicate,
    *,
    timeout: float = 30.0,
    poll_s: float = 0.1,
    clock=None,
    sleep=None,
) -> dict:
    """Poll ``directory.hosts(tokens)`` until ``predicate`` holds (harness wait_until)."""
    import time

    clock, sleep = clock or time.monotonic, sleep or time.sleep
    deadline = clock() + timeout
    last: Optional[dict] = None
    while True:
        last = directory.hosts(tokens)
        if predicate(last):
            return last
        if clock() >= deadline:
            raise KVCMError(f"KVCM placement deadline exceeded; last={last}")
        sleep(poll_s)
