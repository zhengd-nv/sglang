#!/usr/bin/env python3
"""Real GPU producer dependency: delayed writes, explicit stream, no device sync."""
import ctypes as C
import json
import torch
from sglang.srt.mem_cache.l1_transfer.transport import (
    NvlinkEngine, Opcode, TransferRequest, TransferState, BusyError,
)


def main():
    owner, sender = NvlinkEngine(0), NvlinkEngine(1)
    size = (1 << 20) + 37
    sink = owner.allocate_memory(size)
    owner.upload(sink, b'\xa5' * size)
    desc = owner.export_memory(sink, 'write')
    remote = sender.import_memory(desc)
    source = sender.allocate_memory(size)
    sender.upload(source, b'\x00' * size)
    cuda = C.CDLL('libcuda.so.1')
    cuda.cuMemsetD8Async.argtypes = [C.c_uint64, C.c_ubyte, C.c_size_t, C.c_void_p]
    cuda.cuMemsetD8Async.restype = C.c_int
    zero_timeouts = 0
    stream = torch.cuda.Stream(device=1)
    values = (0x37, 0x49, 0x51, 0x62)
    chunk = 1 << 18
    for index, value in enumerate(values):
        with torch.cuda.device(1), torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            assert cuda.cuMemsetD8Async(source.address, value, size, stream.cuda_stream) == 0
        batch = sender.submit_transfer([
            TransferRequest(Opcode.WRITE, source, remote, 7, 11 + index * chunk, chunk)
        ], producer_stream=stream.cuda_stream)
        try:
            sender.wait(batch, timeout=0)
        except TimeoutError:
            zero_timeouts += 1
            try:
                sender.unregister_memory(source)
            except BusyError:
                pass
            else:
                raise AssertionError('timeout relinquished source ownership')
        result = sender.wait(batch, timeout=30)
        assert result.state == TransferState.SUCCESS
        sender.release_batch(batch)
        # Transfer completion must also imply its preceding producer has completed.
        assert stream.query()
    sender.unregister_memory(remote)
    owner.release_export(desc.lease_id)
    expected = bytearray(b'\xa5' * size)
    for index, value in enumerate(values):
        expected[11+index*chunk:11+(index+1)*chunk] = bytes([value]) * chunk
    assert owner.download(sink) == expected
    sender.unregister_memory(source); owner.unregister_memory(sink)
    sender.close(); owner.close()
    # A delayed/scheduled host can miss the early timeout window; the deterministic
    # native fault test separately guarantees timeout coverage independent of timing.
    print(json.dumps({'pass': True, 'dependent_batches': 4, 'observed_zero_timeouts': zero_timeouts}))


if __name__ == '__main__':
    main()
