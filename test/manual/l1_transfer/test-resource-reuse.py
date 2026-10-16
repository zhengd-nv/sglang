#!/usr/bin/env python3
"""Real CUDA regression: concurrent batches, reverse waits and pool teardown."""
from sglang.srt.mem_cache.l1_transfer.transport import (
    NvlinkEngine, Opcode, TransferRequest, TransferState,
)


def main():
    for generation in range(3):
        owner, reader = NvlinkEngine(0), NvlinkEngine(1)
        rows = []
        for i in range(12):
            source = owner.allocate_memory(32771)
            data = bytes((j + i + generation) % 251 for j in range(source.size))
            owner.upload(source, data)
            desc = owner.export_memory(source, 'read')
            remote = reader.import_memory(desc)
            target = reader.allocate_memory(source.size)
            rows.append((source, desc, remote, target, data))
        for cycle in range(10):
            batches = []
            for _, _, remote, target, _ in rows:
                reader.upload(target, b'\xa5' * target.size)
                batches.append(reader.submit_transfer([
                    TransferRequest(Opcode.READ, target, remote, 1, 3, 16383),
                    TransferRequest(Opcode.READ, target, remote, 20001, 17003, 12001),
                ]))
            # All 12 pairs must stay distinct until their batches are released.
            for batch, (_, _, _, target, data) in reversed(list(zip(batches, rows))):
                status = reader.wait(batch, None if cycle % 2 else 30)
                assert status.state == TransferState.SUCCESS
                reader.release_batch(batch)
                expected = bytearray(b'\xa5' * target.size)
                expected[1:16384] = data[3:16386]
                expected[20001:32002] = data[17003:29004]
                assert reader.download(target) == expected
        for source, desc, remote, target, _ in rows:
            reader.unregister_memory(remote)
            reader.unregister_memory(target)
            owner.release_export(desc.lease_id)
            owner.unregister_memory(source)
        reader.close(); owner.close()
    # READ into local memory, then WRITE those bytes to a different remote region.
    # Batch memcpy may reorder copies, so this path must retain stream ordering.
    owner, reader = NvlinkEngine(0), NvlinkEngine(1)
    source, sink = owner.allocate_memory(1 << 20), owner.allocate_memory(1 << 20)
    data = bytes(range(251)) * 4177 + b'last'
    data = data[:1 << 20]
    data += b'x' * ((1 << 20) - len(data))
    owner.upload(source, data)
    source_desc = owner.export_memory(source, 'read')
    sink_desc = owner.export_memory(sink, 'write')
    remote_source, remote_sink = reader.import_memory(source_desc), reader.import_memory(sink_desc)
    local = reader.allocate_memory(1 << 20)
    for _ in range(20):
        reader.upload(local, b'\xa5' * local.size)
        h = reader.submit_transfer([
            TransferRequest(Opcode.READ, local, remote_source, 0, 0, local.size),
            TransferRequest(Opcode.WRITE, local, remote_sink, 0, 0, local.size),
        ])
        assert reader.wait(h, 30).state == TransferState.SUCCESS
        reader.release_batch(h)
    reader.unregister_memory(remote_source); reader.unregister_memory(remote_sink)
    owner.release_export(source_desc.lease_id); owner.release_export(sink_desc.lease_id)
    assert owner.download(sink) == data
    reader.unregister_memory(local)
    owner.unregister_memory(source); owner.unregister_memory(sink)
    reader.close(); owner.close()
    print('PASS: 360 batches; 12 simultaneous; 3 context-pool lifetimes; 20 dependent READ/WRITE batches')


if __name__ == '__main__':
    main()
