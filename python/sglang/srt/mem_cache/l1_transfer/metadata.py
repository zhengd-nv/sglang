# ruff: noqa: BLE001
# CUDA failures are converted to quarantine, validation failures to safe misses.
"""Internal deferred metadata, never accepted from the wire.

Scheduler captures tensor references and records events without waiting. Workers
may wait/copy on their own thread; a failed CUDA wait/copy is NOT a drain proof.
Frozen CPU token bytes avoid retaining any request, tree, node or allocator.
"""

from dataclasses import dataclass

from .int_vectors import from_trusted_ints


def root_event_keys(tokens, page_size):
    # Exactly the native event chain for the supported unsalted/no-LoRA POC.
    # Do not derive from storage keys or chain the truncated int64 event values.
    return event_keys_and_hashes(tokens, page_size)[0]


def event_keys_and_hashes(tokens, page_size):
    from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

    complete = tokens[: len(tokens) // page_size * page_size]
    hashes = tuple(get_hash_str(complete, page_size=page_size))
    return tuple(hash_str_to_int64(h) for h in hashes), hashes


@dataclass(frozen=True)
class DeferredSlots:
    indices: object
    events: tuple

    @classmethod
    def capture(cls, indices, producer_stream=None, *, retain=None):
        import torch

        events = []
        if retain is not None:
            retain(cls(indices, ()))
        if indices.is_cuda:
            if producer_stream is not None:
                event = torch.cuda.Event()
                events.append(event)
                if retain is not None:
                    retain(cls(indices, tuple(events)))
                event.record(producer_stream)
            event = torch.cuda.Event()
            events.append(event)
            if retain is not None:
                retain(cls(indices, tuple(events)))
            event.record(torch.cuda.current_stream(indices.device))
        return cls(indices, tuple(events))

    def materialize(self):
        """Worker only. Success certifies all metadata and producer accesses drained."""
        self.drain()
        values = tuple(self.indices.cpu().tolist())
        # int64 tensor elements are exact ints; registering the minimum lets
        # later PrivateSlotView checks skip a per-element pass.
        return from_trusted_ints(values, min(values)) if values else ()

    def drain(self):
        """Wait only the captured producer accesses, retaining the original tensor."""
        for event in self.events:
            event.synchronize()
