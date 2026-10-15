# Experimental NVLink PD backend

Select with `--disaggregation-transfer-backend nvlink` on both prefill and decode.
The backend uses the in-tree `mem_cache.l1_transfer` range engine. It does not
instantiate Mooncake TE or use host KV staging.

Current configuration: same model and weights, TP/CP/PP/DP=1, MHA NHD, matching
KV dtype and page size, finalized post-capture VMM pool, early decode allocation,
`--max-running-requests 1 --disable-radix-cache`. Use serial client requests.
No speculative state, role switching, host destinations, prefix reuse, or
cross-layout conversion is supported. Pool compatibility is checked by the
adapter and the serialized layout contract; matching weights is the operator's
responsibility.

The public PD `KVManager/KVSender/KVReceiver` interfaces and common bootstrap
server are reused. Decode exports its pool and sends destination page indices
and scoped region descriptors over ZMQ. Prefill synchronizes its producer,
plans paired page ranges, performs synchronous WRITE batches, and closes every
import before reporting completion. First-token and other auxiliary data use
the existing PD `AuxDataCodec` over ZMQ; KV bytes use CUDA FABRIC mappings.
Decode releases exports before returning `KVPoll.Success` to its scheduler.
PD retains the source request/pages until the sender completes.

This is a correctness prototype. Transfers and computation do not overlap.
An uncertain transfer, published-target timeout, or cancellation with live
exports terminates the affected scheduler process (exit 70), rather than
recycling potentially writable pages. Online recovery is not implemented.

Build the native library using `mem_cache/l1_transfer/build-native.sh`, then set
`NVLINK_TRANSFER_LIB`. See `test/manual/l1_transfer/run-pd.sh` for the complete
model test: two PD GPUs plus one reference GPU, six greedy requests including
multi-chunk prefill and repeated input. `summarize-pd.py <output>` validates token
IDs, transfer bytes, chunk completion, and destination release logs. Unit tests
are in `test/registered/unit/disaggregation/test_nvlink_pd.py`.

The manual runner targets the project's 0.5.21/cu130 container and explicitly
skips the kernel version gate: its PyTorch 2.13/kernel 0.4.7 pair predates the
current main requirement. This exception is confined to that test runner;
the backend does not change SGLang's normal version checks.
