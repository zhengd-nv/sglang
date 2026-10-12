# Standalone EGM store (experimental)

CUDA Driver HOST_NUMA + FABRIC storage, without Mooncake. The Python adapter
is loaded through HiCache's existing `dynamic` backend; no factory changes
are required. Build the native library on a CUDA development machine:

```bash
g++ -std=c++17 -O2 -fPIC -shared native.cpp -I"$CUDA_HOME/include" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda -o libegm.so
export EGM_NATIVE_LIB="$PWD/libegm.so"
```

Use `hicache_storage_backend="dynamic"` with extra config:

```json
{"backend_name":"egm","module_path":"sglang.srt.mem_cache.storage.egm.hicache","class_name":"EGMHiCacheStorage","interface_v1":true,"capacity_bytes":67108864,"namespace":"model-weight-revision","numa":0}
```

Requires CUDA HOST_NUMA/FABRIC support and an accessible IMEX channel.
Supports standard FULL MHA, TP/PP/CP=1. Copies are synchronous; objects are
immutable and capacity is fixed. Explicit reader pins block clear/close.
Uncertain copy failures quarantine memory; there is no orphan-pin recovery.
Unix-socket peers currently support processes on one node. The Direct Linker
adapter is fixture-tested and is not connected to an Engine CLI selector.
The Engine path uses ordinary host memory for L2 and EGM for L3.

Manual tests live in `test/manual/egm/` relative to the repository root:

```bash
python test/manual/egm/test_egm_store.py --output "$EGM_RUNROOT"
python test/manual/egm/test_egm_engine.py
```

Set `EGM_RUNROOT` to a fresh output directory and `EGM_NATIVE_LIB` to the built
library. The Engine test generates a tiny dummy-weight model, writes cache,
flushes local cache, and checks actual storage restoration and output equality.

For testing against an installed SGLang runtime, optionally set
`EGM_TEST_STORAGE_PATH="$PWD/python/sglang/srt/mem_cache/storage"`.
The manual-test bootstrap overlays only this backend package (also in spawned
children), preserving the installed runtime. Such a run validates the backend
against that installed version, not the complete checkout. Production use must
install the package in the intended runtime and does not use this bootstrap.
