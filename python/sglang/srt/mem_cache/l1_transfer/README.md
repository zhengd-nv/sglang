# Experimental NVLink transport and PD integration

This base branch contains a CUDA Driver range transport, same-layout page planner,
SGLang MHA/NHD pool adapter, and a synchronous PD backend. It does not contain
Peer prefix restoration, source leases, target reservations, or changes to the
radix cache / allocator / ordinary Engine. The package path `l1_transfer` is kept
stable so the L1 sharing branch can build on this implementation.

- `transport/`: region registration/export/import, READ/WRITE batches, completion,
  layout planning, and finalized post-capture VMM pool adapter.
- `../../disaggregation/nvlink/`: PD backend; PD owns request/page lifetimes.
- `test/registered/unit/mem_cache/l1_transfer/`: CPU transport/planner/adapter tests.
- `test/registered/unit/disaggregation/test_nvlink_pd.py`: PD control lifecycle tests.
- `test/manual/l1_transfer/`: GPU range tests and Qwen2-1.5B PD model comparison.

Build from the repository root in a CUDA development environment:

```bash
bash python/sglang/srt/mem_cache/l1_transfer/build-native.sh
```

The default artifact is `$XDG_CACHE_HOME/sglang/l1_transfer/libnvlink_transfer.so`
(or `$HOME/.cache/sglang/l1_transfer/...`). Set `NVT_BUILD` and
`NVLINK_TRANSFER_LIB` for a custom build location. No full sgl-kernel rebuild is
needed. The transport has no Mooncake or EGM runtime dependency; CUDA FABRIC VMM
and compatible NVLink devices are required. See the license and provenance section below.

With SGLang runtime dependencies installed:

```bash
PYTHONPATH=python python -m unittest discover \
  -s test/registered/unit/mem_cache/l1_transfer -v
PYTHONPATH=python python test/registered/unit/disaggregation/test_nvlink_pd.py -v
bash test/manual/l1_transfer/run-pd.sh /shared/unique-output /path/to/model
```

The PD runner builds the native transport and compares six greedy requests against
an independent ordinary Engine, including multi-chunk prefill. It needs three
GPUs. See `disaggregation/nvlink/README.md` for supported settings and fail-stop
semantics. The runner explicitly carries compatibility overrides for the older
0.5.21/cu130 experimental image; these are not normal framework defaults.

Transport GPU tests can be run with `test/manual/l1_transfer/test-gpu.py --local
--output OUTPUT`, or one process on each of two nodes with `--rank 0/1` and a
shared output directory. Set PYTHONPATH and NVLINK_TRANSFER_LIB first.

## License and provenance

This distribution contains code adapted from the local SGLang experimental
implementation python/sglang/srt/mem_cache/storage/egm/native.cpp.
Copyright contributors to SGLang project
Licensed under the Apache License, Version 2.0 (see the [repository LICENSE](../../../../../LICENSE)).

Reference repository: https://github.com/sgl-project/sglang
Local reference revision: 31ff9b5d42a6898b33d379a787ab0cd07d15ba73
Original local addition: a2276a0ee8b6aee0400d506ce03071f18d617ace
These are local experimental commits, not a claim of upstream acceptance
or availability from that remote.

Adapted portions: CUDA Driver error handling, context/region RAII, VMM
allocation/export/import/mapping and C ABI exception conversion patterns.
Modifications: device-only allocation, single-device context, asynchronous
range batches, CUDA event completion, active-region tracking and nvt_* ABI.

Mooncake Transfer Engine and SGLang PD adapters were design references for
range/batch interfaces and completion contracts. No Mooncake, NIXL or MORI
implementation is bundled in this distribution. This notice records the
local adaptation; it is not an upstream SGLang NOTICE file.

The paired-contiguous planning in `transport/layout.py` follows SGLang
python/sglang/srt/disaggregation/common/utils.py::group_concurrent_contiguous
at the same reference revision, under Apache-2.0. The new planner uses only
the Python standard library and adds layout/capacity checks, byte ranges and
region binding. Copyright contributors to SGLang project.
