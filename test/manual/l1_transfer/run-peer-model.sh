#!/usr/bin/env bash
set -euo pipefail
run="${1:?output directory}"
model="${2:?model path}"
shift 2
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
umask 077
ulimit -s 8192
mkdir -p "$run/cache/tmp"
exec > "$run/run.log" 2>&1
export PYTHONPATH="$repo/python" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export NVT_BUILD="$run/build" NVLINK_TRANSFER_LIB="$run/build/libnvlink_transfer.so"
export XDG_CACHE_HOME="${L1_PD_CACHE:-$run/cache}"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export SGLANG_CACHE_DIR="$XDG_CACHE_HOME/sglang" FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export SGLANG_JIT_CACHE_DIR="$XDG_CACHE_HOME/sglang-jit" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
# UNIX-domain sockets have a 107-byte path limit; retain data under output.
export TMPDIR="/tmp/l1peer-${SLURM_JOB_ID:-local}-$$" MAX_JOBS=4
ln -sr "$run/cache/tmp" "$TMPDIR"
trap 'rm -f "$TMPDIR"' EXIT
sha256sum "$repo/python/sglang/srt/mem_cache/l1_transfer/"*.py \
    "$repo/python/sglang/srt/mem_cache/l1_transfer/peer/"*.py \
    "$repo/python/sglang/srt/mem_cache/l1_transfer/transport/"*.* \
    "$repo/python/sglang/srt/mem_cache/"{source_lease,target_reservation}.py \
    "$here/test-peer-model.py" "$here/run-peer-model.sh" > "$run/source-sha256.txt"
if [[ -f "$repo/.source-commit" ]]; then
    cp "$repo/.source-commit" "$run/sglang-commit.txt"
else
    git -C "$repo" rev-parse HEAD > "$run/sglang-commit.txt"
    git -C "$repo" diff > "$run/sglang-working.patch"
fi
# The 0.5.21 image has kernel 0.4.7; PyPI 0.4.9 requires torch 2.14.1.
# This FlashInfer/Qwen smoke test validates the image's existing kernel at runtime.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
if [[ ! -f "$XDG_CACHE_HOME/cutlass-complete/.ready" ]]; then
    python -m pip install --no-deps --target "$XDG_CACHE_HOME/cutlass-complete" \
        nvidia-cutlass-dsl==4.8.0 nvidia-cutlass-dsl-libs-core==4.8.0 nvidia-cutlass-dsl-libs-base==4.8.0 nvidia-cutlass-dsl-libs-cu12==4.8.0 nvidia-cutlass-dsl-libs-cu13==4.8.0
    touch "$XDG_CACHE_HOME/cutlass-complete/.ready"
fi
export PYTHONPATH="$XDG_CACHE_HOME/cutlass-complete/nvidia_cutlass_dsl/dsl_packages:$XDG_CACHE_HOME/cutlass-complete:$PYTHONPATH"
python -c 'import torch, importlib.metadata as m; print("RUNTIME", torch.__version__, {k: m.version(k) for k in ("sglang-kernel", "flashinfer-python", "nvidia-cutlass-dsl", "transformers")})'
export SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1
python -m unittest discover -s "$repo/test/registered/unit/mem_cache/l1_transfer" -v
hostname
nvidia-smi --query-gpu=index,uuid,driver_version --format=csv
bash "$repo/python/sglang/srt/mem_cache/l1_transfer/build-native.sh"
python "$here/test-peer-model.py" --model "$model" --output "$run" "$@"
