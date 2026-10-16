#!/usr/bin/env bash
set -euo pipefail
run="${1:?output directory}"
logs="${2:-$run}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
umask 077
mkdir -p "$run/cache" "$logs"
exec > "$logs/run.log" 2>&1
export PYTHONPATH="$repo/python" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$run/cache" SGLANG_CACHE_DIR="$run/cache/sglang"
export NVT_BUILD="$run/build" NVLINK_TRANSFER_LIB="$run/build/libnvlink_transfer.so"
export OMP_NUM_THREADS=1
hostname > "$run/hostname.txt"
nvidia-smi -q > "$run/gpu-before.txt"
nvidia-smi topo -m > "$run/topology.txt"
python --version > "$run/python.txt"
sha256sum "$here/bench-layout.py" "$here/run-layout-bench.sh" "$repo/python/sglang/srt/mem_cache/l1_transfer/transport/"*.* > "$run/source-sha256.txt"
bash "$repo/python/sglang/srt/mem_cache/l1_transfer/build-native.sh"
nvidia-smi nvlink -gt d > "$run/nvlink-before.txt" 2>&1 || true
python "$here/bench-layout.py" --output "$run/results"
nvidia-smi nvlink -gt d > "$run/nvlink-after.txt" 2>&1 || true
nvidia-smi -q > "$run/gpu-after.txt"
