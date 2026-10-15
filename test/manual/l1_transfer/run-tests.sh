#!/usr/bin/env bash
# Run on two allocated GPU nodes, one task per node, shared output directory.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
pkg="$repo/python/sglang/srt/mem_cache/l1_transfer"
unit="$repo/test/registered/unit/mem_cache"
run="${1:?absolute unique output directory}"
[[ "$run" == /* ]] || { echo 'Output path must be absolute' >&2; exit 2; }
rank="${SLURM_PROCID:?one Slurm task per node}"
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
umask 077
ulimit -s 8192
[[ ! -e "$run/rank-$rank" ]] || { echo 'Refusing reused rank output' >&2; exit 2; }
mkdir -p "$run/rank-$rank"
exec > "$run/rank-$rank/run.log" 2>&1
trap 'touch "$run/failed-rank-'"$rank"'"' ERR
export NVT_BUILD="$run/build" NVLINK_TRANSFER_LIB="$run/build/libnvlink_transfer.so"
export PYTHONPATH="$repo/python:$unit" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$run/torch-extensions"
export SGLANG_CACHE_DIR="$run/cache/rank-$rank" XDG_CACHE_HOME="$run/cache/rank-$rank"
export FLASHINFER_WORKSPACE_BASE="$run/cache/rank-$rank/flashinfer"
export TMPDIR="$run/cache/rank-$rank/tmp"
mkdir -p "$SGLANG_CACHE_DIR" "$TMPDIR"
hostname
nvidia-smi --query-gpu=index,uuid,driver_version --format=csv
if [[ "$rank" == 0 ]]; then
    sha256sum "$pkg/"*.py "$pkg/transport/"*.py "$pkg/peer/"*.py "$pkg/transport/native.cpp" \
        "$pkg/build-native.sh" "$unit/l1_transfer/"*.py "$here/"*.py "$here/run-tests.sh" \
        "$repo/python/sglang/srt/mem_cache/source_lease.py" \
        "$repo/python/sglang/srt/mem_cache/target_reservation.py" > "$run/source-sha256.txt"
    git -C "$repo" rev-parse HEAD > "$run/sglang-commit.txt"
    git -C "$repo" diff > "$run/sglang-working.patch"
    bash "$pkg/build-native.sh"
    python3 -m unittest discover -s "$unit/l1_transfer" -v
    python3 "$unit/test_source_lease.py" -v
    python3 "$unit/test_target_reservation.py" -v
    python3 -c 'from pathlib import Path; import sglang.srt.mem_cache.l1_transfer as m; print("L1_MODULE", Path(m.__file__).resolve())'
    touch "$run/ready"
fi
for ((i=0;i<300;i++)); do
    if compgen -G "$run/failed-rank-*" > /dev/null; then exit 1; fi
    [[ -f "$run/ready" ]] && break
    sleep 1
done
test -f "$run/ready"
python3 "$here/test-sglang-pool.py" --output "$run/cross-node" --rank "$rank"
python3 "$here/test-peer.py" --output "$run/peer" --rank "$rank"
# Keep transport-only tests independent of tree ownership tests.
if [[ "$rank" == 0 ]]; then
    python3 "$here/test-gpu.py" --local --output "$run/transport-local"
    touch "$run/local-ready"
fi
for ((i=0;i<300;i++)); do
    if compgen -G "$run/failed-rank-*" > /dev/null; then exit 1; fi
    [[ -f "$run/local-ready" ]] && break
    sleep 1
done
test -f "$run/local-ready"
python3 "$here/test-gpu.py" --output "$run/transport-cross-node" --rank "$rank"
