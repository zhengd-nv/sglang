#!/usr/bin/env bash
# CUDA Driver-only extension; no Mooncake, torch extension or sgl-kernel build.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${NVT_BUILD:-${XDG_CACHE_HOME:-$HOME/.cache}/sglang/l1_transfer}"
cuda="${CUDA_HOME:-/usr/local/cuda}"
mkdir -p "$build"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared \
    "$here/transport/native.cpp" -I"$cuda/include" \
    -L"$cuda/lib64/stubs" -lcuda -o "$build/libnvlink_transfer.so"
printf '%s\n' "$build/libnvlink_transfer.so"
