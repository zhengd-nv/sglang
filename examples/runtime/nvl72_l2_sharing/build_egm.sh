#!/usr/bin/env bash
# Build libegm.so for BACKEND=egm (a few seconds) and write $WORK/egm-env.sh,
# which start_egm_store.sh and start_server.sh source. Run once in the
# container; it needs nvcc and the CUDA driver headers.
#
#   WORK=/shared/l2-sharing ./build_egm.sh
#
# Optional: CUDA_HOME (/usr/local/cuda).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:?set WORK to a directory every node sees}"
SRC="$(cd "$HERE/../../../python/sglang/srt/mem_cache/storage/egm" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OUT="$WORK/egm"
mkdir -p "$OUT"

# sm_100 code also runs on GB300 (sm_103); the PTX is a fallback for newer parts.
"$CUDA_HOME/bin/nvcc" -O2 -std=c++17 -shared -Xcompiler -fPIC -DEGM_WITH_GATHER \
  -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100 \
  "$SRC/native.cpp" "$SRC/egm_gather.cu" -I"$CUDA_HOME/include" \
  -L"$CUDA_HOME/lib64/stubs" -lcuda -o "$OUT/libegm.so.tmp"
mv "$OUT/libegm.so.tmp" "$OUT/libegm.so"

cat > "$WORK/egm-env.sh" <<EOF
export EGM_NATIVE_LIB="$OUT/libegm.so"
EOF
echo "built $OUT/libegm.so; wrote $WORK/egm-env.sh"
