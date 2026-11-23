#!/usr/bin/env bash
# Run from a SGLang test environment; no experiment workspace is required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODE="${1:-cpu}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
L1=test/registered/unit/mem_cache/l1_transfer
READER=test/registered/unit/mem_cache
case "$MODE" in
  cpu|collect)
    # Run on CPU CI workers; upstream test_utils rejects an empty mask.
    unset CUDA_VISIBLE_DEVICES
    args=()
    for file in "$L1"/test_*.py; do
      [[ "$file" == *_cuda.py ]] || args+=("$file")
    done
    args+=("$READER/test_kv_peer_read.py" "$READER/test_fixed_fabric_pool.py")
    # Local evidence-harness contracts live beside the harness, outside CI.
    args+=(test/manual/l1_transfer/test_*.py test/manual/slotguard_linker/test_*.py)
    [[ "$MODE" != collect ]] || args+=(--collect-only)
    ;;
  cuda)
    args=("$L1/test_kv_peer_engine_cuda.py" "$L1/test_slotguard_linker_cuda.py" "$READER/test_kv_peer_read.py")
    ;;
  *) echo "Usage: $0 {cpu|collect|cuda} [pytest arguments...]" >&2; exit 2 ;;
esac
exec "${PYTHON:-python3}" -m pytest -q -rfE -p no:cacheprovider "${args[@]}" "$@"
