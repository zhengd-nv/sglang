#!/usr/bin/env bash
# BACKEND=egm needs no native build any more: the copies go through
# sglang.srt.mem_cache.kv_peer_read (cuda-python and a JIT kernel compiled on
# first use). This only writes $WORK/egm-env.sh, which start_egm_store.sh and
# start_server.sh source, so the launch scripts stay unchanged.
#
#   WORK=/shared/l2-sharing ./build_egm.sh
set -euo pipefail

: "${WORK:?set WORK to a directory every node sees}"
mkdir -p "$WORK"
cat > "$WORK/egm-env.sh" <<'ENV'
# EGM over kv_peer_read: nothing to export.
ENV
echo "wrote $WORK/egm-env.sh"
