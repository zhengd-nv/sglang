#!/usr/bin/env bash
# BACKEND=egm: start one EGM store service per NUMA node on this host. Each
# owns a slab of that NUMA's Grace memory and serves a Unix socket (Engines on
# this node) and TCP (Engines on other nodes, which must present the job
# secret). Stays in the foreground; Ctrl-C or SIGTERM stops everything.
#
#   WORK=/shared/l2-sharing SECRET_FILE=/shared/l2-sharing/egm.secret NUMAS="0 1" \
#     ./start_egm_store.sh
#
# The secret file is created (mode 0600) if missing; every node of the job must
# use the same file. Optional: POOL_GIB per NUMA (64), EGM_PORT_BASE (29600;
# NUMA n listens on base + n), SOCK_DIR (/tmp), PROVIDER_GPUS (0,1,2,3), LOG_DIR.
set -uo pipefail

: "${WORK:?set WORK to the directory build_egm.sh wrote}"
: "${SECRET_FILE:?set SECRET_FILE to a job-private file every node sees}"
source "$WORK/egm-env.sh"
NUMAS="${NUMAS:-0}"
EGM_PORT_BASE="${EGM_PORT_BASE:-29600}"
SOCK_DIR="${SOCK_DIR:-/tmp}"
LOG_DIR="${LOG_DIR:-$WORK/logs/$(hostname -s)}"
mkdir -p "$LOG_DIR"

if [[ ! -s "$SECRET_FILE" ]]; then
  (umask 077; python3 -c 'import secrets; print(secrets.token_hex(24))' > "$SECRET_FILE")
fi

PIDS=()
declare -A STORE_PID
cleanup() { kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; }
trap 'cleanup; exit 0' TERM INT

for numa in $NUMAS; do
  sock="$SOCK_DIR/egm-numa$numa.sock"
  rm -f "$sock" "$LOG_DIR/egm-store-numa$numa.json"
  # The slab is reached through the NUMA's entry GPU, so the service must see
  # all GPUs of the node, not only one.
  CUDA_VISIBLE_DEVICES="${PROVIDER_GPUS:-0,1,2,3}" python3 -m sglang.srt.mem_cache.storage.egm.service \
    --numa "$numa" --capacity-gib "${POOL_GIB:-64}" --listen "$sock" \
    --tcp "0.0.0.0:$((EGM_PORT_BASE + numa))" --secret-file "$SECRET_FILE" \
    --ready "$LOG_DIR/egm-store-numa$numa.json" \
    --stats-out "$LOG_DIR/egm-store-numa$numa.stats.json" \
    > "$LOG_DIR/egm-store-numa$numa.log" 2>&1 &
  PIDS+=($!); STORE_PID[$numa]=$!
done
for numa in $NUMAS; do
  for _ in $(seq 300); do
    [[ -e "$LOG_DIR/egm-store-numa$numa.json" ]] && break
    kill -0 "${STORE_PID[$numa]}" 2>/dev/null || break
    sleep 1
  done
  if [[ ! -e "$LOG_DIR/egm-store-numa$numa.json" ]]; then
    echo "EGM store on NUMA $numa failed; see $LOG_DIR/egm-store-numa$numa.log" >&2
    cleanup; exit 1
  fi
done
echo "STORE_READY backend=egm numas=[$NUMAS] sockets=$SOCK_DIR/egm-numa<n>.sock tcp_port_base=$EGM_PORT_BASE"
wait
