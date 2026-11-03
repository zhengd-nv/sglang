#!/usr/bin/env bash
# Start the Mooncake master (with its HTTP metadata server) and one EGM
# provider per NUMA node on this host. Stays in the foreground; Ctrl-C or
# SIGTERM stops everything.
#
#   WORK=/shared/l2-sharing NUMAS="0 1" ./start_store.sh
#
# Optional: HOST_IP, MASTER_PORT (50051), METADATA_PORT (18080), METRICS_PORT (19003),
#           PROVIDER_PORT_BASE (17000), POOL_GIB per NUMA (64), LOG_DIR.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:?set WORK to the directory build_mooncake.sh wrote}"
source "$WORK/env.sh"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-50051}"
METADATA_PORT="${METADATA_PORT:-18080}"
METRICS_PORT="${METRICS_PORT:-19003}"
PROVIDER_PORT_BASE="${PROVIDER_PORT_BASE:-17000}"
NUMAS="${NUMAS:-0}"
LOG_DIR="${LOG_DIR:-$WORK/logs/$(hostname -s)}"
mkdir -p "$LOG_DIR"
METADATA_URL="http://$HOST_IP:$METADATA_PORT/metadata"

PIDS=()
declare -A PROVIDER_PID
cleanup() { kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; }
trap 'cleanup; exit 0' TERM INT

"$MOONCAKE_MASTER_BIN" --rpc_address=0.0.0.0 --rpc_port="$MASTER_PORT" --metrics_port="$METRICS_PORT" \
  --enable_http_metadata_server=true --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port="$METADATA_PORT" --logtostderr=1 > "$LOG_DIR/master.log" 2>&1 &
MASTER_PID=$!
PIDS+=($MASTER_PID)
for _ in $(seq 60); do
  if ! kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "mooncake_master exited (port in use?); see $LOG_DIR/master.log" >&2
    exit 1
  fi
  # Any HTTP status (404 for a missing key) means the metadata server is up.
  python3 -c 'import sys, urllib.request as u
try: u.urlopen(sys.argv[1], timeout=2)
except u.HTTPError: pass' "$METADATA_URL?key=probe" 2>/dev/null && break
  sleep 1
done

for numa in $NUMAS; do
  # EGM pages are reached through an entry GPU, so the provider must see all
  # GPUs of the node, not only one.
  CUDA_VISIBLE_DEVICES="${PROVIDER_GPUS:-0,1,2,3}" python3 "$HERE/egm_provider.py" \
    --master "$HOST_IP:$MASTER_PORT" --metadata "$METADATA_URL" --host "$HOST_IP" \
    --port $((PROVIDER_PORT_BASE + numa)) --numa "$numa" --pool-gib "${POOL_GIB:-64}" \
    > "$LOG_DIR/provider-numa$numa.log" 2>&1 &
  PIDS+=($!); PROVIDER_PID[$numa]=$!
done
for numa in $NUMAS; do
  for _ in $(seq 300); do
    grep -q "EGM provider ready" "$LOG_DIR/provider-numa$numa.log" 2>/dev/null && break
    kill -0 "${PROVIDER_PID[$numa]}" 2>/dev/null || break
    sleep 1
  done
  if ! grep -q "EGM provider ready" "$LOG_DIR/provider-numa$numa.log"; then
    echo "EGM provider on NUMA $numa failed; see $LOG_DIR/provider-numa$numa.log" >&2
    cleanup; exit 1
  fi
done
echo "STORE_READY master=$HOST_IP:$MASTER_PORT metadata=$METADATA_URL numas=[$NUMAS]"
wait
