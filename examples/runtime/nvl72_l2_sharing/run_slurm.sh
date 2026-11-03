#!/usr/bin/env bash
# Two-node run under Slurm: node 0 hosts the store and the writer, node 1 the
# reader and the client. Both nodes must be in the same NVL72 domain.
#
#   srun -N2 --ntasks-per-node=1 --segment=2 \
#     --container-image=<sglang image> --container-mounts=<shared dir> \
#     env WORK=<shared dir>/l2-sharing MODEL=<model> ./run_slurm.sh
#
# The nodes rendezvous through files under $WORK/runs/<job id>, so WORK must
# be on a filesystem both nodes see. Optional: NUMAS, WRITER_GPU, READER_GPU,
# PREFIX_TOKENS, SERVER_ARGS (extra launch_server flags).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:?}" "${MODEL:?}"
RANK="${SLURM_PROCID:?run this under srun}"
RUN="$WORK/runs/${SLURM_JOB_ID:-manual}"
LOGS="$RUN/$(hostname -s)"
mkdir -p "$LOGS"
read -ra EXTRA <<< "${SERVER_ARGS:-}"

wait_for() {  # wait_for <timeout seconds> <pid to watch, or 0> <command...>
  local deadline=$((SECONDS + $1)) pid=$2; shift 2
  until "$@" 2>/dev/null; do
    (( SECONDS < deadline )) || return 1
    [[ "$pid" == 0 ]] || kill -0 "$pid" 2>/dev/null || return 1
    sleep 2
  done
}
healthy() {
  python3 -c 'import sys, urllib.request as u; u.urlopen(sys.argv[1] + "/health", timeout=5)' "$1"
}
nvlink_bytes() {  # per-GPU Tx/Rx bytes summed over all links
  nvidia-smi nvlink -gt d 2>/dev/null | python3 -c '
import json, re, sys
res, gpu = {}, None
for line in sys.stdin:
    m = re.match(r"GPU (\d+):", line)
    if m: gpu = m.group(1); res[gpu] = [0, 0]; continue
    m = re.search(r"Data (Tx|Rx): (\d+) KiB", line)
    if m and gpu is not None: res[gpu][m.group(1) == "Rx"] += int(m.group(2)) * 1024
print(json.dumps(res))'
}

if [[ "$RANK" == 0 ]]; then
  "$HERE/prepare_sglang.sh" > "$LOGS/prepare_sglang.log" 2>&1 \
    || { echo "prepare_sglang.sh failed; see $LOGS" >&2; touch "$RUN/abort"; exit 1; }
  NUMAS="${NUMAS:-0}" LOG_DIR="$LOGS" "$HERE/start_store.sh" > "$LOGS/store.out" 2>&1 &
  STORE=$!
  if ! wait_for 600 $STORE grep -q STORE_READY "$LOGS/store.out"; then
    echo "store failed; see $LOGS" >&2; touch "$RUN/abort"; kill $STORE; exit 1
  fi
  hostname -I | awk '{print $1}' > "$RUN/store_ip"
  MASTER_HOST=$(cat "$RUN/store_ip") GPU="${WRITER_GPU:-0}" \
    "$HERE/start_server.sh" "${EXTRA[@]}" > "$LOGS/writer.log" 2>&1 &
  WRITER=$!
  if wait_for 900 $WRITER healthy "http://127.0.0.1:${PORT:-30000}"; then
    touch "$RUN/writer_ready"
    wait_for 3600 0 test -e "$RUN/done" -o -e "$RUN/abort"
  else
    echo "writer failed; see $LOGS/writer.log" >&2; touch "$RUN/abort"
  fi
  kill $WRITER $STORE 2>/dev/null; wait 2>/dev/null
  exit 0
fi

wait_for 1800 0 test -e "$RUN/writer_ready" -o -e "$RUN/abort"
[[ -e "$RUN/writer_ready" ]] || { echo "node 0 did not come up" >&2; exit 1; }
STORE_IP=$(cat "$RUN/store_ip")
MASTER_HOST="$STORE_IP" GPU="${READER_GPU:-0}" \
  "$HERE/start_server.sh" "${EXTRA[@]}" > "$LOGS/reader.log" 2>&1 &
READER=$!
rc=1
if wait_for 900 $READER healthy "http://127.0.0.1:${PORT:-30000}"; then
  before=$(nvlink_bytes)
  python3 "$HERE/demo_client.py" --writer "http://$STORE_IP:${PORT:-30000}" \
    --reader "http://127.0.0.1:${PORT:-30000}" --prefix-tokens "${PREFIX_TOKENS:-16384}" \
    > "$RUN/demo.json"
  rc=$?
  after=$(nvlink_bytes)
  # Every NVLink byte the reader node received during the demo came from the
  # other node's EGM; compare with prefix pages x bytes per page.
  python3 -c 'import json, sys
a, b = json.loads(sys.argv[1] or "{}"), json.loads(sys.argv[2] or "{}")
print("READER_NODE_NVLINK_DELTA", json.dumps({g: [b[g][0] - a[g][0], b[g][1] - a[g][1]] for g in b if g in a}))' \
    "$before" "$after" | tee -a "$RUN/demo.json"
  grep DEMO_RESULT "$RUN/demo.json"
else
  echo "reader failed; see $LOGS/reader.log" >&2
fi
touch "$RUN/done"
kill $READER 2>/dev/null; wait 2>/dev/null
exit $rc
