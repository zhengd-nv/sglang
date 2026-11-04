#!/usr/bin/env bash
# Two-node run under Slurm: node 0 runs the writer, node 1 the reader and the
# client. Both nodes must be in the same NVL72 domain.
#
#   srun -N2 --ntasks-per-node=1 --segment=2 \
#     --container-image=<sglang image> --container-mounts=<shared dir> \
#     env WORK=<shared dir>/l2-sharing MODEL=<model> [BACKEND=egm] ./run_slurm.sh
#
# BACKEND=mooncake (default): node 0 also hosts the Mooncake master and the EGM
#   providers; the reader reads node 0's Grace memory through Mooncake.
# BACKEND=egm: each node runs its own EGM store services; each server writes to
#   its own node's store and looks up both, so the reader finds the prefix in
#   node 0's store, reads it over NVLink, and does not write it again.
#
# The nodes rendezvous through files under $WORK/runs/<job id>, so WORK must
# be on a filesystem both nodes see. Optional: NUMAS, WRITER_GPU, READER_GPU,
# PREFIX_TOKENS, SERVER_ARGS (extra launch_server flags), EGM_PORT_BASE.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WORK:?}" "${MODEL:?}"
RANK="${SLURM_PROCID:?run this under srun}"
BACKEND="${BACKEND:-mooncake}"
RUN="$WORK/runs/${SLURM_JOB_ID:-manual}"
LOGS="$RUN/$(hostname -s)"
mkdir -p "$LOGS"
read -ra EXTRA <<< "${SERVER_ARGS:-}"
export NUMAS="${NUMAS:-0}" EGM_PORT_BASE="${EGM_PORT_BASE:-29600}"
export SECRET_FILE="$RUN/egm.secret" SOCK_DIR="${SOCK_DIR:-/tmp}"
MY_GPU=$([[ "$RANK" == 0 ]] && echo "${WRITER_GPU:-0}" || echo "${READER_GPU:-0}")

wait_for() {  # wait_for <timeout seconds> <pid to watch, or 0> <command...>
  local deadline=$((SECONDS + $1)) pid=$2; shift 2
  until "$@" 2>/dev/null; do
    (( SECONDS < deadline )) || return 1
    [[ "$pid" == 0 ]] || kill -0 "$pid" 2>/dev/null || return 1
    sleep 2
  done
}
fail() { echo "$1" >&2; touch "$RUN/abort"; kill "${PIDS[@]}" 2>/dev/null; exit 1; }
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
egm_stores() {  # this GPU's NUMA store first (the write target), then the other stores
  local peer_ip=$1 local_numa=$((MY_GPU / 2)) out numa
  out="$SOCK_DIR/egm-numa$local_numa.sock"
  [[ " $NUMAS " == *" $local_numa "* ]] || { echo "NUMAS must include $local_numa (GPU $MY_GPU)" >&2; return 1; }
  for numa in $NUMAS; do
    [[ "$numa" == "$local_numa" ]] || out+=",$SOCK_DIR/egm-numa$numa.sock"
  done
  for numa in $NUMAS; do out+=",tcp://$peer_ip:$((EGM_PORT_BASE + numa))"; done
  echo "$out"
}
egm_stats() {  # objects held by every store of the job, after the demo
  python3 - "$SECRET_FILE" "$@" <<'EOF'
import json, socket, sys
secret = open(sys.argv[1]).read().strip()
for address in sys.argv[2:]:
    host, port = address.removeprefix("tcp://").rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=10) as sock:
        stream = sock.makefile("rwb")
        for req in ({"op": "hello", "secret": secret}, {"op": "stats"}):
            stream.write(json.dumps(req).encode() + b"\n"); stream.flush()
            reply = json.loads(stream.readline())
        keep = ("objects", "used_bytes", "capacity_bytes", "leases", "evicted")
        print("EGM_STORE", address, json.dumps({k: reply["value"].get(k) for k in keep}))
EOF
}

PIDS=()
OTHER=$((1 - RANK))

# ---- stores ----
if [[ "$RANK" == 0 ]]; then
  "$HERE/prepare_sglang.sh" > "$LOGS/prepare_sglang.log" 2>&1 || fail "prepare_sglang.sh failed; see $LOGS"
fi
if [[ "$BACKEND" == egm ]]; then
  if [[ "$RANK" == 0 ]]; then
    "$HERE/build_egm.sh" > "$LOGS/build_egm.log" 2>&1 || fail "build_egm.sh failed; see $LOGS"
    (umask 077; python3 -c 'import secrets; print(secrets.token_hex(24))' > "$SECRET_FILE")
    touch "$RUN/egm_built"
  else
    wait_for 600 0 test -e "$RUN/egm_built" -o -e "$RUN/abort"
    [[ -e "$RUN/egm_built" ]] || exit 1
  fi
  LOG_DIR="$LOGS" "$HERE/start_egm_store.sh" > "$LOGS/store.out" 2>&1 &
elif [[ "$RANK" == 0 ]]; then
  LOG_DIR="$LOGS" "$HERE/start_store.sh" > "$LOGS/store.out" 2>&1 &
fi
if [[ "$BACKEND" == egm || "$RANK" == 0 ]]; then
  STORE=$!; PIDS+=($STORE)
  wait_for 600 $STORE grep -q STORE_READY "$LOGS/store.out" || fail "store failed; see $LOGS"
fi
hostname -I | awk '{print $1}' > "$RUN/ip.$RANK"

# ---- servers ----
if [[ "$BACKEND" == egm ]]; then
  # Every server connects to every store at startup, so wait for the other node's.
  wait_for 900 0 test -e "$RUN/ip.$OTHER" -o -e "$RUN/abort"
  [[ -e "$RUN/ip.$OTHER" ]] || fail "the other node's store did not come up"
  PEER_IP=$(cat "$RUN/ip.$OTHER")
  STORES=$(egm_stores "$PEER_IP") || fail "bad NUMAS"
  SERVER_ENV=(BACKEND=egm "EGM_STORES=$STORES")
else
  wait_for 900 0 test -e "$RUN/ip.0" -o -e "$RUN/abort"
  SERVER_ENV=(BACKEND=mooncake "MASTER_HOST=$(cat "$RUN/ip.0")")
fi

if [[ "$RANK" == 0 ]]; then
  env "${SERVER_ENV[@]}" GPU="$MY_GPU" "$HERE/start_server.sh" "${EXTRA[@]}" > "$LOGS/writer.log" 2>&1 &
  WRITER=$!; PIDS+=($WRITER)
  wait_for 900 $WRITER healthy "http://127.0.0.1:${PORT:-30000}" || fail "writer failed; see $LOGS/writer.log"
  touch "$RUN/writer_ready"
  wait_for 3600 0 test -e "$RUN/done" -o -e "$RUN/abort"
  kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null
  exit 0
fi

wait_for 1800 0 test -e "$RUN/writer_ready" -o -e "$RUN/abort"
[[ -e "$RUN/writer_ready" ]] || fail "node 0 did not come up"
WRITER_IP=$(cat "$RUN/ip.0")
env "${SERVER_ENV[@]}" GPU="$MY_GPU" "$HERE/start_server.sh" "${EXTRA[@]}" > "$LOGS/reader.log" 2>&1 &
READER=$!; PIDS+=($READER)
rc=1
if wait_for 900 $READER healthy "http://127.0.0.1:${PORT:-30000}"; then
  before=$(nvlink_bytes)
  python3 "$HERE/demo_client.py" --writer "http://$WRITER_IP:${PORT:-30000}" \
    --reader "http://127.0.0.1:${PORT:-30000}" --prefix-tokens "${PREFIX_TOKENS:-16384}" \
    > "$RUN/demo.json"
  rc=$?
  after=$(nvlink_bytes)
  # Every NVLink byte the reader node received during the demo came from the
  # other node's Grace memory; compare with prefix pages x bytes per page.
  python3 -c 'import json, sys
a, b = json.loads(sys.argv[1] or "{}"), json.loads(sys.argv[2] or "{}")
print("READER_NODE_NVLINK_DELTA", json.dumps({g: [b[g][0] - a[g][0], b[g][1] - a[g][1]] for g in b if g in a}))' \
    "$before" "$after" | tee -a "$RUN/demo.json"
  if [[ "$BACKEND" == egm ]]; then
    addresses=()
    for ip in "$WRITER_IP" "$(cat "$RUN/ip.1")"; do
      for numa in $NUMAS; do addresses+=("tcp://$ip:$((EGM_PORT_BASE + numa))"); done
    done
    egm_stats "${addresses[@]}" | tee -a "$RUN/demo.json"
  fi
  grep DEMO_RESULT "$RUN/demo.json"
else
  echo "reader failed; see $LOGS/reader.log" >&2
fi
touch "$RUN/done"
kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null
exit $rc
