#!/usr/bin/env bash
# I4 production GLM 2P1D gate; derived from sglang2 axes-serve.sh.
# Fixed SlotGuard dynamic/prefetch, real KVCM, no test-admin.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SG_SRC="$(cd "$here/../../.." && pwd)"
RUN="$(realpath -m "${1:?shared run directory}")"; ROLE="${2:?p0, p1 or d0}"
case "$ROLE" in p0) RANK=0;; p1) RANK=1;; d0) RANK=2;; *) echo "invalid role" >&2; exit 2;; esac
: "${MODEL:?local GLM-5.2-NVFP4 path}" "${KVCM_BIN:?}" "${INDEXER_BIN:?}" "${BRIDGE_BIN:?}" "${CACHE_KEY:?image or environment identity}"
ID="$(basename "$RUN")"
MODE_ARGS=()
ROLES=(p0 p1 d0)
SH="$RUN/shared"; OUT="$RUN/$ROLE-$(hostname)"
mkdir -p "$SH"
mkdir "$OUT" || { echo "role output already exists; use a fresh run directory" >&2; exit 2; }
exec > "$OUT/role.log" 2>&1
read -r -a EXTRA <<< "${MANIFEST_EXTRA:-}"; MODE_ARGS+=("${EXTRA[@]}")
read -r -a SERVER_ARGS <<< "${SERVER_EXTRA:-}"
HCAS="${IB_DEVICES:?comma-separated RDMA devices}"
HTTP_PORT=30000 BOOT_PORT=8998 EVENT_PORT=28758 KVCM_PORT=40017 INDEXER_PORT=50051
CONTROL_BASE=(39000 39100)
INSTANCE="slotguard_${ID//-/_}"
CTX_LEN="${CTX_LEN:-524288}"
# gate: byte oracle and traces on; perf: both off, run perf.py instead.
# TRACE=1 turns traces on for perf too (stage timing), still without the byte oracle.
DRIVER="${DRIVER:-gate}"
case "$DRIVER" in gate) CHECK_ENV=(SGLANG_NVLINK_TRACE_PATH="$OUT/trace/nvlink" SGLANG_NVLINK_VERIFY_BYTES=1);; perf) CHECK_ENV=();; *) echo "invalid DRIVER" >&2; exit 2;; esac
if [[ "$DRIVER" == perf && "${TRACE:-0}" == 1 ]]; then CHECK_ENV=(SGLANG_NVLINK_TRACE_PATH="$OUT/trace/nvlink"); fi
SG_MAX_PIN_PAGES="${SG_MAX_PIN_PAGES:-8192}"
SG_MAX_REMOTE_PINNED_PAGES="${SG_MAX_REMOTE_PINNED_PAGES:-16384}"
PIN_ARGS=()
PIN_ARGS=(${SG_MAX_PIN_PAGES:+--max-pin-pages "$SG_MAX_PIN_PAGES"} ${SG_MAX_REMOTE_PINNED_PAGES:+--max-remote-pinned-pages "$SG_MAX_REMOTE_PINNED_PAGES"})
PGIDS=()
step() { echo "== [$(date -Is)] $*"; }
die() { echo "FATAL: $*"; touch "$SH/failed-$ROLE"; exit 1; }
cleanup() { for g in "${PGIDS[@]}"; do kill -TERM -- "-$g" 2>/dev/null; done; sleep 5;
            for g in "${PGIDS[@]}"; do kill -KILL -- "-$g" 2>/dev/null; done; }
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then touch "$SH/failed-$ROLE"; fi
  cleanup
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
spawn() { local name=$1; shift; setsid "$@" > "$OUT/$name.log" 2>&1 & PGIDS+=($!); echo "$name pid=$!"; }
wait_file() { for _ in $(seq "${2:-1800}"); do [ -e "$1" ] && return 0; ls "$SH"/failed-* >/dev/null 2>&1 && return 1; sleep 1; done; return 1; }
wait_http() { for _ in $(seq "${2:-2400}"); do curl -sf --max-time 5 -o /dev/null "$1" && return 0
              [ -n "${3:-}" ] && ! kill -0 "$3" 2>/dev/null && return 1; sleep 1; done; return 1; }

export SG_SRC SGLANG_SRC="$SG_SRC" PYTHONPATH="$SG_SRC/python" PYTHONDONTWRITEBYTECODE=1
source "$here/runtime.sh"
setup_runtime "$ROLE" || die "runtime setup"
python3 -c 'import sglang; print(sglang.__file__)' > "$OUT/source-path.txt"
export WITH_NVIDIA_PEERMEM="${WITH_NVIDIA_PEERMEM:-0}"

step "evidence role=$ROLE"
IP=${HOST_IP:-$(python3 -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); print(s.getsockname()[0])')}
nvidia-smi -q > "$OUT/nvidia-smi-q.txt"
DOMAIN=$(python3 - "$OUT/nvidia-smi-q.txt" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
u = set(re.findall(r"ClusterUUID\s*:\s*(\S+)", t)); c = set(re.findall(r"CliqueId\s*:\s*(\S+)", t))
print(f"{u.pop()}/{c.pop()}" if len(u) == 1 and len(c) == 1 else "MIXED")
PY
)
{ echo "host=$(hostname) ip=$IP domain=$DOMAIN topo=${SLURM_TOPOLOGY_ADDR:-}"; if [ -e "$SG_SRC/.git" ]; then git -C "$SG_SRC" rev-parse HEAD; else cat "$SG_SRC/.source-commit"; fi; } | tee "$OUT/node.txt"
HTTP_PORT=$(python3 - "$IP" "$HTTP_PORT" <<'PY'
import socket, sys
for port in range(int(sys.argv[2]), int(sys.argv[2]) + 100):
    s = socket.socket()
    try:
        s.bind((sys.argv[1], port)); print(port); break
    except OSError:
        pass
    finally:
        s.close()
PY
) || { echo "FATAL: no free HTTP port on $(hostname)"; touch "$SH/failed-$ROLE"; exit 1; }
[ -n "$HTTP_PORT" ] || { echo "FATAL: no free HTTP port on $(hostname)"; touch "$SH/failed-$ROLE"; exit 1; }
echo "http_port=$HTTP_PORT" | tee -a "$OUT/node.txt"
EVENT_PORT=$(python3 - "$EVENT_PORT" <<'PY'
import socket, sys
for port in range(int(sys.argv[1]), int(sys.argv[1]) + 100):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port)); print(port); break
    except OSError:
        pass
    finally:
        s.close()
PY
) || { echo "FATAL: no free event port on $(hostname)"; touch "$SH/failed-$ROLE"; exit 1; }
[ -n "$EVENT_PORT" ] || { echo "FATAL: no free event port on $(hostname)"; touch "$SH/failed-$ROLE"; exit 1; }
echo "event_port=$EVENT_PORT" | tee -a "$OUT/node.txt"
read -r BOOT_PORT INDEXER_PORT LOCAL_CONTROL < <(python3 - <<'PORTS'
import socket
def pick(base, count):
    for port in range(base, 60000, count):
        sockets = []
        try:
            for value in range(port, port + count):
                sock = socket.socket(); sockets.append(sock)
                sock.bind(("0.0.0.0", value))
            return port
        except OSError:
            pass
        finally:
            for sock in sockets: sock.close()
    raise RuntimeError("no free port range")
print(pick(8998, 1), pick(50051, 1), pick(39000, 4))
PORTS
)
[ -n "${LOCAL_CONTROL:-}" ] || die "ancillary ports unavailable"
printf '%s\n' "$BOOT_PORT" > "$SH/boot-$ROLE"
printf '%s\n' "$INDEXER_PORT" > "$SH/indexer-$ROLE"
printf '%s\n' "$LOCAL_CONTROL" > "$SH/control-$ROLE"
echo "$HTTP_PORT" > "$SH/port-$ROLE.tmp" && mv "$SH/port-$ROLE.tmp" "$SH/port-$ROLE"
echo "$DOMAIN" > "$SH/domain-$ROLE"
# Publish readiness only after all node metadata is complete.
echo "$IP" > "$SH/ip-$ROLE.tmp" && mv "$SH/ip-$ROLE.tmp" "$SH/ip-$ROLE"
for r in "${ROLES[@]}"; do wait_file "$SH/ip-$r" 600 || die "no ip from $r"; done
IP0=$(cat "$SH/ip-p0"); IP1=$(cat "$SH/ip-p1")
INDEXER_PORT=$(cat "$SH/indexer-p0")
CONTROL_BASE=("$(cat "$SH/control-p0")" "$(cat "$SH/control-p1")")

if [ "$ROLE" = p0 ]; then

  step "manifests, token, KVCM, indexer"
  [ "$(cat "$SH/domain-p0")" = "$(cat "$SH/domain-p1")" ] && [ "$DOMAIN" != MIXED ] || die "P0/P1 fabric domains differ"
  KVCM_PORT=$(python3 - "$KVCM_PORT" <<'PY'
import socket, sys
def bind(port):
    try:
        s = socket.socket(socket.AF_INET6)
    except OSError:
        s = socket.socket(socket.AF_INET); s.bind(("0.0.0.0", port)); return s
    s.bind(("::", port)); return s
for base in range(int(sys.argv[1]), 60000, 10):
    socks = []
    try:
        for port in range(base, base + 4):
            socks.append(bind(port))
        print(base); break
    except OSError:
        pass
    finally:
        for s in socks: s.close()
PY
) || die "no free KVCM port range"; echo "kvcm_port=$KVCM_PORT" | tee -a "$OUT/node.txt"
  ( umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$SH/control-token" )
  # /pause_generation and /continue_generation are admin_force: refused unless an admin key is set.
  ( umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$SH/admin-key" )
  B0=$(python3 -c 'import uuid; print(uuid.uuid4())'); B1=$(python3 -c 'import uuid; print(uuid.uuid4())')
  R0="P0-$B0-tp0"; R1="P1-$B1-tp0"; echo "$R0" > "$SH/reporter-p0"; echo "$R1" > "$SH/reporter-p1"
  for pair in "P0 $B0 $IP0 $R0 ${CONTROL_BASE[0]} P1 $B1 $IP1 $R1 ${CONTROL_BASE[1]} p0" \
              "P1 $B1 $IP1 $R1 ${CONTROL_BASE[1]} P0 $B0 $IP0 $R0 ${CONTROL_BASE[0]} p1"; do
    set -- $pair
    python3 -m sglang.srt.mem_cache.l1_transfer.tp_manifest "${MODE_ARGS[@]}" "${PIN_ARGS[@]}" \
      --local-engine "$1" --local-incarnation "$2" --local-host "$3" --local-reporter "$4" --local-base-port "$5" \
      --peer-engine "$6" --peer-incarnation "$7" --peer-host "$8" --peer-reporter "$9" --peer-base-port "${10}" \
      --bind-host "$3" --domain "$DOMAIN" --instance "$INSTANCE" --model "$MODEL" \
      --kvcm-endpoint "$IP0:$KVCM_PORT" --output "$SH/${11}.json" || die "tp_manifest ${11}"

  done
  python3 "$here/kvcm_serve.py" --workdir "$OUT/kvcm" --binary "$KVCM_BIN" \
    --sg-src "$SG_SRC" --instance "$INSTANCE" --rpc-port "$KVCM_PORT" > "$OUT/kvcm-serve.log" 2>&1 || die "kvcm"
  PGIDS+=("$(python3 -c "import json;print(json.load(open('$OUT/kvcm/kvcm-state.json'))['pid'])")")
  spawn indexer env KV_INDEXER_LISTEN_ADDR="$IP0:$INDEXER_PORT" KV_INDEXER_BACKEND=tair_kvcache_kvcm \
    KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT="http://$IP0:$KVCM_PORT" KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID="$INSTANCE" \
    KV_INDEXER_TAIR_KVCACHE_KVCM_STORAGE_TYPE=l1p5 KV_INDEXER_TAIR_KVCACHE_KVCM_REPORT_TIERS=HBM \
    KV_INDEXER_TAIR_KVCACHE_KVCM_AUTO_REGISTER=0 "$INDEXER_BIN"
  sleep 3; kill -0 "${PGIDS[-1]}" || die "indexer exited"
  touch "$SH/meta-ready"
fi

if [[ "$ROLE" = p* ]]; then
  wait_file "$SH/meta-ready" 900 || die "metadata services not ready"
  step "bridge + prefill $ROLE"
  REPORTER=$(cat "$SH/reporter-$ROLE")
  spawn bridge env -u KV_INDEXER_CACHE_COMPONENTS -u KV_INDEXER_SWA_WINDOW_TOKENS \
    KV_INDEXER_WORKER_ID="$REPORTER" KV_INDEXER_WORKER_ADDRESS="http://$IP:$HTTP_PORT" \
    KV_INDEXER_ENDPOINT="http://$IP0:$INDEXER_PORT" SGLANG_KV_EVENT_ENDPOINT="tcp://$IP:$EVENT_PORT" \
    SGLANG_KV_EVENT_TOPIC=kv-events "$BRIDGE_BIN"
  mkdir -p "$OUT/trace"
  SGLANG_NVLINK_CONTROL_TOKEN=$(cat "$SH/control-token"); export SGLANG_NVLINK_CONTROL_TOKEN
  spawn server env SGLANG_NVLINK_CONFIG="$SH/$ROLE.json" SGLANG_GLM_DSA_FIXED_FABRIC_POOL=1 \
    SGLANG_ADMIN_API_KEY="$(cat "$SH/admin-key")" \
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=0 \
    "${CHECK_ENV[@]}" \
    python3 -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
      --host "$IP" --port "$HTTP_PORT" --tp-size 4 --kv-cache-dtype fp8_e4m3 --page-size 64 \
      --max-total-tokens 2560000 --context-length "$CTX_LEN" --chunked-prefill-size 8192 \
      --enable-unified-cache-external-linker --unified-cache-external-linker-backend dynamic \
      --unified-cache-external-linker-config '{"linker_module_path":"sglang.srt.mem_cache.l1_transfer.slotguard_linker","linker":"SlotGuardLinker","load_mode":"prefetch","prefetch_policy":"wait_complete"}' \
      --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
      --disaggregation-bootstrap-port "$BOOT_PORT" --disaggregation-ib-device "$HCAS" \
      --kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$EVENT_PORT\",\"topic\":\"kv-events\"}" \
      "${SERVER_ARGS[@]}"
  unset SGLANG_NVLINK_CONTROL_TOKEN
else
  step "decode"
  spawn server python3 -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
    --host "$IP" --port "$HTTP_PORT" --tp-size 4 --kv-cache-dtype fp8_e4m3 --page-size 64 \
    --max-total-tokens 2560000 --context-length "$CTX_LEN" \
    --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device "$HCAS" "${SERVER_ARGS[@]}"
fi
SERVER_PG=${PGIDS[-1]}
wait_http "http://$IP:$HTTP_PORT/health" 3600 "$SERVER_PG" || die "$ROLE server not healthy (or exited)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$OUT/gpu-mem-ready.csv"
touch "$SH/ready-$ROLE"; step "$ROLE ready"

if [ "$ROLE" = p0 ]; then
  for r in p1 d0; do wait_file "$SH/ready-$r" 3600 || die "$r not ready"; done
  python3 - "$SH" <<PY
import json, sys
sh = sys.argv[1]
ip = {r: open(f"{sh}/ip-{r}").read().strip() for r in ("p0", "p1", "d0")}
port = {r: open(f"{sh}/port-{r}").read().strip() for r in ("p0", "p1", "d0")}
cfg = {"model": "$MODEL", "backend": "nvlink",
       "prefill": [{"url": f"http://{ip[r]}:{port[r]}", "bootstrap_port": int(open(f"{sh}/boot-{r}").read()),
                    "nvlink_config": f"{sh}/{r}.json"} for r in ("p0", "p1")],
       "decode": [{"url": f"http://{ip['d0']}:{port['d0']}"}],
       "control_token_file": f"{sh}/control-token", "admin_key_file": f"{sh}/admin-key",
       "kvcm": {"endpoint": f"{ip['p0']}:$KVCM_PORT", "instance_id": "$INSTANCE"}}
json.dump(cfg, open(f"{sh}/client.json", "w"), indent=1)
PY
  touch "$SH/all-ready"; step "all ready; production HTTP $DRIVER"
  ( trap 'touch "$SH/stop"' EXIT
    python3 "$here/$DRIVER.py" --run-dir "$RUN" --config "$SH/client.json" > "$OUT/$DRIVER.log" 2>&1
  ) &
  RUNNER=$!
fi

while [ ! -e "$SH/stop" ] && ! ls "$SH"/failed-* >/dev/null 2>&1; do
  kill -0 "$SERVER_PG" 2>/dev/null || die "$ROLE server died"
  sleep 2
done
step "stop requested"; [ -n "${RUNNER:-}" ] && kill "$RUNNER" 2>/dev/null
[ -f "$RUN/complete.txt" ] && ! ls "$SH"/failed-* >/dev/null 2>&1
