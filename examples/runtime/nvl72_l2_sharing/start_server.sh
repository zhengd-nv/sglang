#!/usr/bin/env bash
# Launch an SGLang server whose radix cache offloads to and loads from the
# shared Mooncake EGM pool through the direct external linker.
#
#   WORK=/shared/l2-sharing MASTER_HOST=<store node IP> MODEL=Qwen/Qwen2-1.5B \
#     GPU=0 PORT=30000 ./start_server.sh [extra launch_server flags]
#
# Optional: MASTER_PORT (50051), METADATA_PORT (18080), HOST_IP,
#           TE_PORT (17200; this server's Mooncake transfer-engine port).
set -euo pipefail

: "${WORK:?set WORK to the directory build_mooncake.sh wrote}"
: "${MASTER_HOST:?set MASTER_HOST to the node running start_store.sh}"
: "${MODEL:?set MODEL to a model path or Hugging Face id}"
source "$WORK/env.sh"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-50051}"
METADATA_PORT="${METADATA_PORT:-18080}"

STORE_CONFIG=$(printf '{"master_server_address": "%s:%s", "metadata_server": "http://%s:%s/metadata", "protocol": "nvlink", "device_name": "", "global_segment_size": 0}' \
  "$MASTER_HOST" "$MASTER_PORT" "$MASTER_HOST" "$METADATA_PORT")

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
# The transfer-engine endpoint other clients use to reach this server.
export MOONCAKE_LOCAL_HOSTNAME="$HOST_IP:${TE_PORT:-17200}"

# page_size 64: the store keeps one object per page, so small pages mean many
# small objects. Overlap scheduling is off because only that mode is validated.
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port "${PORT:-30000}" \
  --page-size 64 --disable-overlap-schedule \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-backend-extra-config "$STORE_CONFIG" \
  "$@"
