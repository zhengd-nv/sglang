#!/usr/bin/env bash
# Launch an SGLang server whose radix cache offloads to and loads from Grace
# memory (EGM) in the NVL72 domain through the direct external linker.
#
# BACKEND=mooncake (default): a Mooncake store started by start_store.sh.
#   WORK=/shared/l2-sharing MASTER_HOST=<store node IP> MODEL=Qwen/Qwen2-1.5B \
#     GPU=0 PORT=30000 ./start_server.sh [extra launch_server flags]
#   Optional: MASTER_PORT (50051), METADATA_PORT (18080), HOST_IP,
#             TE_PORT (17200; this server's Mooncake transfer-engine port),
#             LOAD_LAYERS_PER_CALL (4) / LOAD_INFLIGHT (2): how the linker batches
#             layer-wise reads (1 / 1 is one read per layer, one at a time).
#
# BACKEND=egm: EGM store services started by start_egm_store.sh on each node.
#   WORK=/shared/l2-sharing BACKEND=egm MODEL=Qwen/Qwen2-1.5B GPU=0 \
#     EGM_STORES=/tmp/egm-numa0.sock,tcp://<other node IP>:29600 \
#     SECRET_FILE=/shared/l2-sharing/egm.secret ./start_server.sh
#   EGM_STORES: comma-separated; the first one is where this server writes, so
#   list this node's store for this GPU's NUMA first. Lookups ask all of them.
#   Optional: NAMESPACE (default MODEL; must differ between weight versions).
set -euo pipefail

: "${WORK:?set WORK to the directory the build script wrote}"
: "${MODEL:?set MODEL to a model path or Hugging Face id}"
BACKEND="${BACKEND:-mooncake}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

egm_linker_config() {  # <comma-separated stores> <secret file> <namespace>
  python3 -c 'import json, sys
stores, secret, namespace = sys.argv[1:]
print(json.dumps(dict(linker="EGMDirectLinker",
                      linker_module_path="sglang.srt.mem_cache.storage.egm.linker",
                      linker_extra_config=dict(stores=[s for s in stores.split(",") if s],
                                               secret_file=secret, namespace=namespace))))' "$@"
}

case "$BACKEND" in
  mooncake)
    : "${MASTER_HOST:?set MASTER_HOST to the node running start_store.sh}"
    source "$WORK/env.sh"
    HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
    MASTER_PORT="${MASTER_PORT:-50051}"
    METADATA_PORT="${METADATA_PORT:-18080}"
    # More reads in flight load faster but slow this server's own kernel launches;
    # 4 layers x 2 in flight measured best for a 16k-token prefix on GB200.
    STORE_CONFIG=$(printf '{"master_server_address": "%s:%s", "metadata_server": "http://%s:%s/metadata", "protocol": "nvlink", "device_name": "", "global_segment_size": 0, "load_layers_per_call": %d, "load_inflight": %d}' \
      "$MASTER_HOST" "$MASTER_PORT" "$MASTER_HOST" "$METADATA_PORT" \
      "${LOAD_LAYERS_PER_CALL:-4}" "${LOAD_INFLIGHT:-2}")
    # The transfer-engine endpoint other clients use to reach this server.
    export MOONCAKE_LOCAL_HOSTNAME="$HOST_IP:${TE_PORT:-17200}"
    LINKER_ARGS=(--unified-cache-external-linker-backend mooncake
                 --hicache-storage-backend-extra-config "$STORE_CONFIG")
    ;;
  egm)
    : "${EGM_STORES:?set EGM_STORES with the local store first}"
    : "${SECRET_FILE:?set SECRET_FILE to the file start_egm_store.sh uses}"
    source "$WORK/egm-env.sh"
    LINKER_CONFIG=$(egm_linker_config "$EGM_STORES" "$SECRET_FILE" "${NAMESPACE:-$MODEL}")
    LINKER_ARGS=(--unified-cache-external-linker-backend dynamic
                 --unified-cache-external-linker-config "$LINKER_CONFIG")
    ;;
  *) echo "BACKEND must be mooncake or egm" >&2; exit 2 ;;
esac

# page_size 64: the store keeps one object per page, so small pages mean many
# small objects. Overlap scheduling is off because only that mode is validated.
exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port "${PORT:-30000}" \
  --page-size 64 --disable-overlap-schedule \
  --enable-unified-cache-external-linker \
  --hicache-write-policy write_through \
  "${LINKER_ARGS[@]}" \
  "$@"
