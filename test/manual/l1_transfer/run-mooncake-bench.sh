#!/usr/bin/env bash
set -euo pipefail
run="${1:?data directory}"
logs="${2:?log directory}"
mcsrc="${3:?Mooncake source checkout}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
mkdir -p "$run/cache" "$logs"
exec > "$logs/run.log" 2>&1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export XDG_CACHE_HOME="$run/cache" SGLANG_CACHE_DIR="$run/cache/sglang"
export NVT_BUILD="$run/build" NVLINK_TRANSFER_LIB="$run/build/libnvlink_transfer.so"
export MC_FORCE_MNNVL=1 MC_USE_TEV1=1
unset MC_USE_TENT MC_FORCE_TCP MC_FORCE_HCA MC_FORCE_SHM MC_USE_NVLINK_IPC MC_INTRANODE_NVLINK
hostname > "$run/hostname.txt"
nvidia-smi -q > "$run/gpu-before.txt"
nvidia-smi topo -m > "$run/topology.txt"
sha256sum "$here/bench-layout.py" "$here/run-mooncake-bench.sh" "$repo/python/sglang/srt/mem_cache/l1_transfer/transport/"*.* > "$run/source-sha256.txt"
bash "$repo/python/sglang/srt/mem_cache/l1_transfer/build-native.sh"
# Build the missing yaml dependency in an isolated prefix from pinned input archives.
yaml_src=("$run/inputs/yaml-cpp/"*)
ylt_src=("$run/inputs/ylt/"*)
cmake -S "${yaml_src[0]}" -B "$run/yaml-build" -DCMAKE_BUILD_TYPE=Release \
  -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_BUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="$run/deps" > "$logs/yaml.log" 2>&1
cmake --build "$run/yaml-build" -j 8 >> "$logs/yaml.log" 2>&1
cmake --install "$run/yaml-build" >> "$logs/yaml.log" 2>&1
# Source remains read-only. Only build TE and its binding; no Store/TENT/EP install.
cmake -S "$mcsrc" -B "$run/te-build" -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DUSE_MNNVL=ON -DUSE_TCP=OFF -DUSE_TENT=OFF \
  -DUSE_ETCD=OFF -DUSE_REDIS=OFF -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_EP=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
  -DENABLE_DEBUG_SYMBOLS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DCMAKE_MODULE_LINKER_FLAGS="-L$run/deps/lib" -DCMAKE_SHARED_LINKER_FLAGS="-L$run/deps/lib" \
  -DCMAKE_CXX_FLAGS="-I$run/deps/include" -DCMAKE_PREFIX_PATH="$run/deps" -DFETCHCONTENT_SOURCE_DIR_YALANTINGLIBS="${ylt_src[0]}" \
  > "$logs/cmake.log" 2>&1
cmake --build "$run/te-build" --target engine -j 8 > "$logs/build.log" 2>&1
mkdir -p "$run/te-python/mooncake"
touch "$run/te-python/mooncake/__init__.py"
cp "$run/te-build/mooncake-integration/"engine*.so "$run/te-python/mooncake/"
export LD_LIBRARY_PATH="$run/deps/lib:$run/deps/lib64:$run/te-build/mooncake-transfer-engine/src:$run/te-build/mooncake-common:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$run/te-python:$repo/python"
python -c 'import mooncake.engine as e; print(e.__file__)' > "$run/te-binding.txt"
sha256sum "$run/te-python/mooncake/"*.so > "$run/te-binary-sha256.txt"
# Fail fast on unavailable transport or mapping problems; full byte/sentinel oracle.
python "$here/bench-layout.py" --backend mooncake --case raw-32768 \
  --rounds 1 --samples 2 --warmup 1 --output "$run/smoke" > "$logs/smoke.log" 2>&1
# Same allocation/node/GPU pair for both implementations, native A / TE B / native A.
for arm in native mooncake native-repeat; do
  [[ ! -e "$run/$arm" ]] || { echo "Refusing reused arm: $arm" >&2; exit 2; }
done
for arm in native mooncake native-repeat; do
  backend="${arm%-repeat}"
  nvidia-smi nvlink -gt d > "$run/$arm-nvlink-before.txt" 2>&1
  python "$here/bench-layout.py" --backend "$backend" --output "$run/$arm/results" > "$logs/$arm.log" 2>&1
  nvidia-smi nvlink -gt d > "$run/$arm-nvlink-after.txt" 2>&1
  nvidia-smi -q > "$run/$arm-gpu-after.txt"
done
