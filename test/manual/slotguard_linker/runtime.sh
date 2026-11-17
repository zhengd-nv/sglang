#!/usr/bin/env bash
# Writable, per-environment/per-role caches; no cluster or workspace paths.
setup_runtime() {
  ulimit -s 8192
  local c="${CACHE_ROOT:-$RUN/cache}/$CACHE_KEY/$1"
  export XDG_CACHE_HOME="$c/xdg" FLASHINFER_WORKSPACE_BASE="$c/flashinfer" TRITON_CACHE_DIR="$c/triton"
  export TORCH_EXTENSIONS_DIR="$c/torch-ext" TORCHINDUCTOR_CACHE_DIR="$c/inductor" HF_HOME="$c/hf"
  export SGLANG_CACHE_DIR="$c/sglang" SGLANG_JIT_CACHE_DIR="$c/sglang-jit" TVM_FFI_CACHE_DIR="$c/tvm-ffi" CUDA_CACHE_PATH="$c/cuda"
  [[ -w "${HOME:-/nonexistent}" ]] || export HOME="$c/home"
  mkdir -p "$HOME" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" "$TVM_FFI_CACHE_DIR" "$CUDA_CACHE_PATH"
  export TMPDIR="/tmp/sgl-${SLURM_JOB_ID:-local}-$$" PYTHONNOUSERSITE=1
  mkdir -p "$TMPDIR"
}
