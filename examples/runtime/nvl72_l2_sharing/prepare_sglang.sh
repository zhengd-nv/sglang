#!/usr/bin/env bash
# Make this checkout runnable from PYTHONPATH inside the SGLang image.
#
# The image's own SGLang tree carries build outputs that are not in git (the
# Rust extensions and _version.py). A checkout placed ahead of it on
# PYTHONPATH lacks them and fails at startup with "rust_tree_core.mem_cache
# is not bundled". Copy them from the image's tree into this checkout.
# Run once inside the container; the branch only changes Python files, so
# the image's extensions still match.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT_PKG="$(cd "$HERE/../../../python/sglang" && pwd)"
IMAGE_PKG="$(cd / && env -u PYTHONPATH python3 -c \
  'import os, sglang; print(os.path.dirname(sglang.__file__))')"
if [[ "$IMAGE_PKG" == "$CHECKOUT_PKG" ]]; then
  echo "The image already imports this checkout; nothing to copy."
  exit 0
fi

copied=0
while IFS= read -r -d '' file; do
  rel="${file#"$IMAGE_PKG"/}"
  if [[ ! -e "$CHECKOUT_PKG/$rel" ]]; then
    mkdir -p "$(dirname "$CHECKOUT_PKG/$rel")"
    cp -p "$file" "$CHECKOUT_PKG/$rel"
    echo "copied $rel"
    copied=$((copied + 1))
  fi
done < <(find "$IMAGE_PKG" \( -name '*.so' -o -name _version.py \) -print0)
echo "Copied $copied file(s) from $IMAGE_PKG"
