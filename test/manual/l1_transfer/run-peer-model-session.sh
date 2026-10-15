#!/usr/bin/env bash
# Development session: write an attempt name into next; touch finish to end.
set -u
root="${1:?output root}"
model="${2:?model path}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root"
export L1_PD_CACHE="$root/cache"
rc=1
for ((i=0;i<1500;i++)); do
    if [[ -f "$root/next" ]]; then
        attempt="$(cat "$root/next")"
        rm "$root/next"
        [[ "$attempt" =~ ^attempt-[0-9]+$ ]] || exit 2
        [[ ! -e "$root/$attempt" ]] || exit 2
        bash "$here/run-peer-model.sh" "$root/$attempt" "$model"
        rc=$?
        printf '%s\n' "$rc" > "$root/$attempt/exit-code"
    fi
    [[ -f "$root/finish" ]] && exit "$rc"
    sleep 1
done
exit 124
