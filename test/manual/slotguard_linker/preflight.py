"""CPU import/configuration preflight for the standalone SlotGuard gate."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optionally validate the local GLM tokenizer")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    sys.path.insert(0, str(root / "python"))
    import gate
    import kvcm_serve
    from gate_client import config, transfer
    from gate_client.workload import load_tokenizer, transfer_prompt

    import sglang
    from sglang.srt.mem_cache.l1_transfer.slotguard_linker import SlotGuardLinker
    from sglang.srt.mem_cache.unified_cache.linker_config import (
        UnifiedCacheLinkerConfig,
        parse_linker_load_options,
    )

    assert Path(sglang.__file__).resolve().is_relative_to(root)
    for script in here.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    raw = re.search(
        r"--unified-cache-external-linker-config '([^']+)'",
        (here / "launch.sh").read_text(),
    ).group(1)
    mode, policy, data = parse_linker_load_options(json.loads(raw))
    assert mode == "prefetch" and policy == "wait_complete"
    assert UnifiedCacheLinkerConfig.from_dict(data).linker == SlotGuardLinker.__name__
    kvcm_serve.load_setup(str(root))
    if args.model:
        tokenizer = load_tokenizer(args.model)
        for n in (257, 1025):
            a = transfer_prompt(tokenizer, n, "standalone-first")
            b = transfer_prompt(tokenizer, n, "standalone-second")
            assert len(a) == len(b) == n and a[:64] != b[:64]
    assert not any(
        n.startswith(("glm_test_admin", "glm_perf", "client.")) for n in sys.modules
    )
    print(
        json.dumps(
            dict(
                status="PASS",
                source=str(root),
                gate=gate.__file__,
                config=config.__file__,
                transfer=transfer.__file__,
            )
        )
    )


if __name__ == "__main__":
    main()
