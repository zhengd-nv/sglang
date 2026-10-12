# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Opt-in overlay of this backend only; never replace the installed runtime."""
import os
from pathlib import Path

if os.environ.get("EGM_TEST_STORAGE_PATH"):
    import sglang.srt.mem_cache.storage as storage

    source = Path(os.environ["EGM_TEST_STORAGE_PATH"]).resolve(strict=True)
    if not (source / "egm" / "hicache.py").is_file():
        raise RuntimeError(f"EGM source missing: {source}")
    storage.__path__.insert(0, str(source))

    import sglang
    import sglang.srt.mem_cache.storage.egm as egm
    assert Path(egm.__file__).resolve().parent == source / "egm"
    print(f"EGM test overlay: runtime={sglang.__version__}, source={egm.__file__}", flush=True)
