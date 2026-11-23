"""Contracts of the local deployment launcher; not part of the upstream package."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase


class LaunchContractTest(CustomTestCase):
    def test_ready_marker_publishes_complete_node_metadata(self):
        launcher = Path(__file__).resolve().parent / "launch.sh"
        source = launcher.read_text()
        # Execute the actual publication block. Observe at the atomic rename,
        # before the producer can execute any subsequent metadata writes.
        start = source.index("printf '%s\\n' \"$BOOT_PORT\"")
        end = source.index('for r in "${ROLES[@]}"', start)
        observer = r"""set -euo pipefail
SH=$1
ROLE=p1
BOOT_PORT=8998 INDEXER_PORT=50051 LOCAL_CONTROL=39000 HTTP_PORT=30000
IP=127.0.0.1 DOMAIN=test-domain
mv() {
    command mv "$@"
    if [[ "$2" == "$SH/ip-$ROLE" ]]; then
        [[ "$(cat "$SH/boot-$ROLE")" == "$BOOT_PORT" ]]
        [[ "$(cat "$SH/indexer-$ROLE")" == "$INDEXER_PORT" ]]
        [[ "$(cat "$SH/control-$ROLE")" == "$LOCAL_CONTROL" ]]
        [[ "$(cat "$SH/port-$ROLE")" == "$HTTP_PORT" ]]
        [[ "$(cat "$SH/domain-$ROLE")" == "$DOMAIN" ]]
    fi
}
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    observer + source[start:end],
                    "launcher-test",
                    directory,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
