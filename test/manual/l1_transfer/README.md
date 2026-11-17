# Retained contract-test helpers

Use [the standalone SlotGuard entry](../slotguard_linker/README.md) for current
GLM deployment and validation.

This directory retains local evidence helpers and their tests. They are not
part of the upstream package; `run-tests.sh cpu` runs their tests locally:

- `test-prefix-sharing.py`: streaming/evidence parsing, tested by
  `test_evidence_harness.py`.
- `slotguard_receipt_audit.py`: same-rank receipt validation, tested by
  `test_slotguard_receipt_audit.py`.
- `glm_long_context_runtime.py`: fixed historical identity/bound fixtures used by
  the receipt validator. These identities do not certify the current draft.

These helpers are not imported by the standalone gate or by registered tests.

The historical performance campaigns, bootstrap operators, raw-range probes and
their dedicated tests remain available at commit
`ad059d3f33d1432a65e0139c886d8f008a025e46`. They are excluded from this draft.
