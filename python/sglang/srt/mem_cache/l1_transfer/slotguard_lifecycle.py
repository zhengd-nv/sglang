"""Owner counts the NVLink backend reports for its owner proof.

Background workers expose locked counters; this module never reads a Tree,
tensor pool or native engine journal.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendOwners:
    accepted: int
    queued: int
    working: int
    imports: int
    batches: int
    unknown: int
    revision: int

    @property
    def owners(self):
        return sum(
            (
                self.accepted,
                self.queued,
                self.working,
                self.imports,
                self.batches,
                self.unknown,
            )
        )
