"""CPU reservations for whole-pool exports versus local target DMA.

The caller reserves before touching memory; unknown cleanup keeps the owner.
No transport calls or payload hashing run under this lock.
"""

import threading
from dataclasses import dataclass
from enum import Enum

from .protocol import _integer
from .slot_guard import LeaseIdentity


class RoleConflict(RuntimeError):
    pass


class TransportRole(str, Enum):
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


@dataclass(frozen=True)
class RoleSnapshot:
    owners: int
    inbound: int
    outbound: int
    revision: int


class TransportRoleArbiter:
    def __init__(self, *, max_owners=128, max_tombstones=65536):
        _integer("max_owners", max_owners, 1)
        _integer("max_tombstones", max_tombstones, 1)
        self.max_owners, self.max_tombstones = max_owners, max_tombstones
        self._lock = threading.Lock()
        self._owners = {}
        self._completed = {}
        self._bindings = {}
        self._nonces = {}
        self._inbound = self._revision = 0
        self._closed = False

    @staticmethod
    def _validate(role, owner):
        if type(role) is not TransportRole or type(owner) is not LeaseIdentity:
            raise ValueError("exact transport role and lease identity required")

    def acquire(self, role, owner):
        self._validate(role, owner)
        with self._lock:
            if self._closed:
                return False
            bound = self._bindings.get(owner.transfer_id)
            if bound is not None and bound != owner:
                raise RoleConflict("transfer binding conflict")
            nonce_owner = self._nonces.get(owner.nonce)
            if nonce_owner is not None and nonce_owner != owner:
                raise RoleConflict("nonce binding conflict")
            old = self._owners.get(owner) or self._completed.get(owner)
            if old is not None and old is not role:
                raise RoleConflict("owner role conflict")
            if owner in self._owners:
                return True
            if owner in self._completed:
                return False
            if (
                len(self._owners) >= self.max_owners
                or len(self._owners) + len(self._completed) >= self.max_tombstones
                or self._inbound
                or (role is TransportRole.INBOUND and self._owners)
            ):
                return False
            self._owners[owner] = role
            self._bindings[owner.transfer_id] = owner
            self._nonces[owner.nonce] = owner
            self._inbound += role is TransportRole.INBOUND
            self._revision += 1
            return True

    def release(self, role, owner):
        self._validate(role, owner)
        with self._lock:
            if self._completed.get(owner) is role:
                return False
            if self._owners.get(owner) is not role:
                raise RoleConflict("role ownership mismatch")
            del self._owners[owner]
            self._completed[owner] = role
            self._inbound -= role is TransportRole.INBOUND
            self._revision += 1
            return True

    def owner_snapshot(self):
        with self._lock:
            return RoleSnapshot(
                len(self._owners),
                self._inbound,
                len(self._owners) - self._inbound,
                self._revision,
            )

    def owns(self, role, owner):
        self._validate(role, owner)
        with self._lock:
            return self._owners.get(owner) is role

    def assert_quiescent(self):
        with self._lock:
            if self._owners:
                raise RoleConflict("active transport reservations remain")

    def close(self):
        with self._lock:
            if self._owners:
                raise RoleConflict("active transport reservations remain")
            self._closed = True
