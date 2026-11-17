"""Scheduler-owned transfer role, acquired before any transfer resource."""

import threading
from enum import Enum

from .protocol import TransferId, _typed


class SchedulerThreadViolation(RuntimeError):
    pass


class LeaseRejected(ValueError):
    pass


class EngineRole(str, Enum):
    IDLE = "IDLE"
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


class RoleGate:
    """One Scheduler-owned inbound OR outbound attempt, acquired before resources."""

    def __init__(self):
        self.owner_thread = threading.get_ident()
        self._role = EngineRole.IDLE
        self._transfer_id = None

    def check_thread(self):
        if threading.get_ident() != self.owner_thread:
            raise SchedulerThreadViolation("role gate belongs to the Scheduler thread")

    @property
    def role(self):
        self.check_thread()
        return self._role

    def acquire(self, role: EngineRole, transfer_id: TransferId) -> bool:
        self.check_thread()
        _typed("role", role, EngineRole)
        _typed("transfer_id", transfer_id, TransferId)
        if role is EngineRole.IDLE:
            raise ValueError("cannot acquire IDLE")
        if self._role is not EngineRole.IDLE:
            return False
        self._role, self._transfer_id = role, transfer_id
        return True

    def release(self, role: EngineRole, transfer_id: TransferId):
        self.check_thread()
        if self._role is not role or self._transfer_id != transfer_id:
            raise LeaseRejected("role gate ownership mismatch")
        self._role, self._transfer_id = EngineRole.IDLE, None

    def require_owner(self, role: EngineRole, transfer_id: TransferId):
        """Check a Scheduler reservation acquired before allocating resources."""
        self.check_thread()
        if self._role is not role or self._transfer_id != transfer_id:
            raise LeaseRejected("role gate ownership mismatch")
