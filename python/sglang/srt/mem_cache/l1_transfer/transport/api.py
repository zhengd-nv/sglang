# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral range transfer contracts (no CUDA import)."""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence


class Opcode(str, Enum):
    READ = "READ"   # remote -> local
    WRITE = "WRITE" # local -> remote


class TransferState(str, Enum):
    IN_FLIGHT = "IN_FLIGHT"
    SUCCESS = "SUCCESS"
    FAILED_DRAINED = "FAILED_DRAINED"
    FAILED_UNCERTAIN = "FAILED_UNCERTAIN"


class TransferError(RuntimeError):
    pass


class BusyError(TransferError):
    pass


def checked_int(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**63)")
    return value


@dataclass(frozen=True)
class MemoryHandle:
    region_id: str
    size: int
    device: int
    remote: bool
    address: int  # process-local; never use as remote identity


@dataclass(frozen=True)
class RegionDescriptor:
    version: int
    transport: str
    owner_id: str
    region_id: str
    lease_id: str
    size: int
    physical_size: int
    granularity: int
    access: str
    transport_handle: str  # opaque serialized backend metadata; not a CUDA-specific contract

    def validate(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported descriptor schema")
        if not isinstance(self.transport, str) or not self.transport:
            raise ValueError("invalid transport")
        for name in ("owner_id", "region_id", "lease_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 32:
                raise ValueError(f"invalid {name}")
            try:
                int(value, 16)
            except ValueError:
                raise ValueError(f"invalid {name}") from None
        for name in ("size", "physical_size", "granularity"):
            checked_int(getattr(self, name), name, 1)
        if self.physical_size < self.size or self.physical_size % self.granularity:
            raise ValueError("invalid physical extent")
        if self.access not in ("read", "write"):
            raise ValueError("access must be read or write")
        if not isinstance(self.transport_handle, str) or not self.transport_handle:
            raise ValueError("invalid opaque transport handle")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RegionDescriptor":
        obj = cls(**value)
        obj.validate()
        return obj


@dataclass(frozen=True)
class TransferRequest:
    opcode: Opcode
    local: MemoryHandle
    remote: MemoryHandle
    local_offset: int
    remote_offset: int
    length: int


@dataclass(frozen=True)
class BatchHandle:
    batch_id: str


@dataclass(frozen=True)
class TransferStatus:
    state: TransferState
    requested_bytes: int
    completed_bytes: int | None
    transport: str = "nvlink-fabric"
    error: str = ""

    @property
    def drained(self) -> bool:
        return self.state in (TransferState.SUCCESS, TransferState.FAILED_DRAINED)


class TransferEngine(Protocol):
    """Common adapter target for native CUDA, TE, NIXL, etc.

    Calls enqueue/observe batches; neither exception nor timeout implies drain.
    Export lifetime and remote acknowledgement belong to the caller's control plane.
    """
    def allocate_memory(self, size: int) -> MemoryHandle: ...
    def export_memory(self, memory: MemoryHandle, access: str = "read") -> RegionDescriptor: ...
    def import_memory(self, descriptor: RegionDescriptor) -> MemoryHandle: ...
    def submit_transfer(self, requests: Sequence[TransferRequest]) -> BatchHandle: ...
    def get_transfer_status(self, batch: BatchHandle) -> TransferStatus: ...
    def wait(self, batch: BatchHandle, timeout: float | None = None) -> TransferStatus: ...
    def release_batch(self, batch: BatchHandle) -> None: ...
    def unregister_memory(self, memory: MemoryHandle) -> None: ...
    def release_export(self, lease_id: str) -> None: ...
