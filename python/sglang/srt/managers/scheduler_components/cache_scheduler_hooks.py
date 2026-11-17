"""Scheduler steps a cache backend runs in lockstep across its TP ranks.

A backend whose allocator histories must stay identical on every rank installs
these hooks through its external linker. Without them the Scheduler runs each
step on its own rank, unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable, Optional, TypeVar

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

T = TypeVar("T")


class CacheSchedulerHooks(ABC):
    @abstractmethod
    def on_requests_received(
        self,
        *,
        paused: bool,
        batches: Iterable[Optional[ScheduleBatch]],
        chunked_req: Optional[Req],
    ) -> None:
        """Advance backend work after this iteration's requests are queued.

        ``batches`` hold every request with KV in use (running, last and
        in-flight); ``paused`` covers a pause and a graceful exit.
        """

    @abstractmethod
    def agree_prefill_entry(self, prepare: Callable[[], Optional[T]]) -> Optional[T]:
        """Run ``prepare`` and return its result once every rank agrees on it.

        None, on every rank, skips this pass's prefill allocation.
        """

    @abstractmethod
    def run_prefill_allocation(self, allocate: Callable[[], T]) -> T:
        """Run ``allocate`` as one common allocation step on every rank.

        ``allocate`` selects requests and takes their pages but builds no
        forward metadata; the Scheduler finishes the batch after this returns.
        """

    @abstractmethod
    def assert_ready_for_forward(self) -> None:
        """Raise when an allocation step is open or has failed on any rank."""

    @abstractmethod
    def flush(
        self, *, idle: bool, paused: bool, reset_pools: Callable[[], bool]
    ) -> bool:
        """Flush the cache through the backend; ``reset_pools`` does the pool reset.

        Returns False, with the cache unchanged, when any rank is busy.
        """
