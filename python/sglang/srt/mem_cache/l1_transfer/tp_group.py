"""Bounded TP4 consensus, invoked only by the fixed Scheduler control stage.

Only compact phase/identity digests cross the CPU group. Worker Events are local
handoffs; no network, CUDA wait, prompt, descriptor, or collective runs in them.
"""

import threading


def common_hint(hints):
    if len(hints) != 4 or any(h is None for h in hints):
        return None
    if len({h[0] for h in hints}) != 1:
        return None
    return hints[0][0], min(h[1] for h in hints)


def worker_decision(states, cancelled=False):
    if len(states) != 4:
        raise ValueError("TP4 needs four rank observations")
    if any(s[0] == "UNKNOWN" for s in states):
        return "QUARANTINE"
    if all(s[0] == "ENQUEUED" for s in states):
        return "DROP" if cancelled or not all(s[1] for s in states) else "PIN"
    if all(s[0] == "PIN" for s in states):
        return "ABORT" if cancelled or not all(s[1] for s in states) else "COPY"
    if all(s[0] == "DRAINED" for s in states):
        return "RELEASE"
    return None


class WorkerRendezvous:
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.waiting = threading.Event()
        self._state = ("WAIT", False)
        self._decision = None

    def arrive(self, phase, success):
        with self._lock:
            self._state = (phase, bool(success))
            self._decision = None
            self._event.clear()
            self.waiting.set()
        self._event.wait()  # backend worker only; Scheduler never waits here
        with self._lock:
            decision = self._decision
            self._state = (
                {"PIN": "WAIT", "COPY": "COPY"}.get(decision, "RELEASE"),
                False,
            )
            self.waiting.clear()
            return decision

    def snapshot(self):
        with self._lock:
            return self._state

    def permit(self, decision, *, before_release=None):
        expected = {
            "PIN": "ENQUEUED",
            "DROP": "ENQUEUED",
            "COPY": "PIN",
            "ABORT": "PIN",
            "RELEASE": "DRAINED",
        }[decision]
        with self._lock:
            # A worker can leave PIN and reach DRAINED between the collective
            # snapshot and this call. Never apply a stale Pin permit to Unpin.
            if self._state[0] != expected:
                return
            if self._decision is not None:
                return  # the first grant is the linearization point
            if before_release is not None:
                # Optional bounded scalar evidence must precede the worker
                # wakeup, so even a CPU fake cannot drain before its COPY event.
                before_release()
            self._decision = decision
            self._event.set()

    def unknown(self):
        with self._lock:
            self._state = ("UNKNOWN", False)


class SchedulerCollective:
    def __init__(self, gather, rank):
        if type(rank) is not int or not 0 <= rank < 4:
            raise ValueError("TP4 rank must be in [0,4)")
        self.rank, self._gather = rank, gather
        self._owner = threading.get_ident()

    @classmethod
    def from_runtime_tp_group(cls, tp_group, cache_group, rank):
        """Unwrap SGLang's GroupCoordinator and verify cache/TP membership.

        kv_cache_builder supplies a GroupCoordinator, whereas its CacheInitParams
        already contains the CPU ProcessGroup. Legacy callers can supply a raw
        ProcessGroup, but it must still match the cache group's ordered members.
        """
        import torch.distributed as dist

        cpu_group = getattr(tp_group, "cpu_group", tp_group)
        if cpu_group is None or cache_group is None:
            raise ValueError("explicit TP and cache CPU groups are required")
        members = tuple(dist.get_process_group_ranks(cpu_group))
        if len(members) != 4 or members != tuple(
            dist.get_process_group_ranks(cache_group)
        ):
            raise ValueError(
                "TP and cache CPU groups must contain the same ordered TP4 ranks"
            )
        if hasattr(tp_group, "cpu_group") and (
            tp_group.world_size != 4
            or tp_group.rank_in_group != rank
            or tuple(tp_group.ranks) != members
        ):
            raise ValueError("SGLang TP coordinator membership/rank mismatch")
        if dist.get_world_size(cache_group) != 4 or dist.get_rank(cache_group) != rank:
            raise ValueError("cache CPU group rank mapping mismatch")
        return cls.from_process_group(cpu_group, rank)

    @classmethod
    def from_process_group(cls, group, rank):
        import torch.distributed as dist

        if dist.get_world_size(group) != 4 or dist.get_rank(group) != rank:
            raise ValueError("Scheduler CPU group must be exact TP4 rank mapping")
        if str(dist.get_backend(group)).lower() != "gloo":
            raise ValueError("NVLink control collectives require a CPU/Gloo group")

        def gather(value):
            values = [None] * 4
            dist.all_gather_object(values, value, group=group)
            return values

        return cls(gather, rank)

    def gather(self, value):
        if threading.get_ident() != self._owner:
            raise RuntimeError("TP control collective belongs to Scheduler thread")
        values = self._gather(value)
        if len(values) != 4:
            raise RuntimeError("incomplete TP4 collective")
        return values
