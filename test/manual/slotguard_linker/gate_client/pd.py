"""Client-side PD dual dispatch to an explicitly chosen Prefill.

Mirrors SGLang's PD routers so that warm and measured requests can be pinned to
P0 or P1 (a router would pick the prefill itself):

- the same /generate JSON body goes to P and to D, with ``bootstrap_host`` (the
  P hostname, IPv6 bracketed), ``bootstrap_port`` (P's disaggregation bootstrap
  port) and a random ``bootstrap_room`` in [0, 2**63 - 1] added
  (sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py:459-486,
  539-540; Rust router pd_router.rs:236-297, pd_types.rs:60-63);
- both POSTs are issued concurrently and only D's stream is returned to the
  caller; P's response is kept open and drained (mini_lb.py:216-266).

With several decode servers, request i of a group goes to decode[i % n]
(deterministic; ``decode_index``/``decode_url`` are recorded per sample).

Timing uses one client clock (time.monotonic). ``send`` is taken immediately
before the first of the two POSTs is written; ``first_token`` when the first
D frame carrying output ids is read; ``completion`` when the D frame carrying a
finish_reason is read.
"""

import http.client
import ipaddress
import json
import random
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

MAX_BOOTSTRAP_ROOM = 2**63 - 1
DEFAULT_BOOTSTRAP_PORT = 8998  # arg_groups/fields/disagg.py:78-80
_ROOM_RNG = random.SystemRandom()
STREAM_MODES = ("auto", "incremental", "cumulative")


def new_bootstrap_room() -> int:
    return _ROOM_RNG.randint(0, MAX_BOOTSTRAP_ROOM)


def maybe_wrap_ipv6(host: str) -> str:
    try:
        ipaddress.IPv6Address(host)
    except ValueError:
        return host
    return f"[{host}]"


@dataclass(frozen=True)
class Server:
    url: str
    bootstrap_host: Optional[str] = None
    bootstrap_port: int = DEFAULT_BOOTSTRAP_PORT

    @property
    def _parsed(self):
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
            raise ValueError(f"server url must be http://host:port, got {self.url!r}")
        return parsed

    @property
    def host(self) -> str:
        return self._parsed.hostname

    @property
    def port(self) -> int:
        return self._parsed.port

    @property
    def resolved_bootstrap_host(self) -> str:
        return maybe_wrap_ipv6(self.bootstrap_host or self.host)


@dataclass(frozen=True)
class Sampling:
    max_new_tokens: int
    min_new_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False

    def to_json(self) -> dict:
        return dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            ignore_eos=self.ignore_eos,
        )


@dataclass(frozen=True)
class Request:
    rid: str
    prefill: int
    sampling: Sampling
    input_ids: Optional[Sequence[int]] = None
    text: Optional[str] = None
    tag: dict = field(default_factory=dict)

    @property
    def input_tokens(self) -> Optional[int]:
        return None if self.input_ids is None else len(self.input_ids)


def build_body(request: Request, prefill: Server, room: int) -> dict:
    if (request.input_ids is None) == (request.text is None):
        raise ValueError("exactly one of input_ids or text is required")
    body = {"rid": request.rid}
    if request.input_ids is not None:
        body["input_ids"] = list(request.input_ids)
    else:
        body["text"] = request.text
    body.update(
        sampling_params=request.sampling.to_json(),
        stream=True,
        bootstrap_host=prefill.resolved_bootstrap_host,
        bootstrap_port=prefill.bootstrap_port,
        bootstrap_room=room,
    )
    return body


class StreamError(RuntimeError):
    pass


class StreamState:
    """Accumulates /generate SSE frames from either streaming output mode.

    With --incremental-streaming-output the server sends deltas, otherwise the
    cumulative list (tokenizer_manager.py:2638-2680). ``auto`` decides per frame
    from meta_info.completion_tokens; both readings agree on the first frame.
    """

    def __init__(self, mode: str = "auto"):
        if mode not in STREAM_MODES:
            raise ValueError(f"stream mode must be one of {STREAM_MODES}")
        self.mode = mode
        self.ids: list = []
        self.meta: Optional[dict] = None
        self.finish_reason = None
        self.frames = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    def add(self, obj) -> bool:
        """Apply one decoded frame; returns True if new output ids appeared."""
        self.frames += 1
        if not isinstance(obj, dict):
            raise StreamError("non-object stream frame")
        if "error" in obj:
            error = obj["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise StreamError(f"server stream error: {str(message)[:300]}")
        if self.finished:
            raise StreamError("frame after terminal frame")
        ids = obj.get("output_ids") or []
        if not isinstance(ids, list) or not all(type(t) is int for t in ids):
            raise StreamError("invalid output_ids")
        meta = obj.get("meta_info") or {}
        before = len(self.ids)
        self.ids = self._merge(ids, meta.get("completion_tokens"))
        self.meta = meta
        reason = meta.get("finish_reason")
        if reason is not None:
            if isinstance(reason, dict) and reason.get("type") == "abort":
                raise StreamError(
                    f"request aborted: {str(reason.get('message'))[:300]}"
                )
            self.finish_reason = reason
        return len(self.ids) > before

    def _merge(self, ids, completion):
        mode = self.mode
        if mode == "auto":
            if type(completion) is not int:
                raise StreamError("auto stream mode needs meta_info.completion_tokens")
            appended = len(self.ids) + len(ids) == completion
            replaced = len(ids) == completion and ids[: len(self.ids)] == self.ids
            if appended and not replaced:
                mode = self.mode = "incremental"
            elif replaced and not appended:
                mode = self.mode = "cumulative"
            elif appended and replaced:
                # Only possible while nothing is accumulated yet; keep deciding.
                return list(ids)
            else:
                raise StreamError("output_ids inconsistent with completion_tokens")
        if mode == "incremental":
            return self.ids + ids
        if ids[: len(self.ids)] != self.ids:
            raise StreamError("cumulative output regressed or changed")
        return list(ids)


class _Leg:
    """One HTTP POST /generate to P or D, connected before the start barrier."""

    def __init__(self, role: str, server: Server, body: bytes, timeout: float):
        self.role, self.server, self.body = role, server, body
        self.conn = http.client.HTTPConnection(
            server.host, server.port, timeout=timeout
        )
        self.send = self.send_unix = self.first = self.completion = None
        self.http_status = None
        self.state = None
        self.done_marker = False
        self.error: Optional[str] = None
        self.aborted = False

    def connect(self):
        self.conn.connect()
        # Large bodies go out in one write; Nagle would only add latency.
        self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def abort(self):
        self.aborted = True
        sock = self.conn.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def run(
        self,
        *,
        start: threading.Event,
        start_at: list,
        stream_mode: str,
        clock: Callable[[], float],
        on_error: Callable[["_Leg"], None],
    ):
        try:
            start.wait()
            delay = start_at[0] - clock()
            if delay > 0:
                time.sleep(delay)
            self.send_unix, self.send = time.time(), clock()
            self.conn.request(
                "POST",
                "/generate",
                body=self.body,
                headers={"Content-Type": "application/json"},
            )
            response = self.conn.getresponse()
            self.http_status = response.status
            if response.status != 200:
                detail = response.read(512).decode("utf-8", "replace")
                raise StreamError(f"HTTP {response.status}: {detail}")
            self._consume(response, stream_mode, clock)
        except Exception as exc:  # noqa: BLE001 -- recorded per leg, raised by the group
            if self.aborted and self.error is None:
                self.error = "aborted after sibling/deadline failure"
            elif self.error is None:
                self.error = f"{type(exc).__name__}: {str(exc)[:400]}"
            on_error(self)
        finally:
            self.conn.close()

    def _consume(self, response, stream_mode, clock):
        self.state = StreamState(stream_mode)
        while True:
            line = response.readline()
            now = clock()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                self.done_marker = True
                break
            grew = self.state.add(json.loads(data))
            if grew and self.first is None:
                self.first = now
            if self.state.finished and self.completion is None:
                self.completion = now
        if not self.state.finished:
            raise StreamError("stream ended without a finish_reason")


def _meta_view(meta: Optional[dict]) -> Optional[dict]:
    if meta is None:
        return None
    keys = (
        "id",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cached_tokens_details",
        "finish_reason",
        "num_retractions",
    )
    return {k: meta[k] for k in keys if k in meta}


def _sample(
    request: Request,
    prefill: Server,
    decode: Server,
    room: int,
    p: _Leg,
    d: _Leg,
    decode_index: int = 0,
) -> dict:
    sends = [leg.send for leg in (p, d) if leg.send is not None]
    ok = (
        p.error is None and d.error is None and d.state is not None and d.state.finished
    )
    return dict(
        rid=request.rid,
        prefill=request.prefill,
        prefill_url=prefill.url,
        decode_url=decode.url,
        decode_index=decode_index,
        bootstrap_host=prefill.resolved_bootstrap_host,
        bootstrap_port=prefill.bootstrap_port,
        bootstrap_room=room,
        input_tokens=request.input_tokens,
        params=request.sampling.to_json(),
        tag=dict(request.tag),
        send=min(sends) if sends else None,
        send_unix=min((leg.send_unix for leg in (p, d) if leg.send_unix), default=None),
        p_send=p.send,
        d_send=d.send,
        first_token=d.first,
        completion=d.completion,
        p_completion=p.completion,
        output_ids=list(d.state.ids) if d.state else [],
        finish_reason=d.state.finish_reason if d.state else None,
        stream_mode=d.state.mode if d.state else None,
        frames=d.state.frames if d.state else 0,
        terminal=bool(ok),
        # Kept for upstream group_metrics(): CONFIRMED = both legs ended normally.
        remote_completion="CONFIRMED" if ok else "UNKNOWN",
        log_error=None,
        status="COMPLETE" if ok else "ERROR",
        error={"p": p.error, "d": d.error} if not ok else None,
        p_http_status=p.http_status,
        d_http_status=d.http_status,
        p_meta=_meta_view(p.state.meta if p.state else None),
        d_meta=_meta_view(d.state.meta if d.state else None),
    )


def decode_servers(decode) -> list:
    """Normalize one Server or a sequence of Servers to a non-empty list."""
    decodes = [decode] if isinstance(decode, Server) else list(decode)
    if not decodes:
        raise ValueError("at least one decode server is required")
    return decodes


class GroupFailed(RuntimeError):
    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.result = result


def dispatch_group(
    requests: Sequence[Request],
    prefills: Sequence[Server],
    decode,
    *,
    timeout: float = 900.0,
    connect_timeout: float = 30.0,
    before_dispatch: Optional[Callable[[], None]] = None,
    stream_mode: str = "auto",
    start_delay: float = 0.05,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Send C requests behind one common start; returns per-lane samples.

    ``decode`` is one Server or a sequence; request i uses decode[i % len].

    Bodies are encoded and every socket is connected before ``before_dispatch``
    runs, so neither is inside the timed window. Raises GroupFailed (with the
    partial result) if any lane fails or the group deadline passes.
    """
    if not requests:
        raise ValueError("empty request group")
    if len({r.rid for r in requests}) != len(requests):
        raise ValueError("duplicate request ID")
    decodes = decode_servers(decode)
    prepared_at = clock()
    lanes = []
    for i, request in enumerate(requests):
        prefill = prefills[request.prefill]
        d_index = i % len(decodes)
        room = new_bootstrap_room()
        body = json.dumps(
            build_body(request, prefill, room), separators=(",", ":")
        ).encode()
        p = _Leg("prefill", prefill, body, connect_timeout)
        d = _Leg("decode", decodes[d_index], body, connect_timeout)
        lanes.append((request, prefill, room, p, d, d_index))
    sibling = {}
    for _, _, _, p, d, _ in lanes:
        sibling[id(p)], sibling[id(d)] = d, p
    legs = [leg for lane in lanes for leg in lane[3:5]]
    try:
        for leg in legs:
            leg.connect()
            leg.conn.sock.settimeout(timeout)
    except Exception:
        for leg in legs:
            leg.conn.close()
        raise
    prepare_s = clock() - prepared_at

    def on_error(leg):
        sibling[id(leg)].abort()

    start, start_at = threading.Event(), [0.0]
    threads = [
        threading.Thread(
            target=leg.run,
            kwargs=dict(
                start=start,
                start_at=start_at,
                stream_mode=stream_mode,
                clock=clock,
                on_error=on_error,
            ),
            name=f"pd-{leg.role}-{i // 2}",
            daemon=True,
        )
        for i, leg in enumerate(legs)
    ]
    for thread in threads:
        thread.start()
    hook_started = clock()
    try:
        if before_dispatch is not None:
            before_dispatch()
    except BaseException:
        for leg in legs:
            leg.abort()
        start.set()
        raise
    before_hook_s = clock() - hook_started
    start_at[0] = clock() + start_delay
    start.set()
    deadline = start_at[0] + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - clock()))
    timed_out = any(thread.is_alive() for thread in threads)
    if timed_out:
        for leg, thread in zip(legs, threads):
            if thread.is_alive() and leg.error is None:
                leg.error = "group deadline exceeded"
            leg.abort()
        for thread in threads:
            thread.join(5)
    samples = [
        _sample(r, prefill, decodes[di], room, p, d, di)
        for r, prefill, room, p, d, di in lanes
    ]
    sends = [s["send"] for s in samples if s["send"] is not None]
    result = dict(
        samples=samples,
        barrier=start_at[0],
        start_skew_s=(max(sends) - min(sends)) if sends else None,
        prepare_s=prepare_s,
        before_hook_s=before_hook_s,
        timed_out=timed_out,
    )
    failed = [s["rid"] for s in samples if s["status"] != "COMPLETE"]
    if failed:
        raise GroupFailed(f"{len(failed)}/{len(samples)} lanes failed", result)
    return result
