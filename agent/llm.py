"""Thin OpenRouter client. One code path for every model in the harness.

Streaming matters here beyond the live output: it separates queue-and-prefill time
from generation time. A diffusion model and an autoregressive model of the same
nominal tok/s have very different shapes — Mercury refines tokens in parallel, so
its time-to-first-token and its decode rate tell you different things, and a single
end-to-end latency number hides both.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import random
import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests
import requests.adapters

from .config import OPENROUTER_BASE_URL
from .providers import OPENROUTER, Provider, effective_temperature, is_mercury

# Anthropic caches a prompt prefix for five minutes; the loop resends the system
# prompt and task on every iteration, so the prefix is worth a breakpoint.
CACHEABLE_PREFIXES = ("anthropic/",)
# Below Anthropic's minimum cacheable prefix a breakpoint is simply ignored.
MIN_CACHE_TOKENS = 1024
# Status codes worth another attempt, wherever they appear: as an HTTP status, or
# inside an error envelope the provider sends with a 200 or mid-stream.
RETRYABLE_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
# A 402 is usually final (no credit), except this one: OpenRouter caps concurrent requests by
# the account balance and answers the excess with 402 and a Retry-After. Treated as final,
# it killed 12 and 17 runs of two SciCode arms at a $0.09 balance (DESIGN-LOG §81.1).
IN_FLIGHT_CAP_REASON = "in_flight_budget_exhausted"
# Immediate reconnects allowed within one chat() call, before falling back to
# ordinary backoff. One covers the common case — a pooled connection that died
# while idle, which no amount of waiting repairs — without letting a genuinely
# unreachable host be hammered.
RECONNECTS_PER_CALL = 2
# Seconds without a single byte before a streamed attempt is abandoned.
# Time-to-first-token across 10,390 traced calls: median 1.14s, p90 5.46s.
# Waiting the full 180s read timeout for a byte that normally arrives in one
# is how a half-open connection costs a run twelve minutes instead of thirty
# seconds. Generous against p90, still twenty times tighter than `timeout`.
FIRST_BYTE_TIMEOUT_S = 45.0

# A diffusion model (Mercury/Inception) does not generate token by token: it
# spends most of a call's wall time on prefill and parallel refinement before
# committing to any output at all, then delivers the whole thing in a short
# burst. Measured across 84 traced `inception/` calls in one sweep: median
# TTFT 15.2s, p90 25.0s, p99 51.1s -- already past the 45s bound above, which
# was sized against the all-model p90 of 5.46s and would misclassify a
# perfectly healthy slow prefill as a half-open connection and retry it. The
# fix is not to loosen FIRST_BYTE_TIMEOUT_S for everyone (that would blunt
# the exact stall detection it exists for on faster, autoregressive models),
# it's to give this one model family its own, correctly-sized bound.
DIFFUSION_MODEL_PREFIXES = ("inception/",)
DIFFUSION_FIRST_BYTE_TIMEOUT_S = 75.0


# Seconds a Mercury attempt may stream nothing but keep-alive comments before
# it is abandoned and retried. The first-byte bound above cannot see this case:
# OpenRouter answers with headers and `: OPENROUTER PROCESSING` comments at once,
# so the first byte arrives in a second even when the model never starts, and
# the attempt then runs until the provider's own 504 at ~120s. Across 50,924
# single-attempt Mercury calls, time to first content was 1.15s median, 11.7s
# p99, 94 over 30s; 330 retried calls each lost ~120s to a first attempt that
# produced nothing. At 30s: ~29,700s saved against ~3,000s spent re-asking the
# 94 slow-but-healthy calls (DESIGN-LOG §69). Diffusion only: that is what was
# measured.
DIFFUSION_FIRST_CONTENT_TIMEOUT_S = 30.0


def _first_content_timeout_for(model: str) -> float | None:
    return DIFFUSION_FIRST_CONTENT_TIMEOUT_S if _is_diffusion_model(model) else None


def _is_diffusion_model(model: str) -> bool:
    # A direct provider's request names Mercury by its own id (`mercury-2.5`),
    # not the `inception/` id OpenRouter uses; both get Mercury's bounds.
    return model.startswith(DIFFUSION_MODEL_PREFIXES) or is_mercury(model)


def _first_byte_timeout_for(model: str) -> float:
    return DIFFUSION_FIRST_BYTE_TIMEOUT_S if _is_diffusion_model(model) else FIRST_BYTE_TIMEOUT_S


def _keepalive_options() -> list[tuple[int, int, int]]:
    """Socket options that make a vanished peer detectable.

    Enabling SO_KEEPALIVE alone leaves the OS defaults, which on macOS and
    Linux begin probing only after two hours. The per-platform tuning
    constants are what make it useful here, and each is applied only if this
    platform defines it.
    """
    options: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    # Idle-before-probing is TCP_KEEPIDLE on Linux and TCP_KEEPALIVE on
    # macOS; whichever exists is the same setting under two names.
    for names, value in (
        (("TCP_KEEPIDLE", "TCP_KEEPALIVE"), 30),
        (("TCP_KEEPINTVL",), 10),
        (("TCP_KEEPCNT",), 3),
    ):
        option = next((getattr(socket, n) for n in names if hasattr(socket, n)), None)
        if option is not None:
            options.append((socket.IPPROTO_TCP, option, value))
    return options


_KEEPALIVE_OPTIONS = _keepalive_options()


class _KeepAliveAdapter(requests.adapters.HTTPAdapter):
    """An adapter whose connections carry TCP keepalive.

    The characteristic failure of an unstable link is silence, not a reset:
    a WiFi drop or a NAT timeout leaves the connection half-open, the peer
    never sends anything, and nothing below the read timeout notices. A
    probe simulating exactly that burned the whole 420s call budget — seven
    attempts, each waiting out its own 180s read timeout — to discover a
    connection that had been dead the entire time.

    `socket_options` belongs to urllib3's connection, not to HTTPAdapter, so
    it has to be threaded through init_poolmanager rather than passed to the
    constructor.
    """

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs.setdefault("socket_options", _KEEPALIVE_OPTIONS)
        super().init_poolmanager(connections, maxsize, block=block, **kwargs)


def _is_socket_level(exc: BaseException) -> bool:
    """Whether the network dropped under us, as opposed to the provider
    answering with something we should wait out.

    ConnectionError covers a refused, reset or unreachable connection;
    ChunkedEncodingError is a stream that died part-way, which on an unstable
    link is the same event seen later. ReadTimeout is deliberately excluded:
    the server took the request and went quiet, so the connection is probably
    fine and retrying instantly would hammer something already struggling.
    """
    return isinstance(exc, (requests.ConnectionError, requests.exceptions.ChunkedEncodingError))


# Reasoning efforts that mean "do not think", which a model that must reason refuses.
_REASONING_OFF = frozenset({"none", "minimal", "off"})


class LLMError(Exception):
    pass


class ModelRefused(LLMError):
    """The provider answered with a refusal and nothing else (DESIGN-LOG §127.9).

    Sonnet 5.5 refuses some prompts with HTTP 200, `finish_reason: content_filter`
    (OpenRouter's native reason `refusal`), no content, no tool call and zero tokens.
    Read as an ordinary empty reply it looked like a model that did not call a tool,
    and the planner quietly fell back to a one-step plan on every run for two days.
    Unlike §129.4's 400s it names no setting to loosen; the one retry is without the
    known refusal-prone wording (_REFUSED_WORDING), and only when the request carried it.
    Raised when that retry is refused too, or there was nothing to reword."""

    def __init__(self, model: str, provider: str = "", finish_reason: str = "content_filter"):
        self.model, self.provider, self.finish_reason = model, provider, finish_reason
        via = f" via {provider}" if provider else ""
        super().__init__(
            f"{model}{via} refused the request (finish_reason {finish_reason}, no content, "
            "no tool call); the prompt itself is being refused, so retrying will not help"
        )


_REFUSAL_REASONS = frozenset({"content_filter", "refusal"})


# Wording a model may refuse because it asks the model to hand over its reasoning, and what
# replaces it for that model only. Sonnet 5.5 refused every plan call while submit_plan's
# description said "after your chain-of-draft reasoning" and planned without it (§127.9).
# Dropping it for every model cut Mercury's steps a plan (§127.10.1), so only a model that has
# refused is sent the plain wording; every other model's request stays byte-for-byte as it was.
_REFUSED_WORDING = ((", after your chain-of-draft reasoning.", "."),)


def without_refused_wording(tools: list[dict]) -> list[dict] | None:
    """`tools` with the refusal-prone wording replaced, or None when none of it is there.
    New dicts where something changed; the caller's schemas are never mutated."""
    changed = False
    out = []
    for tool in tools:
        fn = tool.get("function") or {}
        text = str(fn.get("description") or "")
        plain = text
        for wording, replacement in _REFUSED_WORDING:
            plain = plain.replace(wording, replacement)
        if plain != text:
            tool = {**tool, "function": {**fn, "description": plain}}
            changed = True
        out.append(tool)
    return out if changed else None


def is_refusal(reply: Reply) -> bool:
    """A reply that is only a refusal: a filter finish reason with nothing in it."""
    return (
        reply.finish_reason in _REFUSAL_REASONS
        and not (reply.content or "").strip()
        and not reply.tool_calls
    )


class BudgetExceeded(Exception):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    # Time until the first content or tool-call delta arrived. Streaming only.
    ttft_s: float = 0.0
    # Prompt tokens served from cache, when the provider reports them.
    cached_tokens: int = 0
    # The share of completion_tokens the model spent reasoning, when the
    # provider reports it. Mercury 2.5 through OpenRouter returns the count
    # but never the reasoning text, and at medium/high effort it is ~90% of
    # completion_tokens even for a one-line answer — so without this field a
    # trace cannot say whether a call reasoned or just wrote.
    reasoning_tokens: int = 0
    # The longest silence between deltas after the first one arrived, streaming
    # only. Mercury reasons in hidden tokens, and a runaway call goes quiet for a
    # minute or more after its first delta (DESIGN-LOG §95); this is what sizes
    # the bound on that silence (`content_stall_s`).
    max_gap_s: float = 0.0

    @property
    def decode_s(self) -> float:
        """Time spent generating, once the first token was out."""
        return max(0.0, self.latency_s - self.ttft_s)

    @property
    def decode_tok_s(self) -> float:
        """Generation rate excluding queue and prefill. 0 when not streamed."""
        if self.ttft_s <= 0 or self.decode_s <= 0:
            return 0.0
        return self.completion_tokens / self.decode_s

    @property
    def end_to_end_tok_s(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return self.completion_tokens / self.latency_s


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)
    # How many attempts this reply cost, and the wall-clock seconds the whole
    # sequence took including every failed attempt and backoff. `usage.latency_s`
    # times only the attempt that succeeded, which is why a call that spent
    # 1,954 seconds being retried recorded itself as 1.17s and the stall was
    # invisible in the trace for as long as it was. `attempts == 1` and
    # `total_s == latency_s` on the ordinary path.
    attempts: int = 1
    total_s: float = 0.0
    # The last retryable error seen, when there was one. Without it a stall is
    # only ever "something took a long time"; with it the trace says whether
    # the provider was rate-limiting, erroring, or holding the stream open.
    retry_error: str = ""
    # What the provider says actually served the call, as opposed to the model id
    # that was requested. A routing layer can serve a different snapshot, or a
    # different upstream provider, under an unchanged requested id, and a
    # benchmark that only records the requested one cannot see it: two sweeps
    # of "the same" model a month apart may be two different models. Empty when
    # the provider does not say.
    served_model: str = ""
    provider: str = ""
    # A provider's note that it changed the request (Inception: "Requested
    # temperature 0.0 is not within [0.5, 1]. Temperature has been set to 1.").
    # OpenRouter drops it; a direct call keeps it so the trace can say so.
    warning: str = ""

    def as_message(self) -> dict:
        """The assistant message to append to history, in OpenAI wire format."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class ChatClient(Protocol):
    """What the phases actually depend on. Declaring it keeps the loop testable
    without a network, and keeps OpenRouterClient replaceable."""

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = ...,
        tool_choice: str | dict | None = ...,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        reasoning_effort: str | None = ...,
    ) -> Reply: ...


def supports_caching(model: str) -> bool:
    return model.startswith(CACHEABLE_PREFIXES)


def apply_cache_control(messages: list[dict], model: str) -> list[dict]:
    """Mark the stable prefix — system prompt plus the task — as cacheable.

    Everything before the breakpoint is reused across the loop's iterations, which
    is most of the prompt for a phase that makes a dozen tool calls. Providers that
    do not support it reject the array form, so this only runs where it works.
    """
    if not supports_caching(model) or not messages:
        return messages
    prefix_chars = sum(len(str(m.get("content", ""))) for m in messages[:2])
    if prefix_chars // 4 < MIN_CACHE_TOKENS:
        return messages

    marked = list(messages)
    # The breakpoint goes on the last message of the stable prefix: the task.
    for index in (1, 0):
        if index >= len(marked):
            continue
        message = dict(marked[index])
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        marked[index] = message
        return marked
    return messages


def strict_schemas(tools: list[dict]) -> list[dict]:
    """`tools` with `strict: true` on each function, otherwise unchanged."""
    return [
        {**tool, "function": {**tool["function"], "strict": True}}
        if tool.get("type") == "function"
        else tool
        for tool in tools
    ]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 6,
        max_call_s: float = 420.0,
        stream: bool = True,
        cache_prompts: bool = True,
        on_delta: Callable[[str], None] | None = None,
        on_call_start: Callable[[], None] | None = None,
        on_call_end: Callable[[], None] | None = None,
        on_reconnect: Callable[[int, str], None] | None = None,
        on_attempt_failed: Callable[[int, float, str], None] | None = None,
        strict_tools: bool = False,
        provider_sort: str = "",
        provider: Provider = OPENROUTER,
        mercury_temperature: float | None = None,
        mercury_max_tokens: int | None = None,
        content_stall_s: float | None = None,
        runaway_step_down: bool = False,
    ):
        # Which API this client talks to, and in which dialect (agent/providers.py).
        # One transport for all of them, so every retry and stall bound applies.
        self.provider = provider
        self.base_url = (base_url or provider.base_url or OPENROUTER_BASE_URL).rstrip("/")
        # Settings.mercury_temperature; see providers.effective_temperature.
        self.mercury_temperature = mercury_temperature
        # What a provider has refused, learned from its 400 and remembered per model so the
        # refusal is paid for once (DESIGN-LOG §129.4): a forced tool_choice, and reasoning off.
        self.refuses_forced_choice: set[str] = set()
        self.refuses_no_reasoning: set[str] = set()
        # Models that refused a request carrying _REFUSED_WORDING (§127.9, §127.11).
        self.refuses_reasoning_wording: set[str] = set()
        # Settings.mercury_max_tokens: a cap for Mercury calls that set none.
        self.mercury_max_tokens = mercury_max_tokens
        # Settings.mercury_content_stall_s: cut a Mercury stream that has said
        # something and then goes silent this long, and retry it. The first-content
        # cut only covers the silence before the first delta; after it, only the
        # attempt deadline and the provider's own 504 at ~120s ended a runaway
        # (DESIGN-LOG §95). None leaves it off.
        self.content_stall_s = content_stall_s
        # Settings.runaway_effort_step_down (DESIGN-LOG §131.12): when a Mercury call at medium
        # or high effort is cut for no content, or ends at the token cap having only reasoned,
        # ask again at low effort instead of repeating the same request. Mercury 3 streams no
        # reasoning, and on SciCode 70 reasoned to the 16k cap without answering in 6 of 7
        # calls at medium or high; at low it answered 2 of 3.
        self.runaway_step_down = runaway_step_down
        # OpenRouter's provider preference ("latency", "throughput", "price"). A model
        # served by many providers varies >10x between them: GLM 5.3 Flash's median
        # time to first token was 0.6s on Together and 5-16s on most others, over 288
        # traced calls (DESIGN-LOG §77). Empty leaves OpenRouter's default routing.
        self.provider_sort = provider_sort
        # Send every tool schema with `strict: true`, asking the provider to
        # keep arguments to the declared types while decoding. Here rather
        # than per tool so every phase's calls get it from one switch
        # (Settings.strict_tool_schemas, DESIGN-LOG §63).
        self.strict_tools = strict_tools
        self.timeout = timeout
        self.max_retries = max_retries
        # Ceiling on one chat() call including every retry and backoff, not
        # just one attempt. See _request for the traced 1,954s stall this
        # exists to bound. Comfortably above a slow-but-real call, well below
        # what seven attempts plus six backoffs would otherwise cost.
        self.max_call_s = max_call_s
        self.stream = stream
        self.cache_prompts = cache_prompts
        # Called with each content delta, for live output.
        self.on_delta = on_delta
        # Bracket one call: on_call_start right before it's sent, on_call_end
        # once there is something else to show instead — the first content
        # delta, or (for a tool-only reply, or on failure) the call simply
        # finishing. A "still working" indicator hangs these; neither
        # callback is required to do anything but must never raise into a
        # retry loop, so both are called defensively.
        self.on_call_start = on_call_start
        self.on_call_end = on_call_end
        # Called when a dropped connection is re-established, so a run's trace
        # can tell an unstable link apart from a slow provider. Both look like
        # a retry from the outside and neither is fixed the same way.
        self.on_reconnect = on_reconnect
        # Called for every attempt that fails, with (attempt, seconds, error).
        # `llm_retried` only fires from trace.llm_call, which only runs when a
        # call eventually succeeds — so a call that burns its whole budget and
        # then raises left no record of where the time went at all. Two stalls
        # in one sweep reported 1,100s and 500s gaps whose shape had to be
        # inferred from arithmetic; this is so the next one does not.
        self.on_attempt_failed = on_attempt_failed
        self.session = requests.Session()
        # One Session is shared by every concurrent step in a run — schedule.py
        # submits a batch to a ThreadPoolExecutor sized to the batch — and
        # urllib3's default pool holds ten. Past that it discards connections
        # on release, reintroducing exactly the per-request handshake that
        # `_drain` exists to avoid. `max_retries=0` is deliberate: urllib3
        # retrying underneath `_request` would multiply the stall rather than
        # bound it, which is the opposite of what this file is trying to do.
        adapter = _KeepAliveAdapter(pool_connections=4, pool_maxsize=32, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # OpenRouter's attribution headers; empty for other providers.
                **provider.headers,
            }
        )

    def _reset_pool(self) -> None:
        """Drop every pooled connection, keeping the Session usable.

        `HTTPAdapter.close()` clears its PoolManager rather than disabling
        it, so the next request opens fresh sockets. That is the whole point
        here: after a link drops, the sockets sitting in the pool are dead,
        and a retry that draws one is a second failure for the same reason.
        """
        with contextlib.suppress(Exception):
            self.session.close()

    def close(self) -> None:
        """Release the connection pool. Idempotent, and safe to call twice.

        `requests.Session` holds its pooled sockets until it is closed or
        garbage-collected, and nothing closed this one. For a single run that
        leaks a handful of keep-alive connections at exit, which is harmless.
        A TUI session is not a single run: every prompt builds a fresh client
        through `_build_run`, so twenty prompts left twenty pools of idle
        sockets open against the provider, none of them ever reused again.
        That is the shape of client a host's connection limits exist to
        refuse, and it costs nothing to not be it.
        """
        with contextlib.suppress(Exception):
            self.session.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Reply:
        payload = messages
        if self.cache_prompts:
            payload = apply_cache_control(messages, model)
        if reasoning_effort in _REASONING_OFF and model in self.refuses_no_reasoning:
            reasoning_effort = "low"

        body: dict[str, Any] = {
            "model": model,
            "messages": payload,
            # Ask OpenRouter to return real dollar cost alongside token counts.
            "usage": {"include": True},
        }
        if tools:
            body["tools"], body["tool_choice"] = self._tools_for(model, tools, tool_choice)
        if self.provider_sort:
            body["provider"] = {"sort": self.provider_sort}
        temperature = effective_temperature(model, temperature, self.mercury_temperature)
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is None and self.mercury_max_tokens and is_mercury(model):
            max_tokens = self.mercury_max_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        ended = False

        def end_once() -> None:
            nonlocal ended
            if not ended:
                ended = True
                if self.on_call_end:
                    with contextlib.suppress(Exception):
                        self.on_call_end()

        if self.on_call_start:
            with contextlib.suppress(Exception):
                self.on_call_start()
        if self.stream:
            body["stream"] = True

        try:
            reply = self._accounted(self._send_adapting(model, body, end_once), model)
            if self._steps_down(model, body) and _reasoned_to_the_cap(reply):
                first = reply
                body["reasoning"] = {"effort": "low"}
                reply = self._accounted(self._send_adapting(model, body, end_once), model)
                _add_usage(reply.usage, first.usage)
                reply.warning = reply.warning or (
                    "reasoned to the token cap without answering; asked again at low effort"
                )
            if is_refusal(reply):
                reply = self._retry_without_refused_wording(model, body, reply, end_once)
            if is_refusal(reply):
                raise ModelRefused(model, reply.provider, reply.finish_reason)
            return reply
        finally:
            # Covers a tool-only reply (no content delta ever arrives to end
            # it early) and every failure path, across every retry — this is
            # the one place a call is guaranteed to actually finish.
            end_once()

    def _send_adapting(
        self, model: str, body: dict[str, Any], end_once: Callable[[], None]
    ) -> Reply:
        """Send `body`; after a 400 that refuses a setting, loosen it and send again.

        At most one retry for each setting a provider can refuse (tool choice, then reasoning
        off, in whichever order it names them): the second 400 is a different one."""

        def send() -> Reply:
            request = self.provider.request(body)
            escape = self._steps_down(model, body)
            if self.stream:
                return self._request(
                    lambda budget: self._stream_once(request, end_once, budget), escape
                )
            return self._request(lambda budget: self._complete_once(request, budget), escape)

        stepped = False
        for _ in range(3):
            try:
                reply = send()
                if stepped:
                    reply.warning = reply.warning or (
                        "no content before the first-content cut; asked again at low effort"
                    )
                return reply
            except _Runaway:
                body["reasoning"] = {"effort": "low"}
                stepped = True
            except LLMError as exc:
                if not self._adapt_to_refusal(model, body, str(exc)):
                    raise
        return send()

    def _steps_down(self, model: str, body: dict[str, Any]) -> bool:
        """Whether a runaway of this request is re-asked at low effort (§131.12)."""
        effort = (body.get("reasoning") or {}).get("effort")
        return self.runaway_step_down and _is_diffusion_model(model) and effort in _STEP_DOWN_FROM

    def _tools_for(
        self, model: str, tools: list[dict], tool_choice: str | dict | None
    ) -> tuple[list[dict], str | dict]:
        """The tool schemas and tool choice as sent, loosened where this model has refused."""
        sent = strict_schemas(tools) if self.strict_tools else tools
        if model in self.refuses_reasoning_wording:
            sent = without_refused_wording(sent) or sent
        if isinstance(tool_choice, dict) and model in self.refuses_forced_choice:
            tool_choice = "auto"
        return sent, tool_choice or "auto"

    def _retry_without_refused_wording(
        self, model: str, body: dict[str, Any], reply: Reply, end_once: Callable[[], None]
    ) -> Reply:
        """After a refusal of a request carrying _REFUSED_WORDING, send it once more without
        that wording and remember the model, as _adapt_to_refusal does for a 400. A refusal
        with nothing to reword, or a second one, is returned for the caller to raise."""
        if model in self.refuses_reasoning_wording:
            return reply
        plain = without_refused_wording(body.get("tools") or [])
        if plain is None:
            return reply
        body["tools"] = plain
        self.refuses_reasoning_wording.add(model)
        retried = self._accounted(self._send_adapting(model, body, end_once), model)
        if not is_refusal(retried):
            retried.warning = retried.warning or (
                "refused submit_plan's reasoning hand-off wording; sent without it from now on"
            )
        return retried

    def _adapt_to_refusal(self, model: str, body: dict[str, Any], error: str) -> bool:
        """After a 400 that names what it refused, loosen that one setting in `body` and say so.

        Models that must reason (Sonnet 5.5) refuse a forced `tool_choice` and reasoning off, and
        the harness sends both at its planning and finishing steps. Rather than a table of models,
        the refusal itself is the signal: retry the call once with `tool_choice: auto` or a low
        reasoning effort, and remember it for the model. A caller that needs the tool re-asks
        when the reply has no tool call (runner._last_word). Anything else is not handled here."""
        if not error.startswith("HTTP 400"):
            return False
        text = error.lower()
        if isinstance(body.get("tool_choice"), dict) and "tool_choice" in text.replace(" ", "_"):
            body["tool_choice"] = "auto"
            self.refuses_forced_choice.add(model)
            return True
        asked = (body.get("reasoning") or {}).get("effort")
        if asked in _REASONING_OFF and ("reasoning" in text or "thinking" in text):
            body["reasoning"] = {"effort": "low"}
            self.refuses_no_reasoning.add(model)
            return True
        return False

    def _accounted(self, reply: Reply, model: str) -> Reply:
        """Cost from the price table when the provider reports none (only
        OpenRouter returns dollars), and the provider's name when the response
        does not carry one."""
        usage = reply.usage
        if not usage.cost_usd:
            usage.cost_usd = self.provider.cost(
                model, usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens
            )
        if not reply.provider and self.provider is not OPENROUTER:
            reply.provider = self.provider.name
        return reply

    # -- transport --------------------------------------------------------

    def _request(self, attempt_fn: Callable[[float], Reply], escape_on_cut: bool = False) -> Reply:
        """Attempt the call, retrying, but never past `max_call_s` in total.

        `self.timeout` bounds one attempt. Nothing bounded the *sequence* of
        them, and the arithmetic of the defaults is worse than it looks:
        seven attempts at 180s apiece plus six backoffs of up to 90s is 1,800
        seconds for a single chat() call. That is not theoretical. A traced
        benchmark run sat for 1,954 seconds between one tool result and the
        next model call, with no trace events in the gap at all, and then
        reported the call's latency as 1.17s — because only the final,
        successful attempt is timed. The run's own record showed 43.6s of
        model latency against 1,998s of wall clock and was not marked as
        having timed out.

        So the retry budget is a wall-clock deadline, and it is a real
        ceiling rather than a "do not start another attempt" gate. Three
        things enforce it, and the first two were missing when this was first
        written: each attempt's own socket timeout is clamped to the budget
        that remains, so an attempt starting just inside the deadline cannot
        then run a full `self.timeout` past it; `_sleep` reserves room for
        the attempt a backoff exists to enable, not merely for itself; and
        the final attempt never sleeps at all, because nothing follows it.
        Without the first of those, defaults of 420s and 180s overshot to
        roughly 780s.
        """
        started = time.time()
        deadline = started + self.max_call_s
        last_error: Exception | None = None
        reconnects = 0
        for attempt in range(self.max_retries + 1):
            remaining = deadline - time.time()
            if attempt and remaining <= 0:
                break
            attempt_started = time.time()
            try:
                reply = attempt_fn(remaining)
            except (_Retryable, requests.RequestException) as exc:
                last_error = exc
                self._report_failed_attempt(exc, attempt, attempt_started, escape_on_cut)
                retry_after = getattr(exc, "retry_after", None)
                # The last attempt has nothing after it, so a backoff here
                # waits for a request that will never be made — 60s of it
                # with these defaults, or 90s behind a Retry-After header.
                if attempt >= self.max_retries:
                    break
                # A socket-level failure is not the provider saying no, and
                # it is the failure an unstable link produces: the pooled
                # connection this call reached for went dead while it was
                # idle. Waiting changes nothing about that — the remedy is a
                # new connection, not a later one — and the pool must be
                # cleared first or the retry can draw a second dead socket
                # from it. Bounded to RECONNECTS_PER_CALL so a genuinely
                # unreachable host still falls through to real backoff
                # rather than spinning on instant reconnects.
                if _is_socket_level(exc) and reconnects < RECONNECTS_PER_CALL:
                    reconnects += 1
                    self._reset_pool()
                    if self.on_reconnect:
                        with contextlib.suppress(Exception):
                            self.on_reconnect(attempt + 1, f"{type(exc).__name__}: {exc}")
                    continue
                if not self._sleep(attempt, retry_after, deadline):
                    break
            else:
                # Stamped on the way out so the trace can tell a slow call
                # from a retried one. They look identical from the outside
                # and are completely different problems. Suppressed rather
                # than assumed: a stand-in reply from a test double is not a
                # reason to fail a request that actually succeeded, and the
                # trace's own readers already default these to "did not
                # retry" when they are absent.
                with contextlib.suppress(AttributeError):
                    reply.attempts = attempt + 1
                    reply.total_s = time.time() - started
                    reply.retry_error = (
                        f"{type(last_error).__name__}: {last_error}"[:200] if last_error else ""
                    )
                return reply
        raise LLMError(
            f"request failed after {self.max_retries + 1} attempts "
            f"or {self.max_call_s:.0f}s: {last_error}"
        )

    def _report_failed_attempt(
        self, exc: Exception, attempt: int, attempt_started: float, escape_on_cut: bool
    ) -> None:
        """Tell on_attempt_failed, then, for a cut for no content when the caller
        will ask again at a lower effort instead of repeating it, leave the retry
        loop (§131.12)."""
        if self.on_attempt_failed:
            with contextlib.suppress(Exception):
                self.on_attempt_failed(
                    attempt + 1,
                    time.time() - attempt_started,
                    f"{type(exc).__name__}: {exc}"[:170] + _stream_detail(exc),
                )
        if escape_on_cut and isinstance(exc, _ContentCut):
            raise _Runaway(str(exc)) from exc

    def _attempt_timeout(self, budget: float | None) -> float:
        """How long this attempt may take, given what is left of the call's
        budget. Clamped so an attempt starting just inside the deadline
        cannot run a full `self.timeout` past it."""
        if budget is None:
            return self.timeout
        return max(1.0, min(self.timeout, budget))

    def _first_byte_timeout(self, budget: float | None, model: str = "") -> tuple[float, float]:
        """(connect, read) for a streamed request, where read means "time to
        the response headers".

        With `stream=True`, `post()` returns as soon as headers arrive, so
        its read timeout is exactly the wait for the provider's first word —
        not the whole generation, which `_iter_sse_chunks` bounds separately
        against its own deadline. Giving it the full 180s was the reason a
        half-open connection cost seven attempts of 180s apiece: the read
        that timed out was this one, long before any first-byte check of
        ours could run.

        `model` widens the bound for a diffusion model (see
        `DIFFUSION_FIRST_BYTE_TIMEOUT_S`) rather than using one figure sized
        against the all-model p90, which its own tail already exceeds.
        """
        attempt = self._attempt_timeout(budget)
        return (min(CONNECT_TIMEOUT_S, attempt), min(_first_byte_timeout_for(model), attempt))

    def _request_timeout(self, budget: float | None) -> tuple[float, float]:
        """(connect, read) for one attempt.

        Split because a scalar sets both alike, and a connect timeout of 180s
        is the wrong bound entirely: a SYN that is dropped rather than
        refused — what a provider edge doing connection-rate limiting looks
        like from here — then costs the full 180s before the retry loop even
        sees a failure, and 1,260s across seven attempts. A TCP connection
        that has not been established in ten seconds is not going to be.
        """
        return (
            min(CONNECT_TIMEOUT_S, self._attempt_timeout(budget)),
            self._attempt_timeout(budget),
        )

    def _complete_once(self, body: dict, budget: float | None = None) -> Reply:
        """One non-streaming attempt, bounded in total rather than per-read.

        `requests`' `timeout` bounds the gap between reads, never the whole
        response, and this path had no deadline check of its own the way the
        streamed path does — so a body arriving as a slow trickle was read
        for as long as the provider cared to keep sending. Measured against a
        local server dripping a 125-byte body one byte every 0.5s, with
        `timeout=2` and `max_call_s=3`, one call took 62.9 seconds. At the
        real defaults the same shape holds a call open for hours. Reached
        whenever `stream` is off (`--no-stream`).

        So the body is streamed and accumulated under the same wall-clock
        deadline the SSE path uses, which is what makes the timeout a total.
        """
        started = time.time()
        deadline = started + self._attempt_timeout(budget)
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=body,
            timeout=self._request_timeout(budget),
            stream=True,
        )
        # A socket timeout cannot bound this on its own: every byte that
        # arrives resets it, so a body dripped one byte at a time is never
        # "idle" and never times out. Measured at 64s against a 3s budget
        # with the deadline checked between reads. Closing the response from
        # a timer is what actually interrupts the blocked read.
        watchdog = threading.Timer(max(0.1, deadline - time.time()), lambda: _abort(resp))
        watchdog.daemon = True
        watchdog.start()
        try:
            self._raise_for_status(resp)
            chunks: list[bytes] = []
            # chunk_size=None yields whatever has arrived rather than blocking
            # to fill a buffer. With a fixed size the deadline below is only
            # consulted once a full chunk lands, so a body arriving one byte
            # at a time blocks inside a single read and the check never runs —
            # measured at 47s against a 3s budget before this was changed.
            for chunk in resp.iter_content(chunk_size=None):
                if time.time() > deadline:
                    raise _Retryable(
                        f"response body still arriving past its "
                        f"{self._attempt_timeout(budget):.0f}s deadline"
                    )
                if chunk:
                    chunks.append(chunk)
        except Exception as exc:
            # A body the watchdog cut off reads as a connection error; say
            # what actually happened so the retry loop and the trace see a
            # deadline rather than a mystery socket failure.
            if time.time() > deadline:
                raise _Retryable(
                    f"response body still arriving past its "
                    f"{self._attempt_timeout(budget):.0f}s deadline"
                ) from exc
            raise
        finally:
            watchdog.cancel()
            resp.close()
        latency = time.time() - started
        try:
            data = json.loads(b"".join(chunks).decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            # Terminal, not retryable: a body that is not JSON will not become
            # JSON on the seventh attempt, and retrying spends the whole call
            # budget to arrive at the same error with the body no longer in it.
            raise LLMError(f"malformed response: {b''.join(chunks)[:500]!r}") from exc
        if "error" in data and not data.get("choices"):
            raise _envelope_error(data["error"])
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"malformed response: {json.dumps(data)[:500]}") from exc
        message = choice.get("message") or {}
        return Reply(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            usage=_usage_from(data.get("usage") or {}, latency, ttft=0.0),
            finish_reason=choice.get("finish_reason") or "",
            raw=data,
            served_model=_served_model(data),
            provider=_provider_name(data),
            warning=str(data.get("warning") or ""),
        )

    def _stream_once(
        self,
        body: dict,
        on_first_content: Callable[[], None] | None = None,
        budget: float | None = None,
    ) -> Reply:
        started = time.time()
        # `timeout=` on a streamed request only bounds the gap between reads —
        # each byte that arrives resets it. OpenRouter's keep-alive comments
        # are sent for exactly that reason, to hold a slow generation open,
        # but the same mechanism means a provider-side stall that keeps
        # emitting keep-alives without ever finishing can hold this loop open
        # indefinitely: a trickle of comments is still "activity" as far as
        # the socket read-timeout is concerned. The deadline _iter_sse_chunks
        # checks is the actual backstop — a request this old is retried like
        # any other timeout, keep-alives or not.
        attempt_timeout = self._attempt_timeout(budget)
        deadline = started + attempt_timeout
        # The read timeout passed to requests bounds the gap between reads,
        # and it has to be generous enough for a slow generation. The wait
        # for the *first* byte does not: a connection that has produced
        # nothing in FIRST_BYTE_TIMEOUT_S against a p90 TTFT of 5.46s is
        # almost certainly half-open, and waiting out the full read timeout
        # is how one dead connection costs a run minutes.
        model = body.get("model", "")
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json=body,
            timeout=self._first_byte_timeout(budget, model),
            stream=True,
        )
        first_byte_deadline = min(
            deadline, time.time() + min(_first_byte_timeout_for(model), attempt_timeout)
        )
        state = _StreamState(
            started=started, on_first_content=on_first_content, on_delta=self.on_delta
        )
        # The same watchdog the non-streaming path needs, for the same
        # reason. `_iter_sse_chunks` checks the deadline when a line
        # *arrives*, and `iter_lines` buffers 512 bytes before yielding one —
        # so a provider trickling 25-byte keepalive comments has to send
        # twenty of them before the check can run at all. Measured against a
        # server that sent keepalives and never any content: a 60s budget
        # took 173s to give up. Nothing inside the read loop can bound that;
        # only closing the socket from outside it can.
        watchdog = threading.Timer(max(0.1, deadline - time.time()), lambda: _abort(resp))
        watchdog.daemon = True
        watchdog.start()
        # A second bound, on the first real output rather than the first byte:
        # keep-alive comments satisfy the first-byte check (see
        # DIFFUSION_FIRST_CONTENT_TIMEOUT_S). Cut only if nothing has arrived.
        content_limit = _first_content_timeout_for(model)
        content_cut: list[bool] = []

        def _cut_if_silent() -> None:
            if state.ttft == 0.0:
                content_cut.append(True)
                _abort(resp)

        # Only when it would fire before the attempt's own deadline: at or past
        # it, the deadline already bounds the attempt and says so in its own words.
        content_watchdog = (
            threading.Timer(content_limit, _cut_if_silent)
            if content_limit and content_limit < attempt_timeout
            else None
        )
        if content_watchdog is not None:
            content_watchdog.daemon = True
            content_watchdog.start()
        # A third bound, on silence after the first delta (DESIGN-LOG §95).
        stall_limit = self.content_stall_s if _is_diffusion_model(model) else None
        stream_done, stall_cut = _watch_for_stall(state, resp, stall_limit)
        try:
            # Inside the try, so a 429 or 500 closes the response rather than
            # relying on _raise_for_status's own `resp.text` to drain it as a
            # side effect of formatting the message. Leaking a connection on
            # the rate-limited path is the worst possible time to do it.
            self._raise_for_status(resp)
            for chunk in _iter_sse_chunks(resp, deadline, attempt_timeout, first_byte_deadline):
                state.apply_chunk(chunk)
            if content_cut:
                # Shutting the socket down can end the read cleanly rather than
                # with an error, which would return an empty reply as if final.
                raise _ContentCut("cut off with no content")
            if stall_cut:
                raise _Retryable(f"silent for {stall_limit:g}s after the stream started")
        except Exception as exc:
            _note_stream_state(exc, state)
            # A stream the watchdog cut off surfaces as a socket error; say
            # what actually happened so the retry loop and the trace see a
            # deadline rather than a mystery reset.
            if content_cut:
                raise _ContentCut(
                    f"no content within {content_limit:.0f}s, only keep-alives "
                    "(a healthy Mercury call starts in about a second)"
                ) from exc
            if stall_cut and not isinstance(exc, _Retryable):
                cut = _Retryable(
                    f"silent for {stall_limit:g}s after the stream started "
                    "(a runaway or stalled reply; a healthy one finishes within seconds)"
                )
                _note_stream_state(cut, state)
                raise cut from exc
            if time.time() > deadline:
                raise _Retryable(
                    f"stream open past its {attempt_timeout:.0f}s deadline "
                    "with no finished response"
                ) from exc
            raise
        finally:
            watchdog.cancel()
            if content_watchdog is not None:
                content_watchdog.cancel()
            stream_done.set()
            resp.close()

        latency = time.time() - started
        return Reply(
            content="".join(state.content),
            tool_calls=[state.partial[i] for i in sorted(state.partial)],
            usage=_usage_from(state.usage_payload, latency, state.ttft, state.max_gap),
            finish_reason=state.finish_reason,
            raw={"streamed": True},
            served_model=state.served_model,
            provider=state.provider,
            warning=state.warning,
        )

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.status_code == 200:
            return
        if resp.status_code in RETRYABLE_CODES:
            raise _Retryable(
                f"HTTP {resp.status_code}: {resp.text[:300]}", resp.headers.get("Retry-After")
            )
        if resp.status_code == 402 and IN_FLIGHT_CAP_REASON in resp.text:
            raise _Retryable(
                f"HTTP 402: {resp.text[:300]}",
                resp.headers.get("Retry-After") or _body_retry_after(resp.text),
            )
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    def _sleep(self, attempt: int, retry_after: str | None, deadline: float | None = None) -> bool:
        """Back off far enough to outlast a per-minute rate limit.

        Provider limits are usually expressed per minute, so a backoff capped at
        twenty seconds gives up inside the very window it needs to wait out.

        Returns False when `deadline` leaves no room to wait and try again —
        the caller should stop rather than sleep past the budget it was
        given. A backoff is only worth paying for if there is time left to
        make the attempt it is waiting for.
        """
        wanted = min(2**attempt + random.random(), 60.0)
        if retry_after:
            with contextlib.suppress(ValueError, TypeError):
                wanted = min(float(retry_after), 90.0)
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= wanted:
                return False
        time.sleep(wanted)
        return True


# Error envelopes that name the failure in words rather than an HTTP number. Inception sends
# {"code": "server_error", "type": "server_error"} inside a 200 or mid-stream; parsed as an int
# it became None, so each one was fatal: 18-20 runs per arm in §89 and five in §87.2 (DESIGN-LOG
# §89.3). Only transient kinds are listed: "free_tier_quota_exceeded" and "insufficient_quota"
# mean no more requests will succeed, so they stay terminal rather than burn the retry budget.
RETRYABLE_ERROR_NAMES = frozenset(
    {
        "server_error",
        "internal_error",
        "internal_server_error",
        "overloaded",
        "overloaded_error",
        "service_unavailable",
        "timeout",
        "rate_limit_exceeded",
        "rate_limit_error",
    }
)
# Out of quota or credit: no retry can succeed, whatever else the body says.
TERMINAL_ERROR_NAMES = frozenset({"free_tier_quota_exceeded", "insufficient_quota"})


def _envelope_error(error: Any) -> Exception:
    """Classify a provider error body as retryable or terminal."""
    raw_code = error.get("code") if isinstance(error, dict) else None
    try:
        code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        code = None
    message = f"provider error: {error}"
    if code in RETRYABLE_CODES:
        return _Retryable(message)
    names = (
        {str(error.get(key) or "").lower() for key in ("code", "type")}
        if isinstance(error, dict)
        else set()
    )
    if names & RETRYABLE_ERROR_NAMES and not names & TERMINAL_ERROR_NAMES:
        return _Retryable(message)
    if code == 402 and IN_FLIGHT_CAP_REASON in json.dumps(error, default=str):
        return _Retryable(message, _body_retry_after(json.dumps(error, default=str)))
    return LLMError(message)


def _body_retry_after(text: str) -> str | None:
    """The Retry-After OpenRouter repeats inside a 402 body's metadata, if any."""
    match = re.search(r'"Retry-After"\s*:\s*"?(\d+(?:\.\d+)?)', text)
    return match.group(1) if match else None


class _Retryable(Exception):
    def __init__(self, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class _ContentCut(_Retryable):
    """A Mercury attempt cut for producing no content within the first-content bound."""


class _Runaway(Exception):
    """Raised out of _request on a content cut when the caller steps effort down instead."""


# Efforts a runaway is stepped down from (§131.12); low and off are left alone.
_STEP_DOWN_FROM = frozenset({"medium", "high", "xhigh", "max"})


def _reasoned_to_the_cap(reply: Reply) -> bool:
    """A reply that ended at the token cap with neither text nor a tool call."""
    return reply.finish_reason == "length" and not reply.content.strip() and not reply.tool_calls


def _add_usage(usage: Usage, earlier: Usage) -> None:
    """Fold an abandoned call's spend into the reply that replaced it, so the
    trace and the spend cap still count it."""
    usage.prompt_tokens += earlier.prompt_tokens
    usage.completion_tokens += earlier.completion_tokens
    usage.cached_tokens += earlier.cached_tokens
    usage.reasoning_tokens += earlier.reasoning_tokens
    usage.cost_usd += earlier.cost_usd
    usage.latency_s += earlier.latency_s


# After `data: [DONE]` a chunked SSE response still has its terminal
# zero-length chunk to send. Lines this many past [DONE] are not arriving,
# and waiting for them is worse than dropping the connection.
_DRAIN_LINES = 8
# Draining is worth about one TLS handshake. Past this it is costing more
# than the connection it is trying to save, so the socket is abandoned.
_DRAIN_BUDGET_S = 1.0
# Distinct from None, which is a legitimate socket timeout value.
_UNSET = object()
# A TCP connection not established in this long is not going to be. Kept
# well under `timeout`, which bounds reads and is far too long for a connect.
CONNECT_TIMEOUT_S = 10.0


def _drain(lines, deadline: float, resp: requests.Response) -> None:
    """Read the last few bytes of a finished stream, so the connection lives.

    urllib3 only returns a connection to its pool once the response body has
    been read to the end. Returning at `[DONE]` leaves the terminal chunk
    unread, so `resp.close()` discards the socket — and, because bytes were
    still outstanding, resets it rather than closing it cleanly. Measured
    against a local chunked SSE server, five sequential calls opened four
    connections and the server logged a ConnectionResetError; the same five
    over a Content-Length response, which is fully read by [DONE], opened
    one.

    That costs a fresh TCP and TLS handshake on most calls, against a
    median time-to-first-token of 1.14s that is already 78% of all model
    latency — and a client that opens a new connection per request is also
    what connection-rate limiting at a provider's edge is designed to catch.

    Bounding this correctly matters more than it looks. `next(lines, None)`
    is a blocking socket read, and checking the clock *before* it bounds
    nothing: a drain that begins with 178 seconds of budget still blocks in
    that read for the whole socket timeout. Measured against a server that
    withheld the terminal chunk for six seconds after `[DONE]`, a reply that
    was already complete took 6.01s to return, and the cost landed inside
    `latency_s` where it reads as a slow model rather than as a drain. Worse,
    `contextlib.suppress` hides the ReadTimeout, so it never surfaces as an
    error at all.

    So the budget here is its own, and it is short: this is a best-effort
    optimisation worth about one handshake, and it must never cost more than
    the connection it is trying to save. Past `_DRAIN_BUDGET_S` the socket is
    abandoned to `resp.close()`, which is exactly the behaviour that existed
    before draining was added.
    """
    stop = min(deadline, time.time() + _DRAIN_BUDGET_S)
    with _socket_timeout(resp, _DRAIN_BUDGET_S), contextlib.suppress(Exception):
        for _ in range(_DRAIN_LINES):
            if time.time() > stop or next(lines, None) is None:
                return


def _abort(resp: requests.Response) -> None:
    """Interrupt a read that is already blocking, from another thread.

    `resp.close()` returns the connection to the pool; it does not wake a
    reader blocked in `recv`. Shutting the socket down does, which is the
    only thing that bounds a body arriving as a slow trickle -- every byte
    resets the socket timeout, so nothing else ever fires.
    """
    for sock in _response_sockets(resp):
        with contextlib.suppress(Exception):
            sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(Exception):
        resp.close()


def _response_sockets(resp: requests.Response) -> list[socket.socket]:
    """Every place the socket behind a streamed response might be.

    Under urllib3 2.x, once a streamed body is being read the connection's own
    `sock` is None; the live socket is the one the underlying
    `http.client.HTTPResponse` reads from. Shutting down only the old path was
    silently a no-op, so the watchdog never woke a blocked read and a provider
    trickling keep-alives held one attempt open for 929s against a 180s
    deadline (DESIGN-LOG §51).
    """
    found: list[socket.socket] = []
    for path in (("_fp", "fp", "raw", "_sock"), ("_connection", "sock")):
        obj: object = resp.raw
        for attr in path:
            obj = getattr(obj, attr, None)
        if isinstance(obj, socket.socket):
            found.append(obj)
    return found


@contextlib.contextmanager
def _socket_timeout(resp: requests.Response, seconds: float):
    """Temporarily shorten the socket read timeout, restoring it after.

    The clock check in `_drain` cannot interrupt a read already blocking, so
    the read itself has to be given the shorter bound. Best-effort by
    design: the socket is several private attributes down inside urllib3 and
    a future version may move it, in which case the drain still terminates
    on its line count and the response is closed as it always was. Reaching
    through is worth it only because the alternative is a read that can block
    for the whole 180s socket timeout on a reply that is already complete.
    """
    sock = None
    # None is what gettimeout() returns for a *blocking* socket, so it cannot
    # double as "nothing captured": treating it that way left the drain's
    # 1s timeout on a pooled connection, and the next streamed call to reuse
    # it timed out almost immediately and retried.
    previous: Any = _UNSET
    with contextlib.suppress(Exception):
        sock = resp.raw._connection.sock
        previous = sock.gettimeout()
        sock.settimeout(seconds)
    try:
        yield
    finally:
        if sock is not None and previous is not _UNSET:
            with contextlib.suppress(Exception):
                sock.settimeout(previous)


def _iter_sse_chunks(
    resp: requests.Response,
    deadline: float,
    timeout: float,
    first_byte_deadline: float | None = None,
):
    """Yield parsed JSON chunks from an OpenRouter SSE stream.

    Skips keep-alive comments (": OPENROUTER PROCESSING") and anything that
    is not a `data:` line; stops cleanly at the `[DONE]` sentinel. Raises
    _Retryable if no line arrives before `deadline` — the actual stall
    backstop, since keep-alives alone would otherwise look like activity
    forever. Pulled out of _stream_once as a pure decomposition: same lines
    read, same exceptions, no change to what is sent over the wire.
    """
    # iter_lines(decode_unicode=True) decodes using resp.encoding, which
    # `requests` defaults to ISO-8859-1 for any text/* content type that
    # omits an explicit charset param (per the old HTTP/1.1 default) --
    # OpenRouter's `text/event-stream` responses do exactly that. Every
    # multi-byte UTF-8 character the model wrote (an ellipsis, an em dash,
    # a currency sign) came back mojibake'd -- 0xe2 0x80 0xa6 read one byte
    # at a time as Latin-1 -- and a run that then fed that corrupted text
    # back to the model on a later turn corrupted it a second time. Fixed
    # by always decoding the raw bytes as UTF-8 ourselves instead of asking
    # requests to guess; decode_unicode=False (the default) keeps iter_lines
    # from touching encoding at all.
    # Bound the socket read for the first byte only, so the deadline below
    # is actually reachable: checking the clock cannot interrupt a read that
    # is already blocking, which is the same trap `_drain` fell into.
    lines = resp.iter_lines()
    if first_byte_deadline is not None:
        budget = max(1.0, first_byte_deadline - time.time())
        try:
            with _socket_timeout(resp, budget):
                first = next(lines, None)
        except Exception as exc:
            # Retryable, not suppressed: swallowing it would leave the loop
            # below to block on the same dead socket for the full read
            # timeout, which is the cost this exists to avoid.
            raise _Retryable(
                f"no first byte within {budget:.0f}s (connection looks "
                "half-open; a live one answers in about a second)"
            ) from exc
        if first is not None:
            lines = itertools.chain([first], lines)
    for raw_line in lines:
        if time.time() > deadline:
            raise _Retryable(
                f"stream open past its {timeout:.0f}s deadline with no "
                "finished response (provider keep-alives can mask a stall)"
            )
        if not raw_line:
            continue
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")
        line = raw_line.strip()
        if line.startswith(":") or not line.startswith("data:"):
            continue
        data_str = line[len("data:") :].strip()
        if data_str == "[DONE]":
            _drain(lines, deadline, resp)
            return
        try:
            yield json.loads(data_str)
        except json.JSONDecodeError:
            continue


def _served_model(payload: dict) -> str:
    return str(payload.get("model") or "")


def _provider_name(payload: dict) -> str:
    """The upstream provider a routed response names, whether as a string or as
    an object with a name."""
    provider = payload.get("provider")
    if isinstance(provider, dict):
        provider = provider.get("name") or provider.get("id")
    return str(provider or "")


@dataclass
class _StreamState:
    """The Reply being assembled from a stream, and the deltas that matter
    while it's in flight. apply_chunk() folds one SSE chunk into it."""

    started: float
    on_first_content: Callable[[], None] | None = None
    on_delta: Callable[[str], None] | None = None
    content: list[str] = field(default_factory=list)
    # Tool calls arrive as deltas keyed by index, with arguments in fragments.
    partial: dict[int, dict] = field(default_factory=dict)
    finish_reason: str = ""
    ttft: float = 0.0
    # Wall-clock time of the last delta, and the longest silence between two
    # deltas once the first had arrived (Usage.max_gap_s).
    last_activity: float = 0.0
    max_gap: float = 0.0
    usage_payload: dict = field(default_factory=dict)
    served_model: str = ""
    provider: str = ""
    warning: str = ""

    def apply_chunk(self, chunk: dict) -> None:
        if "error" in chunk and not chunk.get("choices"):
            # A provider dropping out mid-stream is transient far more often
            # than not; losing the partial response is acceptable.
            raise _envelope_error(chunk["error"])
        if chunk.get("usage"):
            self.usage_payload = chunk["usage"]
        # Every chunk repeats the model and provider; keep the first non-empty.
        self.served_model = self.served_model or _served_model(chunk)
        self.provider = self.provider or _provider_name(chunk)
        self.warning = self.warning or str(chunk.get("warning") or "")
        for choice in chunk.get("choices") or []:
            self._apply_choice(choice)

    def _activity(self) -> None:
        now = time.time()
        if self.last_activity:
            self.max_gap = max(self.max_gap, now - self.last_activity)
        self.last_activity = now

    def _apply_choice(self, choice: dict) -> None:
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        # Any delta that says something counts as activity once the stream has
        # started, a reasoning field included where a provider sends one; a bare
        # role marker or an empty delta does not.
        if self.ttft and (
            choice.get("finish_reason") or any(v for k, v in delta.items() if k != "role")
        ):
            self._activity()

        text = delta.get("content")
        if text:
            if self.ttft == 0.0:
                self.ttft = time.time() - self.started
                self.last_activity = time.time()
                # Content is itself visible progress; a tool-call fragment
                # isn't (nothing prints until the full call is assembled and
                # dispatched), so only actual text ends the "still waiting"
                # indicator early — otherwise it'd vanish into a dead gap.
                if self.on_first_content:
                    self.on_first_content()
            self.content.append(text)
            if self.on_delta:
                self.on_delta(text)

        for fragment in delta.get("tool_calls") or []:
            if self.ttft == 0.0:
                self.ttft = time.time() - self.started
                self.last_activity = time.time()
            _merge_tool_call(self.partial, fragment)


def _merge_tool_call(partial: dict[int, dict], fragment: dict) -> None:
    """Accumulate a streamed tool call. Arguments arrive as a string in pieces."""
    index = fragment.get("index", 0)
    slot = partial.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if fragment.get("id"):
        slot["id"] = fragment["id"]
    if fragment.get("type"):
        slot["type"] = fragment["type"]
    function = fragment.get("function") or {}
    if function.get("name"):
        slot["function"]["name"] = function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]


def _watch_for_stall(
    state: _StreamState, resp: requests.Response, limit: float | None
) -> tuple[threading.Event, list[float]]:
    """Start a watcher that closes `resp` once the stream has been silent `limit`
    seconds after its first delta. Returns the event that stops it and the list it
    appends the silence to when it cuts. One polling thread per attempt, rather
    than a timer re-armed on every chunk; no thread at all when `limit` is None."""
    done = threading.Event()
    cut: list[float] = []
    if not limit:
        return done, cut

    def watch() -> None:
        while not done.wait(min(1.0, limit / 4)):
            if state.ttft and time.time() - state.last_activity > limit:
                cut.append(time.time() - state.last_activity)
                _abort(resp)
                return

    threading.Thread(target=watch, daemon=True).start()
    return done, cut


def _note_stream_state(exc: BaseException, state: _StreamState) -> None:
    """Record on a failed attempt's exception whether its stream had started and
    how long it had then been silent, for the attempt-failed trace event."""
    with contextlib.suppress(Exception):
        # None, not 0.0, for "no content": a first delta at 0.03s is content.
        exc.content_started_s = state.ttft or None  # ty: ignore[unresolved-attribute]
        exc.silent_s = (  # ty: ignore[unresolved-attribute]
            time.time() - state.last_activity if state.last_activity else None
        )


def _stream_detail(exc: BaseException) -> str:
    """' [first content at 1.4s, then silent 118.2s]', ' [no content]', or ''
    for a failure that never reached a stream."""
    if not hasattr(exc, "content_started_s"):
        # A cut raises a fresh _Retryable from the socket error that carries it.
        exc = exc.__cause__ if exc.__cause__ is not None else exc
        if not hasattr(exc, "content_started_s"):
            return ""
    started = getattr(exc, "content_started_s", None)
    if started is None:
        return " [no content]"
    silent = getattr(exc, "silent_s", None) or 0.0
    return f" [first content at {started:.1f}s, then silent {silent:.1f}s]"


def _usage_from(raw: dict, latency: float, ttft: float, max_gap: float = 0.0) -> Usage:
    details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        cost_usd=float(raw.get("cost", 0.0) or 0.0),
        latency_s=latency,
        ttft_s=ttft,
        cached_tokens=int(details.get("cached_tokens", 0) or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
        max_gap_s=max_gap,
    )


def approx_tokens(text: str) -> int:
    """Cheap token estimate. Good enough for budgeting, no tokenizer dependency."""
    return max(1, len(text) // 4)


def messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += approx_tokens(json.dumps(m, default=str))
    return total


def request_tokens(system: str, user: str, schemas: list[dict] | None = None) -> int:
    """What the first call of a loop will actually cost to send.

    Every budget in this codebase used to be set against `approx_tokens(user)`
    alone, which is the smallest part of the request. A traced step logged
    `prompt_tokens_est: 3584` for a call the provider then billed at 12,165
    input tokens — the missing 8,600 were the tool schemas (~3,900 for the
    twenty-eight execution tools), the system prompt (~1,200) and the injected
    skills (up to 3,000), none of which the estimate looked at. Budgets
    calibrated against a third of the real number are not budgets.
    """
    total = approx_tokens(system) + approx_tokens(user)
    for schema in schemas or []:
        total += approx_tokens(json.dumps(schema, default=str))
    return total
