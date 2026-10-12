"""JSONL trace. This is the actual experiment record for comparing models."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_MODEL_SLOT = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "reasoning_tokens": 0,
    "cost_usd": 0.0,
    "latency_s": 0.0,
    "ttft_s": 0.0,
    "decode_s": 0.0,
}


@dataclass
class RunTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: str, usage: Any) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.cached_tokens += usage.cached_tokens
        reasoning = int(getattr(usage, "reasoning_tokens", 0) or 0)
        self.reasoning_tokens += reasoning
        self.cost_usd += usage.cost_usd
        self.latency_s += usage.latency_s
        self.calls += 1
        slot = self.by_model.setdefault(model, dict(_MODEL_SLOT))
        slot["calls"] += 1
        slot["prompt_tokens"] += usage.prompt_tokens
        slot["completion_tokens"] += usage.completion_tokens
        slot["cached_tokens"] += usage.cached_tokens
        slot["reasoning_tokens"] = slot.get("reasoning_tokens", 0) + reasoning
        slot["cost_usd"] += usage.cost_usd
        slot["latency_s"] += usage.latency_s
        slot["ttft_s"] += usage.ttft_s
        slot["decode_s"] += usage.decode_s

    def tok_per_s(self, model: str) -> float:
        """End-to-end rate: completion tokens over total wall time."""
        return tok_per_s(self.by_model.get(model))

    def decode_tok_per_s(self, model: str) -> float:
        """Generation rate with queue and prefill excluded. Streaming only."""
        return decode_tok_per_s(self.by_model.get(model))

    def mean_ttft(self, model: str) -> float:
        """Mean time to first token: queue and prefill, without generation."""
        return mean_ttft(self.by_model.get(model))


# The same three rates, over a plain per-model slot dict rather than a live
# RunTotals. Both shapes exist for a reason — a finished run is read back from
# a serialised totals dict, a running one from the object — but the arithmetic
# should not. It was written twice: these methods, and again inline in
# loop.format_totals, which recomputed decode rate and mean TTFT by hand.


def tok_per_s(slot: dict | None) -> float:
    """End-to-end token rate for one model's slot in a totals dict."""
    if not slot or slot.get("latency_s", 0) <= 0:
        return 0.0
    return slot.get("completion_tokens", 0) / slot["latency_s"]


def decode_tok_per_s(slot: dict | None) -> float:
    """Generation rate with queue and prefill excluded. Streaming only."""
    if not slot or slot.get("decode_s", 0) <= 0:
        return 0.0
    return slot.get("completion_tokens", 0) / slot["decode_s"]


def mean_ttft(slot: dict | None) -> float:
    """Mean seconds to first token across a slot's calls."""
    if not slot or not slot.get("calls"):
        return 0.0
    return slot.get("ttft_s", 0.0) / slot["calls"]


class Trace:
    """A run's JSONL record, shared across every ToolContext sibling.

    When --step-concurrency > 1, several steps' worker threads hold the same
    Trace instance (agent/phases/schedule.py spawns siblings via
    ToolContext.for_step(), which shares self.trace rather than copying it).
    Both mutations below — the file write and the totals counters — are
    unprotected read-modify-writes without this lock: interleaved writes can
    corrupt the JSONL file mid-line, and interleaved `+=` on RunTotals' dict
    counters can silently lose updates.
    """

    def __init__(
        self,
        trace_dir: str,
        run_id: str | None = None,
        echo: bool = True,
        on_event: Callable[[dict], None] | None = None,
    ):
        # The suffix keeps concurrent runs from appending to one another's file.
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.path = Path(trace_dir) / f"run-{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.totals = RunTotals()
        self.echo = echo
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        # A live observer (the TUI) subscribes here instead of polling the file.
        # Optional and best-effort: a UI bug must never take down an agent run,
        # so a raising callback is swallowed rather than propagated.
        self._on_event = on_event

    def event(self, kind: str, **payload: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **payload}
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
        if self._on_event is not None:
            # Outside the lock: a slow UI update must not stall other threads'
            # trace writes when steps run concurrently.
            with contextlib.suppress(Exception):
                self._on_event(record)

    def llm_call(
        self, phase: str, model: str, usage: Any, finish_reason: str = "", reply: Any = None
    ) -> None:
        """Record one model call. `reply` carries the retry detail, if any.

        Taken as the whole Reply rather than as three more parameters because
        every caller already has one, and because a caller that forgets to
        pass it degrades to "this call did not retry" rather than to a
        TypeError.
        """
        attempts = int(getattr(reply, "attempts", 1) or 1)
        total_s = float(getattr(reply, "total_s", 0.0) or 0.0)
        retry_error = str(getattr(reply, "retry_error", "") or "")
        with self._lock:
            self.totals.add(model, usage)
        self.event(
            "llm_call",
            phase=phase,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=int(getattr(usage, "reasoning_tokens", 0) or 0),
            cost_usd=usage.cost_usd,
            latency_s=round(usage.latency_s, 3),
            ttft_s=round(usage.ttft_s, 3),
            max_gap_s=round(float(getattr(usage, "max_gap_s", 0.0) or 0.0), 3),
            tok_per_s=round(usage.end_to_end_tok_s, 1),
            decode_tok_per_s=round(usage.decode_tok_s, 1),
            finish_reason=finish_reason,
            attempts=attempts,
            total_s=round(total_s, 3),
            served_model=str(getattr(reply, "served_model", "") or ""),
            provider=str(getattr(reply, "provider", "") or ""),
            # Only a direct provider returns one (Inception resetting a
            # temperature, DESIGN-LOG §85); absent otherwise.
            **(
                {"warning": warning}
                if (warning := str(getattr(reply, "warning", "") or ""))
                else {}
            ),
        )
        # A call that took many times longer than the attempt that finally
        # worked is a retried call, and the trace used to show only the
        # latter. Benchmark runs sat for 900-3,600 seconds in gaps with no
        # events in them at all, reporting sub-second latencies, and the
        # cause could only be guessed at afterwards. Saying it at the moment
        # it happens makes the next one readable instead of archaeology.
        if attempts > 1:
            self.event(
                "llm_retried",
                phase=phase,
                model=model,
                attempts=attempts,
                total_s=round(total_s, 3),
                latency_s=round(usage.latency_s, 3),
                waited_s=round(max(0.0, total_s - usage.latency_s), 3),
                error=retry_error,
            )
            self.say(
                f"   ~ {phase}: {attempts} attempts over {total_s:.0f}s "
                f"({retry_error or 'no error recorded'})"
            )
        # A reply stopped by its token cap is a silently wrong answer, not a
        # short one: the model was mid-sentence. This went unnoticed twice —
        # clarify handed people half a question, and every single end-of-run
        # suggestion was cut off — because finish_reason was parsed off the
        # wire into Reply and then dropped here. Saying so makes the next
        # instance a line in the log rather than an archaeology exercise.
        if finish_reason == "length":
            self.say(
                f"   ! {phase} hit its token cap at {usage.completion_tokens} "
                "tokens; the reply is cut off"
            )

    def say(self, message: str, detail: bool = False) -> None:
        """Narrate `message` to whoever is watching this run live.

        `detail=True` is for the setup/diagnostic lines a conversation
        doesn't need to keep re-showing — which model is planning, which
        tools resolved, a phase's own banner, where the trace file lives —
        recorded here exactly the same as anything else (still readable back
        from the trace file, still in a saved failure log), just not printed
        to the console or written into the TUI's log panel. Before this, a
        run's visible narration read the way its trace file does: a banner
        and a roll call of settings before anything happens, one for each
        phase in between, and a couple of debug lines like "trace: <path>"
        stapled onto the end — accurate, and nothing like how the same
        run's own real narrative lines ("understood as: ...", "-- step 1/3:
        ...", "review round 1: clean") already read.
        """
        if self.echo and not detail:
            print(message, flush=True)
        self.event("log", message=message, detail=detail)

    def summary(self) -> dict:
        return asdict(self.totals)

    def close(self) -> None:
        self.event("totals", **self.summary())
        self._fh.close()

    def __repr__(self) -> str:
        return f"<Trace {os.path.basename(str(self.path))}>"
