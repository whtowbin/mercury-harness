"""Relevance judging, at whatever scale needs it: which tool results in a
step's own context are still worth keeping, which memory sections apply to
the current goal, which earlier findings a later phase should see.

One job, three backends, same interface (`RelevanceJudge.score`). Jev
(TypeSafe's `typesafe_sdk`) is the fastest and cheapest when a key is
configured — one batched call answers many independent yes/no questions in
parallel, well suited to "is this candidate still relevant" repeated over a
list. It is entirely optional: `agent/relevance.py` and everything that calls
it work with no TypeSafe account at all, falling back to an LLM-backed judge
(any model already configured in this harness — Mercury by default, since it
is already the fast, cheap default everywhere else) or, with no network
budget at all, a free keyword-overlap heuristic. `build_judge` picks the best
available backend; nothing upstream needs to know which one it got.

The principle behind all three, borrowed from how Jev's own compaction
cookbook frames it: this is *selection*, not summarization. A judge decides
what to keep; nothing here rewrites or condenses the kept text. Condensing
genuinely-too-much-relevant-material is a different, rarer job, already
handled by the existing low-reasoning summarization call in
agent/loop.py::_compact_memory — this module never does that.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from typing import Any, Protocol

from .llm import ChatClient
from .memory import keywords_of

DEFAULT_REASONING_EFFORT = "low"
# Above this many candidates in one call, split into chunks: a very long tool
# loop or a large memory file could otherwise build a single request with
# hundreds of questions, which is worth capping even though neither backend
# documents a hard limit.
MAX_BATCH = 24


class RelevanceJudge(Protocol):
    def score(self, context: str, candidates: list[str]) -> list[float]:
        """One relevance score per candidate, each in [0, 1], higher = more
        worth keeping. `context` is what "relevant" is relative to — the
        current step's goal, the file being edited, the task at hand.
        Returns exactly `len(candidates)` scores, same order, always: a
        judge that cannot answer degrades to a safe default rather than
        raising, since callers (compaction, memory selection) must not lose
        a step to a judge's own failure.
        """
        ...


# -- heuristic: free, always available, the default with nothing configured -


class HeuristicJudge:
    """Keyword overlap between `context` and each candidate. Zero network
    calls, zero cost, zero configuration — what every backend degrades to
    when neither an LLM client nor a Jev key is available. The same
    word-overlap idea agent/memory.py already uses for `recall`/`relevant`,
    generalised to plain strings instead of `Section` objects."""

    def score(self, context: str, candidates: list[str]) -> list[float]:
        context_words = keywords_of(context)
        if not context_words:
            return [0.5] * len(candidates)
        scores = []
        for candidate in candidates:
            cand_words = keywords_of(candidate)
            if not cand_words:
                scores.append(0.0)
                continue
            overlap = len(context_words & cand_words)
            scores.append(min(1.0, overlap / max(3, len(context_words) ** 0.5)))
        return scores


# -- LLM-backed: any model already configured in this harness --------------

_LLM_JUDGE_SYSTEM = """You are scoring relevance, not answering questions. For each
numbered candidate, decide how relevant it is to the given context on a 0.0-1.0 scale:
1.0 means clearly still needed, 0.0 means safe to discard. Reply with nothing but a JSON
array of numbers, one per candidate, in the same order — no prose, no markdown fence."""


class LLMJudge:
    """Wraps any ChatClient already in this harness -- Mercury by default
    (already the fast, cheap default everywhere else in this repo), GLM, or
    anything else with a `.chat()`. One call scores every candidate at once
    via a JSON array, the same batching principle Jev uses, without a second
    API key or a new dependency.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        on_call: Callable[[str, Any], None] | None = None,
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        # Called with (model, reply) for every call that answered, so the run's
        # trace records it as an llm_call. Without it these calls (memory recall
        # twice a run, compaction) were model time no trace figure counted
        # (DESIGN-LOG §95).
        self.on_call = on_call
        self.calls = 0
        self.cost_usd = 0.0

    def score(self, context: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        scores: list[float] = []
        for chunk in _chunks(candidates, MAX_BATCH):
            scores.extend(self._score_chunk(context, chunk))
        return scores

    def _score_chunk(self, context: str, candidates: list[str]) -> list[float]:
        listing = "\n".join(f"{i + 1}. {c[:800]}" for i, c in enumerate(candidates))
        user = f"Context:\n{context[:2000]}\n\nCandidates:\n{listing}"
        try:
            reply = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                reasoning_effort=self.reasoning_effort,
            )
            self.calls += 1
            self.cost_usd += reply.usage.cost_usd
            if self.on_call is not None:
                with contextlib.suppress(Exception):
                    self.on_call(self.model, reply)
            parsed = _parse_score_array(reply.content, len(candidates))
        except Exception:
            parsed = None
        # A judge that failed to answer must not make the decision for its
        # caller by default-dropping everything -- keep, the safer failure.
        return parsed if parsed is not None else [1.0] * len(candidates)


def _parse_score_array(text: str, expected_len: int) -> list[float] | None:
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start, end = candidate.find("["), candidate.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        values = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(values, list) or len(values) != expected_len:
        return None
    try:
        return [max(0.0, min(1.0, float(v))) for v in values]
    except (TypeError, ValueError):
        return None


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# -- Jev-backed: fastest and cheapest, when configured ----------------------


class JevJudge:
    """One `typesafe_sdk` `system_one` call per chunk, a `Noul` (yes/no,
    calibrated probability) question per candidate, all answered in parallel
    server-side in one request. Falls back like LLMJudge does on any
    failure — a Jev outage must not be the reason compaction stops working,
    only the reason it gets slower for one call.
    """

    def __init__(self, api_key: str, model: str | None = None):
        from typesafe_sdk import TypeSafeClient  # optional dep; see pyproject.toml

        self._client = TypeSafeClient(api_key=api_key, model=model)
        # Real counts from the API's own usage field, not a computed dollar
        # figure -- Jev's response reports input/output tokens but no price,
        # and a rate typed in here from a blog post would go stale silently.
        # A caller that wants cost can apply its own known rate to these.
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def score(self, context: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        scores: list[float] = []
        for chunk in _chunks(candidates, MAX_BATCH):
            scores.extend(self._score_chunk(context, chunk))
        return scores

    def _score_chunk(self, context: str, candidates: list[str]) -> list[float]:
        from typesafe_sdk import Noul

        questions = {
            f"c{i}": Noul(
                instructions=f"Is candidate {i} still relevant to keep, "
                f"given the context?\n\ncandidate: {c[:800]}"
            )
            for i, c in enumerate(candidates)
        }
        try:
            result = self._client.system_one(state=context[:2000], questions=questions)
        except Exception:
            return [1.0] * len(candidates)
        self.calls += 1
        self.input_tokens += result.usage.input_tokens or 0
        self.output_tokens += result.usage.output_tokens or 0
        return [result.nouls[f"c{i}"].noul for i in range(len(candidates))]


# -- picking a backend -------------------------------------------------------


def build_judge(
    settings, client: ChatClient | None = None, on_call: Callable[[str, Any], None] | None = None
) -> RelevanceJudge:
    """The best judge this run can actually use, in one call.

    Preference order: Jev (fastest, cheapest, needs a key and the optional
    `typesafe-sdk` package) > an LLM already configured for this run (needs
    no new dependency or account, uses whatever model `settings` names) >
    the free heuristic (always available, the floor every other backend
    degrades to on its own failure anyway).

    `settings.relevance_backend`, when set, skips that preference order and
    forces one specific backend -- for an A/B comparison (a bench sweep, a
    manual audit) where the task must stay fixed and only the backend vary.
    An unusable forced backend (e.g. "jev" with no key) falls back exactly
    as the auto path would, rather than raising, since a bench job must not
    fail just because a machine's Keychain doesn't have every backend.
    """
    forced = getattr(settings, "relevance_backend", None)
    jev_key = getattr(settings, "jev_api_key", "")
    model = getattr(settings, "relevance_model", None) or settings.executor_model

    if forced == "heuristic":
        return HeuristicJudge()
    if forced == "llm" and client is not None:
        return LLMJudge(client, model, on_call=on_call)
    # forced == "jev" falls through to the same Jev attempt the auto path
    # makes below; every other case (no force, or a forced backend that
    # isn't actually usable here) also lands in the normal preference order.
    if jev_key:
        try:
            return JevJudge(api_key=jev_key)
        except ImportError:
            pass
    if client is not None:
        return LLMJudge(client, model, on_call=on_call)
    return HeuristicJudge()
