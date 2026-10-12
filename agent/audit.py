"""Model-in-the-loop review over saved traces: the judgment layer `tracetools`
deliberately leaves out. `tracetools` is model-free by design (see its own
docstring) and stays that way — this module is where a trace's narration gets
read by something that can actually understand it, for the two questions
`tracetools`' deterministic facts can't answer:

  - did the agent misunderstand what was asked (a dropped requirement, a
    plan that quietly narrows the goal)?
  - is there a defect a human reviewer would flag by reading the story of a
    run (test-fitting, a workaround masking the real bug, thrashing that
    never converges) that isn't expressible as "declared write X never
    happened"?

**Model choice is GLM, not Mercury, and this was measured, not assumed.**
Both were run against the same 4 real failed platformer-app traces (2026-09,
see DESIGN-LOG.md) with the same prompt. Mercury cost ~$0.0004/trace and
took ~5.5s; GLM cost ~$0.0029/trace and took ~76s — roughly 7x the cost and
14x the wall time. But Mercury's findings were almost entirely restatements
of facts `tracetools` already computes for free (`loop_exhausted`,
`final_ok`, an unwritten declared write, a duplicate directory) — i.e. near-
zero marginal value over the deterministic layer this module sits on top of.
GLM's findings included things Mercury caught none of: the agent editing
its own tests until they passed instead of fixing the code under test, a
dropped "scrolling camera" requirement from the original prompt, and a
root-cause diagnosis of a workaround (copying modules into tests/ instead of
fixing pytest's collection path) rather than just noting the symptom. That
gap in kind, not just depth, is why GLM is the default here even though it
is the slower and pricier of the two — at these absolute dollar amounts
(single cents per trace) the cost difference doesn't matter, but the
difference in what gets caught does. This also means the two-tier "Mercury
extracts, GLM evaluates" design floated before the experiment doesn't earn
its complexity: `tracetools.summarize_trace()` (free, deterministic) is
already the cheap extraction step, so adding a second, LLM-based extraction
pass ahead of GLM would cost money without adding anything GLM doesn't
already get from the same deterministic summary.

**That 76s/trace number is stale as of the follow-up speed pass** — see
`DEFAULT_REASONING_EFFORT` and `DEFAULT_CONCURRENCY` below for what actually
cut it: GLM's `reasoning_effort="low"` reduced per-trace wall time 4-6x for
the same substantive findings (measured, not assumed — two more traces, see
DESIGN-LOG.md), and auditing a directory concurrently instead of one trace
at a time turns "N traces * per-trace latency" into roughly one trace's
latency regardless of N. A batch that took minutes serialized now takes
seconds to tens of seconds.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import tracetools
from .llm import ChatClient

Event = dict[str, Any]

DEFAULT_AUDIT_MODEL = "z-ai/glm-5.3-flash"

# GLM 5.3 Flash's reasoning is mandatory (OpenRouter rejects "none" outright)
# but its *effort* is not: measured on 2 real failed traces, "low" cut wall
# time 4-6x (32.3s->8.1s, 86.9s->14.1s) and cost 6-11x versus leaving it
# unset, for the same substantive findings both times — the model was
# spending tokens re-deriving the same conclusions more verbosely, not
# reaching different ones. See DESIGN-LOG.md's audit-speed section.
DEFAULT_REASONING_EFFORT = "low"

# How many traces to audit at once. Each call is an independent network
# request against the same model/provider, so this is pure wall-clock win —
# a directory's audit no longer costs (traces * per-trace latency), it costs
# roughly one trace's latency regardless of how many there are, up to this
# cap. Matches bench's own `--max-per-model` reasoning (see docs/benchmarking
# .md): too high and concurrent requests queue at the provider instead of
# failing, which looks like a hang rather than help.
DEFAULT_CONCURRENCY = 4

# Which trace event kinds carry the narration/error signal an audit prompt
# needs. Everything else (llm_call, tool_call, write_file, ...) is either
# already folded into tracetools.summarize_trace()'s structured fields or is
# noise for this purpose.
_RELEVANT_KINDS = (
    "log",
    "assistant_text",
    "loop_exhausted",
    "repeated_call",
    "batching_nudge",
    "final_check",
    "tool_call_failed",
    "tool_error",
    "review_result",
    "fix_result",
    "final_fix_no_progress",
    "run_tests",
    "step_verify",
)

_NARRATION_CHAR_CAP = 6_000
_SUMMARY_CHAR_CAP = 4_000
_MAX_NARRATION_LINES = 120

AUDIT_PROMPT_TEMPLATE = """You are auditing a coding agent's execution trace to find real defects.

You are given a structured summary of one run (goal, plan size, cost, tool \
usage, declared-vs-actual file writes), narration lines and error events \
pulled from the raw trace, and a set of DETERMINISTIC CHECKS already computed \
for you — treat those as established facts, not things to re-derive. Your \
job is the part those checks cannot do: judging whether the agent \
misunderstood the goal, whether a passing check still hides a real defect \
(test-fitting, a workaround masking the real bug), or anything else a human \
reviewer would flag reading the story of the run.

Report, as a JSON list of objects with fields `category` (one of \
"harness_bug", "model_weakness", "prompt_misinterpretation", "build_failure", \
"other"), `summary` (one sentence), `evidence` (a short quote or fact \
from the input that supports it), and, for `harness_bug` findings only, \
`proposed_fix` (one sentence naming the actual mechanism to change — a \
prompt, a budget, a check the harness is missing — not "investigate \
further"; omit or leave empty for every other category):

- Anything that looks like a real bug in the produced code or the process \
that built it.
- Anywhere the agent seems to have misunderstood what was asked — the \
GOAL WORDS WITH NO PLAN MATCH list below is a candidate list only, not a \
verdict: confirm each one against the narration before reporting it, and \
ignore any that are just paraphrased differently.
- Anywhere the model's own capability (not the harness) seems to be the \
limiting factor.
- A deterministic check that found something (a lint-only verification, a \
missing declared signature/test) is already a fact — don't re-report it \
as your own discovery, but DO use it as a lead into the narration for what \
caused it or what it masked, and report that story if there is one.

Be concise. If nothing is wrong, return an empty list. Do not invent issues \
not supported by the input. Return ONLY the JSON list, no prose before or \
after it.

--- TRACE SUMMARY ---
{summary}

--- DETERMINISTIC CHECKS ---
{checks}

--- NARRATION / ERRORS ---
{narration}
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _narration_lines(events: list[Event]) -> list[str]:
    lines: list[str] = []
    for e in events:
        kind = e.get("kind")
        if kind not in _RELEVANT_KINDS:
            continue
        if kind == "log":
            if e.get("detail"):
                continue
            lines.append(f"[log] {e.get('message', '')[:300]}")
        elif kind == "step_verify":
            if not e.get("passed"):
                lines.append(
                    f"[step_verify FAILED] step={e.get('step')} "
                    f"cmd={e.get('command')} output={str(e.get('output'))[:300]}"
                )
        else:
            payload = {k: v for k, v in e.items() if k not in ("ts", "kind")}
            lines.append(f"[{kind}] {json.dumps(payload)[:400]}")
    return lines


def _deterministic_checks(events: list[Event], workspace_root: Path | str | None) -> dict:
    """Everything `tracetools` can establish without a model call, gathered
    into one block so the audit prompt can hand it over as fact rather than
    make the model re-derive it from raw narration. Cuts real ground the
    model would otherwise have to cover itself — `verify_coverage_gaps`
    alone reproduced, at zero model cost, a finding GLM previously had to
    read the whole narration to reach (a step "verified" by lint alone that
    masked a later regression); see DESIGN-LOG.md's audit-speed section."""
    checks: dict[str, Any] = {
        "declared_vs_actual": tracetools.declared_vs_actual(events),
        "undeclared_writes": tracetools.undeclared_writes(events),
        "verify_coverage_gaps": tracetools.verify_coverage_gaps(events),
        "goal_words_with_no_plan_match": tracetools.goal_requirement_gaps(events),
    }
    if workspace_root is not None:
        checks["new_files_not_in_plan"] = tracetools.new_files_not_in_plan(events, workspace_root)
        checks["plan_conformance"] = tracetools.plan_conformance(events, workspace_root)
    return checks


def build_audit_prompt(events: list[Event], workspace_root: Path | str | None = None) -> str:
    """The prompt handed to the audit model for one trace's events.

    `workspace_root` is optional and enables the two checks that need the
    project's current files on disk (shadow files, plan-vs-code
    conformance) — without it, the audit still runs, just without those two
    deterministic leads.
    """
    summary = tracetools.summarize_trace(events)
    checks = _deterministic_checks(events, workspace_root)
    narration = "\n".join(_narration_lines(events)[:_MAX_NARRATION_LINES])
    return AUDIT_PROMPT_TEMPLATE.format(
        summary=json.dumps(summary, indent=2)[:_SUMMARY_CHAR_CAP],
        checks=json.dumps(checks, indent=2)[:_SUMMARY_CHAR_CAP],
        narration=narration[:_NARRATION_CHAR_CAP],
    )


def parse_findings(text: str) -> list[dict]:
    """Lenient parse of a model's reply into a findings list.

    Models asked for "only JSON" still sometimes wrap it in a ```json fence
    or add a stray leading/trailing sentence — this tolerates both rather
    than failing the whole audit over formatting the prompt already asked
    it not to use.
    """
    candidate = text.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("["), candidate.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    return [f for f in parsed if isinstance(f, dict) and "summary" in f]


def audit_trace(
    events: list[Event],
    client: ChatClient,
    model: str = DEFAULT_AUDIT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    workspace_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run one trace's events through the audit model. Returns findings plus
    the cost/model metadata needed to report on the audit itself."""
    prompt = build_audit_prompt(events, workspace_root)
    reply = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    return {
        "model": model,
        "cost_usd": reply.usage.cost_usd,
        "findings": parse_findings(reply.content),
        "raw": reply.content,
    }


def audit_traces(
    trace_dir: Path | str,
    client: ChatClient,
    pattern: str = tracetools.DEFAULT_PATTERN,
    failed_only: bool = True,
    model: str = DEFAULT_AUDIT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    concurrency: int = DEFAULT_CONCURRENCY,
    workspace_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Audit every trace in a directory (failed runs only, by default — that's
    where nearly all real findings concentrate, and skipping clean runs keeps
    a directory-wide audit's cost proportional to what's actually wrong).

    Traces are audited concurrently, up to `concurrency` at once — each is an
    independent model call, so a directory of N failing traces costs roughly
    one trace's latency instead of N of them serialized (measured: 4 traces
    at ~14-30s each serialized is 1-2 minutes; concurrent, it's whichever one
    is slowest). Order of `results` matches `outcomes`' order regardless of
    which call finishes first.

    `workspace_root`, when given, is passed to every trace's prompt so the
    shadow-file and plan-conformance checks can run — same current-state
    caveat as `tracetools.new_files_not_in_plan`: most reliable against the
    most recent trace for that workspace.
    """
    outcomes = tracetools.run_outcomes(trace_dir, pattern)
    rows = [row for row in outcomes if not failed_only or row["ok"] is False]
    if not rows:
        return []

    def _run(row: dict) -> dict[str, Any]:
        events = tracetools.load_trace(Path(trace_dir) / row["trace"])
        try:
            result = audit_trace(
                events,
                client,
                model=model,
                reasoning_effort=reasoning_effort,
                workspace_root=workspace_root,
            )
        except Exception as exc:
            # One trace hitting a provider stall (seen in practice: a 180s
            # streaming deadline on the audit call itself) must not discard
            # every other trace's findings — pool.map raises on first
            # exception and stops yielding, so an unguarded call here turns
            # "audit N traces" into "audit however many finished before the
            # first flaky one." Recorded as a zero-cost, zero-finding result
            # with the error visible, not silently dropped.
            return {
                "trace": row["trace"],
                "model": model,
                "cost_usd": 0.0,
                "findings": [],
                "raw": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"trace": row["trace"], **result}

    with ThreadPoolExecutor(max_workers=min(concurrency, len(rows))) as pool:
        return list(pool.map(_run, rows))
