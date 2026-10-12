"""Deterministic analysis over saved trace JSONL files.

This module never calls a model. It exists to turn a directory of traces into
structured facts a model (or a human) can review quickly, and to answer the
specific questions a real trace-mining session on this project kept asking by
hand: what did a step actually write versus declare, does the same defect
recur across runs, what failed and why. Every function here reads events
already on disk; where a question needs the *current* state of the workspace
(a shadow-file collision, for instance), the caller supplies `workspace_root`
explicitly rather than this module guessing which workspace a trace ran on.

The composable core is `find_calls` + `next_event`: nearly every finding from
manually mining eight real traces of the platformer app was built from "find
this tool call, then look at what happened right after it," done by hand with
jq and a one-off Python loop each time. Everything else here is a thin layer
on top of that pair.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .editcalls import CHANGE_TOOLS, EDIT_TOOLS

Event = dict[str, Any]

DEFAULT_PATTERN = "run-*.jsonl"


# -- loading ------------------------------------------------------------------


def load_trace(path: Path | str) -> list[Event]:
    """One trace file's events, in order. A malformed line is skipped, not
    fatal — a trace from a killed process is still worth reading up to where
    it stops, rather than discarding everything before the truncation."""
    events: list[Event] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def load_traces(trace_dir: Path | str, pattern: str = DEFAULT_PATTERN) -> dict[str, list[Event]]:
    """Every trace file in a directory, keyed by filename, oldest first by name."""
    trace_dir = Path(trace_dir)
    return {p.name: load_trace(p) for p in sorted(trace_dir.glob(pattern))}


# -- the composable core: find a call, then see what happened next -----------


def find_calls(
    events: list[Event], tool: str | None = None, phase: str | None = None
) -> list[tuple[int, Event]]:
    """(index, event) for every tool_call matching tool/phase, in order. The
    index is kept because most callers need it to inspect what followed."""
    out = []
    for i, ev in enumerate(events):
        if ev.get("kind") != "tool_call":
            continue
        if tool is not None and ev.get("tool") != tool:
            continue
        if phase is not None and ev.get("phase") != phase:
            continue
        out.append((i, ev))
    return out


def next_event(
    events: list[Event], index: int, kinds: tuple[str, ...] | None = None, within: int = 3
) -> Event | None:
    """The first event after `index` matching one of `kinds` (any kind, if
    None), searched up to `within` events ahead."""
    for ev in events[index + 1 : index + 1 + within]:
        if kinds is None or ev.get("kind") in kinds:
            return ev
    return None


def pair_calls_with_outcomes(events: list[Event], tool: str) -> list[dict]:
    """Every call to `tool`, paired with the failure event (if any) that
    immediately followed it.

    `outcome` is "failed" when a `tool_call_failed` or `tool_error` event
    appears within the search window, "ok" otherwise. `tool_call_failed` is
    what runner.py's `_dispatch` now logs whenever a tool returns its own
    "error: ..." string rather than raising — a trace recorded before that
    existed can only ever show "ok" or a raised `tool_error` here, not a
    soft failure the tool reported by returning text.
    """
    results = []
    for i, call in find_calls(events, tool=tool):
        failure = next_event(events, i, kinds=("tool_call_failed", "tool_error"))
        results.append(
            {
                "iteration": call.get("iteration"),
                "phase": call.get("phase"),
                "outcome": "failed" if failure else "ok",
                "failure": failure,
            }
        )
    return results


# -- retroactive declared-vs-actual / shadow-file scan ------------------------


def written_paths(events: list[Event]) -> list[str]:
    """Every path this trace's tools actually wrote to, in first-touched
    order — the trace-file equivalent of `ctx.touched`."""
    seen: list[str] = []
    seen_set: set[str] = set()

    def _add(path: str | None) -> None:
        if path and path not in seen_set:
            seen.append(path)
            seen_set.add(path)

    for ev in events:
        kind = ev.get("kind")
        if kind in ("write_file", "edit_file", "delete_file"):
            _add(ev.get("path"))
        elif kind == "move_file":
            _add(ev.get("src"))
            _add(ev.get("dst"))
    return seen


def _last_plan_event(events: list[Event]) -> dict:
    """The trace's most recent `plan` event, or {} if none.

    A run can emit more than one: `_plan_until_accepted` re-plans up to
    `MAX_REPLANS` times on feedback, and an optional plan-audit phase can
    resubmit a patched plan after the original. The last one is always the
    plan that was actually executed, so every reader here wants it, not the
    first draft.
    """
    return next((e for e in reversed(events) if e.get("kind") == "plan"), {})


def _declared_writes(events: list[Event]) -> dict[str, set[str]]:
    """{step_id: declared writes} from the trace's last `plan` event, or {} if none."""
    plan_event = _last_plan_event(events)
    if not plan_event:
        return {}
    return {
        step.get("id", ""): set(step.get("writes") or []) for step in plan_event.get("steps", [])
    }


def declared_vs_actual(events: list[Event]) -> list[dict]:
    """Per step: declared writes vs. what the run actually touched by the
    time it finished. The read-only, post-hoc counterpart to
    `execute._unwritten_declared_files`, for scanning traces already on disk
    that predate that check (or to confirm it fired correctly on one that
    doesn't)."""
    touched = set(written_paths(events))
    return [
        {"step": step_id, "declared_writes": sorted(writes), "unwritten": sorted(writes - touched)}
        for step_id, writes in _declared_writes(events).items()
    ]


def undeclared_writes(events: list[Event]) -> list[str]:
    """Files this trace's tools actually touched that no step, anywhere in
    the plan, ever declared in `writes` — the direction `declared_vs_actual`
    does not check.

    `declared_vs_actual` answers "did the step do the job it declared"
    (`writes` minus what was touched); this answers the opposite question,
    "did the run touch something no step owned up to." Built from a repeated
    pattern in real audited traces: a step declares `writes=[convert.py]`,
    edits `convert.py` *and* `test_convert.py` in the same step, and
    `declared_vs_actual` reports a clean run because `convert.py` was written
    as promised — the undeclared edit to an already-existing test file is
    invisible to it, since that check only ever looks for what's *missing*.
    An audit model reading the same trace had to notice this by hand, in
    narration, in run after run.

    Deliberately plan-wide rather than per-step, unlike the live
    `_undeclared_shadow_files` check it complements: reconstructing which
    step a given write happened during would need step boundaries this
    trace format doesn't record on write events, and the plan-wide answer
    (something not owned by *any* step) is what an auditor actually needs to
    decide whether to look closer — same scope as `new_files_not_in_plan`,
    which this generalizes from "brand-new files only" to every write, edit,
    move and delete the run made.

    Not a verdict: a test file edited alongside its implementation in one
    step is usually the right call, and execute.py's own `_undeclared_shadow_files`
    deliberately does not gate on this for exactly that reason (see its
    docstring) — this exists so the fact is visible to a reader or an audit
    model, not to block or blame the run for it.
    """
    declared: set[str] = set()
    for writes in _declared_writes(events).values():
        declared.update(writes)
    return sorted(p for p in written_paths(events) if p not in declared)


def new_files_not_in_plan(
    events: list[Event], workspace_root: Path | str | None = None
) -> list[dict]:
    """Files this trace's tools created (`write_file` with `overwrote=False`)
    that no step declared in its `writes`, optionally checked against the
    current workspace for a basename collision elsewhere in the tree.

    The read-only, post-hoc counterpart to `execute._undeclared_shadow_files`
    — built from the real trace where a step declared writes=[coin.py,
    sprites.py] and actually created six files, including a second
    sprites.py inside a tests/ directory it invented. `__init__.py` is
    excluded from the collision check for the same reason the live version
    excludes it: one per package is completely ordinary.

    `workspace_root`'s collision check reads the workspace's *current* state,
    not its state as of when this particular trace ran. Calling this over
    several traces from the same project (oldest to newest) can attribute a
    collision to an older trace that did not actually cause it, when a later
    trace is what created the second copy — the live version in execute.py
    does not have this gap, since it always runs against the workspace's real
    state at that exact moment. Most reliable for the most recent trace; for
    an older one, check which trace actually created the colliding path
    (`written_paths` on each) before concluding that run is at fault.
    """
    declared: set[str] = set()
    for writes in _declared_writes(events).values():
        declared.update(writes)

    created = [
        ev.get("path")
        for ev in events
        if ev.get("kind") == "write_file" and ev.get("overwrote") is False
    ]
    undeclared = [p for p in created if p and p not in declared]

    rows = []
    for path in undeclared:
        row: dict[str, Any] = {"path": path, "shadows": []}
        if workspace_root is not None and Path(path).name != "__init__.py":
            root = Path(workspace_root)
            for candidate in root.rglob(Path(path).name):
                rel = str(candidate.relative_to(root))
                if rel != path:
                    row["shadows"].append(rel)
        rows.append(row)
    return rows


# -- plan conformance (deterministic — no model call) -------------------------
#
# These exist to shrink what the model-based audit (agent/audit.py) has to
# figure out for itself. "Did the agent build the function it said it
# would" is a fact whenever the plan already committed to a specific name,
# not a judgment call — checking it by name via ripgrep costs nothing and
# leaves the model-based audit only the genuinely judgment-requiring half
# ("does this correctly-named thing actually do the right thing").

_TEST_COMMAND_RE = re.compile(r"\bpytest\b|\btest\b", re.IGNORECASE)


def verify_coverage_gaps(events: list[Event]) -> list[dict]:
    """Steps whose step_verify command(s) never look like a test run — the
    lint-only-verification pattern found by hand in a real audit: a step
    "verified" with `ruff check` alone has nothing to catch a behavioral
    regression until (if ever) a later full-suite run does. Heuristic (the
    command text doesn't contain a recognizable test-runner word), not a
    parse of the command — a step with no step_verify event at all is not
    reported here, since "never verified" and "verified without tests" are
    different findings and this function only knows about the latter."""
    by_step: dict[str, list[str]] = {}
    for ev in events:
        if ev.get("kind") == "step_verify":
            by_step.setdefault(ev.get("step", ""), []).append(ev.get("command", ""))
    return [
        {"step": step, "commands": commands}
        for step, commands in by_step.items()
        if commands and not any(_TEST_COMMAND_RE.search(c) for c in commands)
    ]


def _plan_steps(events: list[Event]) -> list[dict]:
    plan = _last_plan_event(events)
    return plan.get("steps", []) or []


def _signature_name(signature: str) -> str | None:
    """Best-effort function/method name out of a plan step's free-text
    `signatures` entry ("Game.update(self)" -> "update", "def foo(x)" ->
    "foo"). Best-effort because a plan signature is prose a planner wrote,
    not a parsed AST node — good enough to grep for, not guaranteed to
    parse cleanly for every string a model might produce."""
    before_paren = signature.split("(", 1)[0]
    tokens = re.findall(r"\w+", before_paren)
    return tokens[-1] if tokens else None


def _test_name(test_entry: str) -> str:
    """ "test_foo: does the thing" -> "test_foo"."""
    return test_entry.split(":", 1)[0].strip()


def _name_exists_in_workspace(workspace_root: Path, name: str) -> bool | None:
    """Whether `name` appears anywhere in the workspace's text files, via
    ripgrep — a name search, not a structural one (it matches inside a
    comment or string too, same honest limitation as a plain grep), which is
    the cheap version of "does this exist at all" rather than "is this truly
    defined as a function". Returns None (not False) when ripgrep itself is
    unavailable or too slow, so a missing tool reads as "unchecked" rather
    than a false "missing" finding."""
    try:
        result = subprocess.run(
            ["rg", "--quiet", "--fixed-strings", "--word-regexp", name, str(workspace_root)],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.returncode == 0


def plan_conformance(events: list[Event], workspace_root: Path | str) -> dict[str, list[dict]]:
    """Which of the plan's own declared signatures/tests never turn up
    anywhere in the current workspace, checked by name via ripgrep. Same
    current-workspace caveat as `new_files_not_in_plan`: this reads the
    workspace's state right now, not its state as of when this trace ran —
    most reliable read against the most recent trace for a given workspace.
    A name ripgrep couldn't check (tool unavailable) is silently skipped
    rather than reported as missing.
    """
    root = Path(workspace_root)
    missing_signatures, missing_tests = [], []
    for step in _plan_steps(events):
        step_id = step.get("id", "")
        for sig in step.get("signatures") or []:
            name = _signature_name(sig)
            if name and _name_exists_in_workspace(root, name) is False:
                missing_signatures.append({"step": step_id, "signature": sig, "name": name})
        for test_entry in step.get("tests") or []:
            name = _test_name(test_entry)
            if name and _name_exists_in_workspace(root, name) is False:
                missing_tests.append({"step": step_id, "test": test_entry, "name": name})
    return {"missing_signatures": missing_signatures, "missing_tests": missing_tests}


_GOAL_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "or",
        "but",
        "with",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "it",
        "its",
        "as",
        "by",
        "from",
        "when",
        "then",
        "than",
        "so",
        "if",
        "not",
        "no",
        "do",
        "does",
    }
)


def goal_requirement_gaps(events: list[Event]) -> list[str]:
    """Words from the run's stated goal that never turn up anywhere in the
    plan's own step titles/intents/acceptance text — a cheap, imperfect
    candidate list for "did the plan quietly drop part of what was asked",
    not a verdict. A word missing here might just be paraphrased differently
    (a real gap still needs a human or the model-based audit to confirm); a
    word present proves nothing about whether it was actually *built*, only
    that the plan mentions it. Exists to shrink what the model-based audit
    has to read cold: it can check this short candidate list against the
    narration instead of holding the entire goal in mind unaided while
    reading everything else.
    """
    run_start = next((e for e in events if e.get("kind") == "run_start"), {})
    plan = _last_plan_event(events)
    goal = run_start.get("goal") or plan.get("goal") or ""
    plan_text = " ".join(
        f"{s.get('title', '')} {s.get('intent', '')} {s.get('acceptance', '')}"
        for s in plan.get("steps", []) or []
    ).lower()

    seen: set[str] = set()
    gaps = []
    for word in re.findall(r"[a-zA-Z]{4,}", goal.lower()):
        if word in _GOAL_STOPWORDS or word in seen:
            continue
        seen.add(word)
        if word not in plan_text:
            gaps.append(word)
    return gaps


# -- cross-trace pattern mining -----------------------------------------------


def grep_traces(
    trace_dir: Path | str, pattern: str, pattern_glob: str = DEFAULT_PATTERN
) -> dict[str, list[str]]:
    """Which trace files mention `pattern` (a plain substring, case-
    insensitive) and where, one line per matching event.

    Built directly from the manual step that found the pygame-font-not-
    initialized defect recurring across four separate traces: reading one
    trace at a time never shows a defect whose pattern only becomes visible
    across several. `pattern` is a literal substring, not a regex — the
    traces this exists to search are prose and code snippets, where regex
    metacharacters in the search term are far more often a mistake than a
    feature.
    """
    rx = re.compile(re.escape(pattern), re.IGNORECASE)
    hits: dict[str, list[str]] = {}
    for name, events in load_traces(trace_dir, pattern_glob).items():
        matches = [
            f"{ev.get('kind')}: {json.dumps(ev, default=str)[:200]}"
            for ev in events
            if rx.search(json.dumps(ev, default=str))
        ]
        if matches:
            hits[name] = matches
    return hits


def run_outcomes(trace_dir: Path | str, pattern_glob: str = DEFAULT_PATTERN) -> list[dict]:
    """One row per trace: goal, final ok/fail, cost, call count — the
    overview table a real trace-mining session on this project kept
    rebuilding by hand with a jq one-liner over every file in the directory."""
    rows = []
    for name, events in load_traces(trace_dir, pattern_glob).items():
        goal = next((e.get("goal") for e in events if e.get("kind") == "run_start"), "")
        finals = [e for e in events if e.get("kind") == "final_check"]
        final = finals[-1] if finals else None
        totals = next((e for e in reversed(events) if e.get("kind") == "totals"), None)
        rows.append(
            {
                "trace": name,
                "goal": (goal or "")[:100],
                "ok": final.get("ok") if final else None,
                "cost_usd": round(totals["cost_usd"], 4) if totals else None,
                "calls": totals.get("calls") if totals else None,
            }
        )
    return rows


# -- per-run statistics --------------------------------------------------------


def tool_usage(events: list[Event]) -> dict[str, int]:
    """Count of each tool called, most-used first."""
    counts: dict[str, int] = {}
    for ev in events:
        if ev.get("kind") == "tool_call":
            name = ev.get("tool", "?")
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def file_statistics(events: list[Event]) -> dict[str, dict[str, int]]:
    """Per file: how many times it was read vs. edited/written this run."""
    stats: dict[str, dict[str, int]] = {}

    def _bump(path: str | None, key: str) -> None:
        if not path:
            return
        stats.setdefault(path, {"reads": 0, "edits": 0})[key] += 1

    for ev in events:
        kind = ev.get("kind")
        if kind == "tool_call" and ev.get("tool") in ("read_file", "read_files"):
            args = ev.get("args") or {}
            path = args.get("path")
            if path:
                _bump(path, "reads")
        elif kind in ("write_file", "edit_file"):
            _bump(ev.get("path"), "edits")
    return stats


def cost_breakdown(events: list[Event]) -> dict[str, dict[str, float]]:
    """Cost and call count per phase (the part before ':' in a phase name
    that carries a step id, e.g. "execute:fix-grounding" -> "execute")."""
    totals: dict[str, dict[str, float]] = {}
    for ev in events:
        if ev.get("kind") != "llm_call":
            continue
        phase = str(ev.get("phase", "?")).split(":", 1)[0]
        slot = totals.setdefault(phase, {"cost_usd": 0.0, "calls": 0})
        slot["cost_usd"] += ev.get("cost_usd", 0.0)
        slot["calls"] += 1
    for slot in totals.values():
        slot["cost_usd"] = round(slot["cost_usd"], 4)
    return totals


def summarize_trace(events: list[Event]) -> dict[str, Any]:
    """One structured summary of a single trace — the unit this project's
    Mercury-extraction / GLM-evaluation pipeline is meant to hand a model,
    with every field sourced from a deterministic reading of the trace
    rather than left for the model to reconstruct from raw events."""
    run_start = next((e for e in events if e.get("kind") == "run_start"), {})
    finals = [e for e in events if e.get("kind") == "final_check"]
    plan = _last_plan_event(events)
    totals = next((e for e in reversed(events) if e.get("kind") == "totals"), {})
    return {
        "goal": run_start.get("goal", ""),
        "workspace": run_start.get("workspace", ""),
        "steps_planned": len(plan.get("steps", [])),
        "final_ok": finals[-1].get("ok") if finals else None,
        "cost_usd": round(totals.get("cost_usd", 0.0), 4),
        "calls": totals.get("calls", 0),
        "tool_usage": tool_usage(events),
        "declared_vs_actual": declared_vs_actual(events),
    }


# -- holds: what the model did after a gate refused a call --------------------
#
# Four gates hold a call once and say how to proceed (the tdd gate, the two
# premise holds, the planner's probe and reproduction gates). §49.1, §54.3 and
# §57.1 each read by hand what the model did next, and each found the same
# uncomfortable shape (it re-sends the call); the third time is the trigger for
# a tool. Everything here reads events already on disk.

# The event each gate emits, and which family of call it holds.
HOLD_EVENTS: dict[str, str] = {
    "tdd_gate_held": "edit",
    "repro_passed_held": "edit",
    "pinned_test_held": "edit",
    "premise_gate_held": "edit",
    "probe_gate_held": "plan",
    "repro_gate_held": "plan",
    "callsite_gate_held": "plan",
    "test_infrastructure_held": "edit",
}


_TEST_PATH = re.compile(r"(^|/)(test_[^/]*|[^/]*_test)\.py$")


def _held_family_tool(family: str, tool: str) -> bool:
    return tool in CHANGE_TOOLS if family == "edit" else tool == "submit_plan"


def _edit_target(event: Event) -> str:
    """The file an edit's tool_call event targets; "" for a batch whose args the
    trace abbreviated too far to read."""
    args = event.get("args")
    if isinstance(args, dict):
        return str(args.get("path") or "")
    match = re.search(r"""['"]path['"]:\s*['"]([^'"]+)['"]""", str(args or ""))
    return match.group(1) if match else ""


def _classify_resend(event: Event, later: Event) -> str:
    """How the next call of the held family relates to the held call."""
    if event.get("kind") == "tdd_gate_held" and _TEST_PATH.search(_edit_target(later)):
        # The model answered a "run a test first" hold by editing the test, which
        # is the reaction that hold asks for. Counting it as a re-send hid that 18
        # of 30 held steps in sweep J went on to a failing test run before their
        # source edit. Only this hold: for a hold on an edit *to* a test, another
        # test edit is the same action again, not a reaction.
        return "wrote_test"
    held_fp, sent_fp = event.get("fp"), later.get("fp")
    if held_fp and sent_fp:
        return "resent_unchanged" if held_fp == sent_fp else "resent_changed"
    return "resent"


def _after_hold(events: list[Event], index: int, family: str) -> tuple[str, str, list[str]]:
    """(response, next tool, tools used in between) for the hold at `index`."""
    event = events[index]
    phase = event.get("phase")
    between: list[str] = []
    for later in events[index + 1 :]:
        if later.get("kind") in ("llm_call", "tool_call") and later.get("phase") != phase:
            break
        if later.get("kind") != "tool_call":
            continue
        tool = str(later.get("tool", ""))
        if _held_family_tool(family, tool):
            return _classify_resend(event, later), tool, between
        between.append(tool)
    return "gave_up", "", between


def hold_responses(events: list[Event]) -> list[dict[str, Any]]:
    """What the model did after each hold in one trace.

    For each hold event, the calls made in the same phase before the next call of
    the held family (another edit, or another `submit_plan`):

    - `at_once`: nothing in between, so the model re-sent immediately;
    - `between`: the tools it used in between, in order (a test run or a read means
      it reacted to the message rather than re-sending by reflex);
    - `response`: `wrote_test` when the next edit after a tdd-gate hold targets a
      test file (the reaction that hold asks for); `resent_unchanged` /
      `resent_changed` when the held call and the next one carry a fingerprint
      (`fp`) and it matches or does not; `resent` when the trace predates
      fingerprints so sameness is unknown; and `gave_up` when the phase made no
      further call of that family.
    """
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        family = HOLD_EVENTS.get(str(event.get("kind")))
        if family is None:
            continue
        response, next_tool, between = _after_hold(events, index, family)
        rows.append(
            {
                "kind": event.get("kind"),
                "phase": event.get("phase"),
                "iteration": event.get("iteration"),
                "tool": event.get("tool"),
                "paths": event.get("paths") or [],
                "response": response,
                "at_once": response != "gave_up" and not between,
                "between": between,
                "next_tool": next_tool,
            }
        )
    return rows


def summarize_holds(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """{hold kind: {response: count}}, with `at_once` and `investigated` split out
    of the resent responses since that split is the question."""
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        slot = table.setdefault(str(row["kind"]), {})
        if row["response"] == "gave_up":
            label = "gave_up"
        else:
            how = "at_once" if row["at_once"] else "after_investigating"
            label = f"{row['response']}:{how}"
        slot[label] = slot.get(label, 0) + 1
    return table


def holds_in(
    trace_dir: Path | str, pattern: str = DEFAULT_PATTERN, recursive: bool = False
) -> list[dict[str, Any]]:
    """Every hold response across a directory's traces, each tagged with its trace."""
    trace_dir = Path(trace_dir)
    paths = sorted(trace_dir.rglob(pattern) if recursive else trace_dir.glob(pattern))
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in hold_responses(load_trace(path)):
            rows.append({**row, "trace": path.name})
    return rows


# -- benchmark leakage (harness trace leaks) ----------------------------------


CODE_TOOLS = ("run_python", "probe_python", "trace_python", "run_bash")


def web_hits(
    events: list[Event], deny: tuple[str, ...], query: str | None = None
) -> list[dict[str, Any]]:
    """Every fetch_url, web_search and code-tool call (`CODE_TOOLS`, by a deny pattern in
    its code, DESIGN-LOG §93) that reached for a benchmark's sources.

    A fetch counts when its URL contains a `deny` pattern (the same match the
    web tools refuse at run time, `web.denied_by`); a search counts when its
    query matches the `query` regex. `returned` is how many characters the
    page held, 0 for one that failed or was refused, and `shown` how much of it
    reached the model (at most the call's `max_chars`): the size its own
    `fetch_url` event recorded, matched by URL, since in a batch of calls the
    next `fetch_url` event can belong to a sibling call. Text back is not proof
    of an answer (a directory listing counts too); read the URLs. Built at the
    second hand-run of this check (DESIGN-LOG §79.1, §80).
    """
    from .tools.web import MAX_FETCH_CHARS, denied_by

    query_re = re.compile(query, re.IGNORECASE) if query else None
    hits: list[dict[str, Any]] = []
    for i, call in find_calls(events):
        args = call.get("args") or {}
        tool = call.get("tool")
        if tool == "fetch_url":
            target = str(args.get("url", ""))
            if not denied_by(target, deny):
                continue
            returned = _fetched_chars(events, i, target)
            # The event records the whole page; the model saw at most what it asked for.
            asked = int(args.get("max_chars") or MAX_FETCH_CHARS)
            shown = min(returned, asked)
        elif tool in CODE_TOOLS:
            target = str(args.get("code") or args.get("command") or "")
            if not denied_by(target, deny):
                continue
            # What the code fetched is not traced; read the code.
            returned = shown = 0
        elif tool == "web_search" and query_re:
            target = str(args.get("query", ""))
            if not query_re.search(target):
                continue
            returned = shown = 0
        else:
            continue
        hits.append(
            {
                "phase": call.get("phase"),
                "tool": tool,
                "target": target,
                "returned": returned,
                "shown": shown,
            }
        )
    return hits + _search_result_hits(events, deny)


def _search_result_hits(events: list[Event], deny: tuple[str, ...]) -> list[dict[str, Any]]:
    """Search results the model was shown whose URL contains a `deny` pattern: it read
    their snippets without fetching anything. Only traces from §93.2 on record `hits`."""
    from .tools.web import denied_by

    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("kind") != "web_search":
            continue
        for hit in ev.get("hits") or []:
            url = str(hit.get("url", ""))
            if denied_by(url, deny):
                shown = len(str(hit.get("snippet", "")))
                out.append(
                    {
                        "phase": ev.get("phase"),
                        "tool": "web_search result",
                        "target": url,
                        "returned": shown,
                        "shown": shown,
                    }
                )
    return out


def _fetched_chars(events: list[Event], index: int, url: str) -> int:
    """Characters the fetch_url call at `index` returned: its own `fetch_url`
    event before the next model call, or 0 when there is none (it failed)."""
    for ev in events[index + 1 :]:
        kind = ev.get("kind")
        if kind == "llm_call":
            break
        if kind == "fetch_url" and ev.get("url") == url:
            return int(ev.get("chars", 0))
    return 0


def leaks_in(
    trace_dir: Path | str, deny: tuple[str, ...], query: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """`web_hits` for every trace under `trace_dir` (recursively), keyed by the
    trace's path relative to it; traces with no hits are left out. A trace file
    seen twice under one name (the bench keeps a copy in `failed-chats/`) is
    read once."""
    trace_dir = Path(trace_dir)
    out: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for path in sorted(
        trace_dir.rglob(DEFAULT_PATTERN), key=lambda p: ("failed-chats" in p.parts, p)
    ):
        if path.name in seen:
            continue
        seen.add(path.name)
        hits = web_hits(load_trace(path), deny, query)
        if hits:
            out[str(path.relative_to(trace_dir))] = hits
    return out


# -- SQL over traces (harness trace sql) -------------------------------------
# Columns pulled out of every event with a type, so a query needs no schema.
# Anything else is `data->>'field'` (text) or `data->'field'` (JSON). Reading
# each line as one raw JSON value, not letting DuckDB infer columns, is the
# point: ~50 event shapes share a file and inference gave up on them three
# times in one session (DESIGN-LOG §66).
_EVENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ts", "DOUBLE"),
    ("kind", "VARCHAR"),
    ("phase", "VARCHAR"),
    ("tool", "VARCHAR"),
    ("step", "VARCHAR"),
    ("iteration", "INTEGER"),
    ("latency_s", "DOUBLE"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("cost_usd", "DOUBLE"),
    ("finish_reason", "VARCHAR"),
    ("error", "VARCHAR"),
    ("result", "VARCHAR"),
    ("command", "VARCHAR"),
)
_EVENT_LISTS = ("encoded", "mistyped", "optional")


def sql_connection(trace_dir: Path | str, results: list[Path] | None = None) -> Any:
    """A DuckDB connection with an `events` view over every run-*.jsonl under
    `trace_dir`, one row per event, and a `results` view over bench result
    files when given. `events.run` is the trace file's path as globbed and
    `events.file` its name; join `parse_filename(results.trace_path) =
    events.file`, since `run` matches `trace_path` only when TRACE_DIR was
    written in the same form the bench recorded."""
    import duckdb  # the [analysis] extra: only this command needs it

    con = duckdb.connect()
    pattern = str(Path(trace_dir) / "**" / "run-*.jsonl").replace("'", "''")
    typed = ", ".join(
        f"TRY_CAST(json->>'{name}' AS {kind}) AS {name}" for name, kind in _EVENT_COLUMNS
    )
    lists = ", ".join(f"TRY_CAST(json->'{name}' AS VARCHAR[]) AS {name}" for name in _EVENT_LISTS)
    con.execute(
        f"CREATE VIEW events AS SELECT filename AS run, parse_filename(filename) AS file, "
        f"{typed}, {lists}, json AS data "
        f"FROM read_json_objects('{pattern}', format='newline_delimited', filename=true, "
        "ignore_errors=true) WHERE json_valid(json)"
    )
    if results:
        files = ", ".join("'" + str(p).replace("'", "''") + "'" for p in results)
        con.execute(
            f"CREATE VIEW results AS SELECT * FROM read_json([{files}], "
            "format='newline_delimited', union_by_name=true)"
        )
    return con


# -- what follows an edit (harness trace after-edit) --------------------------

_EDIT_TOOLS = sorted(EDIT_TOOLS)
_CHECK_TOOLS = ["run_tests", "check_code", "run_python", "run_bash"]

# One row per model turn: a turn is an llm_call and the tool calls logged after
# it, numbered within (run, phase) because parallel steps interleave their calls
# in one trace. Each edit turn that did not finish is paired with the phase's
# next turn.
_AFTER_EDIT_SQL = """
with ev as (
  select run, ts, kind, phase, tool,
    sum(case when kind = 'llm_call' then 1 else 0 end)
      over (partition by run, phase order by ts rows unbounded preceding) turn
  from events
  where kind in ('llm_call', 'tool_call')
    and list_contains($phases, split_part(phase, ':', 1))
), turns as (
  select run, phase, turn, count(tool) n,
    bool_or(list_contains($edit_tools, tool)) has_edit,
    bool_or(tool = 'run_tests') has_tests,
    bool_or(tool = 'finish_step') has_finish,
    bool_and(list_contains($check_tools, tool)) filter (where tool is not null) check_only
  from ev group by all
), paired as (
  select *, lead(n) over w next_n, lead(has_edit) over w next_edit,
    lead(has_finish) over w next_finish, lead(check_only) over w next_check_only
  from turns window w as (partition by run, phase order by turn)
)
select
  case when has_tests then 'edit + run_tests' else 'edit alone' end edit_turn,
  count(*) turns,
  round(100.0 * avg(case when next_check_only then 1 else 0 end), 1) pct_next_check_only,
  round(100.0 * avg(case when next_finish and next_n = 1 then 1 else 0 end), 1)
    pct_next_finish_only,
  round(100.0 * avg(case when next_edit then 1 else 0 end), 1) pct_next_edit,
  round(100.0 * avg(case when next_n is null then 1 else 0 end), 1) pct_phase_ended
from paired
where has_edit and not coalesce(has_finish, false)
group by all order by edit_turn
"""


def after_edit(
    trace_dir: Path | str, phases: tuple[str, ...] = ("execute", "fix")
) -> list[dict[str, Any]]:
    """What the model's next turn was after a turn that edited without finishing,
    split by whether that turn also ran the tests. A next turn that only runs a
    check after `edit alone` is a round trip the harness could have folded into
    the edit's own result; after `edit + run_tests` it is a re-run of a result the
    model already had."""
    params = {"phases": list(phases), "edit_tools": _EDIT_TOOLS, "check_tools": _CHECK_TOOLS}
    rel = sql_connection(trace_dir).execute(_AFTER_EDIT_SQL, params)
    columns = [d[0] for d in rel.description]
    return [dict(zip(columns, row, strict=True)) for row in rel.fetchall()]


# -- where model time went (harness trace latency) ----------------------------

# A runaway is a call that reasoned most of Mercury's 65,536-token ceiling away
# before answering (DESIGN-LOG §83, §95). 30,000 sits in the empty band between
# ordinary calls and the ~62,000-token runaways.
RUNAWAY_REASONING_TOKENS = 30_000

_LATENCY_CALLS = """
with c as (
  select {period} period, latency_s,
    coalesce(try_cast(data->>'total_s' as double), latency_s) total_s,
    try_cast(data->>'attempts' as int) attempts,
    try_cast(data->>'reasoning_tokens' as double) reasoning,
    try_cast(data->>'cached_tokens' as double) cached,
    try_cast(data->>'ttft_s' as double) ttft,
    try_cast(data->>'decode_tok_per_s' as double) decode_tps,
    try_cast(data->>'max_gap_s' as double) max_gap,
    prompt_tokens, completion_tokens
  from events
  where kind = 'llm_call' and coalesce(data->>'model', '') like $model and completion_tokens > 0
)
select period, count(*) calls,
  round(sum(latency_s) / 3600, 2) model_h,
  round(sum(total_s) / 3600, 2) with_retries_h,
  round(100 * (sum(total_s) - sum(latency_s)) / nullif(sum(total_s), 0), 1) retry_pct,
  count(*) filter (where attempts > 1) retried_calls,
  count(*) filter (where reasoning > $runaway) runaway_calls,
  round(sum(latency_s) filter (where reasoning > $runaway) / 3600, 2) runaway_h,
  round(avg(completion_tokens)) mean_out, round(median(completion_tokens)) median_out,
  round(avg(prompt_tokens)) mean_prompt,
  round(100 * sum(cached) / nullif(sum(prompt_tokens), 0), 1) cached_pct,
  round(median(ttft) filter (where prompt_tokens < 8000 and completion_tokens < 150), 2)
    ttft_small_s,
  round(median(decode_tps)) decode_tps,
  round(quantile_cont(max_gap, 0.99), 1) max_gap_p99_s,
  count(*) filter (where max_gap > 30) max_gap_over_30s
from c group by 1 order by 1
"""

_LATENCY_RETRIES = """
select {period} period,
  case when error like '%silent for%' then 'silent after first content (stall bound)'
       when error like '%no content within%' then 'no first content (30s cut)'
       when error like '%504%' or error like '%idle timeout%' then '504 idle timeout'
       when error like '%deadline%' then 'attempt deadline'
       when error like '%Rate limit%' or error like '%429%' then '429 rate limit'
       when error like '%Connection%' or error like '%Chunked%' or error like '%ReadTimeout%'
         then 'connection'
       else 'other' end cause,
  count(*) retried_calls,
  round(sum(try_cast(data->>'waited_s' as double)) / 3600, 2) waited_h,
  round(median(try_cast(data->>'waited_s' as double)), 1) median_wait_s
from events
where kind = 'llm_retried' and coalesce(data->>'model', '') like $model
group by 1, 2 order by 1, waited_h desc
"""

_LATENCY_TTFT = """
select {period} period,
  case when prompt_tokens < 8000 then 'a <8k' when prompt_tokens < 16000 then 'b 8-16k'
       when prompt_tokens < 32000 then 'c 16-32k' else 'd 32k+' end prompt_band,
  count(*) calls,
  round(median(try_cast(data->>'ttft_s' as double)), 2) median_ttft_s,
  round(median(latency_s), 2) median_latency_s
from events
where kind = 'llm_call' and coalesce(data->>'model', '') like $model and completion_tokens > 0
group by 1, 2 order by 1, 2
"""

_PERIODS = {
    "day": "strftime(to_timestamp(ts), '%Y-%m-%d')",
    "week": "strftime(date_trunc('week', to_timestamp(ts)), 'week of %Y-%m-%d')",
}


def latency(
    trace_dir: Path | str, model: str = "mercury", by: str = "week"
) -> dict[str, list[dict[str, Any]]]:
    """Where model time went, by period: model time with and without failed attempts,
    retry waits by cause, runaway reasoning, and first-token time by prompt size.

    `latency_s` on an llm_call counts only the attempt that succeeded; `total_s`
    includes the attempts before it and the backoff between them. Reading only the
    first hid a quarter of last week's Mercury time (DESIGN-LOG §95). `model` is a
    substring of the model id ("" for every model)."""
    if by not in _PERIODS:
        raise ValueError(f"by must be one of {sorted(_PERIODS)}")
    con = sql_connection(trace_dir)
    params = {"model": f"%{model}%", "runaway": RUNAWAY_REASONING_TOKENS}
    tables = {}
    for name, sql in (
        ("calls", _LATENCY_CALLS),
        ("retries", _LATENCY_RETRIES),
        ("first_token", _LATENCY_TTFT),
    ):
        query = sql.format(period=_PERIODS[by])
        used = {k: v for k, v in params.items() if f"${k}" in query}
        rel = con.execute(query, used)
        columns = [d[0] for d in rel.description]
        tables[name] = [dict(zip(columns, row, strict=True)) for row in rel.fetchall()]
    return tables
