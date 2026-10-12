"""Process signals from a run's trace, so a sweep says *how* a config worked.

`bench report` answers outcome questions — did it pass, what did it cost — and
those have been at a ceiling for most of this project's history: eight of nine
configs at 100% accuracy in one sweep, every arm at 100% in the red-start
sweep. When two configs both pass everything, the outcome columns are the same
number twice and the interesting difference is in *how* they got there: one
burned sixteen calls cycling edit/test, the other took nine.

Every counter here was hand-mined from traces with a throwaway script at least
twice before it moved into this module, which is this repo's stated trigger for
building the tool instead ("second time, build the tool"). They are deterministic reads of the
event log — no model call, no judgment — and they are what the failure-mode
work in DESIGN-LOG §43-44 was measured with.

Cheap enough to run on every bench record: one pass over a file that is
typically a few hundred lines.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from agent import failsig
from agent.editcalls import EDIT_TOOLS

# Tool calls that gather information rather than change the workspace. Used
# only for "how much did it look before it leapt", so `run_tests` is
# deliberately not here: running the suite is what produces the failure being
# reacted to, not a read of the code.
READ_TOOLS = frozenset(
    {
        "read_file",
        "read_files",
        "grep",
        "glob_files",
        "list_dir",
        "show_changes",
        "inspect_module",
        "run_python",
        "trace_python",
        "git_diff",
        "git_log",
    }
)
_FILE_READS = frozenset({"read_file", "read_files"})
# The line agent.phases.finalcheck.new_type_errors_text writes. A bare "ty: " also matches
# "empty: " and "Property: " in pytest output (§105 smoke: 3 false reports in arms without it).
_TYPE_REPORT = re.compile(r"(?m)^ty: \d+ type error\(s\) this run introduced")
_CHECK_KINDS = frozenset(
    {"tool_call_failed", "tool_arg_parse_error", "final_check", "after_edit_check"}
)
_TEST_PATH = re.compile(r"(^|/)(test_[^/]*|[^/]*_test)\.py$")
NUDGE_KINDS = (
    "no_progress_nudge",
    "repeated_probe_nudge",
    "same_failure_nudge",
    "failure_count_rose_nudge",
    "batching_nudge",
)


@dataclass
class Diagnostics:
    """One run's process signals. Every field is a count unless named otherwise."""

    loops_exhausted: int = 0
    nudges: int = 0
    step_retries: int = 0
    repeated_calls: int = 0
    tool_errors: int = 0
    tool_calls: int = 0
    test_runs: int = 0
    failed_test_runs: int = 0
    collection_errors: int = 0
    # Edits made after a failing test run that did not reduce the failure
    # count. The residue §44 identified: an edit that changed nothing about
    # the failure, which is what an edit/test cycle is made of.
    unproductive_edits: int = 0
    # The longest streak of those in a row — one wasted edit is ordinary, four
    # in a row is the loop worth naming.
    longest_unproductive_streak: int = 0
    # Read-only calls before the run's first edit. Low means it leapt.
    reads_before_first_edit: int = 0
    # Tool calls made while planning, which is the phase §44 found deciding a
    # fix after a single `read_files`.
    plan_tool_calls: int = 0
    # probe_python calls while planning (Settings.plan_probe): whether the
    # planner used the ability to run code at all. trace_python (§43.1) was
    # offered in 15 runs and called in none, so uptake is checked first.
    plan_probes: int = 0
    # First submit_plan calls held by runner.ProbeGate (plan_probe="require").
    probe_holds: int = 0
    # Runs where plan_probe was on but the workspace had no Python to probe,
    # so the tool, its prompt block and the gate were all left out.
    probe_withheld: int = 0
    # plan_probe="repro": what the planner's reproduce snippet did. reproduced
    # (failed: the problem is real), passed (the report did not reproduce, so
    # the plan went back), broken (died before reaching the code), none (left
    # empty on new work), missing (empty on a bugfix: sent back once),
    # not_run (denied or timed out).
    repro_reproduced: int = 0
    repro_passed: int = 0
    repro_broken: int = 0
    repro_none: int = 0
    # Empty snippet on a goal the planner itself called a bugfix: sent back once.
    repro_missing: int = 0
    repro_not_run: int = 0
    # Settings.premise_gate holds: a source edit after the plan's reproduction
    # passed, and an edit removing or weakening a pre-existing test (the premise-gate plan
    # prices these beside wasted edits).
    repro_holds: int = 0
    pinned_test_holds: int = 0
    # Plans the call-site check sent back for leaving out a file that uses a
    # renamed name (Settings.callsite_check, DESIGN-LOG §66).
    callsite_holds: int = 0
    # Edits to the machinery that runs the tests (tests/runtests.py, conftest.py,
    # the test settings module), held once per file (DESIGN-LOG §73).
    test_infra_holds: int = 0
    # Whether the run's own final suite was green when it stopped: 1 yes, 0 no,
    # -1 it never ran one. What separates "the agent knew it had not finished"
    # from "the agent was satisfied and wrong", the second of which is the
    # failure mode a hidden test exists to catch.
    final_green: int = -1
    # Execute steps that edited non-test source, and how many of those ran a
    # failing test before their first such edit. The executor prompt says to
    # write the test, watch it fail, then fix; on 1,285 traced steps that
    # edited source in existing-code runs, 13% did (85% ran no test at all
    # before editing). Counted per step so a run can be compared on how much of
    # the rule it followed.
    source_edit_steps: int = 0
    red_first_steps: int = 0
    # Reviews whose verdict said "pass" while listing findings (3.3% of 1,289
    # traced reviews). Findings, not the verdict, drive the fix rounds, so these
    # were acted on anyway; counted so a rise is visible.
    verdict_contradicts_findings: int = 0
    # Edits to pre-existing tests that carried an acceptable reason, and those that
    # went through with none (agent/testchange.py; DESIGN-LOG §59).
    test_changes_justified: int = 0
    test_changes_unjustified: int = 0
    # Times the red-first gate held a step's first source edit.
    tdd_holds: int = 0
    # What the executed plan looks like, from its own `plan` event. These are
    # the mechanism metrics for a change to how the planner drafts: the draft
    # never reaches the executor, so it can only help through the steps written
    # after it, and named tests / signatures per step are the free, deterministic
    # measure of how specific those steps are.
    plan_steps: int = 0
    plan_draft_words: int = 0
    plan_named_tests: int = 0
    plan_signatures: int = 0
    plan_declared_files: int = 0
    # Edit calls the tool rejected — almost always an `old_string` that did
    # not match the file byte for byte. Folded into tool_errors before, so a
    # change aimed at targeted edits had no mechanism metric of its own.
    edit_misses: int = 0
    # A test run straight after a failing one with no edit in between: the
    # suite re-run to look at a failure again rather than to test a change.
    reruns_without_edit: int = 0
    # Hidden reasoning tokens by phase, from each llm_call's usage. Mercury
    # returns the count but never the text (~90% of completion tokens at
    # medium/high effort), so this is the only view of how hard it thought.
    plan_reasoning_tokens: int = 0
    execute_reasoning_tokens: int = 0
    review_reasoning_tokens: int = 0
    # Model calls by phase, and output, from each llm_call (DESIGN-LOG §83). The
    # tail arms act on some phases only, so a total can hide a real change in
    # one phase under noise in another (§82.2: execute/fix/review -2.1 calls,
    # planning +1.4).
    plan_calls: int = 0
    execute_calls: int = 0
    fix_calls: int = 0
    review_calls: int = 0
    out_tokens: int = 0
    max_call_out_tokens: int = 0
    # Replies stopped by max_tokens, and loops that gave up after two replies
    # with no tool call: what an output cap could cost.
    length_stops: int = 0
    stalls: int = 0
    # web_search calls, and those that came back with nothing. A search backend whose
    # every engine was blocked answered 0 results for two days of sweeps, recorded only
    # as `results: 0` (DESIGN-LOG §89.1).
    web_searches: int = 0
    empty_searches: int = 0
    # DESIGN-LOG §105's mechanism metrics. File reads and turns that edit, in execute and
    # fix; edit_blocks calls; edit calls whose arguments did not parse (edit_files' list of
    # objects, 4% historically); whether the first final check found new type errors; and
    # after-edit checks that reported type errors (types_after_edit).
    exec_reads: int = 0
    edit_turns: int = 0
    edit_blocks_calls: int = 0
    edit_parse_errors: int = 0
    first_final_type_errors: int = 0
    after_edit_type_reports: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# The agent's own parser, not a second copy of it: a sweep that counted
# failures differently from the harness it is measuring would be comparing two
# definitions of "worse".
_failed_count = failsig.failed_count


def _edited_path(event: dict) -> str:
    """The file an edit tool call targets, from a dict or a stringified args."""
    args = event.get("args")
    if isinstance(args, dict):
        return str(args.get("path", ""))
    match = re.search(r"""['"]path['"]:\s*['"]([^'"]+)['"]""", str(args or ""))
    return match.group(1) if match else ""


# Event kinds that are only ever a tally. Later entries win, which preserves the
# precedence the old if/elif chain had: loop_exhausted, then NUDGE_KINDS, then
# the rest.
_COUNTER_FIELD: dict[str, str] = {
    "step_retry": "step_retries",
    "repeated_call": "repeated_calls",
    "tool_error": "tool_errors",
    "tool_call_failed": "tool_errors",
    "tool_arg_parse_error": "tool_errors",
    "tdd_gate_held": "tdd_holds",
    "repro_passed_held": "repro_holds",
    "pinned_test_held": "pinned_test_holds",
    "test_change_justified": "test_changes_justified",
    "test_change_unjustified": "test_changes_unjustified",
    "probe_gate_held": "probe_holds",
    "repro_gate_held": "probe_holds",
    "callsite_gate_held": "callsite_holds",
    "test_infrastructure_held": "test_infra_holds",
    "probe_withheld": "probe_withheld",
    **dict.fromkeys(NUDGE_KINDS, "nudges"),
    "loop_exhausted": "loops_exhausted",
    "stalled": "stalls",
}


# Which reasoning counter an llm_call's phase feeds, by the phase name before
# any ":step-id" suffix. The fixer is the executor model working a finding, so
# it is counted with execution.
_REASONING_FIELD: dict[str, str] = {
    "plan": "plan_reasoning_tokens",
    "execute": "execute_reasoning_tokens",
    "fix": "execute_reasoning_tokens",
    "review": "review_reasoning_tokens",
}


class _EventFold:
    """The state one pass over a run's events carries between events."""

    def __init__(self) -> None:
        self.result = Diagnostics()
        self.seen_edit = False
        # None outside a failing test run; otherwise the failure count to beat.
        self.pending: int | None = None
        self.edited_since_test = False
        self.streak = 0
        # Per execute step: has it edited source yet, and what did its first test
        # run before that say. `awaiting` is the step whose run_tests call is in
        # flight, since the run_tests *event* carries no phase of its own.
        self.steps: dict[str, dict] = {}
        self.awaiting: str | None = None
        self.edit_turn_keys: set[tuple[str, object]] = set()
        self.final_seen = False

    def feed(self, event: dict) -> None:
        kind = event.get("kind", "")
        if kind in _CHECK_KINDS:
            self._check_event(kind, event)
        elif kind == "llm_call":
            self._reasoning(event)
            self._llm_call(event)
        counter = _COUNTER_FIELD.get(kind)
        if counter is not None:
            setattr(self.result, counter, getattr(self.result, counter) + 1)
        elif kind == "plan":
            self._plan(event)
        elif kind == "repro":
            field = f"repro_{event.get('outcome', '')}"
            if hasattr(self.result, field):
                setattr(self.result, field, getattr(self.result, field) + 1)
        elif kind == "review_result":
            if event.get("verdict") == "pass" and event.get("findings"):
                self.result.verdict_contradicts_findings += 1
        elif kind == "final_tests":
            self.result.final_green = 1 if event.get("exit_code") == 0 else 0
        elif kind == "tool_call":
            self._tool_call(event)
        elif kind == "run_tests":
            self._run_tests(event)
        elif kind == "web_search":
            self.result.web_searches += 1
            if not event.get("results"):
                self.result.empty_searches += 1

    def _plan(self, event: dict) -> None:
        # The last plan event is the one that was executed; a replan or a plan
        # audit can emit another.
        planned = [step for step in event.get("steps") or [] if isinstance(step, dict)]
        result = self.result
        result.plan_steps = len(planned)
        result.plan_draft_words = sum(len(str(note).split()) for note in event.get("draft") or [])
        result.plan_named_tests = sum(len(step.get("tests") or []) for step in planned)
        result.plan_signatures = sum(len(step.get("signatures") or []) for step in planned)
        result.plan_declared_files = sum(len(step.get("files") or []) for step in planned)

    def _reasoning(self, event: dict) -> None:
        phase = str(event.get("phase", "")).split(":", 1)[0]
        field = _REASONING_FIELD.get(phase)
        if field is not None:
            tokens = int(event.get("reasoning_tokens") or 0)
            setattr(self.result, field, getattr(self.result, field) + tokens)

    def _llm_call(self, event: dict) -> None:
        result = self.result
        phase = str(event.get("phase", "")).split(":", 1)[0]
        field = f"{phase}_calls"
        if phase in ("plan", "execute", "fix", "review"):
            setattr(result, field, getattr(result, field) + 1)
        out = int(event.get("completion_tokens") or 0)
        result.out_tokens += out
        result.max_call_out_tokens = max(result.max_call_out_tokens, out)
        if event.get("finish_reason") == "length":
            result.length_stops += 1

    def _tool_call(self, event: dict) -> None:
        result = self.result
        result.tool_calls += 1
        tool = event.get("tool", "")
        phase = str(event.get("phase", ""))
        if phase == "plan":
            result.plan_tool_calls += 1
            result.plan_probes += tool == "probe_python"
        if phase.startswith("execute:"):
            state = self.steps.setdefault(phase, {"source": False, "first_test": None})
            if tool in EDIT_TOOLS and not _TEST_PATH.search(_edited_path(event)):
                if not state["source"]:
                    state["source"] = True
                    result.source_edit_steps += 1
                    result.red_first_steps += state["first_test"] == "red"
            elif tool == "run_tests":
                self.awaiting = phase
        if phase.startswith(("execute:", "fix:")):
            self._work_call(tool, phase, event)
        if tool in EDIT_TOOLS:
            self.seen_edit = True
            if self.pending is not None:
                self.edited_since_test = True
        elif tool in READ_TOOLS and not self.seen_edit:
            result.reads_before_first_edit += 1

    def _check_event(self, kind: str, event: dict) -> None:
        """Edit failures, the first final check's type errors, after-edit type reports."""
        result = self.result
        if kind == "tool_call_failed" and event.get("tool") in EDIT_TOOLS:
            result.edit_misses += 1
        elif kind == "tool_arg_parse_error" and event.get("tool") in EDIT_TOOLS:
            result.edit_parse_errors += 1
        elif kind == "final_check" and not self.final_seen:
            self.final_seen = True
            typed = event.get("new_type_errors")
            result.first_final_type_errors = int(typed not in (0, None, [], "0"))
        elif kind == "after_edit_check" and _TYPE_REPORT.search(str(event.get("result", ""))):
            result.after_edit_type_reports += 1

    def _work_call(self, tool: str, phase: str, event: dict) -> None:
        result = self.result
        result.exec_reads += tool in _FILE_READS
        result.edit_blocks_calls += tool == "edit_blocks"
        if tool in EDIT_TOOLS:
            key = (phase, event.get("iteration"))
            if key not in self.edit_turn_keys:
                self.edit_turn_keys.add(key)
                result.edit_turns += 1

    def _run_tests(self, event: dict) -> None:
        result = self.result
        result.test_runs += 1
        if self.awaiting is not None:
            state = self.steps[self.awaiting]
            self.awaiting = None
            if not state["source"] and state["first_test"] is None:
                state["first_test"] = "red" if event.get("exit_code") not in (0, None) else "green"
        if event.get("collection_error"):
            result.collection_errors += 1
        if self.pending is not None and not self.edited_since_test:
            result.reruns_without_edit += 1
        now = _failed_count(event.get("summary"))
        failed = event.get("exit_code") not in (0, None)
        if failed:
            result.failed_test_runs += 1
        # Judge the *previous* failure against this result, but only if an edit
        # happened in between: two test runs with no edit between them say
        # nothing about an edit.
        if self.pending is not None and self.edited_since_test:
            if now is not None and now < self.pending:
                self.streak = 0
            else:
                result.unproductive_edits += 1
                self.streak += 1
                result.longest_unproductive_streak = max(
                    result.longest_unproductive_streak, self.streak
                )
        if not failed:
            self.streak = 0
        self.pending = now if failed else None
        self.edited_since_test = False


def from_events(events: list[dict]) -> Diagnostics:
    """Read one run's events into counters. Tolerant of missing fields: a
    trace from an older harness simply contributes zero to what it lacks."""
    fold = _EventFold()
    for event in events:
        fold.feed(event)
    return fold.result


def served_models(events: list[dict]) -> list[str]:
    """Distinct "requested -> served @ provider" strings for a run's model calls.

    Requested is what the config asked for; served is what the provider says
    answered. They differ when a routing layer substitutes a snapshot or an
    upstream provider under an unchanged id, which is invisible in a sweep that
    records only the requested one. Older traces carry no served id and
    contribute nothing here.
    """
    seen: dict[str, None] = {}
    for event in events:
        if event.get("kind") != "llm_call" or not event.get("served_model"):
            continue
        provider = event.get("provider") or "?"
        seen[f"{event.get('model', '?')} -> {event['served_model']} @ {provider}"] = None
    return list(seen)


def portable_path(path: str | Path) -> str:
    """How a results file stores a trace's path: relative to the working directory when the
    trace lies under it (the repo root, where bench runs), else `~/...` under the home
    directory, else absolute. An absolute home path put the author's username into 172 lines
    of two committed results files, and breaks on any other machine."""
    if not str(path):
        return ""
    resolved = Path(path).expanduser().resolve()
    for base, prefix in ((Path.cwd().resolve(), ""), (Path.home().resolve(), "~/")):
        try:
            return prefix + str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def trace_file(stored: str | Path) -> Path:
    """The trace a results file names, whichever form `portable_path` stored it in."""
    return Path(stored).expanduser()


def served_models_from_trace(path: str | Path) -> list[str]:
    try:
        with trace_file(path).open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return served_models(events)


def task_kind(events: list[dict]) -> str:
    """The executed plan's `task_kind`, or "" when the planner gave none."""
    kind = ""
    for event in events:
        if event.get("kind") == "plan":
            kind = str(event.get("task_kind") or "")
    return kind


def task_kind_from_trace(path: str | Path) -> str:
    try:
        with trace_file(path).open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return task_kind(events)


def from_trace(path: str | Path) -> Diagnostics:
    """Counters for one trace file. An unreadable or missing trace yields empty
    counters rather than raising: a sweep must not fail over its own telemetry."""
    try:
        with trace_file(path).open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return Diagnostics()
    return from_events(events)
