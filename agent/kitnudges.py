"""Nudges that point the model at a tool built for what it is doing by hand.

In 3,915 bench runs an optional tool was called on 4 of the ~3,100 step kits
that carried one (DESIGN-LOG §62): being offered a tool does not get it used,
and forcing trace_python into every kit got it zero calls (§43.1). What has
changed this model's habits is a sentence attached to the tool result at the
moment the habit costs something (no_progress_nudge, the batching nudge). So
these watch for the by-hand version of a job a tool does, and say so then.

WIDENING_PHASES names the loops where a nudge may add a tool the step was not
offered (Registry.add; the trace_python nudge in runner.py does): execute and
fix, whose kit is a trigger-narrowed execution kit. The planner and reviewer
kits are read-only by design. The ast_grep nudge that also lived here is
shelved (DESIGN-LOG §62.1: nudged 22 times, never followed).

Nothing here costs a model call; each nudge rides a tool result already being
sent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .projectconfig import foreign_linters
from .tools import Registry
from .trace import Trace

WIDENING_PHASES = ("execute:", "fix:")
# run_bash commands that a core tool does better, and why it does. The test
# watcher now reads a run_bash pytest result too (runner._test_result_text),
# so the pytest pointer is about what run_tests adds on top: output cut to
# the failures, and a passing run the step's verify does not repeat.
_SHELL_REDIRECTS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "run_tests",
        re.compile(r"\bpytest\b"),
        "run_tests runs the same suite and returns just the failures, and a passing "
        "run is remembered so the step's own verify does not run it again. Use "
        "run_tests for tests.",
    ),
    (
        "check_code",
        re.compile(r"\bruff\s+check\b|\bty\s+check\b"),
        "check_code runs ruff and the type checker on the files you name in one "
        "call, and can apply ruff's safe fixes. Use it for lint and type checks.",
    ),
)


@dataclass
class KitWatch:
    """State for the nudges below, one per tool loop."""

    redirected: set[str] = field(default_factory=set)
    foreign_lint_warned: bool = False
    # nudge_about_search_churn
    searches_since_fetch: int = 0
    queries: list[str] = field(default_factory=list)
    search_nudges: int = 0


def _calls(tool_calls: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for call in tool_calls:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            out.append((fn.get("name") or "", args))
    return out


def nudge_about_shell_duplicates(
    watch: KitWatch,
    tool_calls: list[dict],
    messages: list[dict],
    registry: Registry,
    trace: Trace,
    phase: str,
) -> int:
    """Point a run_bash pytest/ruff/ty command at the core tool that does it,
    once per tool per loop."""
    results = messages[-len(tool_calls) :] if tool_calls else []
    sent = 0
    for (name, args), message in zip(_calls(tool_calls), results, strict=False):
        if name != "run_bash" or message.get("role") != "tool":
            continue
        command = str(args.get("command", ""))
        for tool, pattern, advice in _SHELL_REDIRECTS:
            if tool in watch.redirected or tool not in registry or not pattern.search(command):
                continue
            watch.redirected.add(tool)
            message["content"] = str(message.get("content", "")) + f"\n\n[harness] {advice}"
            trace.event("shell_redirect_nudge", phase=phase, tool=tool)
            sent += 1
            break
    return sent


# A Ruff invocation that rewrites code rather than only reporting on it.
_RUFF_REWRITE = re.compile(r"\bruff\b[^|;&]*?(?:--fix|--select|--unsafe-fixes|\bformat\b)")


def nudge_about_foreign_lint(
    watch: KitWatch,
    tool_calls: list[dict],
    messages: list[dict],
    root: Path,
    trace: Trace,
    phase: str,
) -> int:
    """Once per loop: a run_bash Ruff fix, select or format in a project that lints
    with something else gets told so.

    On django (flake8 and isort in setup.cfg) the executor ran `ruff check --select
    DTZ,UP031` and `--fix`, rewrote the module's imports and string formatting, and
    the f-string conversion turned `\\w{3}` into `\\w` in a date regex: five tests that
    had passed broke (DESIGN-LOG §73). The command has already run by the time this
    is seen, so the note is about the diff it left: keep only what the goal needs.
    """
    if watch.foreign_lint_warned:
        return 0
    results = messages[-len(tool_calls) :] if tool_calls else []
    for (name, args), message in zip(_calls(tool_calls), results, strict=False):
        if name != "run_bash" or message.get("role") != "tool":
            continue
        if not _RUFF_REWRITE.search(str(args.get("command", ""))):
            continue
        linters = foreign_linters(root)
        if not linters:
            return 0
        watch.foreign_lint_warned = True
        message["content"] = str(message.get("content", "")) + (
            f"\n\n[harness] This project lints with {', '.join(linters)}, not Ruff. Ruff's "
            "fixes and rule selections rewrite code to a style the project does not use, and "
            "a rewrite can change behaviour (an f-string conversion has broken a regex this "
            "way). Undo any change outside the lines the goal needs, and do not run Ruff "
            "fixes here again."
        )
        trace.event("foreign_lint_nudge", phase=phase, linters=linters)
        return 1
    return 0


# Settings.web_research (DESIGN-LOG §116). On 41 GAIA questions answered right in some runs and
# wrong in others, the wrong runs made 13.5 searches against 8.1, opened their first page after
# 4.8 searches against 2.7, and half their queries rephrased an earlier one (40% in right runs).
SEARCHES_BEFORE_FETCH = 3
MAX_SEARCH_NUDGES = 2
# Word overlap (Jaccard) above which a query counts as a rephrasing of an earlier one.
NEAR_REPEAT = 0.6
_SEARCH_PHASES = ("plan", "execute:", "fix:")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _rephrases(query: str, earlier: list[str]) -> bool:
    words = _words(query)
    return any(
        len(words & _words(q)) / max(1, len(words | _words(q))) > NEAR_REPEAT for q in earlier
    )


def nudge_about_search_churn(
    watch: KitWatch,
    tool_calls: list[dict],
    messages: list[dict],
    trace: Trace,
    phase: str,
) -> int:
    """A web_search result gets one line when the loop is searching instead of reading: the
    third search since the last fetch_url, or a query that rephrases an earlier one. At most
    MAX_SEARCH_NUDGES a loop; it rides the result already being sent, so it costs no call."""
    if not phase.startswith(_SEARCH_PHASES):
        return 0
    results = messages[-len(tool_calls) :] if tool_calls else []
    sent = 0
    for (name, args), message in zip(_calls(tool_calls), results, strict=False):
        if name == "fetch_url":
            watch.searches_since_fetch = 0
            continue
        if name != "web_search" or message.get("role") != "tool":
            continue
        query = str(args.get("query", ""))
        repeat = _rephrases(query, watch.queries)
        watch.queries.append(query)
        watch.searches_since_fetch += 1
        if watch.search_nudges >= MAX_SEARCH_NUDGES:
            continue
        if repeat:
            reason = "repeat"
            advice = (
                "This search rephrases an earlier one, and a rephrased query mostly returns the "
                "same pages. Open the most promising result with fetch_url, or search for a "
                "different entity or source."
            )
        elif watch.searches_since_fetch >= SEARCHES_BEFORE_FETCH:
            reason = "no_fetch"
            advice = (
                f"That is {watch.searches_since_fetch} searches without opening a result. A "
                "snippet is a fragment of a page: open the most promising result above with "
                "fetch_url and read it before searching again."
            )
        else:
            continue
        message["content"] = str(message.get("content", "")) + f"\n\n[harness] {advice}"
        trace.event(
            "search_churn_nudge",
            phase=phase,
            reason=reason,
            searches=watch.searches_since_fetch,
        )
        watch.search_nudges += 1
        sent += 1
    return sent
