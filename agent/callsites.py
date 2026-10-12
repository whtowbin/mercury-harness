"""Every file that uses a name the goal renames or removes, checked against the plan.

The one accuracy failure in the rename-callers A/B (DESIGN-LOG §62.1) was a
plan that assigned three of five caller modules and never mentioned the other
two; the fix rounds then swapped the old and new names back and forth. That is
a finding problem, and finding is deterministic: a word search over the
workspace names every file the plan must write. CallSiteGate sends a plan back
once when it leaves any of them out. No model call; it costs a planning round
trip only on the plans it holds.
"""

from __future__ import annotations

import json
import keyword
import re
from pathlib import Path

from .outline import _python_files

# Goals that change a name every caller must follow, as a whole word: `remove`
# inside `remove_and_report` is a function name, not a request (DESIGN-LOG §66.1).
_CHANGES_CALLERS = re.compile(
    r"(?<![\w.])(rename[sd]?|renaming|replac(?:e|es|ed|ing)|remov(?:e|es|ed|ing)|"
    r"delet(?:e|es|ed|ing)|signature|every call(?:er|-? ?site)s?|all call(?:er|-? ?site)s?)(?!\w)",
    re.IGNORECASE,
)
# A clause that forbids the change instead of asking for it: "without changing
# any public signature", "do not change the class's public signature", "should
# not delete". The negation has to come shortly before the request; "Remove
# make_line entirely; do not leave an alias" still asks for the removal.
_NEGATED = re.compile(r"\b(?:not|never|without|don't|doesn't|mustn't|shouldn't)\b", re.IGNORECASE)
_NEGATION_WINDOW_WORDS = 6
_CLAUSE = re.compile(r"(?<=[.!?;])\s+|\n+")
# A code-looking name: `backticked`, dotted.name, name(, or snake_case. Plain
# words in a goal are not identifiers.
_CODE_NAME = re.compile(
    r"`([A-Za-z_][\w.]*)(?:\(\))?`|\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)|\b([A-Za-z_]\w*)\("
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"
)


def names_to_follow(goal: str, root: Path) -> set[str]:
    """Names that a clause of `goal` asks to rename, remove or re-sign, if the
    workspace defines them; empty otherwise. Only the clauses making that
    request count, and not ones that forbid it."""
    found = set()
    for clause in _CLAUSE.split(goal):
        if not _asks_for_change(clause):
            continue
        for match in _CODE_NAME.finditer(clause):
            dotted = next(g for g in match.groups() if g)
            name = dotted.rsplit(".", 1)[-1]
            if len(name) > 2 and not keyword.iskeyword(name):
                found.add(name)
    if not found:
        return set()
    defined = _defined_names(root)
    return {n for n in found if n in defined}


def _asks_for_change(clause: str) -> bool:
    """Whether `clause` requests a rename/removal/signature change, i.e. has a
    trigger word not preceded within a few words by a negation."""
    for match in _CHANGES_CALLERS.finditer(clause):
        before = clause[: match.start()].split()[-_NEGATION_WINDOW_WORDS:]
        if not _NEGATED.search(" ".join(before)):
            return True
    return False


def _defined_names(root: Path) -> set[str]:
    pattern = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)
    names: set[str] = set()
    for path in _python_files(root):
        try:
            names.update(pattern.findall(path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return names


def files_using(root: Path, names: set[str]) -> dict[str, set[str]]:
    """{name: workspace-relative files mentioning it as a whole word}."""
    words = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    uses: dict[str, set[str]] = {n: set() for n in names}
    for path in _python_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, word in words.items():
            if word.search(text):
                uses[name].add(str(path.relative_to(root)))
    return uses


def uncovered(uses: dict[str, set[str]], steps: list) -> dict[str, list[str]]:
    """Files that use a followed name but no step writes (or lists, when a
    step declares no `writes`)."""
    covered: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        declared = step.get("writes") or step.get("files") or []
        if isinstance(declared, list):
            covered.update(str(f).removeprefix("./") for f in declared)
    missing = {n: sorted(files - covered) for n, files in uses.items()}
    return {n: files for n, files in missing.items() if files}


class CallSiteGate:
    """Holds submit_plan once when the plan leaves out a file that uses a name
    the goal renames or removes (Settings.callsite_check)."""

    event = "callsite_gate_held"

    def __init__(self, goal: str, root: Path, trace) -> None:
        self.names = names_to_follow(goal, root)
        self.uses = files_using(root, self.names) if self.names else {}
        self.trace = trace
        self.held = False
        if self.names:
            trace.event(
                "callsite_check",
                names=sorted(self.names),
                files={n: sorted(f) for n, f in self.uses.items()},
            )

    def observe(self, call: dict) -> None:
        pass

    def hold(self, call: dict) -> str | None:
        fn = call.get("function") or {}
        if self.held or not self.names or fn.get("name") != "submit_plan":
            return None
        try:
            steps = json.loads(fn.get("arguments") or "{}").get("steps") or []
            if isinstance(steps, str):
                steps = json.loads(steps)
        except (json.JSONDecodeError, AttributeError):
            return None
        missing = uncovered(self.uses, steps)
        if not missing:
            return None
        self.held = True
        listing = "; ".join(f"`{n}` is used in {', '.join(files)}" for n, files in missing.items())
        return (
            "Plan not accepted yet. The goal changes a name every caller must follow, and no "
            f"step writes these files: {listing}. Add them to a step's `writes` (a mechanical "
            "change across files can be one step), or resubmit unchanged if they genuinely "
            "need no edit."
        )
