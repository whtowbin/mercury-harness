"""What goes last in the context on each turn (DESIGN-LOG §82-83).

Mercury 2.5 "weights recent context heavily" (Inception's prompt guide), and a
rule moved to the end of the system prompt changed nothing while the same rule
placed after the last message tripled batching (§82.2). So the state a turn
most needs is rebuilt from the loop's own history and sent after the last
message, for that call only: the history, and so the cached prefix, is never
changed.

Each optional section is deterministic, read from tool calls and results
already in `messages`, and appears only when it has something to say:

- ``errors``: after a failing run_tests, the failing tests, where it failed,
  and how many runs in a row have failed the same way.
- ``ledger``: what this step has done so far: files edited and how often, test
  runs and the last result, edit misses.
- ``api``: the planner's looked-up facts (research-first findings) that name
  something this step mentions: the API detail it must not get wrong.
- ``inspect``: library functions the code written this step calls but this step
  has not looked up with inspect_module (numpy and scipy included: on SciCode a
  misused library function is the failure).
- ``key``: the one fact the planner said this step must not get wrong
  (Settings.plan_key_facts adds `key_fact` to each step of submit_plan).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass

from .editcalls import EDIT_TOOLS, edited_paths, written_text
from .failsig import failure_details
from .prompts import EXECUTOR_TAIL, PLANNER_TAIL

# A failing-test list longer than this is cut; the tail must stay short.
TAIL_MAX_FAILING = 3
# The api section's facts are cut to this many characters, whole facts only.
TAIL_API_CHARS = 700
# Library calls the inspect section names at most.
TAIL_MAX_UNINSPECTED = 4


@dataclass(frozen=True)
class TailItems:
    errors: bool = False
    ledger: bool = False
    api: bool = False
    inspect: bool = False
    key: bool = False


@dataclass(frozen=True)
class _Call:
    name: str
    args: dict
    result: str


def _calls(messages: list[dict]) -> list[_Call]:
    """Every tool call in `messages` with its arguments and its result, in order."""
    results = {
        m.get("tool_call_id"): str(m.get("content") or "")
        for m in messages
        if m.get("role") == "tool"
    }
    calls: list[_Call] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(
                _Call(
                    function.get("name", ""),
                    args if isinstance(args, dict) else {},
                    results.get(call.get("id"), ""),
                )
            )
    return calls


def _edited_paths(call: _Call) -> list[str]:
    return edited_paths(call.name, call.args)


def _failed_edits(call: _Call) -> int:
    """Edits in this call that did not apply: the whole call, or each edit_blocks block that
    reported an error."""
    if call.name not in EDIT_TOOLS - {"write_file"}:
        return 0
    if call.result.startswith("error:"):
        return 1
    return call.result.count(" ---\nerror") if call.name == "edit_blocks" else 0


def _test_summary(result: str) -> str:
    """The last non-empty line of a run_tests result: its pytest summary."""
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    return lines[-1][:120] if lines else ""


def ledger_section(calls: list[_Call]) -> str:
    edits: Counter[str] = Counter()
    misses = 0
    tests = [c for c in calls if c.name == "run_tests"]
    for call in calls:
        for path in _edited_paths(call):
            if path:
                edits[path] += 1
        misses += _failed_edits(call)
    if not edits and not tests:
        return ""
    lines = []
    if edits:
        lines.append("edited: " + ", ".join(f"{p} x{n}" for p, n in edits.most_common()))
    if misses:
        lines.append(f"edits that did not apply: {misses} (re-read the region before retrying)")
    if tests:
        lines.append(f"test runs: {len(tests)}; last: {_test_summary(tests[-1].result)}")
    return "<work_so_far>\n" + "\n".join(lines) + "\n</work_so_far>"


def errors_section(calls: list[_Call]) -> str:
    """The latest run_tests result, when it failed: which tests, where, and how
    many runs in a row have failed on the same tests."""
    tests = [c for c in calls if c.name == "run_tests"]
    if not tests:
        return ""
    ids, count, where = failure_details(tests[-1].result)
    if not ids and not count:
        return ""
    # Runs in a row that failed on the same tests. A passing run ends the streak:
    # it has no ids either, and when the latest failure's ids did not parse an
    # empty set would otherwise match every earlier pass, so one failure after
    # two passes read as "failed 3 runs in a row" (code review of 2026-09-30).
    streak = 0
    for call in reversed(tests):
        earlier_ids, earlier_count, _ = failure_details(call.result)
        if (not earlier_ids and not earlier_count) or earlier_ids != ids:
            break
        streak += 1
    names = sorted(ids)[:TAIL_MAX_FAILING]
    lines = [
        f"failing: {', '.join(names) or f'{count} test(s)'}"
        + (" (+more)" if len(ids) > len(names) else "")
    ]
    if where:
        lines.append(f"where: {where}")
    if streak > 1:
        lines.append(
            f"the same tests have failed {streak} runs in a row: your last change did not fix them"
        )
    lines.append(
        "Before editing again, decide whether the test or the code is wrong, from the error above."
    )
    return "<last_test_failure>\n" + "\n".join(lines) + "\n</last_test_failure>"


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def api_section(step, facts: tuple[str, ...]) -> str:
    """Findings that share a name with the step (its title, intent, acceptance or
    signatures), or the first few when none does; whole facts, capped."""
    if not facts:
        return ""
    words = set(
        _WORD.findall(" ".join([step.title, step.intent, step.acceptance, *step.signatures]))
    )
    relevant = [f for f in facts if words & set(_WORD.findall(f))] or list(facts)
    lines: list[str] = []
    used = 0
    for fact in relevant:
        if used + len(fact) > TAIL_API_CHARS:
            break
        lines.append(f"- {fact}")
        used += len(fact)
    if not lines:
        return ""
    return "<api_facts>\n" + "\n".join(lines) + "\n</api_facts>"


_IMPORT_AS = re.compile(r"^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", re.M)
_FROM_IMPORT = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+([^\n(]+)", re.M)
_DOTTED_CALL = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(")


def _third_party(module: str) -> bool:
    top = module.split(".")[0]
    return top not in sys.stdlib_module_names and top not in ("solution", "test_solution")


def library_calls(code: str) -> list[str]:
    """Dotted calls in `code` through a name bound to a third-party module:
    `np.linalg.qr` for `import numpy as np`, `special.jv` for `from scipy import
    special`. Stdlib and local modules are left out, and so are numpy's
    top-level basics (`np.array`, `np.dot`): the smoke run's hint named those,
    and Mercury spent inspect_module calls on functions it does not get wrong.
    numpy counts from its submodules on (`np.linalg.norm`, `np.fft.fft`)."""
    aliases: dict[str, str] = {}
    for module, alias in _IMPORT_AS.findall(code):
        if _third_party(module):
            aliases[alias or module.split(".")[0]] = module if alias else module.split(".")[0]
    for module, names in _FROM_IMPORT.findall(code):
        if _third_party(module):
            for part in names.split(","):
                bits = part.strip().split(" as ")
                if bits[0].strip():
                    aliases[bits[-1].strip()] = f"{module}.{bits[0].strip()}"
    seen: list[str] = []
    for call in _DOTTED_CALL.findall(code):
        head = call.split(".")[0]
        if head not in aliases or call in seen:
            continue
        full = aliases[head] + call[len(head) :]
        if full.startswith("numpy.") and full.count(".") < 2:
            continue
        seen.append(call)
    return seen


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in f"/{path}"


def _written_code(calls: list[_Call]) -> str:
    """Code this step wrote outside test files: a test's np.isclose is not an API
    the solution can get wrong."""
    return "\n".join(
        text
        for call in calls
        for path, text in written_text(call.name, call.args)
        if not _is_test_path(path)
    )


def inspect_section(calls: list[_Call], file_text: str = "") -> str:
    """Library functions called in code written this step (the imports may be in
    the file already, `file_text`) that no inspect_module call this step named."""
    written = _written_code(calls)
    if not written:
        return ""
    looked_up = " ".join(
        " ".join(str(v) for v in c.args.values()) for c in calls if c.name == "inspect_module"
    )
    names = library_calls(file_text + "\n" + written)
    in_new_code = [n for n in names if n in written]
    missing = [n for n in in_new_code if n.split(".")[-1] not in looked_up][:TAIL_MAX_UNINSPECTED]
    if not missing:
        return ""
    return (
        "<unchecked_library_calls>\n"
        f"Your code calls {', '.join(missing)} without having looked them up in this step. "
        "inspect_module gives the exact signature, argument order and return shape in one "
        "call; check any you are not certain of before relying on the result.\n"
        "</unchecked_library_calls>"
    )


def step_tail(
    step,
    iteration: int,
    max_iters: int,
    messages: list[dict] | None = None,
    items: TailItems | None = None,
    facts: tuple[str, ...] = (),
    file_text: str = "",
) -> str:
    """The tail for one executor turn: the step and its rules (always), plus the
    optional sections `items` asks for, when they have anything to say."""
    items = items or TailItems()
    left = max(0, max_iters - iteration)
    base = EXECUTOR_TAIL.format(
        step_id=step.id,
        title=step.title,
        acceptance=step.acceptance or "(see the step)",
        turns_left=left,
        max_turns=max_iters,
        last_turns="; this is one of your last turns" if left <= 2 else "",
    )
    calls = _calls(messages or [])
    key_fact = getattr(step, "key_fact", "")
    sections = [
        f"<key_fact>{key_fact}</key_fact>" if items.key and key_fact else "",
        api_section(step, facts) if items.api else "",
        errors_section(calls) if items.errors else "",
        inspect_section(calls, file_text) if items.inspect else "",
        ledger_section(calls) if items.ledger else "",
    ]
    if not any(sections):
        return base
    # The rules stay last of all: state first, then what to do about it.
    head, _, rules = base.partition("<rules>")
    return head + "".join(s + "\n" for s in sections if s) + "<rules>" + rules


# The goal is cut here in the planner's tail: it is a reminder, the full text
# is in the first message.
PLANNER_TAIL_GOAL_CHARS = 600


def planner_tail(
    goal: str, iteration: int, max_iters: int, messages: list[dict] | None = None
) -> str:
    """The planner's tail for one turn (Settings.planner_tail)."""
    left = max(0, max_iters - iteration)
    short = goal if len(goal) <= PLANNER_TAIL_GOAL_CHARS else goal[:PLANNER_TAIL_GOAL_CHARS] + "..."
    return PLANNER_TAIL.format(
        goal=short,
        turns_left=left,
        max_turns=max_iters,
        last_turns="; submit the plan now" if left <= 2 else "",
    )


def with_key_fact(tool):
    """`submit_plan` with a `key_fact` on each step (Settings.plan_key_facts)."""
    parameters = copy.deepcopy(tool.parameters)
    step_props = parameters["properties"]["steps"]["items"]["properties"]
    step_props["key_fact"] = {
        "type": "string",
        "description": (
            "The one fact this step must not get wrong, in a sentence: a convention, "
            "formula, unit, sign, argument order or return shape, e.g. 'scipy.special.sph_harm "
            "takes (m, n, azimuth, polar): azimuth before polar'. It is repeated to the "
            "executor on every turn of this step. Empty when there is none."
        ),
    }
    return dataclasses.replace(tool, parameters=parameters)
