"""Research before planning: the libraries' own documentation first, then the web.

On SciCode's dev set a Mercury planner made one planning call and one tool call a
problem, never looked anything up, and asked an absent user "Should I proceed?" 18
times; a GLM planner on the same problems made 20 web searches, 53 fetches and 61 skill
reads, drafted 287 words to Mercury's 74, and solved 10 of 14 problems to Mercury's 6 of
15 (DESIGN-LOG §76.1, §77). The research, not only the model, may be what differs.

`Settings.research_first` asks for it, in the order the user asked for: the installed
libraries' own API through `inspect_module` (the exact version in this workspace, offline),
then the skills, and only then the web. `ResearchGate` holds the first `submit_plan` once
when the goal imports a third-party library and planning checked no library's API, naming
the concrete lookups to make. Holds that ask for a
concrete action are followed; ones that ask to reconsider are not (§58).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.base import Tool

PLAN_RESEARCH_FIRST = """
**Settle the conventions the result depends on, before you plan.** Checks of numerical work
compare exact numbers, so an implicit choice read differently gives a wrong answer from code
that looks right. List each choice the goal leaves implicit that changes the output, and decide
it: which random-number API (the `np.random` module functions, which `np.random.seed` controls,
unless the signature passes a generator or seed), what one "step" or "iteration" covers (one
move, or one sweep over every element), whether a returned trajectory includes the initial
state, units and constants, sign conventions, array shape and orientation, inclusive or
exclusive ranges, list or array. Decide from the goal's own words, the function header and its
docstring first, then the field's standard convention. Put each in `findings` as
"decision: ... because ...".

**Then look up what you are not certain of, in this order.**
1. The libraries the goal uses, as installed here: `inspect_module(module, member)` gives the
   exact signatures and docstrings of the version in this workspace (for example
   `inspect_module("scipy.integrate", "solve_ivp")`). Prefer it to memory and to the web: it
   cannot be out of date. Skip what you already know for certain (`np.zeros`, `np.sum`).
2. `read_skill` for any listed skill that matches the work.
3. Only then `web_search` and `fetch_url`, for a method, formula, constant or convention that
   the libraries' own documentation does not settle.
Put each fact the steps depend on in `findings`, with where it came from: the executor sees
`findings` in every step, and it does not see your draft. Do not ask the user whether to proceed:
if the goal is clear enough to plan, plan; ask only a question whose answer would change the
plan.
"""

# Planning calls that count as having checked a library's API. Reading the workspace
# does not: a SciCode planner that read README.md had still checked nothing about scipy.
LIBRARY_LOOKUP_TOOLS = frozenset(
    {"inspect_module", "pypi_info", "read_skill", "web_search", "fetch_url"}
)
_IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M)


def _is_local(top: str, root: Path | None) -> bool:
    """Whether `top` is the workspace's own code, which reading the files already checks."""
    if root is None:
        return False
    return any(
        (base / top).is_dir() or (base / f"{top}.py").is_file() for base in (root, root / "src")
    )


# Libraries whose everyday API the model knows. On SciCode, where every goal imports numpy or
# scipy, the gate forced lookups of `np.zeros` and `np.sqrt`: 2-3 findings a plan, none that
# mattered, and none of 156 failing sub-steps was a misused library call (DESIGN-LOG §79.2).
WELL_KNOWN = frozenset({"numpy", "scipy", "matplotlib", "mpl_toolkits"})


def goal_libraries(goal: str, root: Path | None = None) -> list[str]:
    """Third-party modules the goal's own import lines name, in order, deduplicated.

    The standard library and the workspace's own packages are left out: neither has an
    API the planner could be wrong about for want of a lookup.
    """
    seen: list[str] = []
    for match in _IMPORT.finditer(goal or ""):
        module = match.group(1) or match.group(2)
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top in WELL_KNOWN or _is_local(top, root):
            continue
        if module in seen:
            continue
        seen.append(module)
    return seen


class ResearchGate:
    """Holds the planner's first submit_plan once when the goal imports a third-party
    library and planning never checked any library's API.

    Nothing else triggers it. A first version held any plan made with no lookup at
    all, and on the seeded tasks that was 4 of 25 plan-only runs, all on goals that use
    no library (a greenfield `expr.py`, `semver.py`) where the spec is the whole story
    (DESIGN-LOG §79). On SciCode every goal imports numpy or scipy, so it still holds
    there, where it did most of the research-first gain (§77.2).
    """

    event = "research_gate_held"

    def __init__(self, goal: str, root: Path | None = None) -> None:
        self.libraries = goal_libraries(goal, root)
        self.looked_up = False
        self.held = False

    def observe(self, call: dict) -> None:
        if (call.get("function") or {}).get("name") in LIBRARY_LOOKUP_TOOLS:
            self.looked_up = True

    def hold(self, call: dict) -> str | None:
        name = (call.get("function") or {}).get("name")
        if name != "submit_plan" or self.held or self.looked_up or not self.libraries:
            return None
        self.held = True
        lookups = ", ".join(f'inspect_module("{m}")' for m in self.libraries[:4])
        return (
            "[harness] Plan NOT submitted: the goal uses libraries whose API has not been "
            f"checked. Start with {lookups}, and the members you will call. Then read_skill "
            "for a matching skill, and web_search only for what the libraries' own "
            "documentation does not settle. Then send submit_plan again, with what you found "
            "in `findings`."
        )


# Findings the executor sees, at most this many, in a block of at most this many
# characters. Items are dropped whole, never cut: a signature cut in half is worse
# than none. The draft handoff cut research-first drafts (580-790 characters on
# average, up to 1,264) at 900, and is off by default (DESIGN-LOG §77.2).
FINDINGS_MAX = 8
FINDINGS_HANDOFF_CHARS = 2_000


def with_findings(tool: Tool) -> Tool:
    """`submit_plan` with a `findings` list: facts the steps rely on, with sources."""
    parameters = copy.deepcopy(tool.parameters)
    parameters["properties"]["findings"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            f"Up to {FINDINGS_MAX} facts and decisions the steps rely on, one per entry. A "
            "convention you settled: 'decision: angles in radians, because the header's "
            "default is np.pi / 4'. A fact you looked up, with its source: "
            "'scipy.integrate.solve_ivp(fun, t_span, y0, method=...) returns an object with "
            ".y shaped (n, len(t)) (inspect_module)'. Signatures, "
            "formulas, constants, conventions. The executor sees these in every step; it "
            "never sees your draft. Empty when nothing needed looking up."
        ),
    }
    return dataclasses.replace(tool, parameters=parameters)


def _inspect_args(messages: list[dict]) -> dict[str, dict]:
    """The arguments of each inspect_module call in a tool loop, by call id."""
    args_by_id: dict[str, dict] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "inspect_module":
                continue
            try:
                args_by_id[call.get("id", "")] = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
    return args_by_id


def _signature_line(content: str) -> str:
    prefix = "signature: "
    return next(
        (line.removeprefix(prefix) for line in content.splitlines() if line.startswith(prefix)),
        "",
    )


def looked_up_signatures(messages: list[dict]) -> list[tuple[str, str]]:
    """(module.member, signature line) for each inspect_module result in a tool loop.

    Read from the loop's own message history, so it costs no call: the planner
    already received these results.
    """
    args_by_id = _inspect_args(messages)
    found: list[tuple[str, str]] = []
    for message in messages:
        if message.get("role") != "tool" or message.get("name") != "inspect_module":
            continue
        args = args_by_id.get(message.get("tool_call_id", ""))
        signature = _signature_line(str(message.get("content") or ""))
        if args and signature:
            label = ".".join(part for part in (args.get("module"), args.get("member")) if part)
            found.append((label, signature))
    return found


def complete_findings(
    findings: list[str], lookups: list[tuple[str, str]], plan_text: str
) -> list[str]:
    """`findings`, plus any looked-up signature the plan uses but the planner left out.

    "Uses" means the member's own name appears in the steps' text; a module it
    only browsed is not added. Deduplicated, capped at FINDINGS_MAX.
    """
    out = [f.strip() for f in findings if str(f).strip()]
    stated = " ".join(out)
    for label, signature in lookups:
        name = label.rsplit(".", 1)[-1]
        if not re.search(rf"\b{re.escape(name)}\b", plan_text) or name in stated:
            continue
        # inspect_module prints the qualified name ("json.dumps(obj, ...)"); a bare
        # "dumps(obj, ...)" gets its module in front.
        shown = signature if signature.startswith(label) else f"{label}: {signature}"
        out.append(f"{shown} (inspect_module)")
        stated += " " + name
    return out[:FINDINGS_MAX]


def findings_handoff(findings: list[str]) -> str:
    """The executor's findings block: whole items only, within FINDINGS_HANDOFF_CHARS."""
    lines: list[str] = []
    used = 0
    for item in findings:
        line = f"- {item}"
        if used + len(line) + 1 > FINDINGS_HANDOFF_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return (
        "Looked up while planning (checked facts; prefer these to memory):\n"
        + "\n".join(lines)
        + "\n\n"
    )


# Three ways to get the conventions into `findings` (DESIGN-LOG §87). After a80e769
# took numpy/scipy out of the ResearchGate, plans with findings fell from 11-13 of 15
# (rf4-6, gate held 14/15) to about 1 in 5: the hold, arriving last, was what made
# Mercury write them, and the same request in the system prompt mostly goes unread.

# (A) Settings.plan_findings_last: the request, at the end of the planner's first
# message, the last thing it reads before its first (and usually only) reply.
FINDINGS_LAST = """

Before you call submit_plan: every choice the goal leaves implicit that changes the numbers
(sign and unit conventions, what one step or iteration covers, whether the initial state is
included, which random API, array shapes, inclusive or exclusive ranges) goes in `findings` as
"decision: ... because ...". The executor never sees your draft, only `findings`."""


def with_required_findings(tool: Tool) -> Tool:
    """(B) Settings.plan_findings_required: `findings` required in submit_plan, not optional."""
    parameters = copy.deepcopy(tool.parameters)
    required = list(parameters.get("required") or [])
    if "findings" in parameters.get("properties", {}) and "findings" not in required:
        required.append("findings")
    parameters["required"] = required
    return dataclasses.replace(tool, parameters=parameters)


def _decisions(call: dict) -> int:
    try:
        args = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except json.JSONDecodeError:
        return 0
    findings = args.get("findings") if isinstance(args, dict) else None
    if not isinstance(findings, list):
        return 0
    return sum(1 for f in findings if str(f).strip().lower().startswith("decision"))


class DecisionsGate:
    """(C) Settings.plan_findings_hold: holds the first submit_plan once when its
    findings carry no "decision: ..." item, asking for the conventions. One extra
    planner call when it fires; the ResearchGate's hold, which this replaces for
    goals whose libraries are all well known, was followed in ~80% of runs."""

    event = "decisions_gate_held"

    def __init__(self) -> None:
        self.held = False

    def observe(self, call: dict) -> None:
        return None

    def hold(self, call: dict) -> str | None:
        if (call.get("function") or {}).get("name") != "submit_plan" or self.held:
            return None
        if _decisions(call):
            return None
        self.held = True
        return (
            "[harness] Plan NOT submitted: `findings` has no convention decisions. The hidden "
            "checks compare exact numbers, so settle each choice the goal leaves implicit "
            "(signs, units, what one step covers, initial state included or not, random API, "
            "array shapes, range ends) and send submit_plan again with each as "
            '"decision: ... because ..." in `findings`.'
        )
