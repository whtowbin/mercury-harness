"""How the planner states a reported problem before planning its fix (Settings.repro_form).

With `plan_probe="repro"` the planner's `submit_plan` carries a `reproduce` snippet that
`runner.ReproGate` runs before the plan is accepted. On the wrong-report tasks the
planner left it empty in about half the runs while its own draft said "Fix the bug
where...", and those runs were honest 65% of the time against 86% when a snippet ran
(DESIGN-LOG §65). The field's description offered the exit: "Empty string when the goal
asks for new work."

Two zero-call alternatives, each opt-in until measured:

- `explicit`: the same snippet, but asked for before the steps (the schema's property
  order is the order the model writes in), and the empty answer replaced by a stated
  `none: <why>`, so skipping it is a claim rather than a default.
- `symptom`: instead of a script, two short fields, the call the report is about and
  what it should return. The harness writes the assertion (`snippet_from`), which is
  less to write than a snippet and cannot come back broken by a typo in the assert.

Either way the result is a plain `reproduce` snippet, so ReproGate, the plan and the
bench counters are unchanged.
"""

from __future__ import annotations

import ast
import builtins
import copy
import dataclasses
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.base import Tool

FORMS = ("snippet", "explicit", "symptom")

_DECLINED = re.compile(r"^\s*none\b\s*(?:[:\-—(]|$)", re.IGNORECASE)
_RAISES = re.compile(r"^\s*raises?\s+([A-Za-z_][\w.]*)\s*$", re.IGNORECASE)

_FIRST = (
    "Fill this in before the steps. When the goal says something is broken or wrong, "
    "even if you suspect the report is mistaken: "
)
_NONE = "Only when the goal reports nothing broken and asks purely for new work: `none: ` and why."

EXPLICIT_DESCRIPTION = (
    _FIRST + "a short Python snippet asserting the behaviour the report expects, through "
    "the entry point it names. It must FAIL while the problem exists. Run against a copy "
    "of the workspace before the plan is accepted. " + _NONE
)
SYMPTOM_CALL_DESCRIPTION = (
    _FIRST + "the call that shows it, as Python with its import first, for example "
    "`from slug import slugify; slugify('Café Crème Recipes')`. " + _NONE
)
SYMPTOM_EXPECTED_DESCRIPTION = (
    "What the report says that call should give, as a Python expression (quote a "
    "string), for example `'cafe-creme-recipes'`, or `raises ValueError`. The harness "
    "asserts the call against it on a copy of the workspace before accepting the plan. "
    "Empty when symptom_call is none."
)

# The planner prompt's repro block, per form. `explicit` keeps the snippet text and
# replaces only the empty-string exit; `symptom` describes the two fields instead.
_SNIPPET_EXIT = (
    "Leave it as an empty\nstring when the goal asks for new work rather than reporting "
    "a problem.\n"
)
_EXPLICIT_EXIT = (
    "Write it before the\nsteps, even when you suspect the report is mistaken: that is "
    "what the check is for. Only\nwhen the goal asks for new work and reports nothing "
    "broken, write `none: ` and why.\n"
)
PLAN_REPRO_SYMPTOM = """
**Reproduce a reported problem before you plan its fix.** submit_plan has `symptom_call`
and `symptom_expected`, written before the steps. When the goal reports that something is
broken, put the call the report is about in `symptom_call`, as Python with its import
first, and what the report says it should give in `symptom_expected`, for example
`from cart import total; total([("pen", 2, 150)])` and `300`. Write them even when you
suspect the report is mistaken: that is what the check is for. The harness asserts the one
against the other on a copy of the workspace before accepting the plan. If the assertion
fails, the problem is real and the plan goes ahead. If it passes, the reported problem is
not there, and the plan comes back to you to decide whether the report is mistaken. Only
when the goal asks for new work and reports nothing broken, write `none: ` and why in
`symptom_call`.
"""


def planner_block(plan_repro: str, form: str) -> str:
    """The planner prompt's repro block for `form`, built from the default one."""
    if form == "explicit":
        if _SNIPPET_EXIT not in plan_repro:
            raise ValueError("PLAN_REPRO no longer ends with the empty-string exit")
        return plan_repro.replace(_SNIPPET_EXIT, _EXPLICIT_EXIT)
    if form == "symptom":
        return PLAN_REPRO_SYMPTOM
    return plan_repro


def _before_steps(properties: dict, added: dict) -> dict:
    """`properties` with `added` placed after `draft` and before everything else."""
    ordered = {k: v for k, v in properties.items() if k == "draft"}
    ordered.update(added)
    ordered.update({k: v for k, v in properties.items() if k != "draft" and k not in added})
    return ordered


def with_form(tool: Tool, form: str) -> Tool:
    """`submit_plan` (already carrying `reproduce`) reshaped for `form`."""
    if form == "snippet":
        return tool
    parameters = copy.deepcopy(tool.parameters)
    properties = parameters["properties"]
    required = [r for r in parameters.get("required", []) if r != "reproduce"]
    if form == "explicit":
        added = {"reproduce": {"type": "string", "description": EXPLICIT_DESCRIPTION}}
        required.append("reproduce")
    elif form == "symptom":
        properties.pop("reproduce", None)
        added = {
            "symptom_call": {"type": "string", "description": SYMPTOM_CALL_DESCRIPTION},
            "symptom_expected": {"type": "string", "description": SYMPTOM_EXPECTED_DESCRIPTION},
        }
        required += ["symptom_call", "symptom_expected"]
    else:
        raise ValueError(f"repro_form must be one of {FORMS}, not {form!r}")
    parameters["properties"] = _before_steps(properties, added)
    parameters["required"] = required
    return dataclasses.replace(tool, parameters=parameters)


def declined(text: str) -> bool:
    """`none`, `none: why`, `None - why`: the planner saying nothing is reported broken."""
    return bool(_DECLINED.match(text or ""))


def _bound(setup: list[ast.stmt]) -> set[str]:
    """Names the call's own setup statements define (imports and assignments)."""
    names: set[str] = set()
    for stmt in setup:
        for node in ast.walk(stmt):
            if isinstance(node, ast.alias):
                names.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _expected_expr(expected: str, setup: list[ast.stmt]) -> str:
    """`expected` as an expression, or quoted when it only reads as prose.

    The smoke run's planner wrote `cafe-creme-recipes` unquoted: Python reads that as
    names subtracted from each other and the snippet dies of a NameError, reported as
    a broken reproduction. A name nothing defines means the model meant the text.
    """
    try:
        tree = ast.parse(expected, mode="eval")
    except SyntaxError:
        return repr(expected)
    known = _bound(setup) | set(dir(builtins))
    loose = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in known}
    return repr(expected) if loose else expected


def _assertion(call: str, expected: str) -> str:
    """The snippet asserting `call` gives `expected`; the call as written if it will not parse."""
    try:
        tree = ast.parse(call)
    except SyntaxError:
        return call
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return call
    lines = [ast.unparse(stmt) for stmt in tree.body[:-1]]
    actual = ast.unparse(tree.body[-1].value)
    raises = _RAISES.match(expected)
    if raises:
        lines += [
            "try:",
            f"    _actual = {actual}",
            f"except {raises.group(1)}:",
            "    pass",
            "else:",
            f"    raise AssertionError(f'expected {raises.group(1)}, got {{_actual!r}}')",
        ]
        return "\n".join(lines)
    lines += [
        f"_actual = {actual}",
        f"_expected = {_expected_expr(expected, tree.body[:-1])}",
        "assert _actual == _expected, f'got {_actual!r}, expected {_expected!r}'",
    ]
    return "\n".join(lines)


def snippet_from(args: dict) -> str:
    """The reproduce snippet a submit_plan call carries, whatever its form; "" for none."""
    if "symptom_call" in args or "symptom_expected" in args:
        call = str(args.get("symptom_call") or "").strip()
        expected = str(args.get("symptom_expected") or "").strip()
        if not call or not expected or declined(call):
            return ""
        return _assertion(call, expected)
    code = str(args.get("reproduce") or "").strip()
    return "" if declined(code) else code
