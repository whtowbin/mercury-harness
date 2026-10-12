"""Settings.repro_form: the reproduce check asked for explicitly, or as a symptom the
harness turns into the assertion (agent/symptom.py, DESIGN-LOG §65)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agent import symptom
from agent.prompts import PLAN_REPRO
from agent.tools.control import SUBMIT_PLAN, submit_plan, with_reproduce

SLUG = "def slugify(title):\n    return title.lower().replace(' ', '-')\n"


def _run(code: str, tmp_path) -> subprocess.CompletedProcess:
    """The snippet run for real, as ReproGate's scratch run does, beside a module."""
    (tmp_path / "slug.py").write_text(SLUG)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )


# -- the snippet built from a symptom -------------------------------------------


def test_a_symptom_that_is_real_fails_as_an_assertion(tmp_path):
    code = symptom.snippet_from(
        {
            "symptom_call": "from slug import slugify; slugify('Café Crème')",
            "symptom_expected": "'cafe-creme'",
        }
    )
    ran = _run(code, tmp_path)
    assert ran.returncode != 0
    assert "AssertionError" in ran.stderr and "got 'café-crème'" in ran.stderr


def test_a_symptom_that_is_not_there_passes(tmp_path):
    code = symptom.snippet_from(
        {"symptom_call": "from slug import slugify; slugify('A B')", "symptom_expected": "'a-b'"}
    )
    assert _run(code, tmp_path).returncode == 0


@pytest.mark.parametrize(
    ("expected", "passes"),
    [
        ("a-b", True),  # unquoted, as the smoke run's planner wrote it
        ("the slug a-b", False),  # prose that is not a valid expression
        ("'a' + '-b'", True),  # a real expression still evaluates
    ],
)
def test_an_expected_value_that_only_reads_as_text_is_compared_as_text(tmp_path, expected, passes):
    code = symptom.snippet_from(
        {"symptom_call": "from slug import slugify; slugify('A B')", "symptom_expected": expected}
    )
    ran = _run(code, tmp_path)
    assert "NameError" not in ran.stderr and "SyntaxError" not in ran.stderr
    assert (ran.returncode == 0) is passes


def test_a_name_the_call_imports_is_not_mistaken_for_text(tmp_path):
    code = symptom.snippet_from(
        {
            "symptom_call": "from decimal import Decimal; Decimal('1.50') + Decimal('1')",
            "symptom_expected": "Decimal('2.50')",
        }
    )
    assert _run(code, tmp_path).returncode == 0


def test_an_expected_exception_is_asserted_both_ways(tmp_path):
    raises = {"symptom_call": "int('x')", "symptom_expected": "raises ValueError"}
    assert _run(symptom.snippet_from(raises), tmp_path).returncode == 0
    no_raise = {"symptom_call": "int('3')", "symptom_expected": "raises ValueError"}
    ran = _run(symptom.snippet_from(no_raise), tmp_path)
    assert ran.returncode != 0 and "expected ValueError, got 3" in ran.stderr


@pytest.mark.parametrize(
    "args",
    [
        {"symptom_call": "none: the goal asks for a new report", "symptom_expected": ""},
        {"symptom_call": "None - new work", "symptom_expected": ""},
        {"symptom_call": "from slug import slugify; slugify('x')", "symptom_expected": ""},
        {"symptom_call": "", "symptom_expected": "'x'"},
        {"reproduce": "none: nothing is reported broken"},
        {"reproduce": "NONE"},
        {"reproduce": ""},
    ],
)
def test_declining_or_leaving_it_half_empty_is_no_snippet(args):
    assert symptom.snippet_from(args) == ""


def test_a_snippet_that_starts_with_a_name_like_none_is_still_a_snippet():
    code = "none_left = []\nassert none_left"
    assert symptom.snippet_from({"reproduce": code}) == code


def test_an_unparsable_call_is_run_as_written_so_it_reads_as_broken(tmp_path):
    code = symptom.snippet_from({"symptom_call": "slugify(", "symptom_expected": "'x'"})
    assert code == "slugify("
    assert "SyntaxError" in _run(code, tmp_path).stderr


# -- the schema and the prompt ----------------------------------------------------


def _schema(form: str) -> dict:
    return symptom.with_form(with_reproduce(SUBMIT_PLAN), form).parameters


def test_the_default_form_leaves_the_tool_untouched():
    tool = with_reproduce(SUBMIT_PLAN)
    assert symptom.with_form(tool, "snippet") is tool


@pytest.mark.parametrize(
    ("form", "fields"),
    [("explicit", ["reproduce"]), ("symptom", ["symptom_call", "symptom_expected"])],
)
def test_the_check_is_asked_for_after_the_draft_and_before_the_steps(form, fields):
    schema = _schema(form)
    order = list(schema["properties"])
    assert order[: 1 + len(fields)] == ["draft", *fields]
    assert order.index("steps") > order.index(fields[-1])
    assert set(fields) <= set(schema["required"])
    if form == "symptom":
        assert "reproduce" not in schema["properties"]
        assert "reproduce" not in schema["required"]


def test_the_explicit_form_drops_the_empty_string_exit():
    description = _schema("explicit")["properties"]["reproduce"]["description"]
    assert "Empty string" not in description and "`none: `" in description


def test_the_explicit_prompt_replaces_the_exit_and_keeps_the_rest():
    block = symptom.planner_block(PLAN_REPRO, "explicit")
    assert block != PLAN_REPRO
    assert "empty\nstring" not in block and "`none: ` and why" in block
    assert block.startswith(PLAN_REPRO.split("Leave it as")[0])


def test_the_default_prompt_is_unchanged():
    assert symptom.planner_block(PLAN_REPRO, "snippet") is PLAN_REPRO


def test_an_unknown_form_is_refused():
    with pytest.raises(ValueError):
        symptom.with_form(with_reproduce(SUBMIT_PLAN), "freeform")


# -- what the plan and the gate see ------------------------------------------------


class _Ctx:
    result: dict | None = None
    done = False


def test_submit_plan_turns_the_symptom_into_the_plans_reproduce_snippet():
    ctx = _Ctx()
    submit_plan(
        ctx,  # type: ignore[arg-type]
        draft=["understood: x"],
        steps=[],
        symptom_call="from slug import slugify; slugify('A')",
        symptom_expected="'a'",
    )
    assert ctx.result is not None
    assert "assert _actual == _expected" in str(ctx.result["reproduce"])


def test_submit_plan_records_no_snippet_for_a_stated_none():
    ctx = _Ctx()
    submit_plan(ctx, draft=[], steps=[], reproduce="none: new work")  # type: ignore[arg-type]
    assert ctx.result is not None and "reproduce" not in ctx.result


def test_the_repro_gate_runs_the_snippet_built_from_a_symptom(tmp_path):
    from agent.runner import ReproGate
    from agent.trace import Trace

    ran: list[str] = []

    def run(code: str):
        ran.append(code)
        return 1, "AssertionError: got 'x', expected 'y'"

    gate = ReproGate(run, Trace(str(tmp_path / "t"), echo=False))
    args = {"draft": [], "steps": [], "symptom_call": "f()", "symptom_expected": "'y'"}
    call = {"id": "c1", "function": {"name": "submit_plan", "arguments": json.dumps(args)}}
    assert gate.hold(call) is None
    assert ran and "_actual = f()" in ran[0]
