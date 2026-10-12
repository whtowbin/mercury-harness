"""Harness behaviour the SWE-bench django runs exposed (DESIGN-LOG §71): the type
gate on a project that does not type-check, a planner that runs out of
iterations, and a test failure no id could be read from."""

from __future__ import annotations

import json

import pytest
from test_cost_contract import CountingClient, call, reply

from agent import skills as skills_module
from agent import validate
from agent.approval import Approver
from agent.kitnudges import KitWatch, nudge_about_foreign_lint
from agent.phases.finalcheck import (
    TYPE_DEBT_UNGATED,
    FinalCheck,
    _check_types,
    configures_type_checker,
    type_gate_applies,
)
from agent.phases.plan import Step
from agent.projectconfig import foreign_linters
from agent.runner import PremiseGate, is_test_infrastructure, run_tool_loop
from agent.tools import ToolContext, planning_registry
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _events(ctx) -> list[dict]:
    with open(ctx.trace.path) as f:
        return [json.loads(line) for line in f]


def _diag(n: int) -> validate.Diagnostic:
    return validate.Diagnostic(f"m{n}.py", n, 1, "invalid-assignment", "x")


# -- the type gate ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("pyproject.toml", "[tool.ty]\n"),
        ("pyproject.toml", "[tool.mypy]\nstrict = true\n"),
        ("setup.cfg", "[mypy]\n"),
        ("mypy.ini", ""),
        ("pyrightconfig.json", "{}"),
    ],
)
def test_a_configured_type_checker_is_recognised(tmp_path, name, text):
    (tmp_path / name).write_text(text)
    assert configures_type_checker(tmp_path)


def test_the_gate_holds_where_type_debt_is_small_or_a_checker_is_configured(tmp_path):
    small = frozenset(f"k{i}" for i in range(5))
    large = frozenset(f"k{i}" for i in range(TYPE_DEBT_UNGATED + 1))
    assert type_gate_applies(tmp_path, None)
    assert type_gate_applies(tmp_path, small)  # every bench fixture: debt 0-10
    assert not type_gate_applies(tmp_path, large)  # django: ~15,000, no config
    (tmp_path / "pyproject.toml").write_text("[tool.ty]\n")
    assert type_gate_applies(tmp_path, large)


def test_new_type_errors_in_an_untyped_project_are_reported_not_gated(ctx, monkeypatch):
    """django-11848: all tests passed at every final check, and 1-4 "new" type
    errors in an untyped codebase drove all three fix rounds."""
    old = [_diag(i) for i in range(TYPE_DEBT_UNGATED + 50)]
    new = [_diag(10_000), _diag(10_001)]
    monkeypatch.setattr(validate, "ty_check", lambda *a, **k: old + new)
    from agent.phases.finalcheck import _diag_key

    result = FinalCheck()
    _check_types(ctx, ["m.py"], result, ctx.trace, frozenset(_diag_key(d) for d in old))
    assert result.new_type_errors == []
    assert len(result.type_errors) == len(old) + 2
    skipped = [e for e in _events(ctx) if e["kind"] == "type_gate_skipped"]
    assert skipped and skipped[0]["new"] == 2


def test_new_type_errors_still_gate_a_project_with_little_debt(ctx, monkeypatch):
    new = [_diag(1)]
    monkeypatch.setattr(validate, "ty_check", lambda *a, **k: new)
    result = FinalCheck()
    _check_types(ctx, ["m.py"], result, ctx.trace, frozenset())
    assert result.new_type_errors == new


# -- a planner out of iterations --------------------------------------------------


def test_a_planner_out_of_iterations_is_asked_to_submit_what_it_has(ctx):
    """django-12708's planner found the right functions and ran out at 12
    iterations; the run fell back to a one-step plan holding only the goal."""
    plan = call(
        "submit_plan",
        draft=["understood: fix the index deletion"],
        steps=[
            {"id": "s1", "title": "Fix _delete_composed_index", "intent": "x", "files": ["a.py"]}
        ],
    )
    client = CountingClient([reply(call("list_dir", path=".")) for _ in range(3)] + [reply(plan)])
    result, messages = run_tool_loop(
        client=client,
        model="m",
        phase="plan",
        system="s",
        user="u",
        registry=planning_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
        salvage=True,
    )
    assert client.calls == 4, "three iterations plus exactly one last word"
    assert result is not None and result["steps"][0]["title"] == "Fix _delete_composed_index"
    asked = next(m["content"] for m in messages if "last turn" in str(m.get("content")))
    assert "submit_plan" in asked and "complete plan" in asked
    assert "do not submit a plan that only investigates" in asked


# -- the project's own tooling (DESIGN-LOG §73) ------------------------------------

DJANGO_SETUP_CFG = "[flake8]\nexclude = build\n\n[isort]\nline_length = 79\n"


def test_a_project_that_lints_with_something_else_is_recognised(tmp_path):
    assert foreign_linters(tmp_path) == []  # configures nothing: every bench fixture
    (tmp_path / "setup.cfg").write_text(DJANGO_SETUP_CFG)
    assert foreign_linters(tmp_path) == ["flake8", "isort"]
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
    assert foreign_linters(tmp_path) == []  # Ruff is among its own tools


def test_the_ruff_skill_yields_to_a_projects_own_linters(tmp_path):
    loaded = skills_module.load_merged()
    assert "ruff" in [s.name for s in skills_module.for_project(loaded, tmp_path)]
    (tmp_path / "setup.cfg").write_text(DJANGO_SETUP_CFG)
    names = [s.name for s in skills_module.for_project(loaded, tmp_path)]
    assert "ruff" not in names and "diagnosing" in names


def _bash(command: str) -> tuple[list[dict], list[dict]]:
    tool_call = call("run_bash", command=command)
    return [tool_call], [{"role": "tool", "content": "exit=0"}]


def test_a_ruff_rewrite_in_a_flake8_project_is_flagged_once(ctx):
    (ctx.workspace.root / "setup.cfg").write_text(DJANGO_SETUP_CFG)
    watch = KitWatch()
    calls, messages = _bash("uvx --quiet ruff check --select DTZ,UP031 django/utils/http.py")
    assert nudge_about_foreign_lint(watch, calls, messages, ctx.workspace.root, ctx.trace, "x")
    assert "lints with flake8, isort, not Ruff" in messages[0]["content"]
    calls, messages = _bash("uvx --quiet ruff check --fix django/utils/http.py")
    assert not nudge_about_foreign_lint(watch, calls, messages, ctx.workspace.root, ctx.trace, "x")


def test_a_ruff_report_or_a_ruff_project_is_left_alone(ctx):
    root = ctx.workspace.root
    (root / "setup.cfg").write_text(DJANGO_SETUP_CFG)
    calls, messages = _bash("ruff check django/utils/http.py")  # reports, rewrites nothing
    assert not nudge_about_foreign_lint(KitWatch(), calls, messages, root, ctx.trace, "x")
    (root / "pyproject.toml").write_text("[tool.ruff]\n")
    calls, messages = _bash("ruff check --fix x.py")
    assert not nudge_about_foreign_lint(KitWatch(), calls, messages, root, ctx.trace, "x")


def test_test_infrastructure_is_recognised():
    command = "python tests/runtests.py --verbosity 2 --settings=test_sqlite {targets}"
    assert is_test_infrastructure("tests/runtests.py", command, "tests")
    assert is_test_infrastructure("tests/test_sqlite.py", command, "tests")
    assert is_test_infrastructure("conftest.py")
    assert not is_test_infrastructure("django/core/paginator.py", command, "tests")
    assert not is_test_infrastructure("tests/pagination/tests.py", command, "tests")


def test_an_edit_to_the_test_runner_is_held_once_with_a_concrete_alternative(ctx):
    """django-9296's goal-loop run edited tests/runtests.py in passing (§72)."""
    ctx.test_command = "python tests/runtests.py --settings=test_sqlite {targets}"
    ctx.test_label_root = "tests"
    (ctx.workspace.root / "tests").mkdir()
    (ctx.workspace.root / "tests" / "runtests.py").write_text("a\n")
    edit = call("edit_file", path="tests/runtests.py", old_string="a", new_string="b")
    gate = PremiseGate(ctx, goal="Paginator should implement __iter__")
    held = gate.hold(edit)
    assert held and "make the change in the code the goal is about" in held
    assert gate.event == "test_infrastructure_held"
    assert gate.hold(edit) is None  # once per file: re-sending goes through


def test_a_goal_that_names_the_runner_may_change_it(ctx):
    (ctx.workspace.root / "tests").mkdir()
    (ctx.workspace.root / "tests" / "runtests.py").write_text("a\n")
    edit = call("edit_file", path="tests/runtests.py", old_string="a", new_string="b")
    assert PremiseGate(ctx, goal="runtests.py should accept a --shuffle flag").hold(edit) is None


def test_a_step_is_not_failed_on_type_noise_in_an_untyped_project(ctx, monkeypatch):
    from agent.config import Settings
    from agent.phases.execute import _type_check_step
    from agent.phases.finalcheck import _diag_key

    old = [_diag(i) for i in range(TYPE_DEBT_UNGATED + 50)]
    monkeypatch.setattr(validate, "ty_check", lambda *a, **k: [*old, _diag(99_999)])
    ctx.type_baseline = frozenset(_diag_key(d) for d in old)
    ctx.touched.append("m.py")
    step = Step(id="s", title="t", intent="i", acceptance="a", files=["m.py"])
    assert _type_check_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g")) == ""
    ctx.type_baseline = frozenset()  # a project with no type debt keeps the gate
    assert "type errors" in _type_check_step(
        ctx, ctx.trace, step, Settings(workspace=".", goal="g")
    )


def test_a_conftest_this_run_creates_is_not_held(ctx):
    """53 of 64 bench edits to a conftest-like name created the file (§73)."""
    new = call("write_file", path="conftest.py", content="import sys\n")
    assert PremiseGate(ctx, goal="g").hold(new) is None


# -- DESIGN-LOG §74 ---------------------------------------------------------------


class _RecordingClient(CountingClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.tool_choices: list = []

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    ):
        self.tool_choices.append(tool_choice)
        return super().chat(
            model, messages, tools, tool_choice, temperature, max_tokens, reasoning_effort
        )


def test_the_last_word_requires_the_terminal_tool_rather_than_offering_it(ctx):
    """32 of 188 traced last words came back as prose; one left a django planner's
    run with a one-step fallback plan."""
    client = _RecordingClient([reply(call("list_dir", path=".")) for _ in range(2)])
    run_tool_loop(
        client=client,
        model="m",
        phase="plan",
        system="s",
        user="u",
        registry=planning_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=2,
        salvage=True,
    )
    assert client.tool_choices[-1] == {"type": "function", "function": {"name": "submit_plan"}}


def test_the_quick_answer_check_treats_a_bug_report_as_work():
    from agent.prompts import QUICK_ANSWER_SYSTEM

    text = " ".join(QUICK_ANSWER_SYSTEM.split())
    assert "A report that something is broken is such a request" in text


# -- a lint verify judged on what the run introduced (DESIGN-LOG §74) ----------------


def test_a_lint_verify_is_recognised_only_when_it_is_just_a_ruff_check():
    from agent.phases.execute import lint_verify_parts

    assert lint_verify_parts("uvx --quiet ruff check django/x.py") == ([], ["django/x.py"])
    assert lint_verify_parts("uv run ruff check --select UP031,F a.py b.py") == (
        ["--select", "UP031,F"],
        ["a.py", "b.py"],
    )
    assert lint_verify_parts("ruff check x.py && pytest") is None  # chained: runs as written
    assert lint_verify_parts("ruff check --fix x.py") is None  # rewrites: runs as written
    assert lint_verify_parts("ruff check .") is None
    assert lint_verify_parts("python tests/runtests.py x") is None


def _edit(ctx, rel: str, new: str) -> None:
    path = ctx.workspace.root / rel
    ctx.baseline.capture(rel, path)
    path.write_text(new)


def test_a_lint_verify_passes_over_findings_the_file_already_had(ctx):
    """django-11815's step verified with `ruff check serializer.py`, which arrived
    with 23 UP031 findings; the step restyled the file to pass and broke 14 tests."""
    from agent.phases.execute import run_verify

    old = 'def f(x):\n    return "%s" % x\n'
    (ctx.workspace.root / "m.py").write_text(old)
    _edit(ctx, "m.py", old + "\n\ndef g():\n    return 1\n")
    passed, output = run_verify(ctx, "uvx --quiet ruff check m.py")
    assert passed, output
    assert "no findings this run introduced" in output


def test_a_lint_verify_still_fails_on_a_finding_the_run_introduced(ctx):
    from agent.phases.execute import run_verify

    (ctx.workspace.root / "m.py").write_text("def f():\n    return 1\n")
    _edit(ctx, "m.py", "import os\n\n\ndef f():\n    return 1\n")
    passed, output = run_verify(ctx, "ruff check m.py")
    assert not passed and "F401" in output
