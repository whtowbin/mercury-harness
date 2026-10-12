"""A step's verify in the project's own test form, and a verify that selects no test (§110).

The commands and outputs below are the django runs' own (§94 traces). The run_verify tests use a
real workspace with a small runner that loads unittest labels the way django's runtests.py does,
run as a real subprocess.
"""

from __future__ import annotations

import shlex
import sys

import pytest

from agent import validate
from agent.approval import Approver
from agent.phases.execute import VERIFY_SELECTS_NOTHING, run_verify, unrunnable_note
from agent.phases.finalcheck import _verify_outcome
from agent.tools import ToolContext
from agent.trace import Trace
from agent.verifycmd import normalize, selects_nothing
from agent.workspace import Workspace

PY = shlex.quote(sys.executable)  # the repo path has a space
DJANGO = "python tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 {targets}"
HEAD = "python tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"


@pytest.mark.parametrize(
    ("verify", "expected"),
    [
        # 9296 r3: the label root left on the label.
        (
            "python tests/runtests.py tests.pagination.tests.PaginationTests.test_paginator_iter",
            f"{HEAD} pagination.tests.PaginationTests.test_paginator_iter",
        ),
        # 12708 r1: a pytest node id handed to runtests.py.
        (
            f"{HEAD} migrations.test_operations::SchemaMigrationTests::test_delete",
            f"{HEAD} migrations.test_operations.SchemaMigrationTests.test_delete",
        ),
        # 11964 r1: a flag runtests.py rejects.
        (
            "python tests/runtests.py --verbosity 0 --settings=test_sqlite --parallel 1 -xv "
            "model_fields.tests",
            f"{HEAD} model_fields.tests",
        ),
        # 12708 r2 and r3: pytest on django test files.
        (
            "uv run pytest -q tests/backends/test_migration_index_fix.py",
            f"{HEAD} backends.test_migration_index_fix",
        ),
        (
            "uv run pytest -q tests/schema/tests.py::SchemaTests::test_index",
            f"{HEAD} schema.tests.SchemaTests.test_index",
        ),
        ("python -m pytest tests/pagination/tests.py", f"{HEAD} pagination.tests"),
        # A name pattern is kept: both runners take -k.
        (
            "uv run tests/runtests.py model_fields.tests -k test_to_python",
            f"{HEAD} -k test_to_python model_fields.tests",
        ),
    ],
)
def test_a_verify_aimed_at_the_tests_is_rebuilt_from_the_project_command(verify, expected):
    assert normalize(verify, DJANGO, "tests") == expected


@pytest.mark.parametrize(
    "verify",
    [
        f"{HEAD} pagination.tests",  # already right
        "uvx --quiet ruff check django/db/models/fields/__init__.py",
        "python -c 'import django'",
        "cd tests && python runtests.py pagination",  # compound: runs as written
    ],
)
def test_other_verifies_run_as_written(verify):
    assert normalize(verify, DJANGO, "tests") == verify


def test_nothing_is_rewritten_without_a_project_command_or_under_pytest():
    assert normalize("pytest tests/x.py", "", "") == "pytest tests/x.py"
    assert normalize("uv run pytest x.py", "uv run pytest -q {targets}", "") == "uv run pytest x.py"


NO_MODULE = (
    "ImportError: Failed to import test module: tests\n"
    "ModuleNotFoundError: No module named 'tests'\n\nRan 1 test in 0.000s\n\nFAILED (errors=1)"
)


def test_a_verify_that_selects_no_test_is_recognised():
    assert "does not import" in selects_nothing(f"{HEAD} tests.pagination.tests", NO_MODULE)
    node = "ModuleNotFoundError: No module named 'migrations.test_operations::A::test_b'"
    assert selects_nothing(f"{HEAD} migrations.test_operations::A::test_b", node)
    missing = "AttributeError: module 'pagination.tests' has no attribute 'test_iter'"
    assert "does not exist" in selects_nothing(f"{HEAD} pagination.tests.test_iter", missing)
    assert selects_nothing(HEAD, "runtests.py: error: unrecognized arguments: -xv")
    loader = (
        "  File \"/x/unittest/suite.py\", line 57\nTypeError: 'NoneType' object is not iterable"
    )
    assert "file path" in selects_nothing(f"{HEAD} django/db/tests.py", loader)
    assert selects_nothing("pytest x.py", "ERROR: found no collectors for x.py\n1 error in 0.06s")


def test_a_failing_test_or_a_broken_import_in_the_code_is_a_real_failure():
    failed = "Ran 3 tests in 0.004s\n\nFAILED (failures=3)"
    assert selects_nothing(f"{HEAD} auth_tests.test_forms", failed) == ""
    # The test module exists but the code it imports does not: the run's doing.
    broken = "ModuleNotFoundError: No module named 'pagination.helpers'"
    assert selects_nothing(f"{HEAD} admin_views.tests", broken) == ""
    syntax = (
        "ImportError: Failed to import test module: backends.t\nIndentationError: unexpected indent"
    )
    assert selects_nothing(f"{HEAD} backends.t", syntax) == ""


# -- run_verify against a real runner -------------------------------------------

RUNNER = """import sys, unittest
sys.path.insert(0, 'tests')
labels = [a for a in sys.argv[1:] if not a.startswith('-')]
suite = unittest.defaultTestLoader.loadTestsFromNames(labels)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "tests" / "pagination").mkdir(parents=True)
    (tmp_path / "tests" / "runtests.py").write_text(RUNNER)
    (tmp_path / "tests" / "pagination" / "__init__.py").write_text("")
    (tmp_path / "tests" / "pagination" / "tests.py").write_text(
        "import unittest\n\nclass PaginationTests(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertEqual(1, 1)\n"
        "    def test_bad(self):\n        self.assertEqual(1, 2)\n"
    )
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        test_command=f"{PY} tests/runtests.py {{targets}}",
        test_label_root="tests",
    )


def test_a_pytest_style_verify_runs_through_the_project_runner_and_passes(ctx):
    passed, output = run_verify(
        ctx, "uv run pytest tests/pagination/tests.py::PaginationTests::test_ok"
    )
    assert passed, output
    assert output.startswith("(ran as: ") and "pagination.tests.PaginationTests.test_ok" in output


def test_a_label_with_the_root_left_on_is_fixed_rather_than_failed(ctx):
    passed, output = run_verify(
        ctx, f"{PY} tests/runtests.py tests.pagination.tests.PaginationTests.test_ok"
    )
    assert passed, output


def test_a_verify_naming_a_test_that_does_not_exist_is_unrunnable_not_failed(ctx):
    command = f"{PY} tests/runtests.py pagination.tests.PaginationTests.test_gone"
    passed, output = run_verify(ctx, command)
    assert not passed and output.startswith(VERIFY_SELECTS_NOTHING), output
    assert unrunnable_note(output)
    passed, _output, note = _verify_outcome(ctx, command)
    assert not passed and "selects no test" in note  # the final check: no fix round


def test_a_genuinely_failing_test_still_fails(ctx):
    command = f"{PY} tests/runtests.py pagination.tests.PaginationTests.test_bad"
    passed, output = run_verify(ctx, command)
    assert not passed and not unrunnable_note(output), output
    assert _verify_outcome(ctx, command)[2] == ""


# -- lint verifies -------------------------------------------------------------------


def test_only_serious_ruff_rules_count_as_breakage():
    serious = [
        validate.Diagnostic("m.py", 1, 1, code, "x") for code in ("F401", "E902", "B006", "S101")
    ]
    assert not validate.is_serious(validate.Diagnostic("m.py", 1, 1, "SIM108", "x"))
    style = [
        validate.Diagnostic("m.py", 1, 1, code, "x") for code in ("UP031", "I001", "RUF059", "E501")
    ]
    assert all(validate.is_serious(d) for d in serious)
    assert not any(validate.is_serious(d) for d in style)


def test_a_lint_verify_passes_with_style_findings_reported(tmp_path):
    (tmp_path / "m.py").write_text("def f(x):\n    return 1\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    ctx.baseline.capture("m.py", tmp_path / "m.py")
    (tmp_path / "m.py").write_text('def f(x):\n    return "%s" % x\n')
    passed, output = run_verify(ctx, "uvx --quiet ruff check --select UP031 m.py")
    assert passed, output
    assert "not blocking" in output and "UP031" in output


# -- at plan time ----------------------------------------------------------------------


def test_a_plan_s_verify_is_shown_in_the_project_s_own_form(ctx):
    """The executor is told the command that will run, not the one the planner wrote."""
    from agent.config import Settings
    from agent.phases.plan import finalize_plan

    settings = Settings(
        workspace=str(ctx.workspace.root),
        goal="g",
        test_command=ctx.test_command,
        test_label_root="tests",
    )
    step = {
        "id": "s",
        "title": "t",
        "intent": "i",
        "acceptance": "a",
        "files": ["tests/pagination/tests.py"],
        "verify": "uv run pytest -q tests/pagination/tests.py::PaginationTests::test_ok",
    }
    plan = finalize_plan({"draft": [], "steps": [step]}, settings, ctx, ctx.trace)
    expected = f"{PY} tests/runtests.py pagination.tests.PaginationTests.test_ok"
    assert plan.steps[0].verify == expected


def test_the_final_check_starts_with_a_fresh_shell_budget(ctx):
    """The last step's spent budget made final verifies "fail" before they ran."""
    from agent.config import Settings
    from agent.phases.finalcheck import run_final_check
    from agent.phases.plan import Plan, Step

    ctx.guard.record_subprocess(10_000)
    assert ctx.guard.check_subprocess_budget()
    step = Step(
        id="s",
        title="t",
        intent="i",
        acceptance="a",
        files=[],
        verify=f"{PY} tests/runtests.py pagination.tests.PaginationTests.test_ok",
    )
    settings = Settings(workspace=str(ctx.workspace.root), goal="g")
    check = run_final_check(ctx, settings, Plan(goal="g", draft=[], steps=[step]), ctx.trace)
    assert step.verify_passed is True, check.verify_failures
