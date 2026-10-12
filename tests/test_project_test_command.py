"""A project pytest cannot run: Settings.test_command, and a baseline that says
pytest could not load the suite instead of reporting hundreds of failures to fix
(DESIGN-LOG §69-§70, django on SWE-bench)."""

from __future__ import annotations

import json
import shlex
import sys

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.loop import _failing_tests_context
from agent.phases.finalcheck import (
    BaselineTests,
    _mark_unrunnable,
    collect_baseline,
    fixes_pre_existing,
)
from agent.tools import ToolContext
from agent.tools.checks import as_test_labels, run_tests
from agent.trace import Trace
from agent.workspace import Workspace

# The shape pytest reports for django's suite: whole files failing to collect.
DJANGO_UNDER_PYTEST = [f"tests/app{i}/tests.py" for i in range(8)]


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


# -- labels -----------------------------------------------------------------------


def test_test_files_under_the_label_root_become_dotted_labels():
    labels = as_test_labels(
        ["tests/pagination/tests.py", "./tests/utils_tests/test_http.py", "tests/app/__init__.py"],
        "tests",
    )
    assert labels == ["pagination.tests", "utils_tests.test_http", "app"]


def test_a_package_directory_under_the_root_becomes_its_label():
    assert as_test_labels(["tests/pagination", "tests/pagination/"], "tests") == [
        "pagination",
        "pagination",
    ]


def test_labels_and_files_outside_the_root_pass_through():
    paths = ["pagination.tests", "django/core/paginator.py"]
    assert as_test_labels(paths, "tests/") == paths
    assert as_test_labels(["tests/x/tests.py"], "") == ["tests/x/tests.py"]


# -- a baseline pytest could not load -----------------------------------------------


def test_a_suite_pytest_mostly_cannot_collect_is_unrunnable_and_names_the_runner(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "runtests.py").write_text("")
    baseline = _mark_unrunnable(BaselineTests(failures=list(DJANGO_UNDER_PYTEST)), tmp_path)
    assert baseline.unrunnable
    assert "tests/runtests.py" in baseline.runner_hint


def test_a_few_import_errors_in_a_working_suite_stay_ordinary_failures(tmp_path):
    failures = [f"tests/test_a.py::test_{i}" for i in range(20)] + DJANGO_UNDER_PYTEST[:5]
    assert not _mark_unrunnable(BaselineTests(failures=failures), tmp_path).unrunnable
    assert not _mark_unrunnable(
        BaselineTests(failures=DJANGO_UNDER_PYTEST[:4]), tmp_path
    ).unrunnable


def test_an_unrunnable_baseline_is_not_handed_to_the_planner_as_work():
    baseline = BaselineTests(
        failures=list(DJANGO_UNDER_PYTEST), unrunnable=True, runner_hint="`tests/runtests.py`"
    )
    text = _failing_tests_context(baseline, Settings(workspace=".", goal="g"))
    assert "could not load 8" in text and "tests/runtests.py" in text
    assert "part of the work" not in text and "fix them" not in text


def test_the_fix_rounds_leave_an_unrunnable_baseline_alone():
    settings = Settings(workspace=".", goal="g", fix_existing_failures=True)
    unrunnable = BaselineTests(failures=list(DJANGO_UNDER_PYTEST), unrunnable=True)
    assert not fixes_pre_existing(settings, unrunnable)
    assert fixes_pre_existing(settings, BaselineTests(failures=["tests/t.py::test_a"]))
    assert fixes_pre_existing(settings, None)


# -- the project's own command ------------------------------------------------------


def test_with_a_test_command_there_is_no_whole_suite_baseline(ctx):
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    ctx.test_command = f"{shlex.quote(sys.executable)} -m unittest -v {{targets}}"
    assert collect_baseline(ctx, ctx.trace) is None
    assert any(e["kind"] == "test_baseline_skipped" for e in _events(ctx))


def test_the_planner_is_told_how_tests_run():
    settings = Settings(workspace=".", goal="g", test_command="python tests/runtests.py {targets}")
    text = _failing_tests_context(None, settings)
    assert "python tests/runtests.py" in text and "path" in text


def test_run_tests_with_a_command_and_no_target_asks_for_one(ctx):
    ctx.test_command = f"{shlex.quote(sys.executable)} -m unittest -v {{targets}}"
    assert "name what to run" in run_tests(ctx)


def test_run_tests_runs_the_project_command_and_reads_its_failures(ctx):
    """A real unittest process: the result reads like a pytest run, so the
    FAIL/PASS contract every gate relies on holds."""
    root = ctx.workspace.root
    (root / "test_sample.py").write_text(
        "import unittest\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertEqual(1, 1)\n\n"
        "    def test_bad(self):\n        self.assertEqual(add(1, 1), 3)\n\n"
        "def add(a, b):\n    return a + b\n"
    )
    ctx.test_command = f"{shlex.quote(sys.executable)} -m unittest -v {{targets}}"
    out = run_tests(ctx, path="test_sample")
    assert out.startswith("FAIL")
    assert "1 failed" in out and "1 passed" in out
    event = next(e for e in _events(ctx) if e["kind"] == "run_tests")
    assert event["failing"] == ["test_sample.T.test_bad"]

    (root / "test_sample.py").write_text(
        "import unittest\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertEqual(1, 1)\n"
    )
    assert run_tests(ctx, path="test_sample").startswith("PASS")


def test_the_flags_reach_settings(monkeypatch):
    from typer.testing import CliRunner

    from agent import cli

    seen = {}

    def fake_run_agent(settings, plan_path, stop_after, **_kwargs):
        seen["s"] = settings
        return {"totals": {"by_model": {}}, "trace_path": "t"}

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    result = CliRunner().invoke(
        cli.app,
        ["run", "--test-command", "python t.py {targets}", "--test-label-root", "tests", "goal"],
    )
    assert result.exit_code == 0, result.output
    assert seen["s"].test_command == "python t.py {targets}"
    assert seen["s"].test_label_root == "tests"


def test_node_ids_and_the_whole_suite_are_translated_for_the_runner(ctx):
    labels = as_test_labels(
        ["tests/utils_tests/test_http.py::HttpDateProcessingTests::test_rfc850"], "tests"
    )
    assert labels == ["utils_tests.test_http.HttpDateProcessingTests.test_rfc850"]
    ctx.test_command = f"{shlex.quote(sys.executable)} -m unittest {{targets}}"
    assert "name what to run" in run_tests(ctx, path=".")


def test_a_runner_without_verbose_output_still_reads_as_a_real_result(ctx):
    """django's runner at its default verbosity prints dots, not per-test lines;
    the first live runs read those as "no tests ran" (DESIGN-LOG §70)."""
    root = ctx.workspace.root
    (root / "test_quiet.py").write_text(
        "import unittest\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n\n"
        "    def test_bad(self):\n        self.assertEqual(1, 2)\n"
    )
    ctx.test_command = f"{shlex.quote(sys.executable)} -m unittest {{targets}}"  # no -v
    out = run_tests(ctx, path="test_quiet")
    assert out.startswith("FAIL") and "1 failed" in out and "1 passed" in out
    (root / "test_quiet.py").write_text(
        "import unittest\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n"
    )
    assert run_tests(ctx, path="test_quiet").startswith("PASS — 1 passed")


def test_check_code_fix_leaves_the_style_of_a_project_without_a_formatter(ctx):
    """django configures neither ruff nor black; `check_code(fix=true)` restyled
    18 lines of paginator.py around a 4-line fix (DESIGN-LOG §70)."""
    from agent.tools.checks import check_code, uses_a_formatter

    root = ctx.workspace.root
    source = "def f(a,\n      b):\n    return {'a': a, 'b': b}\n"
    (root / "m.py").write_text(source)
    assert not uses_a_formatter(root)
    out = check_code(ctx, paths=["m.py"], fix=True)
    assert "not ruff format" in out
    assert (root / "m.py").read_text() == source

    (root / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    assert uses_a_formatter(root)
    check_code(ctx, paths=["m.py"], fix=True)
    assert (root / "m.py").read_text() != source


def test_a_failure_that_names_no_test_keeps_its_output_in_the_trace(ctx):
    """django-12708 recorded "1 error" with no test id and no output; rerun, the
    same test parsed fine, so the cause was lost (DESIGN-LOG §71)."""
    script = ctx.workspace.root / "runner.py"
    script.write_text(
        "import sys\nprint('Ran 1 test in 0.1s')\nprint()\n"
        "print('FAILED (errors=1)')\nsys.exit(1)\n"
    )
    ctx.test_command = f"{shlex.quote(sys.executable)} runner.py {{targets}}"
    out = run_tests(ctx, path="anything")
    assert out.startswith("FAIL")
    event = next(e for e in _events(ctx) if e["kind"] == "run_tests")
    assert event["failing"] == []
    assert "(errors=1)" in event["unparsed_tail"]


# -- DESIGN-LOG §75 ---------------------------------------------------------------


def test_django_style_tests_modules_count_as_tests():
    from agent.tools.checks import _is_project_test_file

    assert _is_project_test_file("tests/delete/tests.py", "")
    assert _is_project_test_file("tests/delete/test_models.py", "tests")
    # A helper module inside a test app is not a test module (§75.1: `admin_views.admin`
    # as a runner label failed django's system checks before any test ran).
    assert not _is_project_test_file("tests/admin_views/admin.py", "tests")
    assert not _is_project_test_file("tests/delete/models.py", "tests")
    assert not _is_project_test_file("django/db/models/deletion.py", "tests")


def _tree(root, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_test_modules_importing_a_changed_source_are_found(tmp_path):
    """django-11885 changed deletion.py and broke tests/delete/tests.py, which imports
    Collector from it."""
    from agent.tools.checks import tests_importing

    _tree(
        tmp_path,
        {
            "pkg/models/deletion.py": "class Collector: ...\n",
            "tests/delete/tests.py": "from pkg.models.deletion import Collector\n",
            "tests/other/tests.py": "from pkg.models import deletion\n",
            "tests/plain/tests.py": "import pkg.models.deletion\n",
            "tests/unrelated/tests.py": "from pkg.models import base\n",
            "tests/delete/admin.py": "from pkg.models.deletion import Collector\n",
        },
    )
    found = tests_importing(tmp_path, ["pkg/models/deletion.py"], "tests")
    assert found == ["tests/delete/tests.py", "tests/other/tests.py", "tests/plain/tests.py"]


def test_the_final_check_runs_a_touched_tests_module_and_its_importers(ctx):
    """The final check had recorded `exit 127` (nothing ran) on every django task,
    because it only knew `test_*.py` as a test file name."""
    from agent.phases.finalcheck import _run_project_suite

    root = ctx.workspace.root
    _tree(
        root,
        {
            "pkg/__init__.py": "",
            "pkg/calc.py": "def add(a, b):\n    return a - b\n",
            "tests.py": "import unittest\n\nclass T(unittest.TestCase):\n"
            "    def test_one(self):\n        self.assertEqual(1, 1)\n",
            "calc_tests/__init__.py": "",
            "calc_tests/tests.py": "import unittest\nfrom pkg.calc import add\n\n"
            "class C(unittest.TestCase):\n    def test_add(self):\n"
            "        self.assertEqual(add(2, 2), 4)\n",
        },
    )
    ctx.test_command = f"{shlex.quote(sys.executable)} -m unittest -v {{targets}}"
    ctx.touched.extend(["tests.py", "pkg/calc.py"])
    code, text, _, _ = _run_project_suite(ctx)
    assert code == 1  # the importer's test was run, and it fails
    assert "calc_tests.tests.C.test_add" in text or "test_add" in text


def test_uv_init_over_an_existing_project_is_refused(tmp_path):
    from agent.tools.shell import replaces_project_environment

    assert replaces_project_environment(tmp_path, "uv init") is None  # an empty dir may
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup(name='Django')\n")
    assert "already a Python project" in (replaces_project_environment(tmp_path, "uv init") or "")


def test_installing_the_workspace_s_own_package_from_the_index_is_refused(tmp_path):
    """A django fix round ran `uv add django`: django from PyPI shadowed the working
    copy and the test runner stopped importing (§75)."""
    from agent.tools.shell import replaces_project_environment

    (tmp_path / "setup.cfg").write_text("[metadata]\nname = Django\nversion = 3.1\n")
    for command in ("uv add django", "pip install Django==3.1", "python -m pip install django"):
        assert "shadow the working copy" in (replaces_project_environment(tmp_path, command) or "")
    for command in ("uv add pytz", "pip install -e .", "pip install sqlparse"):
        assert replaces_project_environment(tmp_path, command) is None
