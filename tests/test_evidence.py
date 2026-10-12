"""Settings.bugfix_evidence: a fix's own evidence must show the reported problem going away.

DESIGN-LOG §120. The workspace below is django 11790 in miniature: the base field already puts
`maxlength` in its widget attributes when it has a max_length, but the form sets max_length
after the field has computed them, so the rendered field has none. A fix in `widget_attrs`
is in the wrong place, and a test that calls `widget_attrs()` directly passes on the old
code too. The tests run real pytest and a real project runner on real directories: a mock of
the test run would pass by construction.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.evidence import (
    REPRO_STILL_FAILS,
    EvidenceGate,
    _restore_old_code,
    changed_sources,
    new_tests,
    scratch_target,
    verdict_on_old_code,
)
from agent.phases.execute import _step_gate
from agent.runner import GateChain, ReproGate
from agent.tools import ToolContext
from agent.tools.fs import write_file
from agent.trace import Trace
from agent.workspace import Workspace

FORMS = """class Field:
    def __init__(self, max_length=None):
        self.max_length = max_length
        self.attrs = self.widget_attrs()

    def widget_attrs(self):
        return {"maxlength": self.max_length} if self.max_length is not None else {}


class UsernameField(Field):
    def widget_attrs(self):
        return {**super().widget_attrs(), "autocapitalize": "none"}


class LoginForm:
    def __init__(self):
        self.username = UsernameField()
        self.username.max_length = 150
"""
EXISTING_TESTS = """from forms import LoginForm


def test_username_autocapitalize_is_off():
    assert LoginForm().username.attrs["autocapitalize"] == "none"
"""
# The wrong place: widget_attrs runs before LoginForm sets max_length.
WRONG_FIX = FORMS.replace(
    'return {**super().widget_attrs(), "autocapitalize": "none"}',
    'attrs = {**super().widget_attrs(), "autocapitalize": "none"}\n'
    "        if self.max_length is not None:\n"
    '            attrs["maxlength"] = self.max_length\n'
    "        return attrs",
)
RIGHT_FIX = FORMS.replace(
    "        self.username.max_length = 150\n",
    "        self.username.max_length = 150\n"
    '        self.username.attrs["maxlength"] = self.username.max_length\n',
)
# Passes before and after any fix: widget_attrs() already returns maxlength once the form
# has set max_length.
VACUOUS_TEST = """

def test_username_widget_attrs_have_maxlength():
    form = LoginForm()
    assert form.username.widget_attrs()["maxlength"] == 150
"""
# Through what the goal reports: fails on the old code.
PUBLIC_TEST = """

def test_username_renders_maxlength():
    assert LoginForm().username.attrs.get("maxlength") == 150
"""
REPRO = "from forms import LoginForm\nassert LoginForm().username.attrs.get('maxlength') == 150\n"


def _ctx(root: Path, tmp_path: Path, **kwargs) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "trace"), echo=False),
        **kwargs,
    )


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "forms.py").write_text(FORMS)
    (root / "test_forms.py").write_text(EXISTING_TESTS)
    return _ctx(root, tmp_path)


def _finish() -> dict:
    return {"id": "f1", "function": {"name": "finish_step", "arguments": json.dumps({})}}


def _events(ctx: ToolContext, kind: str) -> list[dict]:
    path = Path(ctx.trace.path)
    return [e for e in map(json.loads, path.read_text().splitlines()) if e.get("kind") == kind]


# -- the run's new tests against the code before the run (real pytest) ------


def test_new_tests_that_pass_on_the_old_code_hold_finish_step_once(ctx):
    write_file(ctx, "forms.py", WRONG_FIX)
    write_file(ctx, "test_forms.py", EXISTING_TESTS + VACUOUS_TEST)
    assert new_tests(ctx) == {"test_forms.py": ["test_username_widget_attrs_have_maxlength"]}
    gate = EvidenceGate(ctx)
    message = gate.hold(_finish())
    assert message is not None
    assert "test_username_widget_attrs_have_maxlength" in message
    assert "pass on the code as it was before this run" in message
    assert _events(ctx, "evidence_tests")[-1]["verdict"] == "green"
    # Once a run: the next finish_step, in this step or another, goes through.
    assert gate.hold(_finish()) is None
    assert EvidenceGate(ctx).hold(_finish()) is None
    # The real workspace was never touched: the old code ran in a copy.
    assert (ctx.workspace.root / "forms.py").read_text() == WRONG_FIX


def test_a_new_test_that_fails_on_the_old_code_lets_the_step_finish(ctx):
    write_file(ctx, "forms.py", RIGHT_FIX)
    write_file(ctx, "test_forms.py", EXISTING_TESTS + PUBLIC_TEST)
    assert EvidenceGate(ctx).hold(_finish()) is None
    assert _events(ctx, "evidence_tests")[-1]["verdict"] == "red"


def test_steps_that_change_no_existing_source_are_never_checked(ctx):
    write_file(ctx, "test_forms.py", EXISTING_TESTS + VACUOUS_TEST)  # tests only
    assert EvidenceGate(ctx).hold(_finish()) is None
    write_file(ctx, "helpers.py", "def helper():\n    return 1\n")  # new code only
    assert EvidenceGate(ctx).hold(_finish()) is None
    assert changed_sources(ctx) == {"helpers.py": None}
    assert _events(ctx, "evidence_check") == []


def test_other_calls_are_never_held(ctx):
    write_file(ctx, "forms.py", WRONG_FIX)
    write_file(ctx, "test_forms.py", EXISTING_TESTS + VACUOUS_TEST)
    call = {"id": "e", "function": {"name": "read_file", "arguments": '{"path": "forms.py"}'}}
    assert EvidenceGate(ctx).hold(call) is None


# -- a project runner that is not pytest, as django's is (real subprocess) --

RUNTESTS = """import sys
import unittest

suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
"""
EXISTING_UNITTEST = """import unittest

from forms import LoginForm


class LoginFormTests(unittest.TestCase):
    def test_username_autocapitalize_is_off(self):
        self.assertEqual(LoginForm().username.attrs["autocapitalize"], "none")
"""
VACUOUS_UNITTEST = """
    def test_username_widget_attrs_have_maxlength(self):
        self.assertEqual(LoginForm().username.widget_attrs()["maxlength"], 150)
"""


def test_a_project_runner_runs_the_copy_with_its_own_code_first_on_the_path(tmp_path):
    """django's runner lives in tests/ and imports django from an editable install, which
    points at the real workspace; the copy has to come first on the path, or the old code
    is never what runs. Here the runner cannot import `forms` at all without it, and the
    new test would read as failing on the old code."""
    root = tmp_path / "ws"
    (root / "tests").mkdir(parents=True)
    (root / "forms.py").write_text(FORMS)
    (root / "tests" / "runtests.py").write_text(RUNTESTS)
    (root / "tests" / "test_forms.py").write_text(EXISTING_UNITTEST)
    ctx = _ctx(
        root,
        tmp_path,
        test_command=f"{shlex.quote(sys.executable)} tests/runtests.py {{targets}}",
        test_label_root="tests",
    )
    write_file(ctx, "forms.py", WRONG_FIX)
    write_file(ctx, "tests/test_forms.py", EXISTING_UNITTEST + VACUOUS_UNITTEST)
    message = EvidenceGate(ctx).hold(_finish())
    assert message is not None and "test_username_widget_attrs_have_maxlength" in message
    assert _events(ctx, "evidence_tests")[-1]["verdict"] == "green"


# -- the plan's own reproduction, re-run (real subprocess) ------------------


def test_a_reproduction_that_still_fails_holds_once_with_its_output(ctx):
    ctx.repro_reproduced = REPRO
    write_file(ctx, "forms.py", WRONG_FIX)
    gate = EvidenceGate(ctx)
    message = gate.hold(_finish())
    assert message is not None
    assert message.startswith(REPRO_STILL_FAILS.split("{snippet}")[0])
    assert "AssertionError" in message
    assert "username.attrs.get('maxlength') == 150" in message  # the snippet itself
    assert _events(ctx, "evidence_repro")[-1]["outcome"] == "still_fails"
    assert gate.hold(_finish()) is None  # no new tests, and the repro hold is spent


def test_a_reproduction_that_now_passes_lets_the_step_finish(ctx):
    ctx.repro_reproduced = REPRO
    write_file(ctx, "forms.py", RIGHT_FIX)
    assert EvidenceGate(ctx).hold(_finish()) is None
    assert _events(ctx, "evidence_repro")[-1]["outcome"] == "fixed"


def test_the_repro_gate_keeps_the_snippet_that_reproduced(tmp_path):
    trace = Trace(str(tmp_path / "t"), echo=False)
    gate = ReproGate(lambda code: (1, "AssertionError"), trace)
    submit = {"function": {"name": "submit_plan", "arguments": json.dumps({"reproduce": REPRO})}}
    assert gate.hold(submit) is None
    assert gate.reproduced == REPRO.strip()
    passing = ReproGate(lambda code: (0, ""), trace)
    passing.hold(submit)
    assert passing.reproduced is None


def test_step_contexts_share_the_run_scoped_evidence_state(ctx):
    ctx.repro_reproduced = REPRO
    step = ctx.for_step()
    assert step.repro_reproduced == REPRO
    assert step.evidence_holds is ctx.evidence_holds


# -- reading a test run's verdict -------------------------------------------


def test_the_verdict_reads_the_runs_failures():
    names = ["test_new"]
    assert verdict_on_old_code(0, "test_new (t.T) ... ok\n", names) == "green"
    assert verdict_on_old_code(1, "FAILED t.T.test_new - AssertionError\n", names) == "red"
    # Another test failed and the new one passed: still no evidence from the new one.
    other = "test_new (t.T) ... ok\nFAILED t.T.test_old - AssertionError\n"
    assert verdict_on_old_code(1, other, names) == "green"
    # The old code could not import what the new test needs: it fails there.
    assert verdict_on_old_code(1, "ImportError: cannot import name 'x'\n", names) == "red"
    # Passed, but the output never shows the new test ran: say nothing.
    assert verdict_on_old_code(0, "Ran 3 tests\nOK\n", names) == "unknown"
    # Nothing ran at all (a label that selected nothing): no verdict either way.
    assert verdict_on_old_code(5, "Ran 0 tests\nNO TESTS RAN\n", names) == "unknown"


# -- the switch -------------------------------------------------------------


def test_off_by_default_and_only_on_adds_the_gate(ctx):
    ws = str(ctx.workspace.root)
    assert Settings(goal="x", workspace=ws).bugfix_evidence is False
    off = _step_gate(Settings(goal="x", workspace=ws), ctx)
    on = _step_gate(Settings(goal="x", workspace=ws, bugfix_evidence=True), ctx)
    assert not isinstance(off, EvidenceGate)
    assert not any(isinstance(g, EvidenceGate) for g in getattr(off, "gates", []))
    assert isinstance(on, GateChain)
    assert any(isinstance(g, EvidenceGate) for g in on.gates)


# -- the scratch copy keeps symlinks: never write through one ---------------


def test_restoring_old_code_never_writes_through_a_symlink(tmp_path):
    """copy_workspace keeps symlinks, so a symlinked file or parent directory in the copy
    points outside it. Restoring the old code through it would overwrite (or delete) the real
    file: the workspace's own fix, before scoring. Real directories, real symlinks."""
    outside = tmp_path / "outside"
    (outside / "pkg").mkdir(parents=True)
    (outside / "mod.py").write_text("FIXED\n")
    (outside / "pkg" / "inner.py").write_text("FIXED\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "mod.py").symlink_to(outside / "mod.py")
    (scratch / "pkg").symlink_to(outside / "pkg", target_is_directory=True)
    (scratch / "real.py").write_text("now\n")

    for rel in ("mod.py", "pkg/inner.py"):
        assert "symlink" in str(scratch_target(scratch, rel))
        assert "symlink" in _restore_old_code(scratch, {"real.py": "old\n", rel: "OLD\n"})
        assert "symlink" in _restore_old_code(scratch, {rel: None})  # a delete too
    for rel in ("/etc/hosts", "../outside/mod.py"):
        assert "refused" in str(scratch_target(scratch, rel))
    assert (outside / "mod.py").read_text() == "FIXED\n"
    assert (outside / "pkg" / "inner.py").read_text() == "FIXED\n"
    assert (scratch / "real.py").read_text() == "now\n"  # a refusal changes nothing
    assert _restore_old_code(scratch, {"real.py": "old\n"}) == ""
    assert (scratch / "real.py").read_text() == "old\n"


def test_a_symlinked_source_in_the_workspace_is_refused_and_left_alone(tmp_path):
    """End to end: the gate's check through copy_workspace on a workspace whose changed
    source is a symlink to a file outside it. No hold, the refusal on the trace, and the
    outside file keeps the fix."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "forms.py").write_text(FORMS)
    root = tmp_path / "ws"
    root.mkdir()
    (root / "forms.py").symlink_to(outside / "forms.py")
    (root / "test_forms.py").write_text(EXISTING_TESTS)
    ctx = _ctx(root, tmp_path)
    ctx.baseline.capture("forms.py", root / "forms.py")
    (outside / "forms.py").write_text(WRONG_FIX)
    write_file(ctx, "test_forms.py", EXISTING_TESTS + VACUOUS_TEST)
    assert EvidenceGate(ctx).hold(_finish()) is None
    event = _events(ctx, "evidence_tests")[-1]
    assert event["verdict"] == "unknown" and "symlink" in event["refused"]
    assert (outside / "forms.py").read_text() == WRONG_FIX
