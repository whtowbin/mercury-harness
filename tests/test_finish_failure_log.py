"""A failure log must be written for a run that fails its own gates, not
only one that raises.

`_finish`'s summary sets `ok = not error and outcomes["all_passed"] and
(check is None or check.ok)`, and the CLI exits non-zero whenever `not ok`
-- but the failure log used to be written only `if error`, so the single
most common kind of failed run (a completed run whose verify/finalcheck
gate found a real defect, with no exception ever raised) exited 1 and left
no readable record behind, unlike a crash.
"""

from __future__ import annotations

from agent.approval import Approver
from agent.loop import _finish
from agent.phases.finalcheck import FinalCheck
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


def _finish_with(tmp_path, *, error: str, check: FinalCheck | None, failure_log_dir: str) -> dict:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    return _finish(
        ctx.trace,
        plan=None,
        findings=[],
        ctx=ctx,
        error=error,
        goal="g",
        failure_log_dir=failure_log_dir,
        show_stats=False,
        check=check,
    )


def test_a_gate_failure_with_no_exception_still_gets_a_failure_log(tmp_path):
    dest = tmp_path / "failed-chats"
    summary = _finish_with(
        tmp_path, error="", check=FinalCheck(untested_changes=["x.py"]), failure_log_dir=str(dest)
    )
    assert summary["ok"] is False
    assert summary["error"] == ""  # unchanged meaning: no harness exception occurred
    assert summary["failure_log_path"] != ""
    assert dest.exists() and list(dest.glob("*.md"))


def test_a_raised_error_still_gets_a_failure_log_as_before(tmp_path):
    dest = tmp_path / "failed-chats"
    summary = _finish_with(tmp_path, error="stopped: boom", check=None, failure_log_dir=str(dest))
    assert summary["failure_log_path"] != ""


def test_a_clean_run_gets_no_failure_log(tmp_path):
    dest = tmp_path / "failed-chats"
    summary = _finish_with(tmp_path, error="", check=FinalCheck(), failure_log_dir=str(dest))
    assert summary["ok"] is True
    assert summary["failure_log_path"] == ""
    assert not dest.exists()


def test_no_failure_log_dir_configured_writes_nothing(tmp_path):
    summary = _finish_with(
        tmp_path, error="", check=FinalCheck(untested_changes=["x.py"]), failure_log_dir=""
    )
    assert summary["failure_log_path"] == ""
