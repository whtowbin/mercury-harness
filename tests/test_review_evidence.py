"""The reviewer's evidence for measured numbers: what run_python / run_bash printed."""

import json

from agent.approval import Approver
from agent.phases.review import COMMAND_EVIDENCE_CHAR_CAP, _command_evidence
from agent.runner import COMMAND_LOG_OUTPUT_CHARS, _log_command
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


def _ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def test_a_command_output_is_logged_with_its_code(tmp_path):
    ctx = _ctx(tmp_path)
    _log_command(ctx, "run_python", json.dumps({"code": "print(rmse)"}), "rmse 0.412\n")
    assert ctx.command_log == [
        {"tool": "run_python", "command": "print(rmse)", "output": "rmse 0.412\n"}
    ]


def test_a_long_output_is_clipped(tmp_path):
    ctx = _ctx(tmp_path)
    _log_command(ctx, "run_bash", json.dumps({"command": "cat big"}), "9" * 10_000)
    assert len(ctx.command_log[0]["output"]) == COMMAND_LOG_OUTPUT_CHARS


def test_the_log_is_shared_by_every_step_of_a_run(tmp_path):
    ctx = _ctx(tmp_path)
    _log_command(ctx.for_step(), "run_bash", "{}", "42")
    assert ctx.command_log and ctx.command_log[0]["output"] == "42"


def test_the_reviewer_sees_only_outputs_that_contain_a_number(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.command_log += [
        {"tool": "run_bash", "command": "ls", "output": "a.py b.py"},
        {"tool": "run_python", "command": "fit()", "output": "RMSE 0.412"},
    ]
    evidence = _command_evidence(ctx)
    assert "RMSE 0.412" in evidence and "a.py b.py" not in evidence


def test_the_reviewer_is_told_when_nothing_printed_a_number(tmp_path):
    assert "none printed a number" in _command_evidence(_ctx(tmp_path))


def test_the_newest_outputs_are_kept_when_over_the_cap(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.command_log += [
        {"tool": "run_python", "command": f"run {i}", "output": f"{i} " + "x" * 1_500}
        for i in range(10)
    ]
    evidence = _command_evidence(ctx)
    assert "run 9" in evidence and "run 0" not in evidence
    assert "older outputs omitted" in evidence
    assert len(evidence) < COMMAND_EVIDENCE_CHAR_CAP + 500
