"""Every tool call is followed by a `tool_done` event carrying how long the tool took
(DESIGN-LOG §91's review, Pass 3): `harness trace sql` can then say which tools cost more
than the model."""

from __future__ import annotations

import json

from agent.approval import Approver
from agent.runner import _dispatch
from agent.tools import ToolContext, planning_registry
from agent.trace import Trace
from agent.workspace import Workspace


def _events(trace: Trace) -> list[dict]:
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def test_a_tool_call_is_followed_by_its_latency(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    registry = planning_registry("terse")
    ok = _dispatch(registry, ctx, "read_file", json.dumps({"path": "a.py"}), trace, "plan")
    bad = _dispatch(registry, ctx, "read_file", json.dumps({"path": "nope.py"}), trace, "plan")
    assert "x = 1" in ok and bad.startswith("error:")
    calls = [e for e in _events(trace) if e["kind"] == "tool_call"]
    done = [e for e in _events(trace) if e["kind"] == "tool_done"]
    assert [d["fp"] for d in done] == [c["fp"] for c in calls]
    assert [d["ok"] for d in done] == [True, False]
    assert all(d["tool"] == "read_file" and 0 <= d["latency_s"] < 5 for d in done)
