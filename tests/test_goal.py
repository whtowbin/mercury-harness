"""agent/goal.py: judge_goal()'s verdict parsing/fail-safes, and
run_until_done()'s iterate-refine-stop loop. run_once, judge_goal and
_self_audit are monkeypatched at the module level throughout -- this file is
about the loop's own control flow (when it stops, what it feeds the next
attempt, that self-audit runs once per iteration), never a real model call.
"""

from __future__ import annotations

import pytest

from agent import goal as goal_module
from agent.config import Settings
from agent.llm import Reply, Usage


class _FakeClient:
    def __init__(self, content=None, raises=None):
        self.content = content
        self.raises = raises
        self.calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        if self.raises:
            raise self.raises
        return Reply(content=self.content, usage=Usage(10, 5, 0.0001, 0.1))


# -- _parse_verdict -----------------------------------------------------


def test_parses_a_clean_json_object():
    v = goal_module._parse_verdict('{"status": "done", "reason": "x", "next_goal": ""}')
    assert v.status == "done" and v.reason == "x"


def test_parses_an_object_inside_a_markdown_fence():
    v = goal_module._parse_verdict(
        '```json\n{"status": "continue", "reason": "y", "next_goal": "z"}\n```'
    )
    assert v.status == "continue" and v.next_goal == "z"


def test_garbage_is_rejected_not_raised():
    assert goal_module._parse_verdict("not json") is None


def test_an_unknown_status_is_rejected():
    assert goal_module._parse_verdict('{"status": "maybe", "reason": "x"}') is None


# -- judge_goal -----------------------------------------------------------


def test_judge_goal_returns_the_parsed_verdict():
    client = _FakeClient(content='{"status": "done", "reason": "all good", "next_goal": ""}')
    v = goal_module.judge_goal(client, "model", "goal", {"ok": True})
    assert v.status == "done"


def test_judge_goal_is_blocked_on_a_client_exception():
    """A judge that failed to answer must not loop on a guess or silently
    claim done -- blocked is the safe failure."""
    client = _FakeClient(raises=RuntimeError("network down"))
    v = goal_module.judge_goal(client, "model", "goal", {"ok": False})
    assert v.status == "blocked"
    assert "network down" in v.reason


def test_judge_goal_is_blocked_on_an_unparseable_reply():
    client = _FakeClient(content="not a verdict at all")
    v = goal_module.judge_goal(client, "model", "goal", {"ok": False})
    assert v.status == "blocked"


def test_judge_goal_sends_low_reasoning():
    client = _FakeClient(content='{"status": "done", "reason": "x", "next_goal": ""}')
    goal_module.judge_goal(client, "model", "goal", {"ok": True})
    assert client.calls[0]["reasoning_effort"] == goal_module.GOAL_JUDGE_REASONING_EFFORT


def test_judge_goal_prefers_the_rendered_session_summary(monkeypatch):
    client = _FakeClient(content='{"status": "done", "reason": "x", "next_goal": ""}')
    goal_module.judge_goal(
        client, "model", "goal", {"ok": True, "session_summary": "human-readable account"}
    )
    assert "human-readable account" in client.calls[0]["messages"][1]["content"]


# -- run_until_done ---------------------------------------------------------


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Stubs build_client (no real network), run_once (canned summaries)
    and _self_audit (spy, no real trace file needed)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(goal_module, "build_client", lambda settings, **kw: _FakeClient())
    audits: list[dict] = []
    monkeypatch.setattr(
        goal_module,
        "_self_audit",
        lambda client, trace_path, root, goal, iteration, fb, trace: audits.append(
            {"trace_path": trace_path, "goal": goal, "iteration": iteration}
        ),
    )
    return {"workspace": str(ws), "audits": audits}


def _settings(workspace, **overrides):
    import dataclasses

    return dataclasses.replace(Settings(workspace=workspace, goal="build a game"), **overrides)


def test_stops_after_one_attempt_when_the_judge_says_done(patched, monkeypatch):
    calls = []
    monkeypatch.setattr(
        goal_module,
        "run_once",
        lambda settings, **kw: (
            calls.append(settings.goal),
            {"ok": True, "trace_path": "t1.jsonl"},
        )[1],
    )
    monkeypatch.setattr(
        goal_module,
        "judge_goal",
        lambda client, model, goal, summary: goal_module.GoalVerdict("done", "looks complete"),
    )
    result = goal_module.run_until_done(_settings(patched["workspace"]))
    assert len(calls) == 1
    assert result["goal_loop"]["status"] == "done"
    assert len(result["goal_loop"]["iterations"]) == 1


def test_continues_with_the_judges_refined_goal(patched, monkeypatch):
    calls = []
    monkeypatch.setattr(
        goal_module,
        "run_once",
        lambda settings, **kw: (
            calls.append(settings.goal),
            {"ok": False, "trace_path": "t.jsonl"},
        )[1],
    )
    verdicts = iter(
        [
            goal_module.GoalVerdict("continue", "missed the jump mechanic", "add double jump"),
            goal_module.GoalVerdict("done", "now complete"),
        ]
    )
    monkeypatch.setattr(
        goal_module, "judge_goal", lambda client, model, goal, summary: next(verdicts)
    )
    result = goal_module.run_until_done(_settings(patched["workspace"]))
    assert calls == ["build a game", "add double jump"]
    assert result["goal_loop"]["status"] == "done"
    assert len(result["goal_loop"]["iterations"]) == 2


def test_stops_immediately_when_the_judge_says_blocked(patched, monkeypatch):
    calls = []
    monkeypatch.setattr(
        goal_module,
        "run_once",
        lambda settings, **kw: (calls.append(1), {"ok": False, "trace_path": "t.jsonl"})[1],
    )
    monkeypatch.setattr(
        goal_module,
        "judge_goal",
        lambda client, model, goal, summary: goal_module.GoalVerdict(
            "blocked", "needs a missing API key"
        ),
    )
    result = goal_module.run_until_done(_settings(patched["workspace"]), max_goal_iterations=3)
    assert len(calls) == 1
    assert result["goal_loop"]["status"] == "blocked"


def test_reaching_max_iterations_without_done_reports_blocked(patched, monkeypatch):
    monkeypatch.setattr(
        goal_module, "run_once", lambda settings, **kw: {"ok": False, "trace_path": "t.jsonl"}
    )
    monkeypatch.setattr(
        goal_module,
        "judge_goal",
        lambda client, model, goal, summary: goal_module.GoalVerdict(
            "continue", "still missing something", "try again"
        ),
    )
    result = goal_module.run_until_done(_settings(patched["workspace"]), max_goal_iterations=2)
    assert len(result["goal_loop"]["iterations"]) == 2
    assert result["goal_loop"]["status"] == "blocked"
    assert "max_goal_iterations" in result["goal_loop"]["reason"]


def test_continue_with_no_next_goal_is_treated_as_blocked(patched, monkeypatch):
    """ "continue" with nothing concrete to try differently would just repeat
    the same attempt unchanged -- treated as stuck, not looped on."""
    calls = []
    monkeypatch.setattr(
        goal_module,
        "run_once",
        lambda settings, **kw: (calls.append(1), {"ok": False, "trace_path": "t.jsonl"})[1],
    )
    monkeypatch.setattr(
        goal_module,
        "judge_goal",
        lambda client, model, goal, summary: goal_module.GoalVerdict(
            "continue", "not sure what's missing", ""
        ),
    )
    result = goal_module.run_until_done(_settings(patched["workspace"]), max_goal_iterations=3)
    assert len(calls) == 1
    assert result["goal_loop"]["status"] == "blocked"


def test_self_audit_runs_once_per_iteration(patched, monkeypatch):
    monkeypatch.setattr(
        goal_module,
        "run_once",
        lambda settings, **kw: {"ok": False, "trace_path": f"t-{settings.goal}.jsonl"},
    )
    verdicts = iter(
        [
            goal_module.GoalVerdict("continue", "x", "refined goal"),
            goal_module.GoalVerdict("done", "y"),
        ]
    )
    monkeypatch.setattr(
        goal_module, "judge_goal", lambda client, model, goal, summary: next(verdicts)
    )
    goal_module.run_until_done(_settings(patched["workspace"]))
    assert len(patched["audits"]) == 2
    assert patched["audits"][0]["iteration"] == 1
    assert patched["audits"][1]["iteration"] == 2


def test_plan_path_is_only_used_on_the_first_attempt(patched, monkeypatch):
    plan_paths = []
    monkeypatch.setattr(
        goal_module,
        "run_once",
        lambda settings, plan_path=None, **kw: (
            plan_paths.append(plan_path),
            {"ok": False, "trace_path": "t.jsonl"},
        )[1],
    )
    verdicts = iter(
        [
            goal_module.GoalVerdict("continue", "x", "refined"),
            goal_module.GoalVerdict("done", "y"),
        ]
    )
    monkeypatch.setattr(
        goal_module, "judge_goal", lambda client, model, goal, summary: next(verdicts)
    )
    goal_module.run_until_done(_settings(patched["workspace"]), plan_path="/tmp/plan.json")
    assert plan_paths == ["/tmp/plan.json", None]
