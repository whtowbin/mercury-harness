"""End-to-end wiring test with a scripted client, so it needs no API key."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.llm import Reply, Usage
from agent.runner import run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


def call(name: str, **args) -> dict:
    return {
        "id": f"c{abs(hash(name)) % 1000}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class ScriptedClient:
    """Returns a canned reply per turn, recording what it was sent."""

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> Reply:
        self.seen.append(list(messages))
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0001, 0.1))
        return self.replies.pop(0)


def reply(*tool_calls, content="") -> Reply:
    return Reply(content=content, tool_calls=list(tool_calls), usage=Usage(100, 50, 0.0002, 0.25))


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def drive(client, ctx, max_iters=6):
    return run_tool_loop(
        client=client,
        model="fake/model",
        phase="test",
        system="sys",
        user="task",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=max_iters,
    )


def test_loop_writes_a_file_and_terminates(ctx):
    client = ScriptedClient(
        [
            reply(call("write_file", path="hello.py", content="print('hi')\n")),
            reply(call("finish_step", summary="wrote hello.py", verified=False)),
        ]
    )
    result, messages = drive(client, ctx)
    assert result["summary"] == "wrote hello.py"
    assert (ctx.workspace.root / "hello.py").read_text() == "print('hi')\n"
    assert ctx.touched == ["hello.py"]
    assert messages[-1]["role"] == "tool"


def test_loop_feeds_edit_failure_back_to_the_model(ctx):
    (ctx.workspace.root / "m.py").write_text("a = 1\n")
    client = ScriptedClient(
        [
            reply(call("edit_file", path="m.py", old_string="a = 2", new_string="a = 3")),
            reply(call("edit_file", path="m.py", old_string="a = 1", new_string="a = 3")),
            reply(call("finish_step", summary="fixed")),
        ]
    )
    drive(client, ctx)
    # The failed attempt must reach the model as a tool result it can act on.
    failure = next(m for m in client.seen[-1] if m.get("role") == "tool")
    assert "not found" in failure["content"]
    assert (ctx.workspace.root / "m.py").read_text() == "a = 3\n"


def test_loop_nudges_once_then_gives_up_when_the_model_stalls(ctx):
    client = ScriptedClient([reply(content="thinking..."), reply(content="still thinking...")])
    result, _ = drive(client, ctx)
    assert result is None
    assert any(
        m.get("role") == "user" and "did not call a tool" in m["content"] for m in client.seen[-1]
    )


def test_loop_respects_the_iteration_cap(ctx):
    client = ScriptedClient([reply(call("list_dir", path=".")) for _ in range(20)])
    result, _ = drive(client, ctx, max_iters=3)
    assert result is None
    assert len(client.seen) == 3


def test_a_cancelled_token_stops_before_the_next_model_call(ctx):
    """The stop-request checkpoint: cancel.check() runs at the top of every
    iteration, so a request to stop takes effect between calls rather than
    needing to interrupt one already in flight."""
    from agent.cancel import Cancelled

    ctx.cancel.cancel()
    client = ScriptedClient([reply(call("list_dir", path="."))])
    with pytest.raises(Cancelled):
        drive(client, ctx)
    assert client.seen == [], "the client must never be called once cancelled"


def test_usage_totals_accumulate_per_model(ctx):
    client = ScriptedClient([reply(call("finish_step", summary="ok"))])
    drive(client, ctx)
    totals = ctx.trace.totals
    assert totals.calls == 1
    assert totals.by_model["fake/model"]["completion_tokens"] == 50
    assert totals.tok_per_s("fake/model") == pytest.approx(200.0)


# -- suggest_next_step -----------------------------------------------------


class SuggestingClient:
    """Returns a canned suggestion, recording what it was asked."""

    def __init__(self, text: str = "add input validation to the new function"):
        self.text = text
        self.seen: list[list[dict]] = []

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    ) -> Reply:
        self.seen.append(list(messages))
        return Reply(content=self.text, usage=Usage(50, 20, 0.0003, 0.4))


def make_plan_with_steps():
    from agent.phases.plan import Plan, Step

    step = Step(id="s1", title="build it", intent="", acceptance="")
    step.summary = "created thing.py"
    return Plan(goal="build a thing", draft=[], steps=[step])


def test_suggest_next_step_returns_a_question_the_person_must_answer(ctx):
    from agent.loop import suggest_next_step

    client = SuggestingClient("Should thing.py be wired into main.py, or stay standalone?")
    settings = Settings(workspace=str(ctx.workspace.root), goal="g", planner_model="fake-planner")
    result = suggest_next_step(client, settings, make_plan_with_steps(), [], ctx, ctx.trace)
    assert result == "Should thing.py be wired into main.py, or stay standalone?"


def test_suggest_next_step_withholds_work_the_run_should_have_done(ctx):
    """An instruction is unfinished work, not a choice to hand over."""
    from agent.loop import suggest_next_step

    client = SuggestingClient("Wire thing.py into main.py.")
    settings = Settings(workspace=str(ctx.workspace.root), goal="g", planner_model="fake-planner")
    assert suggest_next_step(client, settings, make_plan_with_steps(), [], ctx, ctx.trace) == ""


def test_suggest_next_step_includes_the_goal_and_diff_in_its_prompt(ctx):
    from agent.loop import suggest_next_step
    from agent.tools.fs import write_file

    write_file(ctx, "thing.py", "def f():\n    pass\n")
    client = SuggestingClient()
    settings = Settings(workspace=str(ctx.workspace.root), goal="g", planner_model="fake-planner")
    suggest_next_step(client, settings, make_plan_with_steps(), [], ctx, ctx.trace)
    user_message = client.seen[0][1]["content"]
    assert "build a thing" in user_message
    assert "thing.py" in user_message


def test_suggest_next_step_never_raises_on_a_client_failure(ctx):
    from agent.loop import suggest_next_step

    class BrokenClient:
        def chat(self, *a, **k):
            raise RuntimeError("network exploded")

    settings = Settings(workspace=str(ctx.workspace.root), goal="g", planner_model="fake-planner")
    result = suggest_next_step(BrokenClient(), settings, make_plan_with_steps(), [], ctx, ctx.trace)
    assert result == ""


def test_suggest_next_step_records_its_cost_in_the_trace(ctx):
    from agent.loop import suggest_next_step

    client = SuggestingClient()
    settings = Settings(workspace=str(ctx.workspace.root), goal="g", planner_model="fake-planner")
    before = ctx.trace.totals.cost_usd
    suggest_next_step(client, settings, make_plan_with_steps(), [], ctx, ctx.trace)
    assert ctx.trace.totals.cost_usd > before


def test_review_can_be_skipped_without_skipping_the_final_gate(monkeypatch, ctx):
    """The benchmark expressed "no review" as stopping after execute, which
    also skipped the gate — making its no-review configuration a strictly
    smaller agent than its no-gate one, and its timings incomparable.

    Asserted behaviourally: an earlier version of this test matched on source
    text and failed against the comment explaining the bug.
    """
    from agent import loop as loop_module

    called = []
    monkeypatch.setattr(loop_module, "_plan_until_accepted", lambda *a, **k: make_plan_with_steps())
    monkeypatch.setattr(loop_module, "execute_plan", lambda *a, **k: None)
    monkeypatch.setattr(
        loop_module, "_review_rounds", lambda *a, **k: called.append("review") or []
    )
    monkeypatch.setattr(loop_module, "_final_gate", lambda *a, **k: called.append("gate"))

    settings = Settings(
        workspace=str(ctx.workspace.root),
        goal="g",
        review=False,
        quiet=True,
        ask_enabled=False,
        clarify=False,
        quick_answers=False,
        memory=False,
        suggest_next_step=False,
    )
    loop_module.run(settings)
    assert "review" not in called, "review must be skipped"
    assert "gate" in called, "the final gate must still run"


def test_review_is_off_by_default_and_the_gate_still_runs(monkeypatch, ctx):
    from agent import loop as loop_module

    called = []
    monkeypatch.setattr(loop_module, "_plan_until_accepted", lambda *a, **k: make_plan_with_steps())
    monkeypatch.setattr(loop_module, "execute_plan", lambda *a, **k: None)
    monkeypatch.setattr(
        loop_module, "_review_rounds", lambda *a, **k: called.append("review") or []
    )
    monkeypatch.setattr(loop_module, "_final_gate", lambda *a, **k: called.append("gate"))
    settings = Settings(
        workspace=str(ctx.workspace.root),
        goal="g",
        quiet=True,
        ask_enabled=False,
        clarify=False,
        quick_answers=False,
        memory=False,
        suggest_next_step=False,
    )
    loop_module.run(settings)
    assert called == ["gate"]


def test_review_runs_when_asked_for(monkeypatch, ctx):
    from agent import loop as loop_module

    called = []
    monkeypatch.setattr(loop_module, "_plan_until_accepted", lambda *a, **k: make_plan_with_steps())
    monkeypatch.setattr(loop_module, "execute_plan", lambda *a, **k: None)
    monkeypatch.setattr(
        loop_module, "_review_rounds", lambda *a, **k: called.append("review") or []
    )
    monkeypatch.setattr(loop_module, "_final_gate", lambda *a, **k: called.append("gate"))
    settings = Settings(
        workspace=str(ctx.workspace.root),
        goal="g",
        quiet=True,
        ask_enabled=False,
        clarify=False,
        quick_answers=False,
        memory=False,
        suggest_next_step=False,
        review=True,
    )
    loop_module.run(settings)
    assert called == ["review", "gate"]


def test_unchanged_since_execute_is_true_when_review_touches_nothing(monkeypatch, ctx):
    """The gate's redundant-verify skip depends on this being computed
    honestly. Review ran, found nothing, and touched no files -- so the
    fingerprint before and after review must compare equal."""
    from agent import loop as loop_module

    captured = {}
    monkeypatch.setattr(loop_module, "_plan_until_accepted", lambda *a, **k: make_plan_with_steps())
    monkeypatch.setattr(loop_module, "execute_plan", lambda *a, **k: None)
    monkeypatch.setattr(loop_module, "_review_rounds", lambda *a, **k: [])

    def fake_gate(
        client,
        settings,
        plan,
        ctx,
        trace,
        findings,
        baseline=None,
        unchanged_since_execute=False,
        type_baseline=None,
    ):
        captured["unchanged"] = unchanged_since_execute

    monkeypatch.setattr(loop_module, "_final_gate", fake_gate)

    settings = Settings(
        workspace=str(ctx.workspace.root),
        goal="g",
        quiet=True,
        ask_enabled=False,
        clarify=False,
        quick_answers=False,
        memory=False,
        suggest_next_step=False,
    )
    loop_module.run(settings)
    assert captured["unchanged"] is True


def test_unchanged_since_execute_is_false_when_review_edits_a_file(monkeypatch, ctx):
    """Review touching a file -- the ordinary case, applying a fix -- must be
    detected, or the gate would skip re-verifying a step whose result the
    fix might have invalidated."""
    from agent import loop as loop_module

    captured = {}
    monkeypatch.setattr(loop_module, "_plan_until_accepted", lambda *a, **k: make_plan_with_steps())
    monkeypatch.setattr(loop_module, "execute_plan", lambda *a, **k: None)

    def fake_review(client, settings, plan, ctx, trace, baseline=None):
        (ctx.workspace.root / "thing.py").write_text("x = 1\n")
        ctx.mark_touched("thing.py")
        return []

    monkeypatch.setattr(loop_module, "_review_rounds", fake_review)

    def fake_gate(
        client,
        settings,
        plan,
        ctx,
        trace,
        findings,
        baseline=None,
        unchanged_since_execute=False,
        type_baseline=None,
    ):
        captured["unchanged"] = unchanged_since_execute

    monkeypatch.setattr(loop_module, "_final_gate", fake_gate)

    settings = Settings(
        workspace=str(ctx.workspace.root),
        goal="g",
        quiet=True,
        ask_enabled=False,
        clarify=False,
        quick_answers=False,
        memory=False,
        suggest_next_step=False,
        review=True,
    )
    loop_module.run(settings)
    assert captured["unchanged"] is False


def test_the_bench_never_stops_before_the_gate():
    """Checked with ast rather than substrings, so a comment naming the old
    behaviour does not fail the test that forbids it."""
    import ast

    tree = ast.parse(pathlib.Path("bench/runner.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "stop_after" and isinstance(kw.value, ast.Constant):
                assert kw.value.value != "execute", "the bench must run to the gate"


def test_load_plan_from_dict(tmp_path):
    """Test that _plan_from_dict correctly deserializes a plan."""
    from agent import loop as loop_module

    plan_data = {
        "goal": "test goal",
        "draft": ["step 1", "step 2"],
        "steps": [
            {
                "id": "step-1",
                "title": "First step",
                "intent": "Do something first",
                "acceptance": "It works",
                "files": ["file1.py"],
                "verify": "echo ok",
            },
            {
                "id": "step-2",
                "title": "Second step",
                "intent": "Do something second",
                "acceptance": "Still works",
                "depends_on": ["step-1"],
            },
        ],
    }

    settings = Settings(workspace=str(tmp_path), goal="", quiet=True)
    plan = loop_module._plan_from_dict(plan_data, settings)

    assert plan.goal == "test goal"
    assert plan.draft == ["step 1", "step 2"]
    assert len(plan.steps) == 2
    assert plan.steps[0].id == "step-1"
    assert plan.steps[0].title == "First step"
    assert plan.steps[0].verify == "echo ok"
    assert plan.steps[1].depends_on == ["step-1"]


def test_load_plan_from_file(tmp_path):
    """Test that _load_plan correctly reads a plan from a file."""
    from agent import loop as loop_module

    plan_data = {
        "goal": "from file",
        "steps": [{"id": "s1", "title": "Step 1", "intent": "", "acceptance": ""}],
    }

    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan_data))

    settings = Settings(workspace=str(tmp_path), goal="original", quiet=True)
    plan = loop_module._load_plan(str(plan_file), settings)

    # Goal from file takes precedence
    assert plan.goal == "from file"
    assert len(plan.steps) == 1
    assert plan.steps[0].id == "s1"
