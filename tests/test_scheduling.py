"""Step scheduling, the verify gate, and reasoning escalation."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.llm import Reply, Usage
from agent.phases.execute import escalate, execute_step, run_verify
from agent.phases.plan import Plan, Step
from agent.phases.schedule import build_waves, partition_disjoint, plan_schedule, run_waves
from agent.tools import ToolContext
from agent.tools.fs import read_files, write_file
from agent.trace import Trace
from agent.workspace import Workspace


def step(id_, files=(), depends_on=(), verify="") -> Step:
    return Step(
        id=id_,
        title=id_,
        intent="",
        acceptance="",
        files=list(files),
        depends_on=list(depends_on),
        verify=verify,
    )


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- dependency layering --------------------------------------------------


def test_independent_steps_share_one_wave():
    waves = build_waves([step("a"), step("b"), step("c")])
    assert len(waves) == 1
    assert {s.id for s in waves[0]} == {"a", "b", "c"}


def test_a_chain_becomes_one_wave_per_step():
    waves = build_waves([step("a"), step("b", depends_on=["a"]), step("c", depends_on=["b"])])
    assert [[s.id for s in w] for w in waves] == [["a"], ["b"], ["c"]]


def test_a_diamond_layers_correctly():
    waves = build_waves(
        [
            step("root"),
            step("left", depends_on=["root"]),
            step("right", depends_on=["root"]),
            step("join", depends_on=["left", "right"]),
        ]
    )
    assert [s.id for s in waves[0]] == ["root"]
    assert {s.id for s in waves[1]} == {"left", "right"}
    assert [s.id for s in waves[2]] == ["join"]


def test_unknown_dependencies_do_not_deadlock():
    waves = build_waves([step("a", depends_on=["does-not-exist"])])
    assert [s.id for s in waves[0]] == ["a"]


def test_unknown_dependencies_are_reported_when_a_trace_is_given(tmp_path):
    trace = Trace(str(tmp_path / "t"), echo=False)
    build_waves([step("a", depends_on=["does-not-exist"])], trace)
    kinds = [json.loads(line)["kind"] for line in trace.path.read_text().splitlines()]
    assert "schedule_unknown_deps" in kinds


def test_a_cycle_falls_back_to_plan_order():
    waves = build_waves([step("a", depends_on=["b"]), step("b", depends_on=["a"])])
    assert [s.id for wave in waves for s in wave] == ["a", "b"]


def test_a_cycle_is_reported_when_a_trace_is_given(tmp_path):
    trace = Trace(str(tmp_path / "t"), echo=False)
    build_waves([step("a", depends_on=["b"]), step("b", depends_on=["a"])], trace)
    kinds = [json.loads(line)["kind"] for line in trace.path.read_text().splitlines()]
    assert "schedule_cycle" in kinds


def test_every_step_is_scheduled_exactly_once():
    steps = [
        step("a"),
        step("b", depends_on=["a"]),
        step("c", depends_on=["a"]),
        step("d", depends_on=["b", "c"]),
    ]
    scheduled = [s.id for wave in build_waves(steps) for s in wave]
    assert sorted(scheduled) == ["a", "b", "c", "d"]


# -- file-disjointness ----------------------------------------------------


def test_disjoint_files_batch_together():
    batches = partition_disjoint([step("a", ["a.py"]), step("b", ["b.py"])], limit=4)
    assert [[s.id for s in b] for b in batches] == [["a", "b"]]


def test_overlapping_files_are_separated():
    batches = partition_disjoint([step("a", ["shared.py"]), step("b", ["shared.py"])], limit=4)
    assert [[s.id for s in b] for b in batches] == [["a"], ["b"]]


def test_a_step_declaring_no_files_runs_alone():
    batches = partition_disjoint([step("a", ["a.py"]), step("b"), step("c", ["c.py"])], limit=4)
    assert [s.id for s in batches[0]] == ["a"]
    assert [s.id for s in batches[1]] == ["b"]


def test_concurrency_limit_caps_batch_size():
    steps = [step(f"s{i}", [f"{i}.py"]) for i in range(5)]
    batches = partition_disjoint(steps, limit=2)
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == 5


def test_concurrency_of_one_is_pure_sequential_plan_order():
    plan = Plan("g", [], [step("a", ["a.py"]), step("b", ["b.py"])])
    assert plan_schedule(plan, 1) == [[plan.steps[0]], [plan.steps[1]]]


def test_concurrency_above_one_overlaps_independent_steps():
    plan = Plan("g", [], [step("a", ["a.py"]), step("b", ["b.py"])])
    assert [[s.id for s in b] for b in plan_schedule(plan, 4)] == [["a", "b"]]


# -- reasoning ladder -----------------------------------------------------


@pytest.mark.parametrize(
    "start,expected",
    [
        ("none", "low"),
        ("low", "medium"),
        ("medium", "high"),
        ("high", "high"),
        (None, "medium"),
        ("nonsense", "medium"),
    ],
)
def test_escalate_walks_up_the_ladder(start, expected):
    assert escalate(start) == expected


# -- verify gate ----------------------------------------------------------


def test_run_verify_reports_success(ctx):
    passed, output = run_verify(ctx, "echo hello")
    assert passed and "hello" in output


def test_run_verify_reports_failure_with_output(ctx):
    passed, output = run_verify(ctx, "echo boom >&2; exit 3")
    assert not passed and "boom" in output


def test_run_verify_runs_in_the_workspace(ctx):
    (ctx.workspace.root / "marker.txt").write_text("x")
    passed, output = run_verify(ctx, "ls")
    assert passed and "marker.txt" in output


def test_run_verify_is_unrestricted_when_not_jailed(ctx):
    """ctx.writable is None for sequential steps: nothing to be jailed
    against, so a verify command may write wherever it likes."""
    passed, _ = run_verify(ctx, "echo hi > side_effect.txt")
    assert passed
    assert (ctx.workspace.root / "side_effect.txt").exists()


def test_run_verify_rejects_writes_outside_a_concurrent_step_s_jail(ctx):
    """A verify command running in a concurrent batch is just as capable of
    clobbering a sibling step's file as run_bash is; before_write only
    guards edit_file/write_file, so this has to be enforced here too."""
    jailed = ctx.for_step(writable=frozenset({"mine.py"}))
    passed, output = run_verify(jailed, "echo oops > someone_elses_file.py")
    assert not passed
    assert "someone_elses_file.py" in output
    assert "outside this step's declared scope" in output


def test_run_verify_allows_writes_inside_a_concurrent_step_s_jail(ctx):
    jailed = ctx.for_step(writable=frozenset({"mine.py"}))
    passed, _ = run_verify(jailed, "echo 'x = 1' > mine.py")
    assert passed
    assert (ctx.workspace.root / "mine.py").exists()


def test_run_verify_rejects_deleting_a_file_outside_the_jail(ctx):
    (ctx.workspace.root / "shared.py").write_text("VALUE = 1\n")
    jailed = ctx.for_step(writable=frozenset({"mine.py"}))
    passed, output = run_verify(jailed, "rm shared.py")
    assert not passed
    assert "shared.py" in output


# -- step execution, retry and escalation ---------------------------------


def call(name: str, **args) -> dict:
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class RecordingClient:
    """Returns scripted replies and records the reasoning level of each call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.reasoning_levels: list[str | None] = []

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
        self.reasoning_levels.append(reasoning_effort)
        if not self.replies:
            return Reply(content="", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def reply(*tool_calls) -> Reply:
    return Reply(content="", tool_calls=list(tool_calls), usage=Usage(100, 50, 0.0, 0.1))


def settings_for(tmp_path, **overrides) -> Settings:
    overrides.setdefault("goal", "g")
    return Settings(workspace=str(tmp_path), quiet=True, trace_dir=str(tmp_path / "t"), **overrides)


def test_a_passing_step_runs_once(ctx, tmp_path):
    target = step("s", verify="exit 0")
    plan = Plan("g", [], [target])
    client = RecordingClient([reply(call("finish_step", summary="did it"))])
    execute_step(client, settings_for(tmp_path), plan, target, 0, ctx, ctx.trace)
    assert target.attempts == 1
    assert target.verify_passed is True
    assert target.summary == "did it"


def test_a_failing_verify_triggers_a_retry_at_higher_reasoning(ctx, tmp_path):
    target = step("s", verify="exit 1")
    plan = Plan("g", [], [target])
    client = RecordingClient(
        [
            reply(call("finish_step", summary="attempt one")),
            reply(call("finish_step", summary="attempt two")),
        ]
    )
    execute_step(
        client, settings_for(tmp_path, executor_reasoning="low"), plan, target, 0, ctx, ctx.trace
    )
    assert target.attempts == 2
    assert client.reasoning_levels == ["low", "medium"]
    assert target.verify_passed is False


def test_escalation_can_be_switched_off(ctx, tmp_path):
    target = step("s", verify="exit 1")
    plan = Plan("g", [], [target])
    client = RecordingClient([reply(call("finish_step", summary="one"))])
    execute_step(
        client, settings_for(tmp_path, escalate_on_failure=False), plan, target, 0, ctx, ctx.trace
    )
    assert target.attempts == 1


def test_verify_gate_can_be_switched_off(ctx, tmp_path):
    target = step("s", verify="exit 1")
    plan = Plan("g", [], [target])
    client = RecordingClient([reply(call("finish_step", summary="one"))])
    execute_step(
        client, settings_for(tmp_path, verify_steps=False), plan, target, 0, ctx, ctx.trace
    )
    assert target.attempts == 1
    assert target.verify_passed is None


def test_a_step_that_never_finishes_is_retried(ctx, tmp_path):
    target = step("s")
    plan = Plan("g", [], [target])
    # No finish_step call at all, twice over.
    client = RecordingClient([Reply(content="hmm", usage=Usage(1, 1, 0.0, 0.1))] * 6)
    execute_step(client, settings_for(tmp_path), plan, target, 0, ctx, ctx.trace)
    assert target.attempts == 2
    assert "never called finish_step" in target.summary


def test_the_retry_prompt_carries_the_failure(ctx, tmp_path):
    target = step("s", verify="echo 'assertion failed'; exit 1")
    plan = Plan("g", [], [target])
    seen: list[str] = []

    class Capturing(RecordingClient):
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
            seen.append(messages[1]["content"])
            return super().chat(
                model,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )

    client = Capturing(
        [reply(call("finish_step", summary="one")), reply(call("finish_step", summary="two"))]
    )
    execute_step(client, settings_for(tmp_path), plan, target, 0, ctx, ctx.trace)
    assert "ALREADY FAILED" in seen[1]
    assert "assertion failed" in seen[1]


# -- checkpoints -----------------------------------------------------------


def test_run_waves_checkpoints_each_step_in_a_git_workspace(ctx, tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not installed")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=tmp_path, check=True)

    plan = Plan("g", [], [step("a", ["a.py"]), step("b", ["b.py"])])

    def fake_step(client, settings, plan, target, index, context, trace):
        target.summary = "ok"

    run_waves(None, settings_for(tmp_path), plan, ctx, ctx.trace, fake_step)
    kinds = [json.loads(line)["kind"] for line in ctx.trace.path.read_text().splitlines()]
    assert kinds.count("checkpoint") == 2


def test_run_waves_does_not_error_outside_a_git_workspace(ctx, tmp_path):
    """No repo at all: checkpointing must degrade silently, not fail the run."""
    plan = Plan("g", [], [step("a", ["a.py"])])

    def fake_step(client, settings, plan, target, index, context, trace):
        target.summary = "ok"

    run_waves(None, settings_for(tmp_path), plan, ctx, ctx.trace, fake_step)
    kinds = [json.loads(line)["kind"] for line in ctx.trace.path.read_text().splitlines()]
    assert "checkpoint" not in kinds
    assert "checkpoint_error" not in kinds


# -- concurrent execution -------------------------------------------------


def test_parallel_steps_all_run(ctx, tmp_path):
    steps = [step("a", ["a.py"]), step("b", ["b.py"]), step("c", ["c.py"])]
    plan = Plan("g", [], steps)
    ran: list[str] = []

    def fake_step(client, settings, plan, target, index, context, trace):
        ran.append(target.id)
        target.summary = f"{target.id} done"

    run_waves(None, settings_for(tmp_path, step_concurrency=3), plan, ctx, ctx.trace, fake_step)
    assert sorted(ran) == ["a", "b", "c"]


def test_parallel_steps_get_their_own_resource_budget(ctx, tmp_path):
    steps = [step("a", ["a.py"]), step("b", ["b.py"])]
    plan = Plan("g", [], steps)
    # Hold the objects, not their ids: CPython reuses addresses after collection.
    guards: list = []

    def fake_step(client, settings, plan, target, index, context, trace):
        context.guard.record_subprocess(10)
        guards.append(context.guard)
        target.summary = "ok"

    run_waves(None, settings_for(tmp_path, step_concurrency=2), plan, ctx, ctx.trace, fake_step)
    assert len(guards) == 2
    assert guards[0] is not guards[1], "each concurrent step needs its own guard"
    # Neither step's shell budget was charged to the other.
    assert all(g.subprocess_seconds == 10 for g in guards)


def test_parallel_steps_share_the_run_baseline(ctx, tmp_path):
    steps = [step("a", ["a.py"]), step("b", ["b.py"])]
    plan = Plan("g", [], steps)

    def fake_step(client, settings, plan, target, index, context, trace):
        write_file(context, f"{target.id}.py", f"# {target.id}\n")
        target.summary = "ok"

    run_waves(None, settings_for(tmp_path, step_concurrency=2), plan, ctx, ctx.trace, fake_step)
    # Both writes must land in the one shared baseline and touched list.
    assert sorted(ctx.touched) == ["a.py", "b.py"]
    assert sorted(ctx.baseline.snapshots) == ["a.py", "b.py"]


def test_a_cancelled_token_stops_before_the_next_batch(ctx, tmp_path):
    """The other stop-request checkpoint: run_waves checks between batches,
    the same way run_tool_loop checks between iterations, so a step already
    in flight finishes but nothing after it starts."""
    from agent.cancel import Cancelled

    steps = [step("a", ["a.py"]), step("b", ["b.py"])]
    plan = Plan("g", [], steps)
    ran: list[str] = []

    def fake_step(client, settings, plan, target, index, context, trace):
        ran.append(target.id)
        context.cancel.cancel()  # requested mid-first-step, as a real stop would be
        target.summary = "ok"

    with pytest.raises(Cancelled):
        run_waves(None, settings_for(tmp_path), plan, ctx, ctx.trace, fake_step)
    assert ran == ["a"], "the in-flight step finishes; the next one never starts"


def test_for_step_shares_state_but_not_the_guard(ctx):
    child = ctx.for_step()
    assert child.baseline is ctx.baseline
    assert child.cache is ctx.cache
    assert child.touched is ctx.touched
    assert child.lock is ctx.lock
    assert child.guard is not ctx.guard


# -- batched reads --------------------------------------------------------


def test_read_files_returns_several_files_in_one_call(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    write_file(ctx, "b.py", "y = 2\n")
    out = read_files(ctx, ["a.py", "b.py"])
    assert "=== a.py ===" in out and "=== b.py ===" in out
    assert "x = 1" in out and "y = 2" in out


def test_read_files_reports_missing_files_inline(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    out = read_files(ctx, ["a.py", "gone.py"])
    assert "x = 1" in out
    assert "no such file" in out


def test_read_files_rejects_an_empty_list(ctx):
    assert read_files(ctx, []).startswith("error:")


def test_read_files_refuses_to_escape_the_workspace(ctx):
    assert "escapes workspace" in read_files(ctx, ["../outside.py"])


# -- planner fallback -----------------------------------------------------


def test_a_planner_that_returns_nothing_degrades_to_one_step(ctx, tmp_path):
    """Not every tool-capable model drives a structured plan call reliably."""
    from agent.phases.plan import make_plan

    client = RecordingClient([Reply(content="I will help!", usage=Usage(1, 1, 0.0, 0.1))] * 20)
    plan = make_plan(client, settings_for(tmp_path, goal="build a thing"), ctx, ctx.trace)
    assert len(plan.steps) == 1
    assert plan.steps[0].intent == "build a thing"
    assert plan.goal == "build a thing"


def test_a_guard_trip_is_abandoned_not_retried(ctx, tmp_path):
    """The bug the code review found: GuardTripped used to be swallowed inside
    run_tool_loop and returned as an ordinary empty result, so execute_step's
    own retry loop would burn a second attempt on a step already shown to be
    stuck. It must now propagate to _guarded(), which abandons the step outright."""
    from agent.phases.schedule import _guarded

    ctx.guard.max_step_errors = 2
    target = step("s")
    plan = Plan("g", [], [target])
    client = RecordingClient(
        [
            reply(call("read_file", path="missing1.py")),
            reply(call("read_file", path="missing2.py")),
            reply(call("read_file", path="missing3.py")),
        ]
    )

    _guarded(execute_step, client, settings_for(tmp_path), plan, target, 0, ctx, ctx.trace)

    assert target.summary.startswith("abandoned:")
    # Only the calls up to the trip, never a second attempt at higher reasoning.
    assert len(client.reasoning_levels) <= 3


def test_a_retry_at_the_top_of_the_ladder_does_not_claim_an_escalation():
    """`executor_reasoning` defaults to "high", so escalate() returns what it
    was handed and the retry is at identical settings. The narration used to
    say "retrying at reasoning=high" anyway, describing a bump that never
    happened; the retry's real content is the failure fed back."""
    from agent.phases.execute import escalate

    assert escalate("high") == "high"
    assert escalate("low") == "medium"
    assert escalate("medium") == "high"
    assert escalate(None) == "medium"
