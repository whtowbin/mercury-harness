"""What a run is allowed to spend, pinned so a refactor cannot quietly raise it.

Cognitive complexity is worth reducing, but not by trading away cost or wall
time — and the ways a refactor can do that are not caught by ordinary
behavioural tests, because the output stays identical while the bill goes up.
An extra model call, a tool-call batch that stops being a batch, a second
pytest run, a prompt that grew: each produces the same answer, more slowly and
for more money.

So these tests count expensive operations rather than checking results. They
are deliberately strict about numbers that should never change without
somebody deciding they should.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.config import Settings
from agent.llm import Reply, Usage
from agent.phases.execute import _step_prompt
from agent.phases.plan import Plan, Step
from agent.runner import run_tool_loop
from agent.tools import ToolContext, execution_registry
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


class CountingClient:
    """Counts chat() calls and hands back a scripted reply each time."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

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
        self.calls += 1
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def call(name, **args):
    return {"id": f"c{name}{len(args)}", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*tool_calls, content=""):
    return Reply(content=content, tool_calls=list(tool_calls), usage=Usage(100, 50, 0.0, 0.25))


# -- the tool loop spends one model call per turn, not per tool call -----


def test_a_batch_of_tool_calls_costs_one_model_call(ctx):
    """Batching is most of what makes a step affordable: five tool calls in
    one assistant turn must stay one round trip, not become five."""
    client = CountingClient(
        [
            reply(
                call("list_dir", path="."),
                call("list_dir", path="a"),
                call("glob_files", pattern="*.py"),
                call("glob_files", pattern="*.md"),
                call("list_dir", path="b"),
            ),
            reply(call("finish_step", summary="done")),
        ]
    )
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
    )
    assert client.calls == 2, "one call per assistant turn, regardless of batch size"


def test_finishing_early_stops_calling_the_model(ctx):
    client = CountingClient([reply(call("finish_step", summary="done"))])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=20,
    )
    assert client.calls == 1


def test_the_iteration_cap_is_a_hard_ceiling_on_model_calls(ctx):
    """max_iters is the spend ceiling for a step. One call per iteration, and
    never more, however the loop is restructured."""
    client = CountingClient([reply(call("list_dir", path=".")) for _ in range(50)])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=5,
    )
    assert client.calls == 5


def test_a_stall_costs_one_extra_call_then_stops(ctx):
    """A model that answers without tool calls gets exactly one nudge before
    the loop gives up — not an unbounded argument about it."""
    client = CountingClient([reply(content="I am thinking about it") for _ in range(10)])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=10,
    )
    assert client.calls == 2


def test_a_refused_repeat_still_costs_no_extra_model_call(ctx):
    """Refusing a repeated call must not turn into an extra round trip."""
    same = call("list_dir", path=".")
    client = CountingClient([reply(same) for _ in range(8)])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=8,
    )
    assert client.calls == 8


# -- the final check runs each expensive command exactly once ------------


def test_the_final_check_runs_the_suite_once_per_invocation(ctx, monkeypatch):
    from agent.phases import finalcheck

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    runs: list[str] = []
    real_run = finalcheck.toolchain.run

    def counting_run(tool, args, cwd, timeout=30, env=None):
        runs.append(tool)
        if tool == "pytest":
            return (0, "1 passed", "")
        return real_run(tool, args, cwd, timeout=timeout)

    monkeypatch.setattr(finalcheck.toolchain, "run", counting_run)
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    # Coverage off so the count is deterministic. With it on, `_run_suite`
    # tries `uv run --with coverage ... -m pytest` and falls back to a bare
    # `pytest` when that cannot start — two toolchain calls, one suite
    # execution, by design. Which of those happens depends on whether uv's
    # cache is warm, so asserting on it tests the sandbox rather than the
    # harness. The contract here is that the suite executes once.
    finalcheck.run_final_check(
        ctx,
        Settings(workspace=".", goal="g", check_coverage=False),
        Plan(goal="g", draft=[], steps=[]),
        ctx.trace,
    )
    assert runs.count("pytest") == 1, f"suite ran {runs.count('pytest')} times"


def test_each_step_verify_command_runs_once(ctx, monkeypatch):
    from agent.phases import execute as execute_module
    from agent.phases import finalcheck

    commands: list[str] = []
    # run_final_check imports run_verify locally (from .execute import
    # run_verify) rather than at module scope, so the patch target is
    # execute.py's own name, not finalcheck's.
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (commands.append(command), (True, ""))[1]
    )
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    plan = Plan(
        goal="g",
        draft=[],
        steps=[
            Step(id="a", title="t", intent="i", acceptance="a", verify="cmd-a"),
            Step(id="b", title="t", intent="i", acceptance="a", verify="cmd-b"),
        ],
    )
    finalcheck.run_final_check(ctx, Settings(workspace=".", goal="g"), plan, ctx.trace)
    assert commands == ["cmd-a", "cmd-b"], "each verify exactly once, in order"


def test_a_workspace_with_no_tests_never_starts_pytest(ctx, monkeypatch):
    """Wall time not spent is the cheapest optimisation there is."""
    from agent.phases import finalcheck

    runs: list[str] = []
    monkeypatch.setattr(
        finalcheck.toolchain, "run", lambda tool, *a, **k: (runs.append(tool), (5, "", ""))[1]
    )
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    finalcheck.run_final_check(
        ctx, Settings(workspace=".", goal="g"), Plan(goal="g", draft=[], steps=[]), ctx.trace
    )
    assert "pytest" not in runs


def test_the_final_check_makes_no_model_calls_at_all(ctx, monkeypatch):
    """Its whole value is being the one phase that costs nothing to ask."""
    from agent.phases import finalcheck

    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    # No client is passed in at all; this is a signature-level guarantee, and
    # the test exists so that stays true.
    import inspect

    params = inspect.signature(finalcheck.run_final_check).parameters
    assert "client" not in params


# -- prompts are an input cost; they must not drift by accident ----------


def test_the_step_prompt_is_byte_stable(ctx):
    """Prompt text is billed on every iteration of a step and shapes the
    output. A refactor that reworded it is a behaviour change wearing a
    cleanup's clothes, so the shape is pinned here."""
    plan = Plan(
        goal="build a thing",
        draft=[],
        steps=[
            Step(
                id="s1",
                title="First",
                intent="do the first thing",
                acceptance="it is done",
                files=["a.py"],
                verify="pytest -q",
            ),
        ],
    )
    prompt = _step_prompt(ctx, plan, plan.steps[0], 0, 12_000)
    for fragment in (
        "Overall goal:\nbuild a thing",
        "Full plan (for orientation only — do NOT work ahead):",
        "=== YOUR STEP 1 of 1: [s1] First ===",
        "Intent: do the first thing",
        "Done when: it is done",
        "Verify with: pytest -q",
        "Files named by this step: a.py",
        "Do this step now, then call finish_step.",
    ):
        assert fragment in prompt, f"missing from the step prompt: {fragment!r}"


def test_a_retry_prompt_adds_the_failure_and_nothing_else(ctx):
    plan = Plan(
        goal="g", draft=[], steps=[Step(id="s1", title="T", intent="i", acceptance="a", files=[])]
    )
    plain = _step_prompt(ctx, plan, plan.steps[0], 0, 12_000)
    retried = _step_prompt(ctx, plan, plan.steps[0], 0, 12_000, feedback="it broke")
    assert "*** THIS STEP ALREADY FAILED ONCE ***" in retried
    assert "it broke" in retried
    # The retry should be the same prompt plus the feedback block, not a
    # wholly different (and larger) prompt.
    assert len(retried) < len(plain) + 200


def test_preloading_respects_its_token_budget(ctx):
    """The preload budget is the main lever on an execute step's input cost."""
    big = "x = 1\n" * 5_000
    (ctx.workspace.root / "big.py").write_text(big)
    plan = Plan(
        goal="g",
        draft=[],
        steps=[Step(id="s1", title="T", intent="i", acceptance="a", files=["big.py"])],
    )
    prompt = _step_prompt(ctx, plan, plan.steps[0], 0, 2_000)
    # Half the budget is reserved for preloaded content; 2000 tokens is
    # roughly 8000 characters, so the whole 30k-character file must not be in
    # there.
    assert len(prompt) < len(big)


# -- the last-word turn: one extra call, and only where it pays ----------


def test_an_exhausted_step_costs_exactly_one_extra_call_to_report(ctx):
    """45 of 301 traced loops ended without ever calling their terminal tool,
    losing everything the model had worked out. Asking for a report is worth
    one call on a loop that has already spent its whole budget — and must
    never cost two."""
    client = CountingClient(
        [reply(call("list_dir", path=".")) for _ in range(5)]
        + [reply(call("finish_step", summary="ran out; the import is circular"))]
    )
    result, _ = run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=5,
        salvage=True,
    )
    assert client.calls == 6, "five iterations plus exactly one last word"
    assert result is not None
    assert "circular" in result["summary"]


def test_the_last_word_is_not_asked_for_when_the_loop_finished(ctx):
    client = CountingClient([reply(call("finish_step", summary="done"))])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=5,
        salvage=True,
    )
    assert client.calls == 1


def test_a_loop_without_salvage_pays_nothing_extra(ctx):
    """The phases that do not salvage keep their existing cost exactly."""
    client = CountingClient([reply(call("list_dir", path=".")) for _ in range(9)])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=4,
    )
    assert client.calls == 4


def test_a_refused_last_word_does_not_retry(ctx):
    """If the model will not report even when asked, that is the end of it."""
    client = CountingClient(
        [reply(call("list_dir", path=".")) for _ in range(3)] + [reply(content="no tool call")]
    )
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
        salvage=True,
    )
    assert client.calls == 4


# -- the tool kit is a fixed tax on every call of a step -----------------


def test_the_core_kit_is_materially_smaller_than_the_full_one():
    """Tool schemas were ~3,900 tokens of a ~12,300-token executor prompt —
    the largest single block in it, and eleven of the 28 tools went unused
    across 782 traced calls."""
    from agent.llm import approx_tokens
    from agent.tools import execution_registry as registry

    def size(reg):
        return sum(approx_tokens(json.dumps(s)) for s in reg.schemas())

    full = size(registry())
    core = size(registry("add a parser to config.py"))
    assert core < full * 0.7, f"core kit {core} vs full {full}: not a real saving"


def test_a_step_that_mentions_git_gets_the_git_tools():
    from agent.tools import execution_tool_names

    assert "git_commit" in execution_tool_names("commit the change to git")
    assert "git_commit" not in execution_tool_names("add a parser to config.py")


def test_a_step_that_mentions_packages_gets_the_lookup_tools():
    from agent.tools import execution_tool_names

    assert "pypi_info" in execution_tool_names("add the requests dependency")
    assert "pypi_info" not in execution_tool_names("rename the variable")


def test_a_step_that_mentions_debugging_gets_the_tracer():
    from agent.tools import execution_tool_names

    assert "trace_python" in execution_tool_names("debug why this returns the wrong value")
    assert "trace_python" not in execution_tool_names("add a parser to config.py")


def test_force_tools_adds_a_tool_regardless_of_step_text():
    """Settings.force_tools (see bench's mercury-trace-python config): a task
    whose auto-generated step titles never say "debug" still needs a way to
    force the tool present, to measure its effect rather than the trigger
    wording."""
    from agent.tools import execution_registry

    registry = execution_registry(
        "add a parser to config.py", extra_tools=frozenset({"trace_python"})
    )
    assert "trace_python" in registry.names()


def test_no_extra_tools_leaves_the_normal_kit_unchanged():
    from agent.tools import execution_registry

    with_none = execution_registry("add a parser to config.py")
    with_empty = execution_registry("add a parser to config.py", extra_tools=frozenset())
    assert set(with_none.names()) == set(with_empty.names())


def test_a_detected_node_workspace_gets_lint_and_test_tools_unconditionally():
    """Unlike a keyword trigger, this doesn't depend on the step text
    mentioning npm/eslint -- a workspace-wide property shouldn't need a step
    to remember to say so."""
    from agent import langtools
    from agent.tools import execution_tool_names

    names = execution_tool_names("fix the off-by-one", profile=langtools.NODE)
    assert {"lint_project", "test_project"} <= names


def test_a_python_workspace_gets_nothing_extra_from_its_profile():
    """check_code/run_tests already cover Python, in the core kit,
    unchanged -- profile=PYTHON must not add anything on top."""
    from agent import langtools
    from agent.tools import execution_tool_names

    with_profile = execution_tool_names("fix the off-by-one", profile=langtools.PYTHON)
    without_profile = execution_tool_names("fix the off-by-one", profile=None)
    assert with_profile == without_profile
    assert "lint_project" not in with_profile


def test_json_mention_triggers_the_data_format_tool_without_a_project_profile():
    from agent.tools import execution_tool_names

    assert "validate_data_file" in execution_tool_names("write a config.json file")
    assert "validate_data_file" not in execution_tool_names("rename the variable")


def test_the_core_kit_always_holds_what_every_step_needs():
    """A narrowed kit that drops the tools a step is made of would trade
    tokens for round trips, which is the wrong direction."""
    from agent.tools import execution_tool_names

    core = execution_tool_names("")
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "run_bash",
        "run_tests",
        "check_code",
        "finish_step",
        "ask_user",
    ):
        assert name in core, f"{name} must never be selected away"


def test_the_prompt_estimate_counts_the_whole_request():
    """A traced step logged prompt_tokens_est 3584 for a call billed at
    12,165: the estimate looked only at the user message."""
    from agent.llm import approx_tokens, request_tokens
    from agent.tools import execution_registry as registry

    schemas = registry().schemas()
    user, system = "u" * 4_000, "s" * 4_000
    estimate = request_tokens(system, user, schemas)
    assert estimate > approx_tokens(user) * 3
    assert estimate >= approx_tokens(system) + approx_tokens(user) + 3_000


# -- tool schema quality, per the tool-writing guidance ------------------


def test_every_tool_parameter_is_described():
    """Anthropic's tool-writing guidance puts description quality first: "even
    small refinements to tool descriptions can yield dramatic improvements."
    Nineteen parameters had no description at all, including edit_file's
    replace_all — the one whose silent misreading rewrites a whole file."""
    from agent.tools import execution_registry, planning_registry, review_registry

    undescribed = []
    for registry in (execution_registry(), planning_registry(), review_registry()):
        for schema in registry.schemas():
            fn = schema.get("function", schema)
            for name, spec in fn["parameters"].get("properties", {}).items():
                if not str(spec.get("description", "")).strip():
                    undescribed.append(f"{fn['name']}.{name}")
    assert not undescribed, f"parameters with no description: {sorted(set(undescribed))}"


def test_every_tool_says_what_it_does():
    from agent.tools import execution_registry

    for schema in execution_registry().schemas():
        fn = schema.get("function", schema)
        assert len(fn.get("description", "")) > 30, f"{fn['name']} is barely described"


def test_the_discovery_tools_point_at_each_other():
    """The measured failure: one step used glob_files twice, list_dir, then
    `run_bash find` to answer "what files exist". Each tool now says which
    question is actually its own."""
    from agent.tools import execution_registry

    by_name = {
        s.get("function", s)["name"]: s.get("function", s)["description"]
        for s in execution_registry().schemas()
    }
    assert "glob_files" in by_name["list_dir"]
    assert "grep" in by_name["list_dir"]
    assert "glob_files" in by_name["grep"]
    assert "find" in by_name["glob_files"]


def test_the_narrowed_kit_still_costs_less_than_the_original():
    """Descriptions are not free. The saving from narrowing must survive them."""
    from agent.llm import approx_tokens
    from agent.tools import execution_registry

    step = sum(
        approx_tokens(json.dumps(s))
        for s in execution_registry("fix the scoring bug in game.py").schemas()
    )
    assert step < 3_400, f"a typical step's kit is {step} tokens; it was 3,884 before"


# -- the batching nudge must never become a round trip -------------------


def test_the_batching_nudge_costs_no_extra_model_call(ctx):
    """It rides on a tool result that was being sent anyway. If it ever
    becomes a message of its own it costs exactly the round trip it exists to
    save, which would make it worse than useless."""
    for name in ("a.py", "b.py", "c.py"):
        (ctx.workspace.root / name).write_text("x = 1\n")
    client = CountingClient(
        [
            reply(call("read_file", path="a.py")),
            reply(call("read_file", path="b.py")),
            reply(call("read_file", path="c.py")),
            reply(call("finish_step", summary="done")),
        ]
    )
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=8,
    )
    assert client.calls == 4, "one call per assistant turn, nudge or no nudge"


def test_the_nudge_is_capped_so_it_cannot_bloat_a_long_step(ctx):
    """Every repeat is prompt tokens on each subsequent call of the step, and
    a model that ignored it twice will not take it the sixth time."""
    from agent.runner import MAX_BATCHING_NUDGES

    for i in range(12):
        (ctx.workspace.root / f"{i}.py").write_text("x = 1\n")
    client = CountingClient(
        [reply(call("read_file", path=f"{i}.py")) for i in range(12)]
        + [reply(call("finish_step", summary="done"))]
    )
    _, messages = run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=13,
    )
    fired = sum(
        1
        for m in messages
        if m.get("role") == "tool" and "does the same work in one" in str(m.get("content", ""))
    )
    assert fired == MAX_BATCHING_NUDGES


# -- an identical read-only check is replayed, not re-run -----------------


def _counting_pytest(monkeypatch, runs: list[str]):
    """Make run_tests' underlying pytest invocation free and countable."""
    from agent.tools import checks

    def fake_run(tool, args, cwd, timeout=30, env=None):
        runs.append(tool)
        return (0, "1 passed in 0.01s", "")

    monkeypatch.setattr(checks.toolchain, "run", fake_run)


def test_an_identical_run_tests_call_is_not_run_twice(ctx, monkeypatch):
    """The reviewer in one traced run called run_tests on the same file four
    times in a single phase -- twice at 22 seconds -- against a workspace it
    had not edited between them. Same command plus byte-identical files is
    the same answer, so the second call replays the first's output."""
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    runs: list[str] = []
    _counting_pytest(monkeypatch, runs)
    same = call("run_tests", path="test_x.py")
    client = CountingClient(
        [reply(same), reply(same), reply(same), reply(call("finish_step", summary="done"))]
    )
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
    )
    assert len(runs) == 1, f"pytest ran {len(runs)} times for three identical calls"


def test_a_changed_workspace_invalidates_the_replay(ctx, monkeypatch):
    """The cache is keyed on the content of every touched file, so an edit
    between two identical calls must make the second one run for real --
    replaying there would hand the model a stale answer about its own edit."""
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    runs: list[str] = []
    _counting_pytest(monkeypatch, runs)
    same = call("run_tests", path="test_x.py")
    client = CountingClient(
        [
            reply(same),
            reply(
                call("write_file", path="test_x.py", content="def test_x():\n    assert 1 == 1\n")
            ),
            reply(same),
            reply(call("finish_step", summary="done")),
        ]
    )
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=8,
    )
    # `runs` also carries the autolint ruff call the write triggers.
    assert runs.count("pytest") == 2, "an edit between the calls forces a real re-run"


def test_replay_can_be_turned_off(ctx, monkeypatch):
    """Settings.replay_identical_checks reaches the tool loop as
    ToolContext.replay_checks, so a benchmark config can price the saving."""
    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    runs: list[str] = []
    _counting_pytest(monkeypatch, runs)
    ctx.replay_checks = False
    same = call("run_tests", path="test_x.py")
    client = CountingClient([reply(same), reply(same), reply(call("finish_step", summary="done"))])
    run_tool_loop(
        client=client,
        model="m",
        phase="p",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
    )
    assert len(runs) == 2, "with replay off, both calls run for real"


# -- the gate does not re-establish what the suite just proved ------------


def _gate_plan():
    return Plan(
        goal="g",
        draft=[],
        steps=[
            Step(
                id="a",
                title="t",
                intent="i",
                acceptance="a",
                verify="uv run pytest test_x.py -k one",
            ),
            Step(
                id="b",
                title="t",
                intent="i",
                acceptance="a",
                verify="uv run pytest test_x.py::TestTwo",
            ),
        ],
    )


def _count_gate_commands(ctx, monkeypatch, settings):
    """Every verify command the gate actually issues, in order."""
    from agent.phases import execute as execute_module
    from agent.phases import finalcheck

    commands: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (commands.append(command), (True, ""))[1]
    )
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    monkeypatch.setattr(
        finalcheck.toolchain, "run", lambda tool, args, cwd, **kw: (0, "2 passed", "")
    )
    finalcheck.run_final_check(ctx, settings, _gate_plan(), ctx.trace)
    return commands


def test_a_green_suite_settles_a_step_s_own_pytest_selection(ctx, monkeypatch):
    """Both step verifies in the 2026-09-16 trace were pytest selections
    inside the 72-test suite that ran moments later, at 23s a gate pass and
    twice over. A green suite has already run every test either selection
    would, so running them again is arithmetic rather than evidence."""
    (ctx.workspace.root / "test_x.py").write_text("def test_one():\n    assert True\n")
    settings = Settings(workspace=".", goal="g")
    assert _count_gate_commands(ctx, monkeypatch, settings) == [], (
        "a green suite should settle both selections without running either"
    )


def test_turning_the_skip_off_runs_both_again(ctx, monkeypatch):
    """Settings.skip_subsumed_verifies is what the perf-nosubsume benchmark
    config flips, so the saving above can be priced rather than assumed."""
    (ctx.workspace.root / "test_x.py").write_text("def test_one():\n    assert True\n")
    settings = Settings(workspace=".", goal="g", skip_subsumed_verifies=False)
    assert len(_count_gate_commands(ctx, monkeypatch, settings)) == 2


def test_a_red_suite_still_runs_the_deferred_commands(ctx, monkeypatch):
    """The saving is given back exactly when the per-step attribution it
    would have cost is worth having: a failed step verify carries the step id
    and files a fix round needs, and a bare list of failing node ids does
    not."""
    from agent.phases import execute as execute_module
    from agent.phases import finalcheck

    (ctx.workspace.root / "test_x.py").write_text("def test_one():\n    assert True\n")
    commands: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (commands.append(command), (True, ""))[1]
    )
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    monkeypatch.setattr(
        finalcheck.toolchain, "run", lambda tool, args, cwd, **kw: (1, "1 failed", "")
    )
    finalcheck.run_final_check(ctx, Settings(workspace=".", goal="g"), _gate_plan(), ctx.trace)
    assert len(commands) == 2, "a red suite settles nothing; both must run"


def test_a_non_pytest_verify_is_never_deferred(ctx, monkeypatch):
    """A prose or lint check has no relationship to the test suite, so the
    suite's result says nothing about it either way."""
    from agent.phases import execute as execute_module
    from agent.phases import finalcheck

    (ctx.workspace.root / "test_x.py").write_text("def test_one():\n    assert True\n")
    commands: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (commands.append(command), (True, ""))[1]
    )
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    monkeypatch.setattr(
        finalcheck.toolchain, "run", lambda tool, args, cwd, **kw: (0, "2 passed", "")
    )
    plan = Plan(
        goal="g",
        draft=[],
        steps=[
            Step(
                id="a",
                title="t",
                intent="i",
                acceptance="a",
                verify="pandoc -f gfm -t plain -o /dev/null README.md",
            ),
        ],
    )
    finalcheck.run_final_check(ctx, Settings(workspace=".", goal="g"), plan, ctx.trace)
    assert commands == ["pandoc -f gfm -t plain -o /dev/null README.md"]


# -- a step does not re-run the suite it just ran -------------------------


def _green_run_tests(ctx, monkeypatch, targets=(".",)):
    """Record a passing whole-suite run at the current workspace state."""
    from agent.tools.base import workspace_state

    ctx.tests_green = (workspace_state(ctx.workspace.root), tuple(targets))


def test_a_step_verify_the_executor_already_ran_is_not_run_again(ctx, monkeypatch):
    """667 of 960 traced step verifies followed a passing run_tests with no
    file-changing call in between, 403 of them over the whole suite. The
    second run is the same subprocess over byte-identical files -- twenty
    seconds apiece on a real project, for a result that cannot have changed."""
    from agent.phases import execute as execute_module

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    _green_run_tests(ctx, monkeypatch)
    ran: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (ran.append(command), (True, ""))[1]
    )
    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    problem = execute_module._verify_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g"))
    assert problem == ""
    assert ran == [], "the suite had just passed against this exact workspace"
    assert step.verify_passed is True


def test_a_changed_workspace_makes_the_step_verify_run(ctx, monkeypatch):
    """The recorded pass is keyed on a stat manifest of every file, so an
    edit between the run and the verify invalidates it."""
    from agent.phases import execute as execute_module

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    _green_run_tests(ctx, monkeypatch)
    (ctx.workspace.root / "new.py").write_text("x = 1\n")
    ran: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (ran.append(command), (True, ""))[1]
    )
    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    execute_module._verify_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g"))
    assert ran == ["uv run pytest -q"]


def test_a_narrower_recorded_run_does_not_cover_a_broader_verify(ctx, monkeypatch):
    """A run_tests over one file proves nothing about `pytest -q` over all of
    them, so the verify must still run."""
    from agent.phases import execute as execute_module

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    _green_run_tests(ctx, monkeypatch, targets=("test_x.py",))
    ran: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (ran.append(command), (True, ""))[1]
    )
    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    execute_module._verify_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g"))
    assert ran == ["uv run pytest -q"]


def test_a_non_pytest_verify_is_never_skipped_by_a_green_suite(ctx, monkeypatch):
    """A pandoc or ruff verify has no relationship to the test suite."""
    from agent.phases import execute as execute_module

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    _green_run_tests(ctx, monkeypatch)
    ran: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (ran.append(command), (True, ""))[1]
    )
    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run ruff check x.py")
    execute_module._verify_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g"))
    assert ran == ["uv run ruff check x.py"]


def test_the_verify_skip_can_be_turned_off(ctx, monkeypatch):
    from agent.phases import execute as execute_module

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    _green_run_tests(ctx, monkeypatch)
    ran: list[str] = []
    monkeypatch.setattr(
        execute_module, "run_verify", lambda c, command: (ran.append(command), (True, ""))[1]
    )
    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    execute_module._verify_step(
        ctx, ctx.trace, step, Settings(workspace=".", goal="g", skip_verified_by_tests=False)
    )
    assert ran == ["uv run pytest -q"]


def test_review_runs_the_suite_once_for_the_whole_phase(ctx, monkeypatch):
    """ "Run run_tests before you decide anything" was issued per chunk, and
    the reviewer cannot edit anything -- so N chunks mandated N runs of the
    same suite over a workspace that provably could not change between them.
    99 of 662 traced runs made two or more run_tests calls during review, up
    to nine, and the cross-chunk repeats were invisible to every dedupe the
    harness has because each chunk gets a fresh seen_calls counter."""
    from agent.phases import review as review_module

    runs: list[str] = []
    monkeypatch.setattr(
        review_module, "_suite_result", lambda c, t: (runs.append("suite"), "suite output\n")[1]
    )
    monkeypatch.setattr(
        review_module, "_load_batches", lambda c, f, b: [[], [], []]
    )  # three chunks
    monkeypatch.setattr(
        review_module, "run_tool_loop", lambda **kw: ({"verdict": "pass", "findings": []}, [])
    )
    ctx.touched.extend(["a.py", "b.py", "c.py"])
    review_module.review(
        object(),
        Settings(workspace=".", goal="g"),
        Plan(goal="g", draft=[], steps=[]),
        ctx,
        ctx.trace,
    )
    assert runs == ["suite"], f"suite ran {len(runs)} times for three chunks"


def test_review_batching_reserves_room_for_non_file_overhead(ctx, monkeypatch):
    """The traced failure: _load_batches used to be handed review_budget()
    unchanged and know nothing about the diff, suite output, step report,
    system prompt or tool schemas review() appends to every chunk afterward.
    A real run's chunk (14 files) measured ~153,216 tokens against a 133,333
    nominal budget and the provider then refused it for exceeding the
    model's actual 260,000-token limit. The budget _load_batches receives
    must now be well under review_budget() -- reduced for both the measured
    fixed overhead and a safety margin for what a flat char/4 estimate still
    misses on dense content."""
    from agent.phases import review as review_module

    captured: dict = {}
    monkeypatch.setattr(
        review_module, "_suite_result", lambda c, t: "1234 failed in a big pytest dump\n" * 200
    )
    monkeypatch.setattr(
        review_module, "_load_batches", lambda c, f, b: (captured.setdefault("budget", b), [[]])[1]
    )
    monkeypatch.setattr(
        review_module, "run_tool_loop", lambda **kw: ({"verdict": "pass", "findings": []}, [])
    )
    ctx.touched.append("a.py")
    settings = Settings(workspace=".", goal="fix the thing" * 50)
    review_module.review(
        object(),
        settings,
        Plan(
            goal=settings.goal,
            draft=[],
            steps=[
                Step(
                    id="s1",
                    title="t",
                    intent="i",
                    acceptance="a",
                    files=["a.py"],
                    summary="did it" * 50,
                )
            ],
        ),
        ctx,
        ctx.trace,
    )
    nominal = settings.budget.review_budget()
    assert captured["budget"] < nominal
    assert captured["budget"] <= nominal * review_module.REVIEW_BUDGET_SAFETY_MARGIN


def test_review_batching_floors_the_budget_even_with_huge_overhead(ctx, monkeypatch):
    """Overhead large enough to swamp the whole review budget must not hand
    _load_batches a non-positive or vanishingly small number -- there is
    still at least one file that needs a real chunk to land in."""
    from agent.phases import review as review_module

    captured: dict = {}
    monkeypatch.setattr(
        review_module, "_suite_result", lambda c, t: "line\n" * 500_000
    )  # absurdly large
    monkeypatch.setattr(
        review_module, "_load_batches", lambda c, f, b: (captured.setdefault("budget", b), [[]])[1]
    )
    monkeypatch.setattr(
        review_module, "run_tool_loop", lambda **kw: ({"verdict": "pass", "findings": []}, [])
    )
    ctx.touched.append("a.py")
    settings = Settings(workspace=".", goal="g")
    review_module.review(object(), settings, Plan(goal="g", draft=[], steps=[]), ctx, ctx.trace)
    assert captured["budget"] == 10_000


def test_the_reviewer_is_given_the_suite_output_not_an_instruction(ctx, monkeypatch):
    """The grounding move `_known_failures_note` already makes: hand over the
    fact rather than telling the model to go and fetch it."""
    from agent.phases import review as review_module
    from agent.tools import checks

    # _suite_result imports run_tests at call time, so the source module is
    # the patch target.
    monkeypatch.setattr(checks, "run_tests", lambda c: "3 passed in 0.1s")
    text = review_module._suite_result(ctx, ctx.trace)
    assert "3 passed in 0.1s" in text
    assert "already been run for you" in text


# -- a step's verify is about the step, not the whole suite ---------------


def _scoped(step, settings=None):
    from agent.phases.plan import _narrow_whole_suite_verify
    from agent.trace import Trace

    _narrow_whole_suite_verify(step, None, Trace("/tmp/x", echo=False))
    return step.verify


def test_a_whole_suite_step_verify_is_narrowed_to_the_step(tmp_path):
    """429 of 960 traced step verifies were a bare `uv run pytest -q`: the
    entire suite, run to check one step. _synthesize_verify's own docstring
    explains why that is wrong -- a plan whose tests arrive in step 3 fails
    step 1 against tests that do not exist yet -- but only applied it to
    steps whose verify was blank, while PLANNER_SYSTEM offered the
    whole-suite form as its worked example."""
    step = Step(
        id="a",
        title="t",
        intent="i",
        acceptance="a",
        verify="uv run pytest -q",
        files=["test_thing.py"],
        writes=["test_thing.py"],
    )
    assert _scoped(step) == "uv run pytest -q test_thing.py"


def test_a_verify_that_already_says_what_it_means_is_left_alone(tmp_path):
    """A command carrying its own path, pattern, or a non-pytest tool is a
    deliberate choice, not the planner's default."""
    for command in (
        "uv run pytest -q test_x.py",
        "pytest -k slow",
        "uvx --quiet ruff check x.py",
        "pandoc -f gfm -t plain -o /dev/null README.md",
    ):
        step = Step(
            id="a",
            title="t",
            intent="i",
            acceptance="a",
            verify=command,
            files=["x.py"],
            writes=["x.py"],
        )
        assert _scoped(step) == command


def test_a_step_with_nothing_to_narrow_to_keeps_the_whole_suite(tmp_path):
    """No declared files means no scoped command to build, and a step with
    no verify at all is worse than one that is broader than ideal."""
    step = Step(id="a", title="t", intent="i", acceptance="a", verify="uv run pytest -q")
    assert _scoped(step) == "uv run pytest -q"


# -- coverage is not measured on a round that is already lost -------------


def test_a_round_with_a_failed_verify_does_not_pay_for_coverage(ctx, monkeypatch):
    """Whether to wrap the suite in `coverage run` has to be decided before
    it runs, because the data only exists if it was collected. A round with a
    failed verify command is going back through a fix round whatever coverage
    says, so measuring what a doomed round covered buys nothing."""
    from agent.phases import execute as execute_module
    from agent.phases import finalcheck

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    wrapped: list[bool] = []

    def fake_run_suite(c, with_coverage=False, parallel=False):
        wrapped.append(with_coverage)
        return (0, "1 passed", "", False)

    monkeypatch.setattr(finalcheck, "_run_suite", fake_run_suite)
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    monkeypatch.setattr(execute_module, "run_verify", lambda c, command: (False, "boom"))
    plan = Plan(
        goal="g",
        draft=[],
        steps=[
            Step(id="a", title="t", intent="i", acceptance="a", verify="./check.sh"),
        ],
    )
    finalcheck.run_final_check(ctx, Settings(workspace=".", goal="g"), plan, ctx.trace)
    assert wrapped == [False], "a failing round must not pay the coverage wrapper"


def test_a_clean_round_still_measures_coverage(ctx, monkeypatch):
    """Coverage decided the verdict alone in 2 of 12 traced failing gates, so
    the round that would otherwise pass must still be measured."""
    from agent.phases import finalcheck

    (ctx.workspace.root / "test_x.py").write_text("def test_x():\n    assert True\n")
    wrapped: list[bool] = []

    def fake_run_suite(c, with_coverage=False, parallel=False):
        wrapped.append(with_coverage)
        return (0, "1 passed", "", False)

    monkeypatch.setattr(finalcheck, "_run_suite", fake_run_suite)
    monkeypatch.setattr(finalcheck.validate, "ty_check", lambda *a, **k: [])
    finalcheck.run_final_check(
        ctx, Settings(workspace=".", goal="g"), Plan(goal="g", draft=[], steps=[]), ctx.trace
    )
    assert wrapped == [True]


# -- a step is type-checked against the rest of the project ---------------


def test_a_step_that_breaks_a_caller_fails_its_own_step(ctx, monkeypatch):
    """The executor type-checked 16 times across every traced run while the
    reviewer did it 486 times, so a changed signature surfaced one phase
    later than it could have. ty resolves the whole project, so it sees a
    call site in a file the step never opened."""
    from agent.phases import execute as execute_module
    from agent.validate import Diagnostic

    (ctx.workspace.root / "mod.py").write_text("def f(a):\n    return a\n")
    ctx.mark_touched("mod.py")
    ctx.type_baseline = frozenset()
    monkeypatch.setattr(
        execute_module.validate,
        "ty_check",
        lambda root, paths=None, timeout=180: [
            Diagnostic(
                file="caller.py",
                line=3,
                column=1,
                code="E",
                message="f() takes 1 argument but 2 were given",
            )
        ],
    )
    step = Step(
        id="a",
        title="t",
        intent="i",
        acceptance="a",
        verify="",
        files=["mod.py"],
        writes=["mod.py"],
    )
    problem = execute_module._type_check_step(
        ctx, ctx.trace, step, Settings(workspace=".", goal="g")
    )
    assert "call sites" in problem
    assert "takes 1 argument" in problem


def test_type_debt_that_predates_the_run_does_not_fail_a_step(ctx, monkeypatch):
    """ty on a partly-typed project reports pre-existing debt; failing a step
    for inherited debt would make the signal useless."""
    from agent.phases import execute as execute_module
    from agent.phases.finalcheck import _diag_key
    from agent.validate import Diagnostic

    existing = Diagnostic(file="caller.py", line=3, column=1, code="E", message="old debt")
    (ctx.workspace.root / "mod.py").write_text("def f(a):\n    return a\n")
    ctx.mark_touched("mod.py")
    ctx.type_baseline = frozenset({_diag_key(existing)})
    monkeypatch.setattr(
        execute_module.validate, "ty_check", lambda root, paths=None, timeout=180: [existing]
    )
    step = Step(
        id="a",
        title="t",
        intent="i",
        acceptance="a",
        verify="",
        files=["mod.py"],
        writes=["mod.py"],
    )
    assert (
        execute_module._type_check_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g"))
        == ""
    )


def test_a_step_with_no_baseline_never_type_fails(ctx, monkeypatch):
    """Without a baseline there is no way to tell a new error from old debt,
    so the check stays silent rather than guessing."""
    from agent.phases import execute as execute_module
    from agent.validate import Diagnostic

    (ctx.workspace.root / "mod.py").write_text("def f(a):\n    return a\n")
    ctx.mark_touched("mod.py")
    ctx.type_baseline = None
    monkeypatch.setattr(
        execute_module.validate,
        "ty_check",
        lambda root, paths=None, timeout=180: [
            Diagnostic(file="x.py", line=1, column=1, code="E", message="boom")
        ],
    )
    step = Step(
        id="a",
        title="t",
        intent="i",
        acceptance="a",
        verify="",
        files=["mod.py"],
        writes=["mod.py"],
    )
    assert (
        execute_module._type_check_step(ctx, ctx.trace, step, Settings(workspace=".", goal="g"))
        == ""
    )
