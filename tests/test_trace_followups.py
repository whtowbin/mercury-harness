"""Five behaviours a traced session showed were wrong.

Each test names the observation from a traced session that motivated it,
so a later change that reintroduces one of these fails with the reason
attached rather than with a bare assertion.
"""

from __future__ import annotations

from agent.phases.clarify import _parse, looks_truncated
from agent.phases.plan import runnable_verify
from agent.runner import _where_it_went

# -- 1. clarifying questions arrived cut in half --------------------------


def test_a_question_cut_off_mid_sentence_is_dropped():
    """max_tokens=300 clipped every traced clarify reply (completion_tokens
    came back 294, 292, 298) and the casualty was always the last question.
    The person was shown "When it comes to" and replied, reasonably, that they
    could not see the rest of it."""
    assert looks_truncated("Are there specific platforms")
    assert looks_truncated("When it comes to")
    assert looks_truncated("")


def test_a_finished_question_is_kept():
    for question in (
        "Should the cache be shared?",
        "Name the module.",
        "Pick one of these:",
        "It must not crash!",
    ):
        assert not looks_truncated(question), question


def test_the_parser_drops_the_fragment_and_keeps_the_rest():
    parsed = _parse("Q: Should the cache be per-process or shared?\nQ: Are there specific")
    assert [q for q, _ in parsed] == ["Should the cache be per-process or shared?"]


# -- 1b. questions with a choice of answers -------------------------------


def test_options_are_parsed_into_buttons():
    parsed = _parse("Q: Where should the cache live? | per-process | shared via Redis")
    assert len(parsed) == 1
    question, options = parsed[0]
    assert question == "Where should the cache live?"
    assert options == ["per-process", "shared via Redis"]


def test_a_question_without_options_still_parses():
    parsed = _parse("Q: Which module did you mean?")
    assert parsed == [("Which module did you mean?", [])]


def test_clear_still_means_no_questions():
    assert _parse("CLEAR") == []


# -- 5. a verify command the workspace could never run --------------------


def test_a_linter_verify_is_rewritten_to_one_that_always_works():
    """The harness authored `uv run ruff check game.py` against a project
    declaring only pygame and pytest, so it could not start. The step reported
    a broken environment and envfail rebuilt the virtualenv for nothing."""
    assert runnable_verify("uv run ruff check game.py") == "uvx --quiet ruff check game.py"
    assert runnable_verify("uv run ty check .") == "uvx --quiet ty check ."


def test_pytest_is_never_rewritten():
    """pytest must import the code under test and everything it imports, so it
    has to run in the project's own environment."""
    for command in ("uv run pytest -q", "uv run pytest -q tests/", "uv run pytest -k scoring"):
        assert runnable_verify(command) == command


def test_invalid_pytest_wildcard_selector_is_removed():
    assert runnable_verify("pytest -q -k '*'") == "pytest -q"
    assert runnable_verify('uv run pytest -k "*" -q') == "uv run pytest -q"


def test_commands_that_are_not_uv_run_are_left_alone():
    for command in ("ruff check .", "uvx ruff check .", "uv run python x.py", "make test"):
        assert runnable_verify(command) == command


# -- 6. `uv run python -m pytest` silently could-not-verify a step --------


def test_pytest_via_python_dash_m_is_rewritten_to_the_form_that_works():
    """A bench sweep had the model write `uv run python -m pytest` for a
    verify command six separate times, and every one failed with "No module
    named pytest" in the identical workspace where bare `uv run pytest`
    passed. `uv run <name>` resolves `<name>` on PATH; `uv run python ...`
    resolves the interpreter through uv's own toolchain discovery instead,
    reaching a freshly-provisioned interpreter with no packages installed and
    never consulting PATH at all. envfail correctly recognised "No module
    named pytest" as environmental and excused the step rather than blaming
    the model's code, but that also meant the step's verification was
    silently skipped -- a real bug in that step's code would have passed
    through unverified. Rewriting to the form that reliably runs keeps the
    step's own tests in the loop instead of quietly excusing them from it."""
    assert runnable_verify("uv run python -m pytest -q") == "uv run pytest -q"
    assert runnable_verify("uv run python3 -m pytest test_x.py -q") == "uv run pytest test_x.py -q"
    assert runnable_verify("uv run python -m pytest") == "uv run pytest"


def test_pytest_via_python_dash_m_keeps_flags_before_the_interpreter():
    assert (
        runnable_verify("uv run --quiet python -m pytest -k scoring")
        == "uv run --quiet pytest -k scoring"
    )


# -- 4. "ran out of iterations" without saying why ------------------------


def _calls(*names):
    return [{"role": "assistant", "tool_calls": [{"function": {"name": n}}]} for n in names]


def test_exhaustion_names_the_tool_it_got_stuck_on():
    spent = _where_it_went(_calls("run_tests", "run_tests", "run_tests", "run_tests", "read_file"))
    assert "run_tests x4" in spent
    assert "4 calls to run_tests in a row" in spent


def test_steady_progress_is_not_reported_as_stuck():
    spent = _where_it_went(_calls("read_file", "edit_file", "run_tests", "check_code"))
    assert "in a row" not in spent


def test_a_loop_that_called_nothing_says_so():
    assert _where_it_went([]) == "no tool calls at all"


# -- 2. a question answered with a claim instead of an answer -------------


def test_quick_answer_is_given_the_workspace_map():
    """Asked "Which scripts sit in this directory?", a traced run planned,
    executed, called list_dir, and reported "Listed all files in the workspace
    root and tests/ subdirectory" — the claim, without the list. Without the
    map the question is NEEDS_PLAN by definition."""
    import inspect

    from agent.loop import try_quick_answer

    assert "workspace_map" in inspect.signature(try_quick_answer).parameters

    captured = {}

    class _Client:
        def chat(self, **kwargs):
            captured["user"] = kwargs["messages"][-1]["content"]
            from agent.llm import Reply, Usage

            return Reply(content="game.py, main.py, tests/", usage=Usage(1, 1, 0.0, 0.1))

    class _Settings:
        goal = "Which scripts sit in this directory?"
        planner_model = "m"

    class _Trace:
        def llm_call(self, *a, **k):
            pass

        def event(self, *a, **k):
            pass

    answer = try_quick_answer(
        _Client(), _Settings(), _Trace(), workspace_map="game.py\nmain.py\ntests/"
    )
    assert "game.py" in captured["user"], "the map must reach the model"
    assert "game.py" in answer


# -- 3. the end-of-run summary was an undifferentiated block --------------


def test_the_summary_renders_headings_bullets_and_spacing():
    from rich.console import Console

    from agent import summary as summary_module

    data = {
        "plan": {
            "steps": [
                {"id": "fix-scoring", "files": ["game.py"], "summary": "Reordered the pass check."}
            ]
        },
        "steps": {"total": 1, "failed": [], "unverified": [], "abandoned": []},
        "changed_files": ["game.py"],
        "final_check": {
            "ok": True,
            "tests_ran": True,
            "tests_passed": True,
            "tests_summary": "9 passed",
        },
        "suggested_next_step": "Add a test for the tie case.",
    }
    console = Console(width=70, record=True, force_terminal=False)
    console.print(summary_module.render_rich("Fix scoring", data))
    out = console.export_text()

    assert "Built" in out and "Next" in out
    assert "•" in out
    assert "\n\nBuilt" in out, "sections need a blank line before them"
    # The step id survives: it is square-bracketed, which Rich reads as a tag.
    assert "[fix-scoring]" in out


def test_a_step_id_is_not_eaten_as_rich_markup():
    """ "[list-files] Listed all files" — Rich reads [list-files] as a style
    tag, swallows it, and raises on an unknown one."""
    from rich.console import Console

    from agent import summary as summary_module

    data = {
        "plan": {"steps": [{"id": "list-files", "files": [], "summary": "Listed the files."}]},
        "steps": {"total": 1, "failed": [], "unverified": [], "abandoned": []},
        "changed_files": [],
    }
    console = Console(width=70, record=True, force_terminal=False)
    console.print(summary_module.render_rich("What is in this directory?", data))
    assert "[list-files]" in console.export_text()


def test_a_goal_ending_in_a_question_mark_is_not_given_a_full_stop_too():
    from agent import summary as summary_module

    text = summary_module.render("Does it work?", {"steps": {}, "changed_files": []})
    assert "work?." not in text


# -- the token cap: silent truncation is the bug, not the number ----------


def test_a_reply_stopped_by_its_cap_is_reported(tmp_path):
    """finish_reason was parsed off the wire into Reply and then dropped, so
    two separate truncation bugs — half a clarifying question, and every
    end-of-run suggestion — were invisible in the trace. 43 of 43 traced
    `suggest` calls came back at their 200-token cap."""
    from agent.llm import Usage
    from agent.trace import Trace

    trace = Trace(str(tmp_path / "t"), echo=False)
    said: list[str] = []
    trace.say = said.append  # type: ignore[method-assign]
    trace.llm_call("suggest", "m", Usage(10, 200, 0.0, 0.1), finish_reason="length")
    assert any("token cap" in line for line in said)


def test_a_reply_that_finished_normally_says_nothing(tmp_path):
    from agent.llm import Usage
    from agent.trace import Trace

    trace = Trace(str(tmp_path / "t"), echo=False)
    said: list[str] = []
    trace.say = said.append  # type: ignore[method-assign]
    trace.llm_call("suggest", "m", Usage(10, 40, 0.0, 0.1), finish_reason="stop")
    assert said == []


def test_finish_reason_reaches_the_trace_file(tmp_path):
    import json
    import pathlib

    from agent.llm import Usage
    from agent.trace import Trace

    path = tmp_path / "t"
    trace = Trace(str(path), echo=False)
    trace.llm_call("plan", "m", Usage(10, 40, 0.0, 0.1), finish_reason="stop")
    written = [
        json.loads(line) for line in pathlib.Path(trace.path).read_text().splitlines() if line
    ]
    call = next(e for e in written if e.get("kind") == "llm_call")
    assert call["finish_reason"] == "stop"


def test_the_agents_own_reasoning_loops_are_uncapped():
    """The planner, executor, reviewer and fixer pass no max_tokens at all —
    a traced execute turn produced 2,958 tokens. Only the small auxiliary
    completions are capped, and a cap there must never migrate to the loop."""
    import inspect

    from agent import runner

    assert "max_tokens" not in inspect.getsource(runner.run_tool_loop)


# -- 7. a verify command whose tool cannot run, caught before a step pays --


def test_preflight_passes_a_working_tool_through_unchanged(tmp_path, monkeypatch):
    from agent.phases import execute
    from agent.phases import plan as plan_module
    from agent.trace import Trace

    class _Ctx:
        class workspace:
            root = tmp_path

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (True, "pytest 8.0.0"))
    trace = Trace(str(tmp_path / "t"), echo=False)
    issue = plan_module._preflight_verify(_Ctx(), trace, "uv run pytest -q test_x.py")
    assert issue == ""


def test_preflight_reports_an_unrepairable_tool(tmp_path, monkeypatch):
    """The class of failure envfail already recognises from a step's own
    verify command — checked once here, before any step has spent a call on
    work nothing could then verify."""
    from agent.phases import execute
    from agent.phases import plan as plan_module
    from agent.trace import Trace

    class _Ctx:
        class workspace:
            root = tmp_path

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (False, "No module named 'pytest'"))
    trace = Trace(str(tmp_path / "t"), echo=False)
    issue = plan_module._preflight_verify(_Ctx(), trace, "uv run pytest -q test_x.py")
    assert "pytest" in issue.lower() or "missing-tool" in issue.lower()


def test_preflight_accepts_a_tool_that_repair_fixes(tmp_path, monkeypatch):
    from agent.phases import execute
    from agent.phases import plan as plan_module
    from agent.trace import Trace

    class _Ctx:
        class workspace:
            root = tmp_path

    calls = {"n": 0}

    def flaky_run_verify(ctx, cmd):
        calls["n"] += 1
        return (calls["n"] > 1, "" if calls["n"] > 1 else "No module named 'pytest'")

    monkeypatch.setattr(execute, "run_verify", flaky_run_verify)
    monkeypatch.setattr(plan_module.envfail, "repair", lambda root: (True, "uv sync rebuilt it"))
    trace = Trace(str(tmp_path / "t"), echo=False)
    issue = plan_module._preflight_verify(_Ctx(), trace, "uv run pytest -q test_x.py")
    assert issue == ""


def test_a_step_whose_verify_tool_cannot_run_retains_its_command_for_final_gate(
    tmp_path, monkeypatch
):
    """Preflight must not discard the only evidence the final gate can use.
    The command is retained, while execution still avoids blaming the model."""
    from agent.approval import Approver
    from agent.llm import Reply, Usage
    from agent.phases import execute
    from agent.phases import plan as plan_module
    from agent.phases.plan import make_plan
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace
    from tests.test_scheduling import RecordingClient, call, settings_for

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        trace=Trace(str(tmp_path / "t"), echo=False),
        approver=Approver(auto_approve=True),
    )
    plan_call = call(
        "submit_plan",
        draft=["understood"],
        steps=[
            {
                "id": "s1",
                "title": "t",
                "intent": "i",
                "acceptance": "a",
                "verify": "uv run pytest -q test_x.py",
            }
        ],
    )
    client = RecordingClient(
        [Reply(content="", tool_calls=[plan_call], usage=Usage(10, 10, 0.0, 0.1))]
    )
    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (False, "No module named 'pytest'"))
    monkeypatch.setattr(plan_module.envfail, "repair", lambda root: (False, "no pyproject.toml"))

    plan = make_plan(client, settings_for(tmp_path, goal="g"), ctx, ctx.trace)
    assert plan.steps[0].verify == "uv run pytest -q test_x.py"
    assert plan.steps[0].verify_error


def test_preflight_only_probes_each_distinct_tool_once(tmp_path, monkeypatch):
    """Five pytest-verified steps must not pay for the same broken-venv
    repair attempt five times over."""
    from agent.approval import Approver
    from agent.llm import Reply, Usage
    from agent.phases import execute
    from agent.phases.plan import make_plan
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace
    from tests.test_scheduling import RecordingClient, call, settings_for

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        trace=Trace(str(tmp_path / "t"), echo=False),
        approver=Approver(auto_approve=True),
    )
    steps = [
        {
            "id": f"s{i}",
            "title": "t",
            "intent": "i",
            "acceptance": "a",
            "verify": "uv run pytest -q test_x.py",
        }
        for i in range(5)
    ]
    plan_call = call("submit_plan", draft=["understood"], steps=steps)
    client = RecordingClient(
        [Reply(content="", tool_calls=[plan_call], usage=Usage(10, 10, 0.0, 0.1))]
    )
    calls = {"n": 0}

    def counting_run_verify(ctx, cmd):
        calls["n"] += 1
        return (True, "pytest 8.0.0")

    monkeypatch.setattr(execute, "run_verify", counting_run_verify)
    make_plan(client, settings_for(tmp_path, goal="g"), ctx, ctx.trace)
    assert calls["n"] == 1, f"pytest was preflighted {calls['n']} times, not once"
