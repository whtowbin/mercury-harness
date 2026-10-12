"""The plan approval gate, and the planning-time improvements around it."""

from __future__ import annotations

import pytest

from agent import dataformats, langtools
from agent.approval import Approver
from agent.config import Settings
from agent.loop import MAX_REPLANS, _plan_until_accepted
from agent.phases.plan import Plan, Step, _synthesize_verify
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def env(tmp_path):
    """Real Settings/ToolContext/Trace: the stubs these replaced did not
    typecheck against the signatures they were standing in for."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = ToolContext(
        workspace=Workspace(workspace),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(workspace=str(workspace), goal="g", clarify=False, ask_enabled=False)
    return settings, ctx


def plan_named(name: str) -> Plan:
    return Plan(goal=name, draft=[], steps=[Step(id="s", title=name, intent="i", acceptance="a")])


@pytest.fixture
def planning(monkeypatch):
    """Patch make_plan to return a numbered plan and record its prompts."""
    import agent.loop as loop_module

    made: list[str] = []

    def fake_make_plan(client, settings, ctx, trace, extra_context=""):
        made.append(extra_context)
        return plan_named(f"plan-{len(made)}")

    monkeypatch.setattr(loop_module, "make_plan", fake_make_plan)
    monkeypatch.setattr(loop_module, "gather_clarifications", lambda *a, **k: "")
    return made


def accept(_plan) -> str:
    return ""


def test_no_gate_uses_the_first_plan(planning, env):
    settings, ctx = env
    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, None)
    assert plan is not None
    assert plan.goal == "plan-1"
    assert len(planning) == 1


def test_an_accepting_gate_uses_the_first_plan(planning, env):
    settings, ctx = env
    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, accept)
    assert plan is not None
    assert plan.goal == "plan-1"


def test_abort_returns_nothing_so_nothing_is_executed(planning, env):
    settings, ctx = env
    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, lambda _p: "abort")
    assert plan is None


def test_abort_is_case_insensitive(planning, env):
    settings, ctx = env
    assert _plan_until_accepted(None, settings, ctx, ctx.trace, lambda _p: "ABORT") is None


def test_feedback_triggers_a_replan_carrying_that_feedback(planning, env):
    settings, ctx = env
    seen = []

    def gate(plan):
        seen.append(plan.goal)
        return "" if len(seen) > 1 else "use a dataclass instead"

    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, gate)
    assert plan is not None
    assert plan.goal == "plan-2"
    assert "use a dataclass instead" in planning[1]


def test_feedback_accumulates_across_several_rejections(planning, env):
    settings, ctx = env
    rejections = iter(["first note", "second note"])
    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, lambda _p: next(rejections, ""))
    assert plan is not None
    assert plan.goal == "plan-3"
    assert "first note" in planning[2]
    assert "second note" in planning[2]


def test_replanning_is_bounded_rather_than_endless(planning, env):
    """A gate that never accepts must still terminate, using the last plan
    rather than discarding the work."""
    settings, ctx = env
    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, lambda _p: "still not right")
    assert len(planning) == MAX_REPLANS + 1
    assert plan is not None


def test_whitespace_only_feedback_counts_as_acceptance(planning, env):
    settings, ctx = env
    plan = _plan_until_accepted(None, settings, ctx, ctx.trace, lambda _p: "   ")
    assert plan is not None
    assert plan.goal == "plan-1"


def test_clarification_context_reaches_the_planner(monkeypatch, env):
    settings, ctx = env
    import agent.loop as loop_module

    prompts: list[str] = []

    def fake_make_plan(client, settings, ctx, trace, extra_context=""):
        prompts.append(extra_context)
        return plan_named("p")

    monkeypatch.setattr(loop_module, "make_plan", fake_make_plan)
    monkeypatch.setattr(
        loop_module, "gather_clarifications", lambda *a, **k: "\n\nQ: which cache?\nA: redis"
    )
    _plan_until_accepted(None, settings, ctx, ctx.trace, None)
    assert "redis" in prompts[0]


# -- synthesized verify commands -----------------------------------------


def with_files(*files: str) -> Step:
    return Step(id="s", title="t", intent="i", acceptance="a", files=list(files))


def test_a_test_file_gets_a_pytest_command():
    assert _synthesize_verify(with_files("test_thing.py")) == "uv run pytest -q test_thing.py"


def test_a_nested_test_file_is_recognised_by_its_basename():
    assert "pytest" in _synthesize_verify(with_files("pkg/tests/test_thing.py"))


def test_a_trailing_underscore_test_name_also_counts():
    assert "pytest" in _synthesize_verify(with_files("thing_test.py"))


def test_a_plain_module_gets_a_ruff_check():
    assert _synthesize_verify(with_files("mod.py")) == "uv run ruff check mod.py"


def test_several_modules_are_checked_in_one_command():
    assert _synthesize_verify(with_files("a.py", "b.py")) == "uv run ruff check a.py b.py"


def test_a_step_touching_no_python_gets_nothing(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    assert _synthesize_verify(with_files("README.md")) == ""


def test_a_step_naming_no_files_gets_nothing():
    assert _synthesize_verify(with_files()) == ""


def test_a_step_mixing_python_and_a_data_file_checks_both(monkeypatch):
    """Before this, any .py file present made the function return before
    ever looking at the rest of a step's own declared writes -- a step that
    ships a code change alongside its config/doc file (a common shape) had
    the non-Python half of its own job silently unverified."""
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: name == "yq")
    command = _synthesize_verify(with_files("mod.py", "config.json"))
    assert command.startswith("uv run ruff check mod.py")
    assert "yq" in command
    assert "config.json" in command


def test_a_step_mixing_a_test_file_and_markdown_checks_both(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: name == "pandoc")
    command = _synthesize_verify(with_files("test_thing.py", "README.md"))
    assert command.startswith("uv run pytest -q test_thing.py")
    assert "pandoc" in command and "README.md" in command


def test_pure_python_step_is_unaffected_by_the_composition(monkeypatch):
    """No non-Python targets -> no `&&`, exactly today's plain command."""
    assert _synthesize_verify(with_files("mod.py")) == "uv run ruff check mod.py"


def with_writes(*writes: str, files: tuple[str, ...] = ()) -> Step:
    return Step(
        id="s", title="t", intent="i", acceptance="a", files=list(files), writes=list(writes)
    )


def test_writes_take_priority_over_files_for_the_verify_target():
    """The traced bug this closes: a "document findings" step declared
    writes=["README.md"] but files also listed a .py file it only read for
    context. The old version checked `files` and lint-checked that Python
    file, never examining README.md -- the step's actual deliverable -- at
    all."""
    step = with_writes("README.md", files=("README.md", "eval_and_compare.py"))
    assert "eval_and_compare.py" not in _synthesize_verify(step)


def test_a_step_that_never_declared_writes_still_falls_back_to_files():
    step = with_writes(files=("mod.py",))
    assert _synthesize_verify(step) == "uv run ruff check mod.py"


# -- synthesized verify commands: non-Python deliverables -----------------
# _synthesize_verify delegates these to agent.dataformats -- see
# test_dataformats.py for the format-by-format behaviour (yq-first, jq/
# tomllib/xmllint fallbacks, the pandoc prose check). These just confirm the
# delegation actually happens.


def test_json_gets_a_yq_syntax_check_when_yq_is_available(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/yq")
    assert _synthesize_verify(with_writes("config.json")) == "yq '.' config.json"


def test_json_falls_back_to_jq_without_yq(monkeypatch):
    monkeypatch.setattr(
        dataformats.shutil, "which", lambda name: "/usr/bin/jq" if name == "jq" else None
    )
    assert _synthesize_verify(with_writes("config.json")) == "jq empty config.json"


def test_json_check_is_skipped_with_no_validator_installed(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    assert _synthesize_verify(with_writes("config.json")) == ""


def test_toml_gets_a_stdlib_tomllib_check_without_yq(monkeypatch):
    """tomllib is stdlib since 3.11, so unlike jq/yq this needs no
    availability check -- but yq, when present, is tried first."""
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    command = _synthesize_verify(with_writes("pyproject.toml"))
    assert "tomllib" in command
    assert "pyproject.toml" in command


def test_yaml_gets_a_yq_check_only_when_yq_is_installed(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/yq")
    assert "yq" in _synthesize_verify(with_writes("config.yaml"))
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    assert _synthesize_verify(with_writes("config.yaml")) == ""


def test_markdown_gets_a_pandoc_check_only_when_pandoc_is_installed(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/pandoc")
    assert "pandoc" in _synthesize_verify(with_writes("README.md"))
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    assert _synthesize_verify(with_writes("README.md")) == ""


def test_csv_gets_a_yq_check_when_yq_is_available(monkeypatch):
    """yq genuinely validates CSV structure (confirmed locally: it rejects a
    ragged row and an unbalanced quote), so unlike Markdown this is a real
    syntax check, not a fabricated one -- it just has no fallback without
    yq, same as YAML."""
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: "/usr/bin/yq")
    assert _synthesize_verify(with_writes("data.csv")) == "yq '.' data.csv"


def test_csv_is_skipped_without_yq(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    assert _synthesize_verify(with_writes("data.csv")) == ""


def test_mixed_target_types_are_all_checked_together(monkeypatch):
    monkeypatch.setattr(
        dataformats.shutil, "which", lambda name: "/usr/bin/jq" if name == "jq" else None
    )
    command = _synthesize_verify(with_writes("a.json", "b.toml"))
    assert "jq empty a.json" in command
    assert "tomllib" in command and "b.toml" in command
    assert " && " in command


def test_data_and_prose_targets_are_both_checked_when_both_apply(monkeypatch):
    monkeypatch.setattr(
        dataformats.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in ("yq", "pandoc") else None,
    )
    command = _synthesize_verify(with_writes("config.json", "README.md"))
    assert "yq '.' config.json" in command
    assert "pandoc" in command and "README.md" in command
    assert " && " in command


# -- synthesized verify commands: a detected non-Python language profile --


def test_a_detected_node_workspaces_test_files_get_the_node_profile(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    step = with_writes("src/thing.test.ts")
    assert _synthesize_verify(step, profile=langtools.NODE) == "npm test -- src/thing.test.ts"


def test_a_node_profile_does_not_affect_a_step_with_no_matching_files(monkeypatch):
    """A step whose targets don't look like this profile's test files falls
    through to the data-format/prose checks exactly as it would with no
    profile at all."""
    monkeypatch.setattr(dataformats.shutil, "which", lambda name: None)
    step = with_writes("README.md")
    assert _synthesize_verify(step, profile=langtools.NODE) == _synthesize_verify(
        step, profile=None
    )


def test_python_files_still_win_over_a_detected_non_python_profile():
    """Belt and suspenders: a step with .py targets is a Python step
    whatever the workspace's overall detected profile says."""
    step = with_writes("mod.py")
    assert _synthesize_verify(step, profile=langtools.NODE) == "uv run ruff check mod.py"
