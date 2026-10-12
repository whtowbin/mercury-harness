"""Skills the model can see, and skills that fire only when they apply.

Measured across 163 traced skill selections: four of the five skills were
injected on 94-99% of steps, because every one of them declared "*.py" as a
trigger and any Python workspace matches that. Selection was not selecting —
it was a flat 2,240-token tax, so a step to rename a variable arrived with the
whole uv dependency-management skill attached.

Narrowing the triggers creates the opposite problem: a skill whose triggers
miss is now invisible, and the model has no way to know it exists. Hence the
index — a name and a use case per skill, about ten tokens each — and
read_skill, which turns the list into something it can act on.
"""

from __future__ import annotations

from agent import skills as sk
from agent.llm import approx_tokens

LOADED = sk.load()


# -- triggers discriminate now -------------------------------------------


def test_a_dependency_step_gets_uv_and_not_the_rest():
    chosen = {
        s.name
        for s in sk.select(LOADED, "execute", "add the requests dependency", ["pyproject.toml"])
    }
    assert "uv" in chosen
    assert "ty" not in chosen and "complexity" not in chosen


def test_a_refactor_step_gets_complexity():
    chosen = {
        s.name
        for s in sk.select(LOADED, "execute", "refactor update() to reduce nesting", ["game.py"])
    }
    assert "complexity" in chosen


def test_a_typing_step_gets_ty():
    chosen = {
        s.name
        for s in sk.select(LOADED, "execute", "add type annotations to parser.py", ["parser.py"])
    }
    assert "ty" in chosen


def test_an_ordinary_edit_does_not_drag_in_everything():
    """The regression: "*.py" in every trigger list made this all five."""
    chosen = sk.select(LOADED, "execute", "rename a variable in utils.py", ["utils.py"])
    assert len(chosen) < len(LOADED)
    assert approx_tokens(sk.render(chosen)) < 2_000


def test_a_readme_step_gets_documenting_changes():
    chosen = {
        s.name
        for s in sk.select(
            LOADED, "execute", "update the readme with the comparison results", ["README.md"]
        )
    }
    assert "documenting-changes" in chosen


def test_an_ordinary_code_step_does_not_get_documenting_changes():
    chosen = {
        s.name for s in sk.select(LOADED, "execute", "fix a bug in the parser", ["parser.py"])
    }
    assert "documenting-changes" not in chosen


def test_verification_still_applies_to_any_python_step():
    """One skill genuinely is universal: every step that writes code has to
    check it worked."""
    chosen = {s.name for s in sk.select(LOADED, "execute", "make the bird flap", ["game.py"])}
    assert "verifying-your-work" in chosen


# -- the index ------------------------------------------------------------


def test_every_skill_for_the_phase_is_listed_by_name_and_use_case():
    text = sk.index(LOADED, "execute")
    for skill in LOADED:
        if skill.applies_to("execute"):
            assert skill.name in text
            assert skill.description[:30] in text


def test_a_skill_that_is_already_included_is_not_offered_for_reading():
    chosen = sk.select(LOADED, "execute", "add the requests dependency", ["pyproject.toml"])
    text = sk.index(LOADED, "execute", included=chosen)
    uv_line = next(line for line in text.splitlines() if line.startswith("- uv:"))
    assert "read_skill" not in uv_line
    other = next(line for line in text.splitlines() if line.startswith("- ty:"))
    assert "read_skill" in other


def test_the_index_is_cheap_enough_to_always_send():
    """About ten tokens a skill. It has to stay proportional: an index that
    grows past a few hundred tokens is a body budget by another name."""
    per_skill = approx_tokens(sk.index(LOADED, "execute")) / max(len(LOADED), 1)
    assert per_skill < 40, f"{per_skill:.0f} tokens per skill in the index"


def test_render_includes_the_index_when_given_a_phase():
    chosen = sk.select(LOADED, "execute", "add the requests dependency", ["pyproject.toml"])
    assert "read_skill" in sk.render(chosen, LOADED, "execute")


def test_render_without_a_phase_is_unchanged():
    chosen = sk.select(LOADED, "execute", "add the requests dependency", ["pyproject.toml"])
    assert "read_skill" not in sk.render(chosen)


def test_a_step_matching_nothing_still_learns_what_exists():
    chosen = sk.select(LOADED, "execute", "update the changelog", ["CHANGELOG.md"])
    rendered = sk.render(chosen, LOADED, "execute")
    assert "complexity" in rendered and "read_skill" in rendered


# -- read_skill -----------------------------------------------------------


def test_body_of_finds_a_skill_by_name():
    assert sk.body_of(LOADED, "uv")
    assert sk.body_of(LOADED, "UV"), "matching is case-insensitive"


def test_body_of_returns_none_for_an_unknown_name():
    assert sk.body_of(LOADED, "nope") is None


def test_read_skill_returns_the_body(tmp_path):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.tools.notes import read_skill
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    root.mkdir()
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        skills=LOADED,
    )
    assert "uv" in read_skill(ctx, "uv").lower()
    bad = read_skill(ctx, "nope")
    assert bad.startswith("error:") and "Available:" in bad


def test_read_skill_is_offered_to_the_planner_and_reviewer_not_the_executor():
    """Changed with DESIGN-LOG §113: the executor called it once in 59,338 calls, and its schema
    and catalogue cost about 420 tokens of every executor call. Its skills are pushed instead."""
    from agent.tools import execution_registry, planning_registry, review_registry

    assert "read_skill" in planning_registry().names()
    assert "read_skill" in review_registry().names()
    assert "read_skill" not in execution_registry("implement the parser step").names()


# -- the two new skills ---------------------------------------------------


def test_grilling_applies_to_every_clarify_call_and_nowhere_else():
    """Clarify's whole job is asking well, so this is not situational — but a
    trigger-less skill on execute or fix would be a flat tax on every step."""
    grilling = next(s for s in LOADED if s.name == "grilling")
    assert grilling.triggers == []
    assert grilling.phases == ["clarify"]
    for text in ("make the lamps avoid the wall gaps", "add a --verbose flag"):
        assert "grilling" in {s.name for s in sk.select(LOADED, "clarify", text, [])}
    for phase in ("execute", "fix", "plan"):
        assert "grilling" not in {s.name for s in sk.select(LOADED, phase, "anything", ["a.py"])}


def test_research_fires_on_api_work_and_not_on_a_rename():
    api = {
        s.name
        for s in sk.select(LOADED, "execute", "call the stripe API to create a charge", ["pay.py"])
    }
    assert "research" in api
    rename = {
        s.name for s in sk.select(LOADED, "execute", "rename a variable in utils.py", ["utils.py"])
    }
    assert "research" not in rename


def test_the_clarify_phase_is_given_its_skills():
    """It was the only phase running with no reference at all."""
    import inspect

    from agent.phases.clarify import gather_clarifications

    source = inspect.getsource(gather_clarifications)
    assert "skills_module.select" in source
    assert '"clarify"' in source


# -- diagnosing -----------------------------------------------------------

BUG_REPORTS = [
    "The timer never resets when a new round starts",
    "the pause sound was not added to the menu",
    "The icons are still not appearing. Perhaps they are too faint?",
    "bosses are still spawning off the map when they shouldn't",
    "Fix the tile offset logic in grid.py render() method",
    "Traceback (most recent call last): File main.py line 38",
]
NOT_BUG_REPORTS = [
    "add a --verbose flag to the CLI",
    "rename a variable in utils.py",
    "build a todo list with JSON persistence",
    "add the requests dependency",
]


def test_diagnosing_fires_on_how_people_actually_report_bugs():
    """Invented, in the shape of the goals in this harness's own traces. Bug reports are
    phrased as negations and persistence — "never", "still not appearing",
    "when they shouldn't" — not as the word "bug"."""
    for goal in BUG_REPORTS:
        chosen = {s.name for s in sk.select(LOADED, "execute", f"{goal} game.py", ["game.py"])}
        assert "diagnosing" in chosen, goal


def test_diagnosing_stays_out_of_ordinary_feature_work():
    for goal in NOT_BUG_REPORTS:
        chosen = {s.name for s in sk.select(LOADED, "execute", f"{goal} a.py", ["a.py"])}
        assert "diagnosing" not in chosen, goal


def test_the_tautological_anti_pattern_is_named_in_the_verify_skill():
    """The wing failure: a test-file helper reproduced Bird.draw's polygon
    maths and was asserted against hardcoded points. Six tests passed and draw
    was never called."""
    verify = next(s for s in LOADED if s.name == "verifying-your-work")
    assert "autological" in verify.body
    assert "recording double" in verify.body


def test_the_executor_s_skills_block_carries_no_catalogue(tmp_path):
    """The catalogue says "call read_skill(name)", which the executor can no longer do."""
    from agent.approval import Approver
    from agent.phases.execute import _skills_for
    from agent.phases.plan import Step
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        skills=LOADED,
    )
    step = Step(id="s", title="add the requests dependency", intent="i", acceptance="a")
    block = _skills_for(ctx, "execute", step)
    assert block and "read_skill" not in block


def _ctx_with_readme(tmp_path, triggers: str):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    root.mkdir(parents=True)
    (root / "README.md").write_text("# project\n")
    (root / "solution.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        skills=LOADED,
        skill_triggers=triggers,
    )


def test_step_triggers_ignore_the_root_readme_and_the_verify_command(tmp_path):
    """§117: a root README fired documenting-changes, and `uv run pytest` fired uv, on most
    steps."""
    from agent.phases.execute import _skills_for
    from agent.phases.plan import Step

    step = Step(
        id="s",
        title="Implement the solver",
        intent="Write the function",
        acceptance="a",
        files=["solution.py"],
        verify="uv run pytest -q test_solution.py",
    )
    wide = _skills_for(_ctx_with_readme(tmp_path / "a", "workspace"), "execute", step)
    narrow = _skills_for(_ctx_with_readme(tmp_path / "b", "step"), "execute", step)
    assert "## documenting-changes" in wide and "## uv" in wide
    assert "## documenting-changes" not in narrow and "## uv" not in narrow
    assert "## verifying-your-work" in narrow


def test_step_triggers_keep_a_skill_the_step_actually_needs(tmp_path):
    from agent.phases.execute import _skills_for
    from agent.phases.plan import Step

    docs = Step(id="d", title="Update the README", intent="i", acceptance="a", files=["README.md"])
    deps = Step(id="p", title="Add the requests dependency", intent="i", acceptance="a")
    ctx = _ctx_with_readme(tmp_path, "step")
    assert "## documenting-changes" in _skills_for(ctx, "execute", docs)
    assert "## uv" in _skills_for(ctx, "execute", deps)
