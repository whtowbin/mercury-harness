"""Skill loading, trigger matching, and budget-aware selection."""

from __future__ import annotations

import pytest

from agent import skills as skills_module
from agent.skills import Skill, load, render, select

SAMPLE = """---
name: uv
description: Using uv.
triggers: [pyproject.toml, uv, "*.py"]
phases: [execute, fix]
---

# uv

Use `uv add`, never `pip install`.
"""


@pytest.fixture
def skills_dir(tmp_path):
    (tmp_path / "uv.md").write_text(SAMPLE)
    (tmp_path / "ruff.md").write_text(
        "---\nname: ruff\ntriggers: [lint]\nphases: [execute]\n---\n\nRun check_code.\n"
    )
    (tmp_path / "notes.txt").write_text("not a skill")
    (tmp_path / "broken.md").write_text("no frontmatter here")
    return tmp_path


# -- loading --------------------------------------------------------------


def test_loads_skills_with_frontmatter(skills_dir):
    loaded = load(skills_dir)
    assert {s.name for s in loaded} == {"uv", "ruff"}


def test_a_file_without_frontmatter_is_skipped_not_fatal(skills_dir):
    assert all(s.name != "broken" for s in load(skills_dir))


def test_a_missing_directory_yields_nothing():
    assert load("/nonexistent/skills") == []


def test_frontmatter_fields_are_parsed(skills_dir):
    uv = next(s for s in load(skills_dir) if s.name == "uv")
    assert uv.description == "Using uv."
    assert uv.triggers == ["pyproject.toml", "uv", "*.py"]
    assert uv.phases == ["execute", "fix"]
    assert "uv add" in uv.body


def test_multiline_description_is_joined(tmp_path):
    (tmp_path / "s.md").write_text(
        "---\nname: s\ndescription:\n  first line\n  second line\n---\n\nbody\n"
    )
    assert load(tmp_path)[0].description == "first line second line"


# -- the real shipped skills ---------------------------------------------


def test_the_shipped_skills_all_parse():
    shipped = load(skills_module.DEFAULT_SKILLS_DIR)
    assert {s.name for s in shipped} >= {"uv", "ruff", "ty"}
    # A skill with no triggers matches every step, which is only defensible
    # when its phases are narrow enough that "every step" is a small, uniform
    # set — `grilling` applies to clarify, whose entire job is asking well.
    # A trigger-less skill on execute or fix would be a flat tax on every step
    # in the run, which is exactly what "*.py" in all five triggers was.
    broad = {"execute", "fix"}
    for skill in shipped:
        assert skill.body and skill.description
        if not skill.triggers:
            assert skill.phases, f"{skill.name} has neither triggers nor phases"
            assert not (set(skill.phases) & broad), (
                f"{skill.name} has no triggers and runs on {skill.phases}; "
                "it would be injected into every step of the run"
            )


def test_the_shipped_skills_fit_the_default_budget():
    shipped = load(skills_module.DEFAULT_SKILLS_DIR)
    chosen = select(shipped, "execute", "edit app.py and run tests", ["app.py"])
    assert chosen, "a Python step should pull in at least one skill"
    assert sum(s.tokens for s in chosen) <= skills_module.DEFAULT_TOKEN_BUDGET


def test_the_uv_skill_tells_the_model_not_to_install_pytest():
    uv = next(s for s in load(skills_module.DEFAULT_SKILLS_DIR) if s.name == "uv")
    # This was the single most common wasted call in the benchmark.
    assert "pytest" in uv.body and "uv add --dev pytest" in uv.body


# -- trigger matching -----------------------------------------------------


def test_a_word_trigger_matches_step_text():
    skill = Skill(name="uv", triggers=["dependency"])
    assert skill.matches("add a dependency", [])
    assert not skill.matches("write a parser", [])


def test_a_glob_trigger_matches_filenames():
    skill = Skill(name="py", triggers=["*.py"])
    assert skill.matches("", ["main.py"])
    assert not skill.matches("", ["README.md"])


def test_matching_is_case_insensitive():
    assert Skill(name="s", triggers=["Pytest"]).matches("run PYTEST now", [])


def test_a_skill_with_no_triggers_always_matches():
    assert Skill(name="s").matches("anything", [])


def test_phase_gating():
    skill = Skill(name="s", phases=["execute"])
    assert skill.applies_to("execute")
    assert not skill.applies_to("plan")


# -- selection ------------------------------------------------------------


def test_selection_filters_by_phase_and_trigger(skills_dir):
    loaded = load(skills_dir)
    chosen = select(loaded, "fix", "fix the uv config", ["pyproject.toml"])
    # ruff is execute-only, so a fix phase excludes it
    assert [s.name for s in chosen] == ["uv"]


def test_selection_respects_the_token_budget(skills_dir):
    loaded = load(skills_dir)
    assert select(loaded, "execute", "uv lint", ["a.py"], budget_tokens=0) == []


def test_selection_prefers_more_skills_over_bigger_ones():
    small = Skill(name="small", body="x" * 40)
    big = Skill(name="big", body="x" * 4000)
    chosen = select([big, small], "execute", "anything", [], budget_tokens=100)
    assert [s.name for s in chosen] == ["small"]


def test_selection_is_deterministic_in_name_order():
    a = Skill(name="alpha", body="x" * 40)
    z = Skill(name="zulu", body="x" * 40)
    assert [s.name for s in select([z, a], "execute", "x", [])] == ["alpha", "zulu"]


# -- rendering ------------------------------------------------------------


def test_render_is_empty_when_nothing_applies():
    assert render([]) == ""


def test_render_labels_reference_material_not_instructions():
    block = render([Skill(name="uv", body="Use uv add.")])
    assert "# Reference" in block
    assert "not instructions to carry out" in block
    assert "## uv" in block and "Use uv add." in block


# -- trigger matching: word starts, not bare substrings ---------------------
# Measured over 71 distinct real traced goals: substring matching fired `ty`
# on 31 goals of which 30 were words like "party", "empty", "gravity",
# "capacity" and "community", and `ty` is the smallest skill, so smallest-first
# selection spent the budget on it instead of a skill the step needed.


def _skill(triggers: list[str], name: str = "s") -> Skill:
    return Skill(name=name, description="d", triggers=triggers, phases=["execute"], body="body")


def test_a_trigger_does_not_match_inside_a_word():
    assert not _skill(["ty"]).matches("the gravity is empty at the party", [])
    assert not _skill(["error"]).matches("raises ValueError for a negative weight", [])


def test_a_word_start_rule_cannot_save_a_trigger_that_prefixes_a_real_word():
    """`pip` does start the word "pipe", so no matching rule fixes that one —
    which is why the uv skill no longer lists `pip` as a trigger at all."""
    assert _skill(["pip"]).matches("the bird passes a pipe", [])


def test_a_trigger_still_matches_a_plural_or_longer_form():
    """`package` is there to catch "packages", `test` to catch "tests", `bug`
    to catch "bugs" — the suffix is free, only the word start is required."""
    assert _skill(["package"]).matches("do not add packages", [])
    assert _skill(["test"]).matches("run the tests", [])
    assert _skill(["bug"]).matches("two bugs a player can see", [])


def test_a_trigger_matches_the_bare_word_too():
    assert _skill(["ty"]).matches("run ty over it", [])


def test_a_multi_word_trigger_still_matches_the_phrase():
    assert _skill(["not appearing"]).matches("the icons are not appearing", [])


def test_a_glob_trigger_still_matches_filenames_not_text():
    assert _skill(["*.py"]).matches("anything", ["game.py"])
    assert not _skill(["*.py"]).matches("anything", ["notes.md"])


def test_a_skill_with_no_triggers_still_always_matches():
    assert _skill([]).matches("", [])


def test_the_uv_skill_no_longer_fires_on_the_word_pipe():
    """A real, recurring goal shape in a flappy-bird app's traces: the
    workspace's "pipe" pulled 476 tokens of dependency guidance into steps
    about game geometry."""
    uv = next(s for s in load() if s.name == "uv")
    assert not uv.matches("the bird passes a pipe and the score never goes up", [])
    assert uv.matches("add the requests dependency", [])
