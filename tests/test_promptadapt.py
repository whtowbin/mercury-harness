"""The prompts describe the features a run has on, and nothing else (DESIGN-LOG §104).

Request (2026-10-01): make sure the system prompts do not discourage the new tools by steering
away from old behaviour. With check_after_edit on, "Re-run the tests after your change" asks for
the very call the harness now makes; with edit_blocks on, "edit_files" names a tool the kit does
not carry. Every replacement's old text must still be in its prompt, or it silently stops
applying."""

from __future__ import annotations

import dataclasses

from agent import promptadapt
from agent.config import Settings
from agent.phases.execute import _executor_system
from agent.prompts import EXECUTOR_BATCH_LAST, EXECUTOR_SYSTEM, EXECUTOR_TAIL, FIXER_SYSTEM

OFF = dict(check_after_edit=False, edit_blocks=False, read_assist=False)


def _settings(**flags) -> Settings:
    return dataclasses.replace(Settings(workspace=".", goal="g"), **flags)


def test_every_replacement_still_finds_its_text():
    executor_off = _executor_system(_settings(**OFF))
    for old, _ in promptadapt.AFTER_EDIT_EXECUTOR + promptadapt.READS:
        assert old in executor_off, old
    for old, _ in promptadapt.AFTER_EDIT_FIXER:
        assert old in FIXER_SYSTEM, old
    for old, _ in promptadapt.AFTER_EDIT_TAIL + promptadapt.READS_TAIL:
        assert old in EXECUTOR_TAIL, old
    batch_texts = EXECUTOR_SYSTEM + EXECUTOR_BATCH_LAST + EXECUTOR_TAIL
    for old, _ in promptadapt.EDIT_BLOCKS:
        assert old in batch_texts, old


def test_with_every_feature_off_the_prompts_are_unchanged():
    """Byte-for-byte the prompt earlier sweeps used (control arms must not move)."""
    from agent.prompts import EXECUTOR_ORDER_LEGACY, EXECUTOR_ORDER_TDD

    settings = _settings(**OFF)
    legacy = EXECUTOR_SYSTEM.replace(EXECUTOR_ORDER_LEGACY, EXECUTOR_ORDER_TDD)
    assert _executor_system(settings) == legacy
    assert promptadapt.fixer(FIXER_SYSTEM, settings) == FIXER_SYSTEM


def test_the_defaults_do_not_ask_for_the_test_run_the_harness_makes():
    text = _executor_system(_settings())
    assert "Re-run the tests after your change" not in text
    assert "5. run_tests again" not in text
    assert "the harness runs the tests" in text
    assert "sections=" in text
    fixer = promptadapt.fixer(FIXER_SYSTEM, _settings())
    assert (
        "Re-run run_tests afterwards" not in fixer and "re-runs\nthis finding's own check" in fixer
    )


def test_edit_blocks_on_never_names_the_tool_it_replaced():
    text = _executor_system(_settings(edit_blocks=True))
    assert "edit_files" not in text and "edit_blocks" in text
    batch_last = _executor_system(_settings(edit_blocks=True, executor_batch_last=True))
    assert "edit_files" not in batch_last


def test_a_project_with_a_formatter_is_told_not_to_spend_calls_on_layout(tmp_path):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    assert "formatted automatically" in _executor_system(_settings(), ctx)
    assert "formatted automatically" not in _executor_system(_settings())


def test_the_reminder_never_asks_for_the_test_run_the_harness_makes():
    text = promptadapt.tail(EXECUTOR_TAIL, _settings(edit_blocks=True))
    assert "edit_files" not in text and "then\n  run_tests" not in text
    assert "edit_blocks" in text and "harness\n  runs the tests" in text
    assert promptadapt.tail(EXECUTOR_TAIL, _settings(**OFF)) == EXECUTOR_TAIL


# The planner, the skills block and the reviewer (DESIGN-LOG §108).

OFF_ALL = dict(OFF, types_after_edit=False, review_presupplied_checks=False)


def _skill_bodies() -> str:
    from agent import skills

    return skills.render(skills.load(), phase="execute")


def _skill(name: str) -> str:
    from agent import skills

    body = skills.body_of(skills.load(), name)
    assert body, name
    return body


def test_every_new_replacement_still_finds_its_text():
    from agent.prompts import PLANNER_SYSTEM, REVIEWER_SYSTEM

    for old, _ in promptadapt.PLANNER_TYPES:
        assert old in PLANNER_SYSTEM, old
    for old, _ in promptadapt.SKILLS_TYPES:
        assert old in _skill("ty"), old
    for old, _ in promptadapt.SKILLS_EDIT_BLOCKS + promptadapt.SKILLS_AFTER_EDIT:
        assert old in _skill("verifying-your-work"), old
    assert promptadapt.REVIEWER_CHECKS_OLD in REVIEWER_SYSTEM


def test_with_every_feature_off_planner_skills_and_reviewer_are_unchanged():
    from agent.phases.plan import _planner_system
    from agent.prompts import PLANNER_SYSTEM, REVIEWER_SYSTEM

    settings = _settings(**OFF_ALL)
    assert promptadapt.planner(PLANNER_SYSTEM, settings) == PLANNER_SYSTEM
    assert PLANNER_SYSTEM in _planner_system(settings)
    blocks = _skill_bodies()
    assert promptadapt.skills(blocks, settings) == blocks
    assert promptadapt.skills(blocks, settings, fixer=True) == blocks
    assert promptadapt.reviewer(REVIEWER_SYSTEM, settings, types_supplied=True) == REVIEWER_SYSTEM


def test_type_text_needs_both_switches():
    """New type errors ride the after-edit check, so types_after_edit alone changes nothing."""
    from agent.prompts import PLANNER_SYSTEM

    only_types = _settings(**dict(OFF_ALL, types_after_edit=True))
    assert promptadapt.planner(PLANNER_SYSTEM, only_types) == PLANNER_SYSTEM
    assert promptadapt.skills(_skill("ty"), only_types) == _skill("ty")


def test_the_defaults_describe_when_types_and_tests_run():
    from agent.prompts import PLANNER_SYSTEM

    settings = _settings()
    planner = promptadapt.planner(PLANNER_SYSTEM, settings)
    assert "type checks already run on every write" not in planner
    assert "type-checked automatically" not in planner
    assert planner.count("the harness type-checks after every edit") == 2
    ty = promptadapt.skills(_skill("ty"), settings)
    assert "It is not automatic" not in ty and "runs for you after each turn that edits" in ty
    verify = promptadapt.skills(_skill("verifying-your-work"), settings)
    assert "run only that test by name until it passes" not in verify
    assert "the step's tests for you" in verify


def test_a_fixer_is_told_about_its_findings_check_not_ty_or_the_steps_tests():
    """A fix loop's after-edit check is the finding's own; ty runs there only when the
    finding is a type error (runner._check_after_edit skips types with a custom check)."""
    settings = _settings()
    ty = promptadapt.skills(_skill("ty"), settings, fixer=True)
    assert ty == _skill("ty")
    verify = promptadapt.skills(_skill("verifying-your-work"), settings, fixer=True)
    assert "this finding's own check for you" in verify and "step's tests" not in verify


def test_edit_blocks_on_the_skills_never_name_edit_files():
    verify = promptadapt.skills(_skill("verifying-your-work"), _settings(edit_blocks=True))
    assert "edit_files" not in verify and "edit_blocks" in verify


def test_the_reviewer_with_checks_supplied_is_not_pointed_at_run_tests():
    from agent.prompts import REVIEWER_SYSTEM

    settings = _settings(review_presupplied_checks=True)
    both = promptadapt.reviewer(REVIEWER_SYSTEM, settings, types_supplied=True)
    suite = promptadapt.reviewer(REVIEWER_SYSTEM, settings, types_supplied=False)
    assert "run_tests" not in both and "type checker's report" in both
    assert "run_tests" not in suite and promptadapt.REVIEWER_SUITE in suite


def test_the_executor_prompt_names_inspect_module_only_when_its_kit_has_it():
    """Every step was told to use inspect_module, but only research-first steps get it (§113)."""
    for old, _ in promptadapt.INTROSPECT:
        assert old in EXECUTOR_SYSTEM, old
    default = _executor_system(_settings())
    assert "inspect_module" not in default and "inspect.signature" in default
    research = _executor_system(_settings(research_first=True))
    assert "inspect_module gives you the real signature" in research


def test_every_tool_a_default_prompt_names_is_in_that_phase_s_kit():
    """A prompt naming a tool its kit lacks is advice the model cannot follow: the executor named
    inspect_module for months without having it (0 calls in 59,338, §113)."""
    import re

    from agent.prompts import REVIEWER_SYSTEM
    from agent.tools import execution_registry, planning_registry, review_registry

    every = set(execution_registry().names()) | set(planning_registry().names())
    every |= set(review_registry().names())

    def named(text: str) -> set[str]:
        return {n for n in every if re.search(rf"\b{n}\b", text)}

    executor_kit = set(execution_registry("implement the step").names())
    assert named(_executor_system(_settings())) <= executor_kit
    assert named(promptadapt.fixer(FIXER_SYSTEM, _settings())) <= executor_kit
    assert named(REVIEWER_SYSTEM) <= set(review_registry().names())
