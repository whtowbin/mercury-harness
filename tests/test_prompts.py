"""Content guards on prompt text where the exact wording is the bug.

Not exhaustive prompt-content testing in general -- these exist for cases a
real trace showed the model getting wrong because of what a prompt did or
didn't say, the same way the rest of this suite is trace-motivated.
"""

from __future__ import annotations

from agent import skills as skills_module
from agent.prompts import PLANNER_SYSTEM, QUICK_ANSWER_SYSTEM, REVIEWER_SYSTEM


def test_quick_answer_tells_the_model_it_has_real_tool_access_in_general():
    """A real traced run: asked "why do you not have the environment, you
    have access to uv", the quick-answer call replied "I don't have an
    active environment because I'm a language model without access to a
    live Python interpreter" -- true of this one tool-free classification
    call, and false of the harness in general, which the model was never
    told. Without this, any question about the agent's own capabilities
    gets a generic, incorrect LLM self-description instead of an accurate
    one, every time, since this call has no tools to demonstrate otherwise."""
    text = QUICK_ANSWER_SYSTEM.lower()
    assert "shell" in text or "tool" in text
    assert "this one check" in text or "this specific" in text or "this call" in text


def test_reviewer_checks_for_documented_numbers_nothing_produced():
    """A real traced run: the reviewer caught a wording mismatch ("six" vs
    "seven" models) but missed that the RMSE values in the same README table
    were unchanged hardcoded placeholders, never produced by running the
    newly-added model -- a following run fixed only the wording and closed
    the finding, leaving the fabricated numbers in place until a person
    caught it. Without an explicit instruction to trace a documented number
    back to a real tool call, the reviewer has no reason to look past
    whether the prose is internally consistent."""
    text = REVIEWER_SYSTEM.lower()
    assert "rmse" in text or "placeholder" in text
    assert "run" in text and ("produced" in text or "confirm" in text)


def test_keeping_tests_fast_teaches_sharing_a_real_computation_not_just_mocking():
    """Mocking guidance alone would tell the model to fake away exactly the
    computation a test is supposed to check for real (an actual RMSE, not a
    stand-in). The complementary lesson -- compute an expensive, deterministic
    result once and share it across the tests that need it, via a session or
    module fixture -- is a distinct technique the mocking section doesn't
    cover on its own."""
    loaded = skills_module.load()
    skill = next(s for s in loaded if s.name == "keeping-tests-fast")
    text = skill.body.lower()
    assert "session" in text or "module" in text
    assert "fixture" in text
    assert "once" in text


def test_no_prompt_names_a_tool_that_does_not_exist():
    """A renamed or removed tool that a prompt still recommends costs a failed
    tool call per attempt, and nothing else in the suite would notice: prompts
    are strings, and the registry they describe is built somewhere else.
    """
    import re

    from agent import prompts as prompts_module
    from agent.tools import _ALL_EXECUTION, planning_registry, review_registry

    real = {tool.name for tool in _ALL_EXECUTION}
    for registry in (planning_registry(), review_registry()):
        real |= {tool.name for tool in registry.tools()}
    # Names that look like this project's tools (snake_case, called or quoted)
    # but are not registered anywhere. Prose about a user's own code —
    # `show_preferences()` in the worked example — has no underscore-joined
    # verb_noun shape shared with a real tool, so it does not trip this.
    known_non_tools = {
        "depends_on",
        "show_preferences",
        "old_string",
        "new_string",
        "expected_result",
        "base_y",
        "per_key",
        "pipe_spacing",
    }
    offenders: dict[str, set[str]] = {}
    for name, value in vars(prompts_module).items():
        if not name.isupper() or not isinstance(value, str):
            continue
        tokens = set(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\(", value))
        tokens |= set(re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", value))
        missing = {t for t in tokens if t not in real and t not in known_non_tools}
        if missing:
            offenders[name] = missing
    assert not offenders, f"prompts name tools that do not exist: {offenders}"


def test_the_planner_is_told_not_to_assert_a_cause_it_cannot_run():
    """Audited runs (DESIGN-LOG §44) planned a specific fix after one
    `read_files`, for goals that reported a symptom and named no defect."""
    from agent.phases.plan import _planner_system

    class On:
        symptom_first_planning = True

    class Off:
        symptom_first_planning = False

    text = _planner_system(On())
    assert "reports a symptom rather than naming the defect" in text
    assert "reproduce before it fixes" in text
    # Off is what `mercury-nosymptom` measures against; it must really be the
    # bare prompt, or the sweep compares a config with itself.
    assert _planner_system(Off()) == PLANNER_SYSTEM
    assert "reports a symptom" not in PLANNER_SYSTEM


def test_the_draft_schema_does_not_contradict_the_planner_prompt():
    """The tool schema said "five words or fewer each" while PLANNER_SYSTEM asks
    for a first note that restates the whole request and allows later notes of a
    few sentences. A model can obey only one; across 283 traced plans the first
    note's median was 14 words and 29% of later notes exceeded five."""
    from agent.tools.control import SUBMIT_PLAN

    description = SUBMIT_PLAN.parameters["properties"]["draft"]["description"]
    assert "five words" not in description
    assert "understood:" in description
    assert "understood:" in PLANNER_SYSTEM


def test_expected_result_does_not_claim_a_check_that_does_not_exist():
    """The description said "a reviewer, not you, checks it against what actually
    happened". Nothing does: a mismatch is logged as `expectation_mismatch` and
    never fails a step. A prompt that describes a check nobody performs teaches
    the model to trust a safeguard that is not there."""
    from agent.tools.control import FINISH_STEP

    text = FINISH_STEP.parameters["properties"]["expected_result"]["description"]
    assert "a reviewer, not you" not in text
    assert "does not fail" in text


# -- draft_style: terse (default) vs prose ---------------------------------


class _Style:
    def __init__(self, draft_style="terse", symptom_first_planning=False):
        self.draft_style = draft_style
        self.symptom_first_planning = symptom_first_planning


def test_the_default_planner_prompt_is_unchanged_by_the_existence_of_prose():
    """`terse` must remain byte-for-byte what it was, or every earlier sweep's
    control arm stops being the control."""
    from agent.phases.plan import _planner_system

    assert _planner_system(_Style()) == PLANNER_SYSTEM


def test_prose_replaces_the_chain_of_draft_paragraph_rather_than_adding_to_it():
    from agent.phases.plan import _planner_system
    from agent.prompts import DRAFT_PROSE, DRAFT_TERSE

    assert DRAFT_TERSE in PLANNER_SYSTEM
    text = _planner_system(_Style("prose"))
    assert DRAFT_PROSE in text
    assert DRAFT_TERSE not in text
    # Two instructions about how to draft, one saying "not full paragraphs", is
    # the contradiction this style exists to remove.
    assert "not full paragraphs" not in text


def test_prose_keeps_the_rest_of_the_planner_prompt_intact():
    from agent.phases.plan import _planner_system
    from agent.prompts import (
        DRAFT_HANDOFF_PROSE,
        DRAFT_HANDOFF_TERSE,
        DRAFT_PROSE,
        DRAFT_PROSE_REWORDINGS,
        DRAFT_TERSE,
    )

    restored = (
        _planner_system(_Style("prose"))
        .replace(DRAFT_PROSE, DRAFT_TERSE)
        .replace(DRAFT_HANDOFF_PROSE, DRAFT_HANDOFF_TERSE)
    )
    for terse, prose in DRAFT_PROSE_REWORDINGS:
        restored = restored.replace(prose, terse)
    assert restored == PLANNER_SYSTEM


def test_prose_never_asks_for_terse_notes():
    """The handoff sentence used to survive the swap, so a planner asked for
    full paragraphs was then told to submit "those terse notes"."""
    from agent.phases.plan import _planner_system

    assert "terse" not in _planner_system(_Style("prose")).lower()


def test_prose_still_asks_for_the_understood_restatement():
    from agent.phases.plan import _planner_system

    assert "prefixed `understood:`" in _planner_system(_Style("prose"))


def test_prose_says_code_in_the_draft_never_reaches_the_executor():
    """The guard against the counter-risk: a wrong body sketched in a draft
    leaking into `intent`, which the executor would follow rather than notice."""
    from agent.prompts import DRAFT_PROSE

    assert "never shown to the executor" in DRAFT_PROSE
    assert "`intent`" in DRAFT_PROSE


def test_the_schema_and_the_prompt_change_together():
    from agent.tools import planning_registry

    def draft_text(style):
        schema = next(
            t for t in planning_registry(style).schemas() if t["function"]["name"] == "submit_plan"
        )
        return schema["function"]["parameters"]["properties"]["draft"]["description"]

    assert "one paragraph per item, full sentences" in draft_text("prose")
    assert "full sentences" not in draft_text("terse")
    assert "understood:" in draft_text("prose") and "understood:" in draft_text("terse")


def test_the_terse_schema_object_is_the_shared_default_not_a_copy():
    from agent.tools.control import SUBMIT_PLAN, submit_plan_tool

    assert submit_plan_tool("terse") is SUBMIT_PLAN
    assert submit_plan_tool("prose") is not SUBMIT_PLAN
    # And building the prose variant must not mutate the shared one.
    assert "full sentences" not in SUBMIT_PLAN.parameters["properties"]["draft"]["description"]


def test_a_long_draft_is_truncated_on_the_terminal_but_kept_whole_in_the_plan():
    from agent.phases.plan import DRAFT_NARRATION_CHARS, Plan, _narrate_understanding

    said: list[str] = []

    class T:
        def say(self, text, **_):
            said.append(text)

    long = "word " * 400
    plan = Plan(goal="g", draft=["understood: fix it", long], steps=[])
    _narrate_understanding(plan, T())
    shown = next(line for line in said if line.startswith("draft:"))
    assert len(shown) < DRAFT_NARRATION_CHARS + 60
    assert "full draft in the trace" in shown
    assert plan.draft[1] == long


def test_every_prose_rewording_finds_its_terse_text_in_the_planner_prompt():
    """A rewording whose terse half has drifted out of PLANNER_SYSTEM silently does
    nothing; §60.2 found the prose arm still told to use "notes" this way."""
    from agent.prompts import DRAFT_PROSE_REWORDINGS

    for terse, _prose in DRAFT_PROSE_REWORDINGS:
        assert terse in PLANNER_SYSTEM, terse


def test_the_prose_planner_prompt_no_longer_speaks_of_notes():
    from types import SimpleNamespace

    from agent.phases.plan import _planner_system

    text = _planner_system(SimpleNamespace(draft_style="prose", plan_probe="off"))
    for phrase in ("first note", "those notes", "that one note", "terse notes", "brief notes"):
        assert phrase not in text, phrase
    assert "Open the draft with your reading of the request" in text
    assert "one paragraph per item" in text


def test_the_terse_planner_prompt_is_unchanged_by_the_rewordings():
    from types import SimpleNamespace

    from agent.phases.plan import _planner_system

    assert _planner_system(SimpleNamespace(draft_style="terse", plan_probe="off")) == PLANNER_SYSTEM
