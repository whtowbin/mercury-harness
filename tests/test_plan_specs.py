"""A plan that specifies, and a plan that only goes looking.

The planner reads the codebase, works out where things live, and then throws
that away: its step intents were one or two sentences and the executor started
by re-reading the same files. Measured across the traced runs, the planner read
game.py and its tests, then the executor read tests/test_game.py another 13
times; and 13 of 62 planned steps had acceptance criteria like "files read
successfully" or "test output captured" — steps producing no code, costing 67
model calls and 1.1M prompt tokens to learn things discarded at the step
boundary.

Two changes: a step can now carry the shapes to write and the tests that would
prove it done, and a step whose acceptance admits it changes nothing is dropped.
"""

from __future__ import annotations

from agent.phases.plan import Step, is_reconnaissance

# Acceptance strings copied from real traced plans.
RECON = [
    "Files read successfully.",
    "Test output captured.",
    "File contents identified",
    "Current code state understood",
    "File content reviewed",
    "Files read and content available for analysis.",
    "Dead code markers identified and documented.",
    "Performance reviewed and recommendations documented",
]
REAL = [
    "Items draw at y = 160, 200, 240, 280, 320, 360, 400, 440",
    "Clicking at the visual location activates the input",
    "The suite passes with the new wing test",
    "pytest exits 0 and score increments when a pipe passes",
    "All tests pass or failures are documented",
    "Edge cases identified and tested",
    "The README documents the new flag",
    "Preferences persist across a reset",
]


def _step(acceptance: str, **kw) -> Step:
    return Step(id="s", title="t", intent=kw.pop("intent", "i"), acceptance=acceptance, **kw)


# -- telling the two apart ------------------------------------------------


def test_reconnaissance_acceptances_are_recognised():
    for acceptance in RECON:
        assert is_reconnaissance(_step(acceptance)), acceptance


def test_real_acceptances_survive():
    for acceptance in REAL:
        assert not is_reconnaissance(_step(acceptance)), acceptance


def test_a_step_that_claims_work_was_done_is_kept():
    """ "Edge cases identified and tested" wrote six real tests; "test output
    captured" wrote nothing. The past participle is the tell."""
    assert not is_reconnaissance(_step("Edge cases identified and tested"))
    assert is_reconnaissance(_step("Test output captured."))


def test_an_intent_that_starts_with_read_is_not_enough_to_drop_a_step():
    """Judged on acceptance, never intent: "Read the handler and fix the
    off-by-one" is real work that happens to begin with a looking verb."""
    assert not is_reconnaissance(
        _step(
            "The off-by-one no longer fires at the boundary",
            intent="Read the handler and fix the off-by-one",
        )
    )


def test_an_empty_acceptance_is_not_dropped():
    assert not is_reconnaissance(_step(""))


# -- the plan carries a specification -------------------------------------


def test_a_step_carries_signatures_and_tests():
    step = Step.from_dict(
        {
            "id": "fix-prefs",
            "title": "t",
            "intent": "i",
            "acceptance": "a",
            "files": ["game.py"],
            "signatures": ["Game.show_preferences(self, screen)", "Game.input_items: list"],
            "tests": ["test_prefs_draw_at_distinct_y: each of the 8 items has its own y"],
        },
        0,
    )
    assert step.signatures == ["Game.show_preferences(self, screen)", "Game.input_items: list"]
    assert step.tests and "distinct_y" in step.tests[0]


def test_a_plan_written_before_these_fields_existed_still_loads():
    step = Step.from_dict({"id": "s", "title": "t", "intent": "i", "acceptance": "a"}, 0)
    assert step.signatures == [] and step.tests == []


def test_the_executor_is_shown_them(tmp_path):
    from agent.approval import Approver
    from agent.phases.execute import _step_prompt
    from agent.phases.plan import Plan
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    root.mkdir()
    (root / "game.py").write_text("x = 1\n")
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    step = Step(
        id="s",
        title="t",
        intent="i",
        acceptance="a",
        files=["game.py"],
        signatures=["Game.show_preferences(self, screen)"],
        tests=["test_prefs_y: items draw at 160, 200, 240"],
    )
    prompt = _step_prompt(ctx, Plan(goal="g", draft=[], steps=[step]), step, 0, 12_000)
    assert "Game.show_preferences(self, screen)" in prompt
    assert "test_prefs_y" in prompt


def test_a_step_without_them_adds_no_empty_headings(tmp_path):
    from agent.approval import Approver
    from agent.phases.execute import _step_prompt
    from agent.phases.plan import Plan
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    root.mkdir()
    ctx = ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    step = Step(id="s", title="t", intent="i", acceptance="a")
    prompt = _step_prompt(ctx, Plan(goal="g", draft=[], steps=[step]), step, 0, 12_000)
    assert "Shapes to write" not in prompt
    assert "Tests this step should produce" not in prompt
