"""Suggestions the harness should keep to itself.

Across seventeen traced runs on one project, ten goals were the previous run's
suggestion accepted with a keystroke — because the TUI pre-filled it into the
prompt box, which made the machine's idea the default. Two of those suggestions
were "run the test suite to verify", which phase 4 had already done. Meanwhile
the feature the person kept asking for went unbuilt for four runs.
"""

from __future__ import annotations

from agent.loop import is_chore

# -- work the run should have finished itself -----------------------------


def test_re_running_the_tests_is_not_a_next_goal():
    """Verbatim from the traces."""
    assert is_chore(
        "Run the test suite to verify the updated wing coordinates and "
        "cloud spawning logic pass all existing checks in tests/test_game.py."
    )
    assert is_chore(
        "Run the existing unit tests in tests/test_game.py to ensure the "
        "updated wing positions pass with the new 2-frame animation logic."
    )


def test_verification_phrasings_are_all_caught():
    for text in (
        "Verify that the fix works.",
        "Check the tests still pass.",
        "You should confirm the new behaviour.",
        "Ensure cloud_height is an int.",
        "Make sure the suite passes.",
        "Re-run pytest.",
        "Now validate the preferences pane.",
    ):
        assert is_chore(text), text


def test_a_bulleted_or_numbered_chore_is_still_a_chore():
    assert is_chore("- Run the tests again.")
    assert is_chore("1. Verify the output.")


def test_nothing_obvious_is_treated_as_no_suggestion():
    for text in ("NOTHING OBVIOUS", "Nothing obvious.", "none", "N/A"):
        assert is_chore(text), text


# -- real questions must survive ------------------------------------------


def test_a_statement_is_an_instruction_and_is_withheld():
    """The shape rule. An instruction is work the run should have finished —
    all seventeen suggestions in the traces were statements, and ten of them
    became the next goal."""
    for text in (
        "Update Game.reset() in game.py to reset cloud_spawn_counter to zero.",
        "Fix the syntax error in tests/test_game.py that is preventing collection.",
        "Add validation in game.py to ensure cloud_height remains an integer.",
        "Implement the actual wing animation logic in Bird.update().",
    ):
        assert is_chore(text), text


def test_the_prose_form_of_nothing_obvious_is_withheld():
    for text in (
        "The task is complete as the reset logic already exists and no code changes "
        "were required. No further action is needed.",
        "The goal has been fully achieved with all tests passing.",
        "The task appears genuinely complete as the diff confirms validation was added.",
    ):
        assert is_chore(text), text


def test_a_question_that_asks_permission_for_the_harnesss_own_job_is_withheld():
    for text in (
        "Should I run the test suite to confirm?",
        "Do you want me to verify the wing coordinates?",
        "Shall we check that the tests still pass?",
        "Would you like me to run pytest again?",
    ):
        assert is_chore(text), text


def test_a_genuine_design_question_is_kept():
    assert not is_chore(
        "Should pipes scroll faster once the score passes ten, or should "
        "the gap narrow instead? The goal does not say which."
    )


def test_a_new_feature_suggestion_is_kept():
    assert not is_chore(
        "The cloud spawner could take a seed so runs are reproducible — "
        "worth adding to Game.__init__?"
    )
    assert not is_chore("Should the bird flip direction when falling, or always face right?")


def test_a_question_that_merely_mentions_tests_is_kept():
    """Matching is on the opening clause: "run the tests" as the subject is a
    chore, a question that happens to mention them later is not."""
    assert not is_chore(
        "Should the preferences pane persist to disk between sessions? "
        "That would need tests for the load path too."
    )


def test_an_empty_suggestion_is_a_chore():
    assert is_chore("")


# -- the TUI must not pre-fill --------------------------------------------


def test_the_tui_leaves_the_prompt_box_empty():
    import inspect

    from agent.tui import RunApp

    source = inspect.getsource(RunApp._finish)
    assert 'value=""' in source, "the next-goal box must start empty"
    assert "value=suggestion" not in source
