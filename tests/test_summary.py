"""The end-of-run account shown to a person.

Derived entirely from the run summary, so these tests are also a check that
the summary dict carries everything a reader needs.
"""

from __future__ import annotations

from agent.summary import render


def base(**overrides) -> dict:
    summary = {
        "ok": True,
        "changed_files": ["calc.py", "test_calc.py"],
        "steps": {"total": 2, "failed": [], "abandoned": [], "unverified": []},
        "final_check": {
            "ok": True,
            "tests_ran": True,
            "tests_passed": True,
            "tests_summary": "4 passed",
        },
        "plan": {
            "steps": [
                {"id": "s1", "summary": "Added divide()", "files": ["calc.py"]},
                {"id": "s2", "summary": "Added tests", "files": ["test_calc.py"]},
            ]
        },
        "suggested_next_step": "",
        "quick_answer": "",
        "error": "",
    }
    summary.update(overrides)
    return summary


# -- the paragraph --------------------------------------------------------


def test_a_clean_run_says_what_it_did_and_that_it_passed():
    text = render("Add divide to calc.py", base())
    assert "Add divide to calc.py" in text
    assert "2 steps ran" in text
    assert "4 passed" in text


def test_a_failed_run_says_so_rather_than_claiming_success():
    text = render(
        "g",
        base(
            ok=False,
            steps={"total": 2, "failed": ["s2"], "abandoned": [], "unverified": []},
            final_check={
                "ok": False,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "1 failed",
                "new_failures": ["t.py::x"],
                "verify_failures": [],
            },
        ),
    )
    assert "final check failed" in text
    assert "does not yet pass" in text


# -- a goal pasted from a bordered terminal panel --------------------------
# A traced run's goal was copy-pasted from a previous run's own rendered
# failure panel, box-drawing characters and embedded newlines included:
# "Fix these issues:•   step [x] failed its verify command │  │  •   verify
# still fails: ...". The summary that followed read as broken, multi-line
# text: "Goal: Fix these issues:• step [x] failed... │  │ verify still
# fails ." -- a dangling space-then-period from goal.strip()[:160] cutting
# mid-whitespace, on top of the embedded newlines never being collapsed.


def test_a_goal_with_embedded_newlines_renders_as_one_line():
    goal = "Fix these issues:\n•   step [x] failed\n│  │\n•   verify still fails"
    text = render(goal, base())
    opening = text.splitlines()[0]
    assert "\n" not in opening
    assert opening.startswith("Goal: Fix these issues: •")


def test_a_goal_truncated_mid_whitespace_does_not_leave_a_dangling_period():
    """The exact traced artifact: a 160-char cut landing on a run of spaces
    left `trimmed[-1]` a space, which isn't in "?!:", so the punctuation
    branch appended a period after it -- "fails ." instead of "fails."."""
    goal = "word " * 40  # well past the 160-char cutoff, cuts mid-whitespace
    text = render(goal, base())
    opening = text.splitlines()[0]
    assert " ." not in opening
    assert opening.endswith(".")


def test_a_quick_answer_goal_with_embedded_newlines_also_renders_flat():
    goal = "Fix these issues:\n• step failed\n│ verify fails"
    text = render(goal, base(quick_answer="try pandoc"))
    assert "\n" not in text.splitlines()[0]


def test_a_harness_error_is_reported_in_the_paragraph():
    text = render("g", base(ok=False, error="BudgetExceeded: cap reached"))
    assert "did not finish" in text
    assert "BudgetExceeded" in text


def test_a_quick_answer_run_says_nothing_was_built():
    text = render("what is a monad?", base(quick_answer="A monad is..."))
    assert "without planning" in text
    assert "no files were changed" in text


def test_a_run_that_did_nothing_says_nothing_happened():
    text = render(
        "g",
        base(
            changed_files=[],
            plan=None,
            steps={"total": 0, "failed": [], "abandoned": [], "unverified": []},
            final_check=None,
        ),
    )
    assert "Nothing was executed" in text


def test_many_changed_files_are_abbreviated():
    text = render("g", base(changed_files=[f"f{i}.py" for i in range(9)]))
    assert "9 files changed" in text
    assert "…" in text


def test_a_single_file_is_not_pluralised():
    text = render("g", base(changed_files=["only.py"]))
    assert "1 file changed" in text
    assert "1 files" not in text


# -- the bullets ----------------------------------------------------------


def test_built_lists_each_step_with_its_files():
    text = render("g", base())
    assert "[s1] (calc.py) Added divide()" in text
    assert "[s2] (test_calc.py) Added tests" in text


def test_failed_lists_failed_and_abandoned_steps():
    text = render(
        "g",
        base(ok=False, steps={"total": 3, "failed": ["s2"], "abandoned": ["s3"], "unverified": []}),
    )
    assert "step [s2] failed its verify command" in text
    assert "step [s3] was abandoned" in text


def test_an_abandoned_step_is_not_also_listed_as_built():
    text = render(
        "g",
        base(
            ok=False,
            steps={"total": 1, "failed": [], "abandoned": ["s1"], "unverified": []},
            plan={"steps": [{"id": "s1", "summary": "abandoned: too many errors", "files": []}]},
        ),
    )
    built = text.split("built:")[1].split("failed:")[0] if "built:" in text else ""
    assert "abandoned: too many errors" not in built


def test_newly_failing_tests_are_named():
    text = render(
        "g",
        base(
            ok=False,
            final_check={
                "ok": False,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "1 failed",
                "new_failures": ["tests/test_x.py::test_y"],
                "verify_failures": [],
            },
        ),
    )
    assert "tests/test_x.py::test_y" in text


def test_a_clean_run_has_no_failed_section():
    assert "failed:" not in render("g", base())


def test_the_suggested_next_step_appears_under_next():
    text = render("g", base(suggested_next_step="Add type hints to calc.py"))
    assert "next:" in text
    assert "Add type hints to calc.py" in text


def test_pre_existing_failures_are_mentioned_as_deliberately_untouched():
    text = render(
        "g",
        base(
            final_check={
                "ok": True,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "1 failed, 3 passed",
                "pre_existing_failures": ["tests/old.py::test_z"],
                "new_failures": [],
            }
        ),
    )
    assert "already failing before this run" in text
    assert "tests/old.py::test_z" in text


def test_one_pre_existing_failure_reads_as_singular():
    text = render(
        "g",
        base(
            final_check={
                "ok": True,
                "tests_ran": True,
                "tests_passed": False,
                "pre_existing_failures": ["a::b"],
                "new_failures": [],
            }
        ),
    )
    assert "1 test was already failing" in text


def test_a_run_with_no_step_summaries_falls_back_to_changed_files():
    text = render("g", base(plan={"steps": []}))
    assert "changed calc.py" in text


def test_the_summary_is_plain_text_with_no_markup():
    text = render("g", base(suggested_next_step="do the thing"))
    assert "[/" not in text
    assert "[bold]" not in text


def test_a_run_that_left_tests_red_never_claims_the_suite_passes():
    """The reported bug: ok=True meant "nothing got worse", and the summary
    turned that into "The test suite passes (1 error in 0.03s)"."""
    text = render(
        "fix the tests",
        base(
            final_check={
                "ok": True,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "1 error in 0.03s",
                "pre_existing_failures": ["tests/test_flappy.py"],
                "failing_tests": [],
                "new_failures": [],
            }
        ),
    )
    assert "The test suite passes" not in text
    assert "still has failures" in text


def test_a_genuinely_green_suite_still_says_so():
    text = render("g", base())
    assert "The test suite passes (4 passed)" in text


def test_a_plan_only_run_says_it_planned_not_that_steps_ran_and_passed():
    """--stop-after plan once read "3 steps ran, all verified. Everything the plan checked,
    passed" beside "(no changes)": nothing had run (DESIGN-LOG §92.3)."""
    from agent import summary as summary_module

    summary = {
        "steps": {"total": 3, "unverified": ["a", "b", "c"], "failed": [], "abandoned": []},
        "stopped_after": "plan",
        "changed_files": [],
    }
    text = summary_module.render("please sort out those errors?", summary)
    assert "Planned 3 steps and stopped before running them, as asked." in text
    assert "ran" not in text.split("as asked")[0].replace("running", "")
    assert "Everything the plan checked" not in text
    assert "finished without anything confirming them" not in text
