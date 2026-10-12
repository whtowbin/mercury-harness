"""Comparing what the tests execute against what the run changed.

The gap this closes: a real run produced five passing tests about the exact
two features that were broken, by setting internal state by hand rather than
running the new code at all. "Tests exist and are green" did not distinguish
that from working software.
"""

from __future__ import annotations

from agent.phases.coverage import (
    DEFAULT_MIN_COVERAGE,
    MIN_LINES_TO_REPORT,
    CoverageReport,
    changed_lines,
    finding_for,
)

# -- which lines a run actually changed -----------------------------------


def test_a_new_file_counts_entirely_as_changed():
    assert changed_lines(None, "a\nb\nc\n") == {1, 2, 3}


def test_an_untouched_file_has_no_changed_lines():
    same = "a\nb\nc\n"
    assert changed_lines(same, same) == set()


def test_an_appended_line_is_the_only_change():
    assert changed_lines("a\nb\n", "a\nb\nc\n") == {3}


def test_a_modified_line_is_reported_at_its_new_position():
    assert changed_lines("a\nb\nc\n", "a\nBBB\nc\n") == {2}


def test_an_insertion_shifts_later_lines_without_marking_them():
    """Only what really changed: a pure insertion must not mark the whole
    remainder of the file as new, or every edit looks like a rewrite."""
    assert changed_lines("a\nb\n", "a\nNEW\nb\n") == {2}


def test_a_deletion_alone_marks_nothing_new():
    assert changed_lines("a\nb\nc\n", "a\nc\n") == set()


# -- the report -----------------------------------------------------------


def test_percentages_are_computed_across_files():
    report = CoverageReport(measured=True, per_file={"a.py": (2, 4), "b.py": (4, 4)})
    assert report.covered == 6
    assert report.total == 8
    assert report.percent == 75.0


def test_an_empty_report_is_vacuously_complete():
    assert CoverageReport().percent == 100.0


def test_a_well_covered_change_produces_no_finding():
    assert finding_for(CoverageReport(measured=True, per_file={"a.py": (9, 10)})) is None


def test_a_barely_covered_change_produces_a_blocker():
    report = CoverageReport(measured=True, per_file={"a.py": (1, 6)}, unexecuted=["a.py"])
    finding = finding_for(report)
    assert finding is not None
    assert finding["severity"] == "blocker"
    assert "a.py" in finding["issue"]
    assert "1 of 6" in finding["issue"]


def test_the_finding_says_to_use_the_real_entry_point():
    report = CoverageReport(measured=True, per_file={"a.py": (0, 8)}, unexecuted=["a.py"])
    finding = finding_for(report)
    assert finding is not None
    assert "entry point" in finding["fix"]
    assert "internal state by hand" in finding["fix"]


def test_the_finding_admits_that_def_lines_inflate_the_count():
    """`def f(...)` executes on import, so a function nothing calls still
    shows a covered line. The threshold is a fraction for that reason, and
    the finding says so rather than overstating its own number."""
    report = CoverageReport(measured=True, per_file={"a.py": (1, 9)}, unexecuted=["a.py"])
    finding = finding_for(report)
    assert finding is not None
    assert "def" in finding["failure"]


def test_the_threshold_is_a_fraction_not_zero():
    """A rule of "no lines covered" would never fire, because of the above."""
    assert 0.0 < DEFAULT_MIN_COVERAGE < 1.0
    assert MIN_LINES_TO_REPORT >= 1


# -- a function inside a well-covered file that no test ever enters -------

from agent.phases.coverage import (  # noqa: E402
    MIN_FUNCTION_LINES,
    functions_at,
    untouched_functions,
)

MODULE = """class Bird:
    def update(self, prefs):
        self.velocity += prefs.gravity
        self.y += self.velocity
        return self.y

    def draw(self, screen):
        points = [(1, 2), (3, 4)]
        screen.polygon(points)
        screen.flip()


def helper(a):
    return a
"""


def test_functions_are_found_with_qualified_names():
    names = [n for n, _, _ in functions_at(MODULE)]
    assert "Bird.update" in names and "Bird.draw" in names and "helper" in names


def test_a_function_span_starts_at_its_first_statement_not_its_def():
    """`def draw(...)` executes when the module is imported, so counting the
    signature line would make every function look entered — the same trap that
    made "zero lines covered" an unreachable threshold."""
    start = next(s for n, s, _ in functions_at(MODULE) if n == "Bird.draw")
    assert MODULE.splitlines()[start - 1].strip().startswith("points =")


def test_a_changed_function_the_tests_never_entered_is_reported():
    """The traced failure: ten runs edited Bird.draw to make a wing appear, six
    tests about the wing passed, and draw was never executed once — the tests
    called a helper in the test file that reimplemented the drawing arithmetic."""
    executable = {3, 4, 5, 8, 9, 10, 14}
    dead = untouched_functions(
        MODULE, touched=executable, executed={3, 4, 5}, executable=executable
    )
    assert dead == ["Bird.draw"]


def test_a_function_the_tests_do_enter_is_not_reported():
    executable = {3, 4, 5, 8, 9, 10}
    assert (
        untouched_functions(
            MODULE, touched=executable, executed={3, 4, 5, 8, 9, 10}, executable=executable
        )
        == []
    )


def test_an_unchanged_function_is_not_reported():
    """A run that never touched draw must not be blocked by draw."""
    executable = {3, 4, 5, 8, 9, 10}
    dead = untouched_functions(MODULE, touched={3, 4, 5}, executed={3, 4, 5}, executable=executable)
    assert dead == []


def test_a_one_line_change_in_an_untested_function_is_not_worth_blocking():
    executable = {3, 4, 5, 8, 9, 10}
    assert untouched_functions(MODULE, touched={8}, executed={3, 4, 5}, executable=executable) == []
    assert MIN_FUNCTION_LINES >= 2


def test_the_finding_survives_a_file_ratio_that_looks_fine():
    """71% of changed lines ran, so the ratio says nothing; the function check
    still blocks. This is the case the average was hiding."""
    report = CoverageReport(
        measured=True, per_file={"game.py": (12, 17)}, dead_functions=["game.py::Bird.draw"]
    )
    assert report.percent > 70
    assert not report.unexecuted
    finding = finding_for(report)
    assert finding is not None
    assert finding["severity"] == "blocker"
    assert "Bird.draw" in finding["issue"]
    assert "reimplements the logic" in finding["failure"]


def test_the_fix_text_offers_a_way_to_test_a_drawing_function():
    report = CoverageReport(
        measured=True, per_file={"game.py": (12, 17)}, dead_functions=["game.py::Bird.draw"]
    )
    finding = finding_for(report)
    assert finding is not None
    assert "stub" in finding["fix"] or "double" in finding["fix"]


def test_a_dead_function_fails_the_gate():
    from agent.phases.finalcheck import FinalCheck

    check = FinalCheck(
        tests_ran=True,
        tests_passed=True,
        coverage=CoverageReport(
            measured=True, per_file={"g.py": (12, 17)}, dead_functions=["g.py::Bird.draw"]
        ),
    )
    assert check.ok is False
    assert "deadfn:g.py::Bird.draw" in check.signature()
