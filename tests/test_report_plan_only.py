"""Plan-only runs carry no score and must not count as zeros in the history view."""

from bench.report import render_history


def _record(task: str, accuracy: float, **extra: object) -> dict:
    return {
        "config": "mercury-only",
        "task": task,
        "accuracy": accuracy,
        "tests_passed": int(accuracy * 13),
        "tests_total": 13,
        "honest": True,
        "calls": 10,
        "started_at": 1.0,
        **extra,
    }


def test_history_leaves_plan_only_runs_out():
    scored = [_record("rename-callers", 1.0) for _ in range(9)]
    unscored = [_record("rename-callers", 0.0, plan_only=True) for _ in range(9)]
    text = render_history(scored + unscored, "mercury-only")
    row = next(line for line in text.splitlines() if line.startswith("rename-callers"))
    assert "   9" in row and "100%" in row, row
    assert "ceiling" in row


def test_history_with_only_plan_only_runs_says_so():
    text = render_history([_record("x", 0.0, plan_only=True)], "mercury-only")
    assert text.startswith("no scored runs of 'mercury-only'")
