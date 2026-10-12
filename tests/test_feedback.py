"""agent/feedback.py: .harness/feedback.md, the goal loop's self-audit log."""

from __future__ import annotations

from agent.feedback import FeedbackLog


def test_lands_at_dot_harness_feedback_md(tmp_path):
    log = FeedbackLog(tmp_path)
    assert log.path == tmp_path / ".harness" / "feedback.md"


def test_a_harness_bug_finding_is_recorded(tmp_path):
    log = FeedbackLog(tmp_path)
    written = log.record(
        "build a game",
        1,
        "traces/run-x.jsonl",
        [
            {
                "category": "harness_bug",
                "summary": "review chunking ignored overhead",
                "evidence": "chunk 1/1 ~153216 tokens",
                "proposed_fix": "reserve overhead in batching",
            },
        ],
    )
    assert written == 1
    text = log.path.read_text()
    assert "review chunking ignored overhead" in text
    assert "chunk 1/1 ~153216 tokens" in text
    assert "reserve overhead in batching" in text
    assert "iteration 1" in text
    assert "traces/run-x.jsonl" in text


def test_an_empty_findings_list_writes_nothing(tmp_path):
    log = FeedbackLog(tmp_path)
    written = log.record("goal", 1, "t.jsonl", [])
    assert written == 0
    assert not log.path.exists()


def test_a_finding_with_no_summary_is_skipped(tmp_path):
    log = FeedbackLog(tmp_path)
    written = log.record("goal", 1, "t.jsonl", [{"category": "harness_bug", "summary": "  "}])
    assert written == 0
    assert not log.path.exists()


def test_recording_appends_across_iterations(tmp_path):
    log = FeedbackLog(tmp_path)
    log.record("goal", 1, "t1.jsonl", [{"summary": "first bug"}])
    log.record("goal", 2, "t2.jsonl", [{"summary": "second bug"}])
    text = log.path.read_text()
    assert "first bug" in text and "second bug" in text
    assert text.index("first bug") < text.index("second bug")


def test_a_finding_with_no_evidence_or_fix_still_records_the_summary(tmp_path):
    log = FeedbackLog(tmp_path)
    written = log.record("goal", 1, "t.jsonl", [{"summary": "bare finding"}])
    assert written == 1
    assert "bare finding" in log.path.read_text()


def test_an_unwritable_directory_does_not_raise(tmp_path):
    """Same principle as MemoryStore: this is bookkeeping, never a reason to
    fail a run."""
    target = tmp_path / "file-not-a-dir"
    target.write_text("x")
    log = FeedbackLog(target)
    written = log.record("goal", 1, "t.jsonl", [{"summary": "x"}])  # must not raise
    assert written == 0
