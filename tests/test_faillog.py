"""faillog: a readable, separate record of a run that failed, stopped, or
timed out — built from the same trace every run already writes."""

from __future__ import annotations

import json

from agent import faillog


def write_trace(path, *records):
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


# -- save() ----------------------------------------------------------------


def test_save_does_nothing_when_dest_dir_is_falsy(tmp_path):
    trace_path = tmp_path / "run-x.jsonl"
    write_trace(trace_path, {"kind": "log", "message": "hi"})
    assert faillog.save(None, "goal", "boom", trace_path) is None
    assert faillog.save("", "goal", "boom", trace_path) is None


def test_save_writes_a_summary_and_copies_the_trace(tmp_path):
    dest = tmp_path / "failed-chats"
    trace_path = tmp_path / "traces" / "run-20260101-000000-abcdef.jsonl"
    trace_path.parent.mkdir(parents=True)
    write_trace(
        trace_path,
        {"kind": "log", "message": "goal: build a thing"},
        {"kind": "tool_call", "phase": "execute", "tool": "run_python", "args": {}},
    )

    out = faillog.save(str(dest), "build a thing", "stopped: interrupted by user", trace_path)

    assert out is not None
    assert out.parent == dest
    assert out.suffix == ".md"
    text = out.read_text(encoding="utf-8")
    assert "build a thing" in text
    assert "stopped: interrupted by user" in text
    assert "run_python" in text
    # the trace itself is copied alongside the summary, under its own name
    assert (dest / trace_path.name).is_file()


def test_save_returns_none_rather_than_raising_on_an_unwritable_dest(tmp_path):
    # A file where the dest directory should be: mkdir(parents=True) fails.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    trace_path = tmp_path / "run-x.jsonl"
    write_trace(trace_path, {"kind": "log", "message": "hi"})
    assert faillog.save(str(blocked / "sub"), "goal", "boom", trace_path) is None


# -- transcript() ------------------------------------------------------------


def test_transcript_reconstructs_log_and_tool_call_lines(tmp_path):
    trace_path = tmp_path / "run-x.jsonl"
    write_trace(
        trace_path,
        {"kind": "log", "message": "goal: catalog my books"},
        {"kind": "assistant_text", "phase": "execute", "text": "I'll add a Book class."},
        {
            "kind": "tool_call",
            "phase": "execute",
            "tool": "run_python",
            "args": {"code": "print(1)"},
        },
        {"kind": "llm_call", "model": "inception/mercury-2.5"},  # ignored kind
    )
    text = faillog.transcript(trace_path)
    assert "catalog my books" in text
    assert "I'll add a Book class." in text
    assert "run_python" in text


def test_transcript_skips_malformed_lines_without_crashing(tmp_path):
    trace_path = tmp_path / "run-x.jsonl"
    trace_path.write_text('{"kind": "log", "message": "ok"}\nnot json at all\n\n', encoding="utf-8")
    text = faillog.transcript(trace_path)
    assert "ok" in text


def test_transcript_of_a_run_with_no_events_says_so(tmp_path):
    trace_path = tmp_path / "run-x.jsonl"
    trace_path.write_text("", encoding="utf-8")
    assert "no events" in faillog.transcript(trace_path)


def test_transcript_of_a_missing_trace_does_not_raise(tmp_path):
    assert "unreadable" in faillog.transcript(tmp_path / "does-not-exist.jsonl")
