"""Trace.say(detail=True): setup/diagnostic narration kept in the trace file
but out of the live conversation, and every call site it applies to.

Before this, a run's visible narration opened with five lines nobody reading
a conversation asks for -- which models are planning/executing/reviewing,
which tools resolved, the workspace path, the goal echoed straight back --
followed by a "=== phase N: ... ===" banner ahead of every phase and a
"trace: <path>" line stapled onto the end. All of it was real narration
(Trace.say), so none of it could be told apart from the run's own actual
account of what it did ("understood as: ...", "-- step 1/3: ...", "review
round 1: clean"). `detail=True` is that distinction, made real: still
recorded in the trace file (still in a saved failure log, which reads every
"log" event regardless), just not printed live or shown in the TUI panel.
"""

from __future__ import annotations

import json

from agent.trace import Trace


def test_a_detail_line_is_not_printed(tmp_path, capsys):
    trace = Trace(str(tmp_path / "t"), echo=True)
    trace.say("goal: build a thing", detail=True)
    trace.close()
    assert capsys.readouterr().out == ""


def test_a_detail_line_still_reaches_the_trace_file(tmp_path):
    trace = Trace(str(tmp_path / "t"), echo=True)
    trace.say("goal: build a thing", detail=True)
    trace.close()
    lines = [json.loads(line) for line in trace.path.read_text().splitlines()]
    log_events = [line for line in lines if line["kind"] == "log"]
    assert log_events == [
        {"ts": log_events[0]["ts"], "kind": "log", "message": "goal: build a thing", "detail": True}
    ]


def test_an_ordinary_line_is_still_printed_and_marked_not_detail(tmp_path, capsys):
    trace = Trace(str(tmp_path / "t"), echo=True)
    trace.say("understood as: build a thing")
    trace.close()
    assert "understood as: build a thing" in capsys.readouterr().out
    lines = [json.loads(line) for line in trace.path.read_text().splitlines()]
    log_events = [line for line in lines if line["kind"] == "log"]
    assert log_events[0]["detail"] is False


def test_echo_off_prints_neither_kind_but_records_both(tmp_path, capsys):
    """A caller with echo=False (the CLI's --quiet, a non-interactive
    embedder) already suppresses everything; detail must not change that."""
    trace = Trace(str(tmp_path / "t"), echo=False)
    trace.say("understood as: build a thing")
    trace.say("goal: build a thing", detail=True)
    trace.close()
    assert capsys.readouterr().out == ""
    lines = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert len([line for line in lines if line["kind"] == "log"]) == 2


def test_a_saved_failure_log_still_includes_detail_lines(tmp_path):
    """faillog.py reconstructs a plain-text log from every "log" event with
    no detail filter -- a saved failure log is exactly where the setup
    context (which models, which tools) belongs, unlike the live view."""
    from agent import faillog

    trace = Trace(str(tmp_path / "traces"), echo=False)
    trace.say("goal: build a thing", detail=True)
    trace.say("understood as: build a thing")
    trace.close()

    saved = faillog.save(
        str(tmp_path / "failed"), "build a thing", "BudgetExceeded: cap reached", trace.path
    )
    assert saved is not None
    text = saved.read_text()
    assert "goal: build a thing" in text
    assert "understood as: build a thing" in text
