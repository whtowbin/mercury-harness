"""§105's mechanism metrics in bench diagnostics, and the report filling a row's missing
diagnostics from its own trace (DESIGN-LOG §105)."""

from __future__ import annotations

import json

from bench import diagnostics
from bench.cli import _backfill_diagnostics


def call(tool, phase="execute:s", it=0, **args):
    return {"kind": "tool_call", "tool": tool, "phase": phase, "iteration": it, "args": args}


EVENTS = [
    call("read_file", it=0, path="a.py"),
    call("read_files", it=0, paths=["b.py"]),
    call("edit_blocks", it=1, blocks="..."),
    call("edit_file", it=1, path="a.py"),  # same turn: still one turn that edits
    call("edit_file", it=2, path="a.py"),
    {"kind": "tool_call_failed", "tool": "edit_file"},
    {"kind": "tool_arg_parse_error", "tool": "edit_files"},
    {
        "kind": "after_edit_check",
        "result": "PASS — 3 passed\n\nty: 1 type error(s) this run introduced",
    },
    {"kind": "final_check", "new_type_errors": 2},
    {"kind": "final_check", "new_type_errors": 0},  # a later check does not overwrite the first
    call("read_file", phase="plan", path="c.py"),  # planning reads are not counted
]


def test_the_s105_metrics_are_counted():
    d = diagnostics.from_events(EVENTS)
    assert d.exec_reads == 2
    assert d.edit_turns == 2
    assert d.edit_blocks_calls == 1
    assert (d.edit_misses, d.edit_parse_errors) == (1, 1)
    assert d.first_final_type_errors == 1
    assert d.after_edit_type_reports == 1


def test_report_fills_only_the_diagnostics_a_row_lacks(tmp_path):
    trace = tmp_path / "run-1.jsonl"
    trace.write_text("".join(json.dumps(e) + "\n" for e in EVENTS))
    row = {"config": "c", "trace_path": str(trace), "diag_exec_reads": 99}
    _backfill_diagnostics([row])
    assert row["diag_exec_reads"] == 99  # recorded values are kept
    assert row["diag_edit_turns"] == 2 and row["diag_edit_blocks_calls"] == 1


def test_a_word_ending_in_ty_is_not_a_type_report():
    """§105 smoke: pytest output with "empty: " counted as a type report."""
    events = [{"kind": "after_edit_check", "result": "FAIL — 1 failed\nassert queue empty: []"}]
    assert diagnostics.from_events(events).after_edit_type_reports == 0
