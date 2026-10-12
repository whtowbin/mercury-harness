"""harness trace sql: SQL over trace events, against real files and real DuckDB.

Built after ~790 hand-written JSONL loops across past sessions (DESIGN-LOG §66).
The case that matters is mixed event shapes in one file: DuckDB's own schema
inference gave up on real traces three times in one session.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("duckdb")

from agent.tracetools import sql_connection


def _write(path, events):
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


@pytest.fixture
def traces(tmp_path):
    root = tmp_path / "traces"
    (root / "sweep").mkdir(parents=True)
    _write(
        root / "sweep" / "run-a.jsonl",
        [
            {"ts": 1.0, "kind": "run_start", "goal": "g", "budget": {"max_steps": 20}},
            {"ts": 2.0, "kind": "step_attempt", "step": "s1", "tools": 17, "optional": ["x"]},
            {
                "ts": 3.0,
                "kind": "tool_call",
                "phase": "execute:s1",
                "tool": "edit_files",
                "args": {"edits": "[...]"},
                "encoded": ["edits"],
            },
            {
                "ts": 4.0,
                "kind": "llm_call",
                "phase": "execute:s1",
                "latency_s": 1.5,
                "prompt_tokens": 100,
                "finish_reason": "tool_calls",
            },
            {"ts": 5.0, "kind": "log", "message": "free text", "detail": True},
        ],
    )
    (root / "sweep" / "run-b.jsonl").write_text(
        json.dumps({"ts": 1.0, "kind": "tool_call", "phase": "plan", "tool": "read_file"})
        + "\nnot json at all\n"
    )
    return root


def test_mixed_event_shapes_come_back_typed(traces):
    con = sql_connection(traces)
    rows = con.sql(
        "SELECT kind, tool, latency_s, encoded FROM events WHERE run LIKE '%run-a%' ORDER BY ts"
    ).fetchall()
    assert [r[0] for r in rows] == ["run_start", "step_attempt", "tool_call", "llm_call", "log"]
    assert rows[2][1:] == ("edit_files", None, ["edits"])
    assert rows[3][2] == 1.5


def test_any_other_field_is_reachable_through_data(traces):
    con = sql_connection(traces)
    goal = con.sql("SELECT data->>'goal' FROM events WHERE kind='run_start'").fetchone()
    steps = con.sql("SELECT (data->'budget'->>'max_steps')::INT FROM events WHERE kind='run_start'")
    assert goal == ("g",)
    assert steps.fetchone() == (20,)


def test_a_malformed_line_is_skipped_not_fatal(traces):
    con = sql_connection(traces)
    assert con.sql("SELECT count(*) FROM events WHERE run LIKE '%run-b%'").fetchone() == (1,)


def test_results_join_to_their_traces(traces, tmp_path):
    run_a = str(traces / "sweep" / "run-a.jsonl")
    results = tmp_path / "results.jsonl"
    _write(results, [{"config": "c1", "accuracy": 1.0, "trace_path": run_a}])
    con = sql_connection(traces, [results])
    row = con.sql(
        "SELECT r.config, count(*) FROM events e JOIN results r ON r.trace_path = e.run GROUP BY 1"
    ).fetchone()
    assert row == ("c1", 5)


def test_the_cli_runs_a_query(traces):
    from typer.testing import CliRunner

    from agent.cli import app

    out = CliRunner().invoke(
        app, ["trace", "sql", str(traces), "SELECT count(*) AS n FROM events", "--json"]
    )
    assert out.exit_code == 0, out.output
    assert json.loads(out.output.strip()) == {"n": 6}


def test_results_join_on_file_name_however_the_trace_path_was_written(traces, tmp_path):
    """The bench records a relative trace_path; joining on `run` then matches
    nothing when TRACE_DIR is given another way. `file` does not care."""
    results = tmp_path / "results.jsonl"
    _write(results, [{"config": "c1", "trace_path": "traces/bench/sweep/run-a.jsonl"}])
    con = sql_connection(traces.resolve(), [results])
    on_run = con.sql("SELECT count(*) FROM events e JOIN results r ON r.trace_path = e.run")
    on_file = con.sql(
        "SELECT count(*) FROM events e JOIN results r ON parse_filename(r.trace_path) = e.file"
    )
    assert on_run.fetchone() == (0,)
    assert on_file.fetchone() == (5,)


def _turn(ts, phase, *tools):
    """One model turn: its llm_call, then the tool calls it made, in order."""
    events = [{"ts": ts, "kind": "llm_call", "phase": phase}]
    events += [
        {"ts": ts + 0.01 * (i + 1), "kind": "tool_call", "phase": phase, "tool": t}
        for i, t in enumerate(tools)
    ]
    return events


def test_after_edit_pairs_each_edit_with_its_own_phases_next_turn(tmp_path):
    """Two parallel steps interleave their turns in one trace. Numbering turns per
    run alone would pair s1's edit with s2's turn, not s1's next one."""
    from agent.tracetools import after_edit

    s1, s2 = "execute:s1", "execute:s2"
    _write(
        tmp_path / "run-p.jsonl",
        _turn(1, s1, "edit_file")  # edit alone -> s1's next turn only runs a check
        + _turn(2, s2, "edit_file", "run_tests")  # batched -> s2's next turn finishes
        + _turn(3, s1, "run_tests")
        + _turn(4, s2, "finish_step")
        + _turn(5, s1, "edit_files", "run_tests")  # batched -> s1 re-runs the tests
        + _turn(6, s1, "run_tests")
        + _turn(7, s1, "finish_step")
        + _turn(8, "review", "edit_file")  # not an execute/fix phase: not counted
        + _turn(9, "review", "run_tests"),
    )
    rows = {r["edit_turn"]: r for r in after_edit(tmp_path)}
    assert rows["edit alone"]["turns"] == 1
    assert rows["edit alone"]["pct_next_check_only"] == 100.0
    assert rows["edit + run_tests"]["turns"] == 2
    assert rows["edit + run_tests"]["pct_next_check_only"] == 50.0
    assert rows["edit + run_tests"]["pct_next_finish_only"] == 50.0
    assert after_edit(tmp_path, ("review",))[0]["turns"] == 1


def test_after_edit_cli(tmp_path):
    from typer.testing import CliRunner

    from agent.cli import app

    _write(tmp_path / "run-c.jsonl", _turn(1, "fix:f1", "write_file") + _turn(2, "fix:f1", "grep"))
    out = CliRunner().invoke(app, ["trace", "after-edit", str(tmp_path)])
    assert out.exit_code == 0, out.output
    row = json.loads(out.output.strip())
    assert (row["edit_turn"], row["turns"], row["pct_next_check_only"]) == ("edit alone", 1, 0.0)


def test_latency_counts_retried_time_runaways_and_retry_causes(tmp_path):
    """harness trace latency: latency_s is only the attempt that worked; total_s holds
    the rest (DESIGN-LOG §95). Real files in a real directory, read by DuckDB."""
    from agent.tracetools import latency

    day = 1790600000.0
    _write(
        tmp_path / "run-a.jsonl",
        [
            {
                "ts": day,
                "kind": "llm_call",
                "model": "inception/mercury-2.5",
                "phase": "execute",
                "latency_s": 1.0,
                "total_s": 1.0,
                "attempts": 1,
                "prompt_tokens": 5000,
                "completion_tokens": 100,
                "reasoning_tokens": 50,
                "ttft_s": 0.5,
                "max_gap_s": 0.1,
            },
            {
                "ts": day + 10,
                "kind": "llm_call",
                "model": "inception/mercury-2.5",
                "phase": "review",
                "latency_s": 2.0,
                "total_s": 32.0,
                "attempts": 2,
                "prompt_tokens": 20000,
                "completion_tokens": 41000,
                "reasoning_tokens": 40000,
                "ttft_s": 1.4,
                "max_gap_s": 45.0,
            },
            {
                "ts": day + 11,
                "kind": "llm_retried",
                "model": "inception/mercury-2.5",
                "attempts": 2,
                "total_s": 32.0,
                "latency_s": 2.0,
                "waited_s": 30.0,
                "error": "_Retryable: no content within 30s, only keep-alives [no content]",
            },
            {
                "ts": day + 20,
                "kind": "llm_call",
                "model": "z-ai/glm-5.3-flash",
                "latency_s": 9.0,
                "total_s": 9.0,
                "attempts": 1,
                "prompt_tokens": 100,
                "completion_tokens": 10,
            },
            {
                "ts": day + 30,
                "kind": "llm_retried",
                "model": "inception/mercury-2.5",
                "attempts": 2,
                "total_s": 125.0,
                "latency_s": 3.0,
                "waited_s": 122.0,
                "error": "_Retryable: provider error: {'code': 504, "
                "'message': 'Upstream idle timeout exceeded'}",
            },
        ],
    )
    tables = latency(tmp_path, by="day")
    (calls,) = tables["calls"]
    assert calls["calls"] == 2  # GLM left out by the default model filter
    assert calls["retried_calls"] == 1 and calls["runaway_calls"] == 1
    assert calls["retry_pct"] == round(100 * (33 - 3) / 33, 1)
    assert calls["max_gap_over_30s"] == 1
    causes = {r["cause"]: r for r in tables["retries"]}
    assert set(causes) == {"no first content (30s cut)", "504 idle timeout"}
    assert causes["504 idle timeout"]["median_wait_s"] == 122.0
    bands = {r["prompt_band"]: r["calls"] for r in tables["first_token"]}
    assert bands == {"a <8k": 1, "c 16-32k": 1}
    everything = latency(tmp_path, model="", by="day")["calls"][0]
    assert everything["calls"] == 3


def test_latency_cli_prints_tables_and_json(tmp_path):
    from typer.testing import CliRunner

    from agent.cli import app

    _write(
        tmp_path / "run-b.jsonl",
        [
            {
                "ts": 1790600000.0,
                "kind": "llm_call",
                "model": "inception/mercury-2.5",
                "latency_s": 1.0,
                "total_s": 1.0,
                "attempts": 1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        ],
    )
    runner = CliRunner()
    text = runner.invoke(app, ["trace", "latency", str(tmp_path)])
    assert text.exit_code == 0, text.output
    assert "calls" in text.output and "retries" in text.output and "(none)" in text.output
    rows = runner.invoke(app, ["trace", "latency", str(tmp_path), "--json"])
    assert '"table": "calls"' in rows.output
