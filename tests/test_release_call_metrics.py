"""scripts/release_call_metrics.py: per-call metrics leave, text and app runs never do.

Runs the real script over a real directory of trace files (DuckDB reads them as it would the
release set), so what the parquet holds is what a release would publish.
"""

import importlib.util
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_call_metrics", ROOT / "scripts" / "release_call_metrics.py"
)
assert _spec and _spec.loader
rcm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcm)

SECRET_GOAL = "Fix the bug in /Users/someone/private-project/app.py"
SECRET_ERROR = "provider said: account someone@example.com over quota at /Users/someone"


def _write(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def _bench_run() -> list[dict]:
    return [
        {"ts": 1.0, "kind": "run_start", "goal": SECRET_GOAL, "workspace": "/Users/someone/ws"},
        {
            "ts": 2.0,
            "kind": "llm_attempt_failed",
            "attempt": 1,
            "seconds": 30.4,
            "error": "_Retryable: no content within 30s, only keep-alives",
        },
        {"ts": 2.5, "kind": "llm_attempt_failed", "attempt": 2, "error": SECRET_ERROR},
        {
            "ts": 3.0,
            "kind": "llm_call",
            "phase": "execute:add-accent-test",
            "model": "inception/mercury-2.5",
            "prompt_tokens": 6562,
            "completion_tokens": 3327,
            "reasoning_tokens": 3301,
            "cost_usd": 0.00076,
            "ttft_s": 9.1,
            "finish_reason": "tool_calls",
            "provider": "Inception",
        },
        {"ts": 4.0, "kind": "tool_call", "tool": "write_file", "args": {"path": "/Users/x"}},
    ]


def test_only_allowed_fields_from_benchmark_runs_reach_the_parquet(tmp_path, capsys):
    runs = tmp_path / "runs"
    _write(runs / "s126-dj" / "traces" / "11964" / "run-20261006-1.jsonl", _bench_run())
    # An app run of the interactive agent sits at the root and must not be exported.
    _write(runs / "run-20261006-app.jsonl", _bench_run())
    out = tmp_path / "out" / "calls.parquet"

    assert rcm.main([str(out), f"django={runs}"]) == 0

    rows = duckdb.sql(f"SELECT * FROM read_parquet('{out}') ORDER BY ts").fetchall()
    cols = [d[0] for d in duckdb.sql(f"SELECT * FROM read_parquet('{out}')").description]
    assert len(rows) == 3  # two failed attempts and one call, from the benchmark run only
    assert set(cols) == {
        "suite",
        "sweep",
        "task",
        "run",
        "ts",
        "kind",
        "failure_cause",
        *rcm.CALL_FIELDS,
    }
    record = dict(zip(cols, rows[2], strict=True))
    assert (record["suite"], record["sweep"], record["task"], record["run"]) == (
        "django",
        "s126-dj",
        "11964",
        "run-20261006-1",
    )
    assert record["prompt_tokens"] == 6562 and record["provider"] == "Inception"
    assert record["phase"] == "execute"  # the planner's step id is model text and stays out
    causes = [dict(zip(cols, r, strict=True))["failure_cause"] for r in rows[:2]]
    assert causes == ["silent_call_cut", "other"]
    flat = " ".join(str(v) for r in rows for v in r)
    assert "/Users" not in flat and "someone" not in flat and "private-project" not in flat


def test_a_root_with_no_benchmark_runs_exports_nothing(tmp_path):
    runs = tmp_path / "runs"
    _write(runs / "run-20261006-app.jsonl", _bench_run())
    assert rcm.main([str(tmp_path / "calls.parquet"), f"django={runs}"]) == 1


def test_a_run_in_two_roots_is_counted_once_from_the_first(tmp_path):
    live, archive = tmp_path / "live", tmp_path / "archive"
    _write(live / "s117" / "run-1.jsonl", _bench_run())
    _write(archive / "bench" / "s117" / "run-1.jsonl", _bench_run())
    _write(archive / "s83" / "run-2.jsonl", _bench_run())
    out = tmp_path / "calls.parquet"
    assert rcm.main([str(out), f"fixtures={live}", f"fixtures={archive}"]) == 0
    runs = duckdb.sql(
        f"SELECT run, any_value(sweep), count(*) FROM read_parquet('{out}') "
        "GROUP BY run ORDER BY run"
    ).fetchall()
    assert runs == [("run-1", "s117", 3), ("run-2", "s83", 3)]
