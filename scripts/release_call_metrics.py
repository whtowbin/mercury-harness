"""Per-call metrics from harness traces, for release: numbers and categories only, no text.

  uv run python scripts/release_call_metrics.py OUT.parquet SUITE=DIR [SUITE=DIR ...]

Replaces publishing raw traces (decided 9 Oct: the release ships per-call metrics instead). Each
DIR is searched for `run-*.jsonl`; a file directly in DIR (an app run of the interactive agent)
is skipped, so only benchmark sweeps count. Every row is one `llm_call` or `llm_attempt_failed`
event, with an allow-list of fields: no goal, prompt, reply, tool argument, path or error text
leaves this script. A failed attempt keeps only a cause category. The sweep and task names come
from the folders the run sat in, and the run id from its file name.

Example (the release set, 9 Oct):
  uv run python scripts/release_call_metrics.py \\
      ~/.cache/mercury-harness/release-data/calls.parquet \\
      django=~/.cache/mercury-harness/swebench-runs scicode=~/.cache/mercury-harness/scicode-runs \\
      fixtures=traces/bench fixtures=<extracted trace archive>
GAIA runs are never passed in: its terms forbid redistributing per-question data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Numbers and categorical labels only. Adding a field here is a release decision: it must not
# be able to carry text from a prompt, a reply, a tool call or the machine.
CALL_FIELDS = {
    "phase": "VARCHAR",
    "model": "VARCHAR",
    "served_model": "VARCHAR",
    "provider": "VARCHAR",
    "finish_reason": "VARCHAR",
    "prompt_tokens": "BIGINT",
    "completion_tokens": "BIGINT",
    "cached_tokens": "BIGINT",
    "reasoning_tokens": "BIGINT",
    "cost_usd": "DOUBLE",
    "latency_s": "DOUBLE",
    "ttft_s": "DOUBLE",
    "max_gap_s": "DOUBLE",
    "tok_per_s": "DOUBLE",
    "decode_tok_per_s": "DOUBLE",
    "attempts": "BIGINT",
    "total_s": "DOUBLE",
    "attempt": "BIGINT",
    "seconds": "DOUBLE",
}

# A failed attempt's error text can quote a provider message; only its cause leaves.
CAUSE_SQL = r"""
CASE
  WHEN error IS NULL THEN NULL
  WHEN error ILIKE '%no content within%' THEN 'silent_call_cut'
  WHEN error ILIKE '%deadline%' OR error ILIKE '%timed out%' OR error ILIKE '%timeout%'
    THEN 'timeout'
  WHEN regexp_matches(error, '\b(429|500|502|503|504)\b') THEN 'provider_http_error'
  ELSE 'other'
END
"""


def run_parts(path: str, root: str) -> tuple[str, str, str]:
    """(sweep, task, run) from where a run file sat under its root."""
    rel = Path(path).relative_to(root).parts
    folders = [p for p in rel[:-1] if p != "traces"]
    sweep = folders[0] if folders else ""
    task = folders[-1] if len(folders) > 1 else ""
    return sweep, task, Path(path).stem


def collect(
    con: duckdb.DuckDBPyConnection, table: str, suite: str, root: Path, seen: set[str]
) -> int:
    """Load one root's benchmark runs into `table`; return the number of run files.

    A run already loaded from an earlier root (its file name is its id) is skipped, so a
    sweep present both live and in an archive is counted once, from the first root given.
    """
    root = root.expanduser().resolve()
    files = [
        str(p) for p in sorted(root.rglob("run-*.jsonl")) if p.parent != root and p.stem not in seen
    ]
    seen.update(Path(f).stem for f in files)
    if not files:
        return 0
    columns = {"ts": "DOUBLE", "kind": "VARCHAR", "error": "VARCHAR", **CALL_FIELDS}
    # One glob over the root; the join on `files` below drops the app runs at the root itself.
    raw = con.read_json(
        str(root / "**" / "run-*.jsonl"),
        format="newline_delimited",
        columns=columns,
        filename=True,
        ignore_errors=True,
    )
    con.register("raw", raw)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE files "
        "(filename VARCHAR, sweep VARCHAR, task VARCHAR, run VARCHAR)"
    )
    con.executemany(
        "INSERT INTO files VALUES (?, ?, ?, ?)", [(f, *run_parts(f, str(root))) for f in files]
    )
    # A phase is "execute:<step id>"; the step id is model-written text, so only the phase leaves.
    keep = ", ".join(
        "split_part(r.phase, ':', 1) AS phase" if c == "phase" else f"r.{c}" for c in CALL_FIELDS
    )
    # Only module constants and this script's own table name are interpolated; the suite label
    # from the command line goes in as a parameter.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT ? AS suite, f.sweep, f.task, f.run, r.ts, r.kind, {keep},
               {CAUSE_SQL} AS failure_cause
        FROM raw r JOIN files f USING (filename)
        WHERE r.kind IN ('llm_call', 'llm_attempt_failed')
        """,
        [suite],
    )
    con.unregister("raw")
    return len(files)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or any("=" not in a for a in argv[1:]):
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    tables = []
    seen: set[str] = set()
    for i, spec in enumerate(argv[1:]):
        suite, _, folder = spec.partition("=")
        n = collect(con, f"part{i}", suite, Path(folder), seen)
        print(f"{spec}: {n} run files", file=sys.stderr)
        if n:
            tables.append(f"part{i}")
    if not tables:
        return 1
    rel = con.table(tables[0])
    for t in tables[1:]:
        rel = rel.union(con.table(t))
    rel.order("suite, sweep, run, ts").write_parquet(str(out))
    print(
        con.read_parquet(str(out))
        .aggregate("suite, kind, count(*) AS rows, count(DISTINCT run) AS runs", "suite, kind")
        .order("suite, kind")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
