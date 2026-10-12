"""Where did each django run put its fix, where did its plan say to, and what did the
bug-fix evidence check see? DESIGN-LOG §120.

  uv run python scripts/analysis/fix_location.py PREFIX [PREFIX ...]

PREFIX is an iso.py round prefix (`s120-abl`) or any round-directory prefix under the runs
folder (`s107d`, `p1-dj`). One line per run: round directory, task, resolved (from
`results/<PREFIX>.jsonl` or `results/s94-harness-compare.txt`), model calls, the function the
plan targeted, where the final diff put the fix, the evidence check's events, and the run's
test functions. Locations are named for the two tasks whose failures were a misplaced fix:
11790 (`AuthenticationForm.__init__` is gold, `UsernameField.widget_attrs` the trap) and 11951
(`_batched_insert` gold, `bulk_create` the trap). Other tasks print the plan target only.
`--diff` also prints each run's source hunks. Every number comes from the traces through
DuckDB (`agent.tracetools.sql_connection`).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from agent.tracetools import sql_connection  # noqa: E402

RUNS = Path.home() / ".cache" / "mercury-harness" / "swebench-runs"
# (task, label, pattern on the final diff's source hunks, pattern on the plan's targets)
PLACES = {
    "11790": [
        (
            "__init__",
            r"self\.fields\[.username.\]\.widget\.attrs|username_field\.widget\.attrs",
            r"__init__|AuthenticationForm",
        ),
        (
            "widget_attrs",
            r"\+.*attrs\[.maxlength.\] = (str\()?self\.max_length"
            r"|\+\s*.maxlength.: self\.max_length",
            r"widget_attrs",
        ),
    ],
    "11951": [
        ("_batched_insert", r"-\s+batch_size = \(batch_size or max\(", r"_batched_insert"),
        ("bulk_create", r"\+.*connections?(\[self\.db\])?\.ops\.bulk_batch_size", r"bulk_create"),
    ],
}  # fmt: skip


def resolved_by_run(prefix: str) -> dict[tuple[str, str], tuple[bool, int]]:
    out = {}
    path = REPO / "results" / f"{prefix}.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "resolved" in r:
                out[(f"{r['round']}-{r['arm']}", r["task"])] = (bool(r["resolved"]), r["calls"])
        return out
    for line in (REPO / "results" / "s94-harness-compare.txt").read_text().splitlines():
        parts = line.split("\t")
        m = re.match(r"(\d{4,5})\s.*?(\d+) calls", parts[2]) if len(parts) == 3 else None
        if m and parts[0].startswith(prefix):
            out[(f"{parts[0]}-{parts[1]}", m.group(1))] = ("RESOLVED" in parts[2], int(m.group(2)))
    return out


def classify(task: str, diff: str, plan: str) -> tuple[str, str]:
    src = "\n".join(
        line
        for chunk in re.split(r"(?m)^diff --git |^\[(?:modified|new file)\] ", diff)
        if "tests/" not in chunk.split("\n", 1)[0]
        for line in chunk.splitlines()
    )
    places = PLACES.get(task, [])
    fix = ",".join(name for name, pat, _ in places if re.search(pat, src)) or "-"
    target = next((name for name, _, pat in places if re.search(pat, plan)), "") or plan[:40]
    return target, fix


def _source_hunks(diff: str) -> str:
    keep, in_test = [], False
    for line in diff.splitlines():
        if line.startswith(("[modified]", "[new file]", "diff --git", "--- a/", "+++ b/")):
            in_test = "tests/" in line
        elif not in_test and line[:1] in "+-" and line[1:].strip():
            keep.append("    " + line[:120])
    return "\n".join(keep[:30])


QUERY = """
select regexp_extract(run, 'swebench-runs/([^/]+)/traces/([0-9]+)/', 1) rnd,
       regexp_extract(run, 'swebench-runs/([^/]+)/traces/([0-9]+)/', 2) task,
       string_agg(case when kind = 'plan' then (data->'steps')::varchar end, ' ') plan,
       string_agg(case when kind = 'run_diff' then data->>'diff' end, ' ') diff,
       string_agg(case when kind = 'evidence_repro' then 'repro:' || (data->>'outcome')
                       when kind = 'evidence_tests' then 'tests:' || (data->>'verdict')
                       when kind = 'evidence_held' then 'HELD' end, ' ' order by ts) evidence,
       string_agg(case when kind = 'repro' then data->>'outcome' end, ' ') repro
from events group by 1, 2 order by 1, 2
"""


def main(prefixes: list[str]) -> None:
    show = "--diff" in prefixes
    prefixes = [p for p in prefixes if p != "--diff"]
    for prefix in prefixes:
        scored = resolved_by_run(prefix)
        for d in sorted(RUNS.glob(f"{prefix}*/traces")):
            con = sql_connection(d)
            for rnd, task, plan, diff, evidence, repro in con.sql(QUERY).fetchall():
                plan_targets = " ".join(re.findall(r'"signatures": \[([^\]]*)\]', plan or "")) or (
                    plan or ""
                )
                target, fix = classify(task, diff or "", plan_targets)
                ok, calls = scored.get((rnd, task), (None, 0))
                tests = sorted(set(re.findall(r"^\+\s*def (test\w+)", diff or "", re.M)))
                print(
                    f"{rnd:34s} {task:6s} {'RES' if ok else 'not' if ok is False else '?':3s} "
                    f"{calls:3d}c plan={target:16s} fix={fix:24s} repro={repro or '-':10s} "
                    f"ev=[{evidence or ''}] tests={','.join(tests)[:80]}"
                )
                if show:
                    print(_source_hunks(diff or ""))


if __name__ == "__main__":
    main(sys.argv[1:])
