"""Where did each django plan say to look? A plan-only measure of localization (§125).

  uv run python scripts/analysis/plan_targets.py PREFIX [PREFIX ...] [--by arm]

One line per run, then a hit count per task and arm. A plan HITS when any step names the
gold location in the field the task's rule reads (`files` and `writes` are paths, `signatures`
are function signatures). 11951 and 11790 have the gold function and its trap in one file, so
they read `signatures`; 11964's fix is in another file from where most plans go, so it reads
paths. `trap` is the misplaced target §119/§120 named. Prefix as in fix_location.py.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from agent.tracetools import sql_connection  # noqa: E402

RUNS = Path.home() / ".cache" / "mercury-harness" / "swebench-runs"
# task -> (field read, gold pattern, trap pattern or None)
RULES: dict[str, tuple[str, str, str | None]] = {
    "11964": ("paths", r"django/db/models/enums\.py", None),
    "11951": ("signatures", r"_batched_insert", r"bulk_create"),
    "11790": ("signatures", r"AuthenticationForm\.__init__|__init__", r"widget_attrs"),
}

QUERY = """
select regexp_extract(run, 'swebench-runs/([^/]+)/traces/([0-9]+)/', 1) rnd,
       regexp_extract(run, 'swebench-runs/([^/]+)/traces/([0-9]+)/', 2) task,
       string_agg(case when kind = 'plan' then (data->'steps')::varchar end, ' ') plan
from events group by 1, 2 order by 1, 2
"""


def step_fields(plan: str) -> dict[str, str]:
    """The plan's `files`+`writes` (paths) and `signatures`, each as one searchable string."""
    out = {"paths": [], "signatures": []}
    for field, key in (("files", "paths"), ("writes", "paths"), ("signatures", "signatures")):
        for chunk in re.findall(rf'"{field}":\s*\[([^\]]*)\]', plan or ""):
            out[key].append(chunk)
    return {k: " ".join(v) for k, v in out.items()}


def gold_paths(task: str) -> list[str]:
    """The source files the task's upstream fix changes (from the private task data), or []."""
    path = REPO / "traces" / "swebench-data" / f"django__django-{task}.json"
    if not path.exists():
        return []
    patch = json.loads(path.read_text())["patch"]
    return sorted({m for m in re.findall(r"^diff --git a/(\S+)", patch, re.M) if "tests/" not in m})


def classify(task: str, plan: str) -> tuple[bool, bool]:
    """(hit, trap) for a plan. A task with a function-level rule uses it; any other task hits
    when a step's `files` or `writes` name a source file its upstream fix changes."""
    if task not in RULES:
        paths = step_fields(plan)["paths"]
        return any(g in paths for g in gold_paths(task)), False
    field, gold, trap = RULES[task]
    text = step_fields(plan)[field]
    return bool(re.search(gold, text)), bool(trap and re.search(trap, text))


def arm_of(rnd: str) -> str:
    return rnd.rsplit("-", 1)[-1]


def main(prefixes: list[str]) -> None:
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for prefix in prefixes:
        for d in sorted(RUNS.glob(f"{prefix}*/traces")):
            for rnd, task, plan in sql_connection(d).sql(QUERY).fetchall():
                hit, trap = classify(task, plan or "")
                print(f"{rnd:34s} {task:6s} hit={int(hit)} trap={int(trap)}")
                t = tally[(task, arm_of(rnd))]
                t[0] += 1
                t[1] += hit
                t[2] += trap
    print("\ntask   arm    plans  hit  trap")
    for (task, arm), (n, h, tr) in sorted(tally.items()):
        print(f"{task:6s} {arm:6s} {n:5d} {h:4d} {tr:5d}")


if __name__ == "__main__":
    main(sys.argv[1:])
