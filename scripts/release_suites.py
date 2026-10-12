"""Build the gated Hugging Face datasets of benchmark traces, one folder per suite.

  uv run python scripts/release_suites.py {swebench,scicode,gaia} [--out DIR]

Each suite becomes its own dataset (TODO 2.12): GAIA's behind a strict gate (its access terms
allow resharing only inside a gated or private repository on the Hub), SWE-bench's and SciCode's
behind light gates that carry their licenses (MIT and BSD-3 for the django traces, Apache-2.0
for SciCode) and ask that the traces not be trained on. Every file passes through
`scripts/release_traces.py`: paths to placeholders, the person's identity replaced, and a file
that still matches a secret, an email address, a home path or the identity refused. The folder
gets a dataset card (`README.md`, gate settings in its front matter), `LICENSES/` and
`manifest.json`. Nothing is uploaded: read the manifest, then `hf upload <repo> DIR
--repo-type dataset` and turn on access requests (automatic approval) in the dataset settings.

Per-run index: SWE-bench's and SciCode's folders also get `index.jsonl` (one row per run: suite,
arm folder, agent, model, task, outcome, cost, calls, tokens, wall time, the harness trace and the
patch or solution file, the private commit a trace records) and a card with the schema, loading
snippets and the trace event kinds. Per D3 (release plan, phase 5) the harness's own traces are
published as they are, but the other agents (mini-swe-agent, Aider, Goose, Codex) contribute their
result rows and solution files only: their raw logs and transcripts hold machine fingerprints and
third-party prompt text. Each dropped file is listed in the manifest. GAIA is built as before.

The GAIA card names no question, answer or per-question result: a gated dataset's card is
public. `check_card` refuses a card that holds a GAIA question id or answer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release_traces import (  # noqa: E402
    DENIED_NAMES,
    Identity,
    Manifest,
    release_file,
    scan,
    this_machine,
)

RUNS = Path.home() / ".cache" / "mercury-harness"
GOOSE_DB = Path.home() / ".local/share/goose/sessions/sessions.db"
LICENSES = HERE / "release_licenses"
REPO = HERE.parent


@dataclass
class Suite:
    base: str  # folder under RUNS
    runs: str  # regex on run folder names
    globs: list[str]
    licenses: list[str]
    card: str
    deny: tuple[str, ...] = DENIED_NAMES
    goose: str = ""  # regex on Goose session names to export
    extras: list[str] = field(default_factory=list)
    layout: str = ""  # the card's Files section: where this suite's files sit


def _swebench_results(tmp: Path) -> list[tuple[Path, str]]:
    """The score lines of the released rounds, without the OpenRouter balance lines."""
    keep = re.compile(SUITES["swebench"].runs.replace("-[a-z-]+$", ""))
    lines = [
        line
        for line in (REPO / "results" / "s94-harness-compare.txt").read_text().splitlines()
        if keep.match(line.split("\t")[0]) and "\tbalance\t" not in line
    ]
    out = tmp / "rounds.txt"
    out.write_text("\n".join(lines) + "\n")
    return [(out, "results/rounds.txt")]


def _scicode_scores(tmp: Path) -> list[tuple[Path, str]]:
    pat = re.compile(SUITES["scicode"].runs)
    found = []
    for p in sorted((RUNS / "scicode-runs").glob("*.out")):
        if pat.match(p.stem):
            copy = tmp / f"{p.stem}.txt"  # the scrubber takes text by suffix
            shutil.copy(p, copy)
            found.append((copy, f"results/{p.stem}.txt"))
    return found


def _gaia():
    """scripts/gaia/gaia.py by path: a bare `import gaia` can resolve to another module."""
    spec = importlib.util.spec_from_file_location(
        "gaia_runner", REPO / "scripts" / "gaia" / "gaia.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gaia_results(tmp: Path) -> list[tuple[Path, str]]:
    """Per question: level and whether the answer scored right. Inside the gate only."""
    gaia = _gaia()
    qs = gaia.questions()
    found = []
    for run in sorted(
        p for p in (RUNS / "gaia-runs").iterdir() if re.match(SUITES["gaia"].runs, p.name)
    ):
        rows = []
        for goal in sorted(run.glob("*.goal")):
            answer = run / goal.stem / "answer.txt"
            text = answer.read_text().strip() if answer.exists() else None
            ok = text is not None and gaia.question_scorer(text, qs[goal.stem]["answer"])
            rows.append({"task_id": goal.stem, "level": qs[goal.stem]["level"], "right": ok})
        out = tmp / f"{run.name}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        found.append((out, f"results/{run.name}.jsonl"))
    return found


def _goose_sessions(tmp: Path, pattern: str) -> list[tuple[Path, str]]:
    if not pattern or not GOOSE_DB.exists():
        return []
    db = sqlite3.connect(f"file:{GOOSE_DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    found = []
    for s in db.execute("select id, name from sessions order by name"):
        if not re.match(pattern, s["name"] or ""):
            continue
        rows = db.execute("select * from messages where session_id = ? order by id", (s["id"],))
        out = tmp / f"{s['name']}.jsonl"
        out.write_text("".join(json.dumps(dict(r), default=str) + "\n" for r in rows))
        found.append((out, f"goose-sessions/{s['name']}.jsonl"))
    return found


EXTRAS = {
    "swebench_results": _swebench_results,
    "scicode_scores": _scicode_scores,
    "gaia_results": _gaia_results,
}


def collect(suite: Suite) -> list[tuple[Path, str]]:
    """(source, released path) for every run file of the suite, in a stable order."""
    base = RUNS / suite.base
    found = []
    for run in sorted(p for p in base.iterdir() if p.is_dir() and re.match(suite.runs, p.name)):
        for pattern in suite.globs:
            for src in sorted(run.glob(pattern)):
                if src.is_file() and not src.is_symlink():
                    found.append((src, f"traces/{run.name}/{src.relative_to(run)}"))
    return found


def _strip_gaia_ids(out: Path, manifest: Manifest) -> None:
    """GAIA task ids to `<gaia-task>` in a non-GAIA suite: an agent's `ps aux` lists every
    process on the machine, a concurrent GAIA run's workspace path among them."""
    ids = list(_gaia().questions())
    pat = re.compile("|".join(re.escape(i) for i in ids + [i[:8] for i in ids]))
    for entry in manifest.copied:
        path = out / entry["path"]
        text = path.read_text(errors="replace")
        new = pat.sub("<gaia-task>", text)
        if new != text:
            path.write_text(new)
            entry["bytes"] = path.stat().st_size


REPO_URL = "https://github.com/whtowbin/mercury-harness"  # the author's own, credited on purpose


def _common_words() -> set[str]:
    words = Path("/usr/share/dict/words")
    return set(words.read_text().lower().split()) if words.exists() else set()


def check_card(text: str, ident: Identity, gaia_free: bool) -> list[str]:
    """A card may name the repository it comes from; nothing else that identifies the author,
    and for GAIA no question id or answer. A one-word answer that is an ordinary English word
    ("right") is no evidence of GAIA content."""
    problems = scan(text.replace(REPO_URL, "<repo-url>"), ident)
    common = _common_words() if gaia_free else set()
    if gaia_free:
        for tid, q in _gaia().questions().items():
            answer = str(q["answer"]).strip()
            ordinary = answer.lower() in common
            named = re.search(rf"(?<!\w){re.escape(answer.lower())}(?!\w)", text.lower())
            if tid[:8] in text or (len(answer) >= 5 and not ordinary and named):
                problems.append(f"GAIA content ({tid[:8]})")
    return problems


EXTERNAL_AGENTS = ("mini", "aider", "codex", "goose")
DROP_REASON = "external-agent raw log (D3)"
KEPT_EXTERNAL = {
    "solution.py",
    "last_message.txt",
}  # what an external agent contributes besides rows


@dataclass
class SuiteManifest(Manifest):
    """The base manifest plus the files left out on purpose (not refusals: the build still
    succeeds)."""

    dropped: list[dict] = field(default_factory=list)


def agent_of(run: str) -> str:
    """Which agent produced a run folder: an external agent's name is one word of the folder
    name (`s94-r2-mini-haiku`, `s109-sci-aider-glm-r1`); anything else is this harness."""
    words = set(run.split("-"))
    return next((a for a in EXTERNAL_AGENTS if a in words), "harness")


def _label_agent(label: str) -> str:
    """The agent behind a released path (`traces/<run>/...`, `goose-sessions/...`)."""
    parts = label.split("/")
    if parts[0] == "goose-sessions":
        return "goose"
    return agent_of(parts[1]) if parts[0] == "traces" and len(parts) > 1 else "harness"


def split_external(files: Iterable[tuple[Path, str]]):
    """(kept, dropped): an external agent keeps its solution file and final message; every
    other file of its runs (logs, trajectories, session exports) is dropped."""
    kept, dropped = [], []
    for src, label in files:
        raw = _label_agent(label) != "harness" and Path(label).name not in KEPT_EXTERNAL
        (dropped if raw else kept).append((src, label))
    return kept, dropped


def _task_of(label: str) -> str | None:
    m = re.search(r"(?:/mini/|/mini-|/aider-|/out/|/agent-out/)(\d+)(?:[./]|$)", label)
    return m.group(1) if m else None


def _external_facts(src: Path, label: str) -> dict:
    """What a raw log says that the results rows do not: the model and, for Codex, tokens.
    Read before the file is dropped; nothing of the log itself is kept."""
    text = src.read_text(errors="replace")
    facts: dict = {}
    if label.endswith(".traj.json") or label.endswith("/traj.json"):
        try:
            name = json.loads(text)["info"]["config"]["model"]["model_name"]
        except (ValueError, KeyError, TypeError):
            return facts
        facts["model"] = name
    elif m := re.search(r"^Model: (\S+)|Model metadata for `([^`]+)`", text, re.M):
        facts["model"] = m[1] or m[2]
    usage = [
        json.loads(line)["usage"]
        for line in text.splitlines()
        if line.startswith('{"type":"turn.completed"')
    ]
    if usage:
        facts["prompt_tokens"] = sum(u.get("input_tokens", 0) for u in usage)
        facts["cached_tokens"] = sum(u.get("cached_input_tokens", 0) for u in usage)
        facts["completion_tokens"] = sum(u.get("output_tokens", 0) for u in usage)
    if "model" in facts:
        facts["model"] = facts["model"].removeprefix("openrouter/")
    return facts


def _gather_facts(dropped: list[tuple[Path, str]]) -> dict[tuple[str, str], dict]:
    facts: dict[tuple[str, str], dict] = {}
    for src, label in dropped:
        task = _task_of(label)
        parts = label.split("/")
        if task and parts[0] == "traces" and src.suffix in {".json", ".log"}:
            facts.setdefault((parts[1], task), {}).update(_external_facts(src, label))
    return facts


# One row per run. (field, type, meaning): the card's schema table is made from this list.
INDEX_SCHEMA: list[tuple[str, str, str]] = [
    ("suite", "string", "`swebench` (django tasks) or `scicode`."),
    (
        "run_folder",
        "string",
        "The arm or sweep folder: `traces/<run_folder>/` holds the run's files.",
    ),
    (
        "agent",
        "string",
        "`harness` (this repository's agent), `mini`, `aider`, `codex` or `goose`.",
    ),
    (
        "model",
        "string or null",
        "Executor model for the harness (read from the trace's `run_start`); the agent's "
        "model where its log named one; otherwise null.",
    ),
    ("task_id", "string", "SWE-bench: the django ticket number. SciCode: the problem number."),
    (
        "resolved",
        "bool or null",
        "SWE-bench: the task's tests pass. SciCode: every sub-step passed.",
    ),
    ("steps_passed", "int or null", "SciCode: sub-steps passed. Null for SWE-bench."),
    ("steps_total", "int or null", "SciCode: sub-steps in the problem. Null for SWE-bench."),
    (
        "cost_usd",
        "float or null",
        "Spend recorded for the run (results line; the trace's total when the line has none)."
        " Null where the agent's cost was not recorded.",
    ),
    (
        "calls",
        "int or null",
        "Model calls recorded for the run. Codex's count is approximate in SWE-bench rounds. "
        "Null where not recorded.",
    ),
    (
        "prompt_tokens",
        "int or null",
        "Input tokens including cached ones: from the trace's `totals` event, or Codex's own "
        "usage lines.",
    ),
    ("completion_tokens", "int or null", "Output tokens, reasoning included (same sources)."),
    (
        "cached_tokens",
        "int or null",
        "Input tokens served from the provider's cache (same sources).",
    ),
    ("wall_s", "float or null", "Wall time in seconds as the results line recorded it."),
    (
        "trace_file",
        "string or null",
        "Path of the harness's event stream (the last `run-*.jsonl` of the task). Null for "
        "other agents.",
    ),
    (
        "patch_file",
        "string or null",
        "SciCode: the run's final `solution.py`. SWE-bench: null (the diff is in the trace's "
        "`run_diff` event; other agents' patches were not kept per run).",
    ),
    (
        "harness_ref",
        "string or null",
        "Short commit of the (private) repository the trace records in `run_start.harness`. "
        "Null where no trace or no record.",
    ),
    ("harness_dirty", "bool or null", "Whether that checkout had uncommitted changes."),
]
INDEX_FIELDS = [f for f, _, _ in INDEX_SCHEMA]

_SWE_LINE = re.compile(
    r"^(?P<round>\S+)\t(?P<arm>\S+)\t(?P<task>\d+) .*?\s(?P<res>RESOLVED|not)\s+(?P<tail>.*?"
    r"(?P<calls>\d+)\*? calls \$(?P<cost>\d+\.\d+).*)$"
)
_SCI_LINE = re.compile(
    r"^\s*(?P<task>\d+)\s+\S+\s+(?P<res>SOLVED|not)\s+steps (?P<got>\d+)/(?P<of>\d+)\s+"
    r"(?P<min>\d+(?:\.\d+)?) min\s+(?P<calls>\d+) calls \$(?P<cost>\d+\.\d+)"
)


def _blank_row(suite: str, run: str, task: str) -> dict:
    return {f: None for f in INDEX_FIELDS} | {
        "suite": suite,
        "run_folder": run,
        "agent": agent_of(run),
        "task_id": task,
    }


def swebench_rows(text: str) -> list[dict]:
    """Rows from `results/rounds.txt`: resolved, cost, calls and wall time per run."""
    rows = []
    for line in text.splitlines():
        m = _SWE_LINE.match(line)
        if not m:
            continue
        row = _blank_row("swebench", f"{m['round']}-{m['arm']}", m["task"])
        ran = re.search(r"\| ran (\d+)s", line)
        mins = re.match(r"(\d+(?:\.\d+)?) min\b", m["tail"])
        row |= {
            "resolved": m["res"] == "RESOLVED",
            "cost_usd": float(m["cost"]),
            "calls": int(m["calls"]),
            "wall_s": float(ran[1]) if ran else (float(mins[1]) * 60 if mins else None),
        }
        rows.append(row)
    return rows


def scicode_rows(run: str, text: str) -> list[dict]:
    """Rows from one `results/<run>.txt`. An external agent's line carries no time, calls or
    cost (0.0 min, 0 calls, $0.000: not recorded), which become null."""
    rows = []
    for line in text.splitlines():
        m = _SCI_LINE.match(line)
        if not m:
            continue
        row = _blank_row("scicode", run, m["task"])
        recorded = int(m["calls"]) > 0 or float(m["cost"]) > 0
        row |= {
            "resolved": m["res"] == "SOLVED",
            "steps_passed": int(m["got"]),
            "steps_total": int(m["of"]),
            "cost_usd": float(m["cost"]) if recorded else None,
            "calls": int(m["calls"]) if recorded else None,
            "wall_s": round(float(m["min"]) * 60, 1) if recorded else None,
        }
        rows.append(row)
    return rows


def trace_summary(path: Path) -> dict:
    """Model, harness commit and token totals from a released event stream."""
    start: dict = {}
    totals: dict = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("kind") == "run_start" and not start:
            start = event
        elif event.get("kind") == "totals":
            totals = event
    harness = start.get("harness") or {}
    return {
        "model": start.get("executor") or start.get("planner"),
        "harness_ref": harness.get("commit"),
        "harness_dirty": harness.get("dirty"),
        "prompt_tokens": totals.get("prompt_tokens"),
        "completion_tokens": totals.get("completion_tokens"),
        "cached_tokens": totals.get("cached_tokens"),
        "calls": totals.get("calls"),
        "cost_usd": totals.get("cost_usd"),
    }


_TRACE_PATH = re.compile(r"^traces/(?P<run>[^/]+)/traces/(?P<task>[^/]+)/run-[^/]+\.jsonl$")
_SOLUTION_PATH = re.compile(r"^traces/(?P<run>[^/]+)/(?:agent-out/)?(?P<task>[^/]+)/solution\.py$")


def _released_files(manifest: Manifest):
    """(run, task) -> trace path (the last stream) and -> solution path (`<task>/solution.py`
    before the agent's own copy under `agent-out/`)."""
    traces: dict[tuple[str, str], str] = {}
    solutions: dict[tuple[str, str], str] = {}
    for entry in sorted(manifest.copied, key=lambda e: e["path"]):
        path = entry["path"]
        if m := _TRACE_PATH.match(path):
            traces[m["run"], m["task"]] = path
        elif m := _SOLUTION_PATH.match(path):
            solutions.setdefault((m["run"], m["task"]), path)
    return traces, solutions


def _result_rows(name: str, out: Path, manifest: Manifest) -> list[dict]:
    if name == "swebench":
        path = out / "results" / "rounds.txt"
        return swebench_rows(path.read_text()) if path.exists() else []
    rows = []
    for entry in manifest.copied:
        m = re.match(r"^results/(?P<run>[^/]+)\.txt$", entry["path"])
        if m:
            rows += scicode_rows(m["run"], (out / entry["path"]).read_text())
    return rows


def build_index(
    name: str, out: Path, manifest: Manifest, facts: dict[tuple[str, str], dict]
) -> list[dict]:
    """One row per run, from the released results files, the released traces and the facts
    read from the dropped logs. A field that none of them recovers stays null."""
    traces, solutions = _released_files(manifest)
    rows = _result_rows(name, out, manifest)
    for row in rows:
        key = (row["run_folder"], row["task_id"])
        row["patch_file"] = solutions.get(key) if name == "scicode" else None
        row |= {k: v for k, v in facts.get(key, {}).items() if row.get(k) is None}
        if row["agent"] == "harness" and key in traces:
            row["trace_file"] = traces[key]
            seen = trace_summary(out / traces[key])
            row |= {k: v for k, v in seen.items() if row.get(k) is None}
    rows.sort(key=lambda r: (r["run_folder"], int(r["task_id"])))
    return rows


def write_index(rows: list[dict], out: Path, manifest: Manifest) -> None:
    path = out / "index.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest.copied.append({"path": "index.jsonl", "bytes": path.stat().st_size})


TRACE_EVENTS = [
    (
        "run_start",
        "Once per run: the goal text, workspace, planner/executor/reviewer models, budget, "
        "settings and `harness` (`commit`, `dirty`) of the private checkout.",
    ),
    (
        "llm_call",
        "One model call: `phase`, `model`, `prompt_tokens`, `completion_tokens`, "
        "`cached_tokens`, `reasoning_tokens`, `cost_usd`, `latency_s`, `ttft_s`, "
        "`finish_reason`.",
    ),
    (
        "llm_attempt_failed, llm_retried",
        "A call that stalled or errored and the retry: `error`, `attempts`, seconds waited.",
    ),
    (
        "tool_call, tool_done, tool_call_failed",
        "A tool call (`tool`, `args`, `phase`, `iteration`, `batch_size`), its latency and "
        "`ok`, or why it failed.",
    ),
    (
        "plan, schedule",
        "The planner's steps (`steps`, `findings`, `task_kind`) and the order they run in.",
    ),
    (
        "step_attempt, step_verify, final_verify",
        "An execution step starting (`step`, `reasoning`, tool count), and the command that "
        "checked it with `passed` and its output.",
    ),
    (
        "assistant_text, tail",
        "The model's own text in a turn and the reminder appended to its context.",
    ),
    (
        "edit_file, write_file, run_python",
        "File changes (`path`, `added`, `removed`, a preview) and Python runs (`exit_code`).",
    ),
    (
        "run_tests, after_edit_check, final_tests, final_check",
        "Test runs (`exit_code`, `summary`, failing tests), the check after each edit and the"
        " end-of-run verdict (`ok`, new failures, coverage).",
    ),
    (
        "check_code, review_result, fix_result, test_audit",
        "Lint and type checks, the reviewer's `verdict` and `findings`, fixes, and the audit "
        "of the tests the run added.",
    ),
    (
        "checkpoint, run_diff",
        "A git checkpoint (`sha`, `ref`) and the run's final `diff` with the `files` it touches.",
    ),
    (
        "cache_stats, session_summary, totals",
        "Cache `hits`/`misses`, a one-paragraph summary, and the run's totals: tokens, "
        "`cost_usd`, `calls`, `latency_s`, per-model breakdown.",
    ),
    ("log", "Console lines (`message`; `detail` true for the ones the console hides)."),
    (
        "web_denied, loop_exhausted, no_progress_nudge, batching_nudge",
        "Harness interventions: a fetch blocked by the run's deny list, a tool loop that ran "
        "out of iterations, nudges to the model.",
    ),
]


def _table(header: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


CARD_SECTIONS = """
## Files

@LAYOUT@
- `index.jsonl`: one row per run (schema below).
- `manifest.json`: every file included (`copied`), refused by the scrubber (`refused`) and left
  out on purpose (`dropped`, reason "external-agent raw log (D3)").
- `LICENSES/`: license texts.

Only this harness's own traces are published whole. For mini-swe-agent, Aider, Goose and Codex
CLI the dataset holds each run's result row in the index and results files (and a solution file
or final message where the agent wrote one), not their logs or transcripts: those hold
machine-specific output and other projects' prompt text.

## index.jsonl schema

@SCHEMA@

A field the sources do not record is null. Where a harness trace exists, tokens, model and
`harness_ref` come from it.

## Loading

With DuckDB (a trace's events joined to its index row; paths are relative to the dataset root):

```python
import duckdb

print(duckdb.sql('''
    select i.run_folder, i.task_id, i.resolved, e.kind, count(*) as n
    from read_json('traces/*/traces/*/run-*.jsonl', format='newline_delimited', filename=true,
                   ignore_errors=true, columns={kind: 'VARCHAR'}) e
    join read_json('index.jsonl') i on e.filename = i.trace_file
    where i.agent = 'harness'
    group by all order by n desc limit 20
'''))
```

With `datasets` (the index; open a trace with `load_dataset("json", data_files=<trace_file>)`):

```python
from datasets import load_dataset

index = load_dataset("json", data_files="index.jsonl", split="train")
# from the Hub, after accepting the gate: data_files="hf://datasets/<owner>/<dataset>/index.jsonl"
harness_runs = index.filter(lambda r: r["agent"] == "harness")
```

## Trace events

Each line of a `run-*.jsonl` is one JSON object with a `ts` (Unix seconds) and a `kind`. The main
kinds:

@EVENTS@

The harness's `harness trace report|sql` commands read these files unchanged. Traces hold the
event log, not full prompts or tool outputs.
"""


def render_sections(layout: str) -> str:
    schema = _table(("field", "type", "meaning"), [(f"`{f}`", t, m) for f, t, m in INDEX_SCHEMA])
    events = _table(("kind", "holds"), [(f"`{k}`", v) for k, v in TRACE_EVENTS])
    return (
        CARD_SECTIONS.replace("@LAYOUT@", layout.strip("\n"))
        .replace("@SCHEMA@", schema)
        .replace("@EVENTS@", events)
    )


def build(name: str, out: Path, ident: Identity) -> SuiteManifest:
    suite = SUITES[name]
    manifest = SuiteManifest()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        kept, dropped = split_external(collect(suite) + _goose_sessions(tmp, suite.goose))
        facts = _gather_facts(dropped)
        manifest.dropped = [{"path": label, "reason": DROP_REASON} for _, label in dropped]
        for extra in suite.extras:
            kept += EXTRAS[extra](tmp)
        for src, label in kept:
            release_file(src, out / label, ident, manifest, label, suite.deny, redact_emails=True)
    if name != "gaia":
        _strip_gaia_ids(out, manifest)
        write_index(build_index(name, out, manifest, facts), out, manifest)
    (out / "LICENSES").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "LICENSE", out / "LICENSES" / "mercury-harness-Apache-2.0.txt")
    for lic in suite.licenses:
        shutil.copy(LICENSES / lic, out / "LICENSES" / lic)
    card = suite.card.format(
        files=len(manifest.copied), refused=len(manifest.refused), repo_url=REPO_URL
    )
    if name != "gaia":
        card += render_sections(suite.layout)
    problems = check_card(card, ident, gaia_free=name == "gaia")
    if problems:
        raise SystemExit(f"card refused: {problems}")
    (out / "README.md").write_text(card)
    manifest.write(out)
    return manifest


CARD_SWEBENCH = """---
license: other
license_name: mixed-mit-bsd-3-clause-apache-2.0
license_link: LICENSES/
pretty_name: mercury-harness SWE-bench (django) traces
tags: [agent-trajectories, swe-bench, coding-agents, diffusion-llm]
gated: true
extra_gated_heading: "Access the django (SWE-bench) agent traces"
extra_gated_prompt: >-
  These traces contain SWE-bench task text (MIT License), django source code (BSD 3-Clause) and
  the agents' own output (Apache-2.0). The license texts are in LICENSES/. Please keep them with
  any copy, and please do not train models on these traces: they hold solutions to benchmark
  tasks.
extra_gated_fields:
  I will keep the license notices in LICENSES/ with any copy or derivative: checkbox
  I will not use these traces to train a model that is evaluated on SWE-bench: checkbox
extra_gated_button_content: "Agree and access"
---

# mercury-harness: django (SWE-bench) agent traces

Agent runs on ten django tasks from SWE-bench, behind DESIGN-LOG §94, §96 and §109 of the
[mercury-harness]({repo_url}) repository: the same model in
several agents (this harness, mini-swe-agent, Aider, Goose, Codex CLI) and the same harness with
two models (Inception's Mercury 2.5 and GLM 5.3 Flash), three rounds each.

The files and the index schema are described below.

**Known issues, read before using the scores.** DESIGN-LOG §118 voids 11 resolved runs that saw
the upstream fix (from django's history in the worktree, another task's worktree, the task data
or a shared worktree that still held the gold patch): mini-swe-agent with Mercury on 11790 and
11999 in s94-r2 and 11999 in p1-djm-r2; mini-swe-agent with GLM on 11790 in s109-c1 and
s109-c3 and 11815 in s109-c3; Codex on 11885 and 11999 in s109-c1; Goose on 11885 in s109-c1
and 11999 in s109-c3; Aider with GLM on 11885 in s109-c1. `rounds.txt` keeps the scores as
recorded. `p1-dj-r1-mini` is a dropped scout at an unequal time limit (§96.3).

**Redaction.** Paths are rewritten to placeholders (`<runs>/`, `<repo>/`, `<home>/<redacted>`),
the author's identity and every email address are replaced, and command outputs that listed
the author's home folder were blanked before release (`[redacted: ...]`). {files} files are
included; {refused} were refused
by the scrubber and are listed in `manifest.json`.

"""

CARD_SCICODE = """---
license: apache-2.0
pretty_name: mercury-harness SciCode traces
tags: [agent-trajectories, scicode, coding-agents, diffusion-llm]
gated: true
extra_gated_heading: "Access the SciCode agent traces"
extra_gated_prompt: >-
  These traces contain SciCode problems (Apache-2.0, scicode-bench/SciCode) and the agents'
  solutions to them (Apache-2.0). The license texts are in LICENSES/. Please keep them with any
  copy, and please do not train models on these traces: they hold solutions to benchmark
  problems.
extra_gated_fields:
  I will keep the license notices in LICENSES/ with any copy or derivative: checkbox
  I will not use these traces to train a model that is evaluated on SciCode: checkbox
extra_gated_button_content: "Agree and access"
---

# mercury-harness: SciCode agent traces

Runs on the SciCode development split (15 problems, 48-49 sub-steps), behind DESIGN-LOG §96,
§109 and §119 of the [mercury-harness]({repo_url})
repository: this harness with Mercury 2.5 and with GLM 5.3 Flash, mini-swe-agent, Aider, Goose
and Codex CLI with Mercury, and the research-first arms.

The files and the index schema are described below.

Problem 70 was lost to a provider cut in the first GLM runs (scored zero); Codex wrote no
solution file in 11 of 15 problems (DESIGN-LOG §109.5).

**SciCode** is by Tian et al., [scicode-bench/SciCode](https://github.com/scicode-bench/SciCode),
Apache-2.0; its problem text appears in these traces unchanged.

**Redaction.** Paths are rewritten to placeholders, the author's identity and every email
address are replaced, and command outputs that listed the author's home folder were blanked
before release. {files} files are included; {refused} were refused by the scrubber and are
listed in `manifest.json`.
"""

CARD_GAIA = """---
license: other
license_name: gated-benchmark-content
license_link: LICENSES/
pretty_name: mercury-harness GAIA traces
tags: [agent-trajectories, gaia, diffusion-llm]
gated: true
extra_gated_heading: "Access the GAIA agent traces"
extra_gated_prompt: >-
  These traces contain questions from the GAIA validation set (gaia-benchmark/GAIA), which may
  only be shared inside gated or private repositories on the Hugging Face Hub. By requesting
  access you agree not to reshare these files, or GAIA questions and answers derived from them,
  outside a gated or private repository, and not to publish them in a crawlable format.
extra_gated_fields:
  I have accepted the GAIA dataset's terms on Hugging Face: checkbox
  I will not reshare these traces or their GAIA content outside a gated repository: checkbox
  Intended use:
    type: select
    options: [Research, Education, Other]
extra_gated_button_content: "Agree and access"
---

# mercury-harness: GAIA agent traces

Two runs of this harness over the 127 text-only questions of the GAIA validation set, with
Inception's Mercury 2.5 and Brave search, behind DESIGN-LOG §96 and §119 of the
[mercury-harness]({repo_url}) repository: the release
candidate (`p1-gaia-rc`) and the 30 September version run the same day as a control
(`p1-gaia-g7`).

- `traces/<run>/`: the harness's event streams (`traces/<task>/run-*.jsonl`), console logs and
  each question's `answer.txt`.
- `results/<run>.jsonl`: per question, its level and whether the answer scored right.

This card names no question, answer or per-question result, because a gated dataset's card is
public. GAIA's content stays inside the gate, under GAIA's own terms, which you accepted to
reach it.

**Redaction.** Paths are rewritten to placeholders and the author's identity and every email
address are replaced. {files} files are included; {refused} were refused by the scrubber and are
listed in `manifest.json`.
"""

SUITES: dict[str, Suite] = {
    "swebench": Suite(
        base="swebench-runs",
        runs=r"^(s94-r[123]|p1-dj-r[123]|p1-djm-r[123]|s109-c[123])-[a-z-]+$",
        globs=[
            "traces/*/run-*.jsonl",
            "*.log",
            "mini/*.traj.json",
            "out/*/stdout.log",
            "out/*/last_message.txt",
        ],
        licenses=["SWE-bench-MIT.txt", "django-BSD-3-Clause.txt"],
        card=CARD_SWEBENCH,
        goose=r"^s109-s109-c[123]-goose-\d+$",
        extras=["swebench_results"],
        layout="""
- `traces/<run_folder>/traces/<task_id>/run-*.jsonl`: this harness's event streams.
- `traces/<run_folder>/<task_id>.log`: this harness's console log.
- `traces/<run_folder>/out/<task_id>/last_message.txt`: Codex's final message.
- `results/rounds.txt`: one score line per run (round, arm, task, resolved or not, cost, calls,
  diff size, time).
""",
    ),
    "scicode": Suite(
        base="scicode-runs",
        runs=r"^(p1-sci-r[123]|s109-sci-[a-z]+(-[a-z]+)?(-r[123])?|s109-rf-[a-z]+-[a-z]+-r[123])$",
        globs=[
            "traces/*/run-*.jsonl",
            "*.log",
            "*/solution.py",
            "agent-out/*/traj.json",
            "agent-out/*/*.log",
            "agent-out/*/solution.py",
        ],
        licenses=["SciCode-Apache-2.0.txt"],
        card=CARD_SCICODE,
        goose=r"^s109-s109-sci-goose(-r\d)?-\d+$",
        extras=["scicode_scores"],
        layout="""
- `traces/<run_folder>/traces/<task_id>/run-*.jsonl`: this harness's event streams.
- `traces/<run_folder>/<task_id>.log`: this harness's console log.
- `traces/<run_folder>/<task_id>/solution.py`: the run's final solution (also for the other
  agents; `agent-out/<task_id>/solution.py` is the agent's own copy).
- `traces/<run_folder>/agent-out/<task_id>/last_message.txt`: Codex's final message, where kept.
- `results/<run_folder>.txt`: SciCode's per-problem scores (solved, sub-steps passed).
""",
    ),
    "gaia": Suite(
        base="gaia-runs",
        runs=r"^p1-gaia-(rc|g7)$",
        globs=["traces/*/run-*.jsonl", "*.log", "*/answer.txt"],
        licenses=[],
        card=CARD_GAIA,
        deny=tuple(d for d in DENIED_NAMES if d != "gaia"),
        extras=["gaia_results"],
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("suite", choices=sorted(SUITES))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    out = args.out or RUNS / f"release-{args.suite}"
    if out.exists() and any(out.iterdir()):
        print(f"{out} is not empty; choose a fresh directory", file=sys.stderr)
        return 2
    manifest = build(args.suite, out, this_machine())
    total = sum(f["bytes"] for f in manifest.copied)
    print(
        f"{args.suite}: copied {len(manifest.copied)} files ({total / 1e6:.1f} MB), "
        f"refused {len(manifest.refused)}"
    )
    for r in manifest.refused[:20]:
        print("  refused", r["path"], "-", r["reason"])
    print(f"folder: {out}")
    return 0 if manifest.ok else 1


if __name__ == "__main__":
    sys.exit(main())
