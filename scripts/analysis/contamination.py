"""Did a django run see the fix before it wrote it? TODO 2.10.

  uv run python scripts/analysis/contamination.py [--rounds s109-c1,s109-c2 ...] [--all]

Every arm of a round works on one shared django worktree whose git object store holds all of
django's history, beside the other arms' logs and a shared /tmp (scripts/swebench/round.py), so
a run can reach the upstream fix by several routes. This does not guess the route. For each
task it takes the distinctive lines the gold patch adds (stripped, at least 20 characters, not
already in the file at the task's base commit), and for each external-agent run (mini, Codex,
Goose: their tool outputs are recorded) reports a gold line that appeared in a tool output
before the run had written it itself. Hidden-test lines (the test patch) are reported apart.

Routes are flagged too, for every run including the harness, whose traces keep a tool call's
arguments but not its output: reading the benchmark's own data (`swebench-data`), searching git
history beyond HEAD, reading another round's or arm's folder, and reading a /tmp file the run
did not write. A flag is a lead to read, not a verdict; a leak line is the evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from agent_actions import ROOT, codex_rows, goose_rows, mini_rows  # noqa: E402

DATA = Path(os.environ.get("SWE_DATA", REPO / "traces" / "swebench-data"))
RESULTS = REPO / "results" / "s94-harness-compare.txt"
RUNS = ROOT / "swebench-runs"
ARMS = (
    "mini-haiku", "mini-glm", "mini-m3", "mini", "aider-glm", "aider", "goose", "codex",
    "glm-only", "glm-plan", "mercury-only",
    # §120's isolated rounds (scripts/swebench/iso.py), longest names first.
    "rel-no-cae", "rel-no-tail", "rel-no-rpc", "rel", "sep30", "fix",
    # scripts/swebench/iso_s126.py's arms (§126-§131). Missing until §131, so §126's harness arms
    # were never read here; tests/test_contamination.py keeps this in step with that file.
    "h", "h-nr", "h-glm", "h-idle", "h-m3", "p-def", "p-sym", "h-new", "h-direct", "h-d2",
    "h-dsym", "h-tc", "h-s55", "sonnet-plan", "sonnet5-plan", "h-sonnet", "h-haiku",
    "t2", "t3", "t3-nog", "n3", "n3-sd", "f2", "f3", "f3-low", "f-haiku55", "f-luna",
)  # fmt: skip

ROUTES = {
    "answer key": re.compile(r"swebench-data/django__|\[.?(test_)?patch.?\]|gold\.patch"),
    "git history": re.compile(
        r"git\s+(?:-C\s+\S+\s+)?(?:log\b[^|;&]*(?:--all|-S|-G|--grep|\.\.|\s[0-9a-f]{7,})"
        r"|show\s+[0-9a-f]{7,}|rev-list|reflog|cherry-pick|branch\s+-a|checkout\s+[0-9a-f]{7,}"
        r"|diff\s+[0-9a-f]{7,}|fetch|log\s+[^|;&]*origin)"
    ),
    # Network: mini's shell is unconfined (iso_agents.py), so a fetch of the fix is possible.
    "web": re.compile(
        r"\b(?:curl|wget)\b|urllib|requests\.(?:get|post)|\bhttpx\b|git\s+(?:clone|fetch|pull)\b"
        r"|(?:api\.)?github\.com|githubusercontent|djangoproject|huggingface|sourcegraph|grep\.app",
        re.I,
    ),
    "harness files": re.compile(r"scripts/swebench|harness-[0-9a-f]{7}/|results/s94"),
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _base_file(base: str, path: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(DATA / "django.git"), "show", f"{base}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout


def gold_lines(patch: str, base: str) -> list[str]:
    """Lines the patch adds that are distinctive: long enough, and not already in the file."""
    out, before = [], ""
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            before = norm(_base_file(base, line[6:]))
        elif line.startswith("+") and not line.startswith("+++"):
            body = norm(line[1:])
            new = len(body) >= 20 and body not in before and re.search(r"[A-Za-z]{3}", body)
            if new and not _in_repo(base, line[1:].strip()):
                out.append(body)
    return list(dict.fromkeys(out))


def _in_repo(base: str, text: str) -> bool:
    """Whether a line already exists anywhere in the repository at the base commit: a line
    new to its file may sit in another one, where a run sees it legitimately."""
    r = subprocess.run(
        ["git", "-C", str(DATA / "django.git"), "grep", "-q", "-F", "-e", text, base],
        capture_output=True,
        check=False,
    )
    return r.returncode == 0


@dataclass
class Gold:
    fix: list[str]
    test: list[str]


def load_gold(task: str) -> Gold:
    d = json.loads((DATA / f"django__django-{task}.json").read_text())
    return Gold(gold_lines(d["patch"], d["base"]), gold_lines(d["test_patch"], d["base"]))


@dataclass
class Run:
    round: str
    arm: str
    task: str
    agent: str
    actions: list[dict] = field(default_factory=list)


def split_dir(name: str) -> tuple[str, str] | None:
    for arm in sorted(ARMS, key=len, reverse=True):  # `h-glm` before `h`
        if name.endswith("-" + arm):
            return name[: -len(arm) - 1], arm
    return None


def external_runs() -> list[Run]:
    runs: list[Run] = []
    for p in sorted(RUNS.glob("*/mini/*.traj.json")):
        parts = split_dir(p.parent.parent.name)
        if parts:
            runs.append(Run(*parts, p.name.split(".")[0], "mini", list(mini_rows(p))))
    # iso_agents.py (§126): <round>-<arm>/out/<task>/traj.json; `<task>.abandoned-*` is skipped.
    for p in sorted(RUNS.glob("*/out/*/traj.json")):
        parts = split_dir(p.parents[2].name)
        if parts and parts[1].startswith("mini") and p.parent.name.isdigit():
            runs.append(Run(*parts, p.parent.name, "mini", list(mini_rows(p))))
    for p in sorted(RUNS.glob("*-codex/out/*/stdout.log")):
        parts = split_dir(p.parents[2].name)
        if parts:
            runs.append(Run(*parts, p.parent.name, "codex", list(codex_rows(p))))
    by_session: dict[str, list[dict]] = {}
    for r in goose_rows():
        by_session.setdefault(r["src"], []).append(r)
    for src, rows in by_session.items():
        m = re.fullmatch(r"goose:s109-(.+)-(\d{4,5})", src)
        parts = split_dir(m.group(1)) if m else None
        if m and parts:
            runs.append(Run(*parts, m.group(2), "goose", rows))
    return runs


def harness_runs() -> list[Run]:
    runs = []
    for p in sorted(RUNS.glob("*/traces/*/run-*.jsonl")):
        parts = split_dir(p.parents[2].name)
        if not parts or not p.parent.name.isdigit():  # iso.py's `<id>.abandoned-*` folders
            continue
        actions = []
        for line in p.open():
            e = json.loads(line)
            if e.get("kind") == "tool_call":
                actions.append(
                    {"kind": e.get("tool"), "text": str(e.get("args", "")), "output": ""}
                )
        runs.append(Run(*parts, p.parent.name, "harness", actions))
    return runs


def _authored(a: dict) -> str:
    return " ".join(str(a.get(k) or "") for k in ("text", "args"))


def _output(raw: object) -> str:
    """mini's tool message is a JSON object as a string; quotes inside it are escaped."""
    text = str(raw or "")
    try:
        return str(json.loads(text).get("output", text))
    except (json.JSONDecodeError, AttributeError):
        return text


_WRITES = re.compile(
    r"sed\s+-i|>\s*[\w./-]+\.py\b|write_text|open\([^)]*['\"]w|\.write\(|git\s+apply|patch\s+-p"
    r"|apply_patch|str_replace|cat\s*>|tee\s|\bcp\s|\bmv\s"
)


def _is_edit(a: dict) -> bool:
    return a.get("kind") in ("edit", "write", "edit_file", "write_file") or bool(
        _WRITES.search(_authored(a))
    )


def leaks(run: Run, lines: list[str]) -> list[tuple[int, str, str, bool]]:
    """(action index, gold line, command, before the run's first edit) for each gold line
    first seen in an output. A sighting after the run's own edit may be its own work, written
    by a script that never spelled out the whole line; one before any edit cannot be."""
    written, found, seen, edited = "", [], set(), False
    for i, a in enumerate(run.actions):
        written += " " + norm(_authored(a))
        out = norm(_output(a.get("output")))
        for g in lines:
            if g not in seen and g in out and g not in written:
                seen.add(g)
                found.append((i, g, norm(str(a.get("text")))[:160], not edited))
        edited = edited or _is_edit(a)
    return found


def routes(run: Run) -> dict[str, list[str]]:
    own = {f"{run.round}/{run.task}", f"{run.round}-{run.arm}/{run.task}"}
    hits: dict[str, list[str]] = {}
    written_tmp: set[str] = set()
    for a in run.actions:
        cmd = _authored(a)
        for name, pat in ROUTES.items():
            if pat.search(cmd):
                hits.setdefault(name, []).append(norm(cmd)[:160])
        for m in re.finditer(r"swebench-runs/([^/\s\"']+)/(\d{4,5})?", cmd):
            where = f"{m.group(1)}/{m.group(2)}" if m.group(2) else m.group(1)
            if where not in own:
                hits.setdefault("other run folders", []).append(norm(cmd)[:160])
        written_tmp |= set(re.findall(r"(?:>|-o|tee|cp \S+)\s*(/tmp/[\w./-]+)", cmd))
        for t in re.findall(
            r"(?:cat|apply|open\(['\"]|source|patch\s+-p\d\s*<)\s*(/tmp/[\w./-]+)", cmd
        ):
            if t not in written_tmp:
                hits.setdefault("someone else's /tmp file", []).append(norm(cmd)[:160])
    return hits


def resolved() -> dict[tuple[str, str, str], bool]:
    out = {}
    for line in RESULTS.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        m = re.match(r"(\d{4,5})\s", parts[2])
        if m:
            out[(parts[0], parts[1], m.group(1))] = "RESOLVED" in parts[2]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", default="", help="comma-separated round names (default: all)")
    ap.add_argument("--all", action="store_true", help="also list runs with nothing to report")
    args = ap.parse_args(argv)
    keep = set(filter(None, args.rounds.split(",")))
    score = resolved()
    golds: dict[str, Gold] = {}
    rows = []
    for run in external_runs() + harness_runs():
        if keep and run.round not in keep:
            continue
        gold = golds.setdefault(run.task, load_gold(run.task))
        fix, test = leaks(run, gold.fix), leaks(run, gold.test)
        rows.append({
            "round": run.round, "arm": run.arm, "task": run.task, "agent": run.agent,
            "resolved": score.get((run.round, run.arm, run.task)),
            "fix_lines": len(gold.fix), "fix_seen": fix, "test_seen": test, "routes": routes(run),
        })  # fmt: skip
    for r in rows:
        if r["fix_seen"] or r["test_seen"] or r["routes"] or args.all:
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
