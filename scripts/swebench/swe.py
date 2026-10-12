"""SWE-bench Verified (django) spike runner: setup, validate, run, score.

Usage:  python3 swe.py setup|validate|run|score ROUND ID [ID ...]
ROUND names a directory of worktrees (ROUND/<id>); IDs are like 11790.
Nothing here writes to the harness repository.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runlimits  # noqa: E402  scripts/runlimits.py, the run time limit (DESIGN-LOG §96.2)

# Each harness run's wall limit: about 1.5x the slowest django run on record (18.5 min, §107).
RUN_TIMEOUT_S = runlimits.env_seconds("SWE_RUN_TIMEOUT_S", 1500)
# Task JSONs (they hold the gold patches), goal texts and a bare django clone live outside git.
DATA = Path(os.environ.get("SWE_DATA", REPO / "traces" / "swebench-data"))
# Each round's workspaces are full django checkouts with a .venv: kept out of the repo.
RUNS = Path(
    os.environ.get("SWE_RUNS", Path.home() / ".cache" / "mercury-harness" / "swebench-runs")
)
HERE = Path(__file__).resolve().parent
HARNESS = os.environ.get("SWE_HARNESS", str(REPO / ".venv" / "bin" / "harness"))
# Where a django task's answer lives online: the fix commit and PR, later releases' source,
# the ticket and the newer docs (the workspace already holds the docs at the base version),
# and SWE-bench itself. DESIGN-LOG §79.1 found two fixes fetched from these; §80 built --web-deny.
WEB_DENY = [
    "github.com/django/django",
    "githubusercontent.com/django/django",
    "djangoproject.com",
    "swe-bench",
    "swebench",
    "princeton-nlp",
]
TEST_COMMAND = (
    "python tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 {targets}"
)
DEPS = ["asgiref>=3.2,<4", "pytz", "sqlparse>=0.2.2"]

_RESULT = re.compile(r"^(?:.* \.\.\. )?(.*?) \.\.\. (ok|FAIL|ERROR|skipped|expected failure)")
_NAME = re.compile(r"^(test\w* \([\w.]+\))$")


def parse(out: str) -> dict[str, str]:
    status: dict[str, str] = {}
    pending = None
    for line in out.splitlines():
        named = _NAME.match(line)
        if named:
            pending = named.group(1)
            continue
        m = _RESULT.match(line)
        if m:
            status[m.group(1)] = m.group(2)
            if pending:
                status[pending] = m.group(2)
        pending = None
    for m in re.finditer(r"^(FAIL|ERROR): (test\w* \([\w.]+\))", out, re.M):
        status[m.group(2)] = m.group(1)
    return status


def inst(i: str) -> dict:
    return json.loads((DATA / f"django__django-{i}.json").read_text())


def sh(cmd: list[str], cwd: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def setup(root: Path, i: str) -> None:
    d = inst(i)
    ws = root / i
    if not ws.exists():
        sh(["git", "worktree", "add", "-q", "--detach", str(ws), d["base"]], DATA / "django.git")
    if not (ws / ".venv").exists():
        sh(["uv", "venv", "-q", "-p", d.get("python", "3.8"), ".venv"], ws)
        env = {**os.environ, "VIRTUAL_ENV": str(ws / ".venv")}
        r = sh(["uv", "pip", "install", "-q", *DEPS, "-e", "."], ws, env=env)
        if r.returncode:
            print(i, "install failed", r.stderr[-300:])
    (DATA / f"goal-{i}.txt").write_text(d["problem"])


def labels(d: dict) -> list[str]:
    line = next(line for line in d["eval"].splitlines() if "runtests" in line)
    return line.split("--parallel 1", 1)[1].split()


def base_commit(ws: Path, d: dict) -> str:
    """The task's base commit in this workspace: its django SHA in a worktree of the full
    clone, or the root commit of an isolated workspace made from the base tree (iso.py),
    where django's own commits do not exist."""
    if sh(["git", "cat-file", "-e", f"{d['base']}^{{commit}}"], ws).returncode == 0:
        return d["base"]
    roots = sh(["git", "rev-list", "--max-parents=0", "HEAD"], ws).stdout.split()
    return roots[-1] if roots else d["base"]


def score_state(ws: Path, d: dict, gold: bool = False) -> tuple[bool, str]:
    """SWE-bench's way: the task's test files reset to base_commit, the hidden test
    patch applied, the task's own test labels run; the workspace restored after."""
    files = [line[6:] for line in d["test_patch"].splitlines() if line.startswith("+++ b/")]
    saved = {f: (ws / f).read_text() if (ws / f).exists() else None for f in files}
    base = base_commit(ws, d)
    tracked = [f for f in files if sh(["git", "cat-file", "-e", f"{base}:{f}"], ws).returncode == 0]
    try:
        sh(["git", "checkout", "-q", base, "--", *tracked], ws)
        # From stdin: a `.<id>_tp.diff` beside the worktrees was readable by the next run
        # (DESIGN-LOG §118, TODO 2.11).
        applied = sh(["git", "apply", "-"], ws, input=d["test_patch"])
        if gold:
            sh(["git", "apply", "-"], ws, input=d["patch"])
        out = sh(
            [
                str(ws / ".venv/bin/python"),
                "tests/runtests.py",
                "--verbosity",
                "2",
                "--settings=test_sqlite",
                "--parallel",
                "1",
                *labels(d),
            ],
            ws,
            timeout=900,
        )
        st = parse(out.stdout + out.stderr)
        f2p = [st.get(t, "missing") for t in d["f2p"]]
        bad = [t for t in d["p2p"] if st.get(t) not in ("ok", "skipped")]
        ok = applied.returncode == 0 and all(s == "ok" for s in f2p) and not bad
        note = f"f2p {f2p} p2p_not_ok {len(bad)}/{len(d['p2p'])}" + (
            f" e.g. {bad[:2]}" if bad else ""
        )
        if applied.returncode:
            note = "TEST PATCH DID NOT APPLY " + applied.stderr[:150]
        return ok, note
    finally:
        if gold:
            sh(["git", "apply", "-R", "-"], ws, input=d["patch"])
        for f, content in saved.items():
            if content is None:
                (ws / f).unlink(missing_ok=True)
            else:
                (ws / f).write_text(content)


def validate(root: Path, i: str) -> None:
    d = inst(i)
    ws = root / i
    before_ok, before = score_state(ws, d)
    gold_ok, gold = score_state(ws, d, gold=True)
    valid = (not before_ok) and gold_ok
    print(f"{i}: {'VALID' if valid else 'INVALID'} | before fix: {before} | gold: {gold}")


def run(root: Path, ids: list[str]) -> None:
    procs = []
    goal_prefix = os.environ.get("SWE_GOAL", "goal")
    for i in ids:
        ws = root / i
        log = open(root / f"{i}.log", "w")  # noqa: SIM115 - the subprocess holds it until wait()
        # SWE_ASK: clarify on, questions answered by the simulated issue author (amb.py).
        asking = (
            [
                "--clarify",
                "--ask",
                "--answer-command",
                f"python3 {HERE / 'amb.py'} answer {i}",
            ]
            if os.environ.get("SWE_ASK")
            else ["--no-clarify", "--no-ask"]
        )
        procs.append(
            (
                log,
                runlimits.start(
                    [
                        HARNESS,
                        "run",
                        (DATA / f"{goal_prefix}-{i}.txt").read_text(),
                        "-w",
                        str(ws),
                        *os.environ.get("SWE_MODELS", "--all-mercury").split(),
                        "-y",
                        "--max-usd",
                        os.environ.get("SWE_MAX_USD", "2.0"),  # a per-problem spend cap (§109)
                        "--no-memory",
                        *asking,
                        *[arg for pat in WEB_DENY for arg in ("--web-deny", pat)],
                        "--no-animate",
                        "--test-command",
                        TEST_COMMAND,
                        "--test-label-root",
                        "tests",
                        "--trace-dir",
                        str(root / "traces" / i),
                    ],
                    cwd=ws,
                    env={**os.environ, "AMB_LOG_DIR": str(root)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
            )
        )
    deadline = time.monotonic() + RUN_TIMEOUT_S
    for log, p in procs:
        if not runlimits.wait(p, deadline, log):
            print(f"killed at the {RUN_TIMEOUT_S:.0f}s run limit: {log.name}", flush=True)
        else:  # §109.3: one run ended silently mid-plan with nothing in its log; keep the code
            log.write(f"\n[harness exited {p.returncode}]\n")
            log.flush()


def score(root: Path, i: str) -> None:
    d = inst(i)
    ws = root / i
    ok, note = score_state(ws, d)
    diff = sh(["git", "diff", "--shortstat", "--", "django"], ws).stdout.strip()
    events = [
        json.loads(line)
        for t in (root / "traces" / i).glob("run-*.jsonl")
        for line in t.read_text().splitlines()
    ]
    calls = [e for e in events if e.get("kind") == "llm_call"]
    wall = (max(e["ts"] for e in events) - min(e["ts"] for e in events)) / 60 if events else 0
    cost = sum(e.get("cost_usd") or 0 for e in calls)
    print(
        f"{i} {d.get('difficulty', '?'):<15} {'RESOLVED' if ok else 'not     '} {wall:5.1f} min "
        f"{len(calls):4d} calls ${cost:.3f} | {diff} | {note}"
    )


if __name__ == "__main__":
    cmd, round_name, *ids = sys.argv[1:]
    root = RUNS / round_name
    root.mkdir(exist_ok=True)
    if cmd == "run":
        run(root, ids)
    else:
        for i in ids:
            {"setup": setup, "validate": validate, "score": score}[cmd](root, i)
