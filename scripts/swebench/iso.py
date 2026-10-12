"""Isolated django rounds (TODO 2.11, DESIGN-LOG §118, §120): harness arms only.

  python iso.py build SHA                 # a slim, frozen harness tree for an arm
  python iso.py run PREFIX --ids 11790,11951 --arms rel,rel-no-cae --repeats 3 \
      [--parallel 4] [--max-spend 1.50]
  python iso.py table PREFIX              # resolved, calls and cost by arm and task

What §118 found reachable, and what this does instead:
- **django's whole history and other runs' checkpoints** sat in every worktree. Each run here
  gets a fresh repository made from the task's base tree (`git archive`), one commit, no
  other refs; HEAD's tree is checked against the base commit's tree and the status must be
  clean, or the run does not start.
- **A shared worktree per task, reset between arms.** Each run has its own directory,
  `RUNS/<PREFIX>-r<k>-<arm>/<task>/`, never reused.
- **The answer key, other rounds' folders, `traces/`, the scorer's `.<id>_tp.diff`, a shared
  /tmp.** The harness runs under `sandbox-exec` with reading and writing of the home folder
  denied except: its own run directory (workspace, trace and temp folders), its frozen
  harness tree (code only: no design log, results or docs), and the uv, git and keychain
  files a run needs. TMPDIR is the run's own temp folder. The scorer applies the hidden test
  patch from stdin, so no `.<id>_tp.diff` is ever written.

Arms are pinned to slim harness trees (`build`: agent/, bench/ and the lockfile from one
commit, `uv sync --frozen`), so a run cannot read the repository's design log either.
Runs are interleaved: for each repeat, each task, the arms in a rotated order, and the job
list is worked through in that order by `--parallel` workers. Each finished run appends one
JSON line to `results/<PREFIX>.jsonl` and its score line to `results/s94-harness-compare.txt`
in round.py's format (`<PREFIX>-r<k>`, arm), so `contamination.py` and `table.py` read it.
New runs stop being started once the measured spend reaches `--max-spend`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import runlimits  # noqa: E402
from swe import DATA, DEPS, REPO, RUNS, TEST_COMMAND, WEB_DENY, inst, score_state  # noqa: E402

HOME = Path.home()
CACHE = HOME / ".cache" / "mercury-harness"
RESULTS_TXT = REPO / "results" / "s94-harness-compare.txt"
RUN_TIMEOUT_S = float(os.environ.get("ISO_RUN_TIMEOUT_S", "1200"))
MAX_USD_A_RUN = os.environ.get("ISO_MAX_USD", "0.25")  # round.py's CAPS for mercury-only
# What the harness may still read in the home folder: uv's interpreters, cache and tools,
# git's own config, and the keychain the OpenRouter key comes from.
HOME_READ = [
    ".cache/uv",
    ".local/share/uv",
    ".local/bin",
    ".config/uv",
    ".gitconfig",
    ".config/git",
    "Library/Keychains",
    "Library/Preferences",
]
HOME_WRITE = [".cache/uv", "Library/Keychains"]


def harness_tree(sha: str) -> Path:
    return CACHE / f"iso-harness-{sha[:7]}"


# arm -> (commit of its harness tree, extra harness flags). Every arm runs Mercury in every
# phase (--all-mercury) with the defaults of its commit, plus these flags.
RELEASE = "343bca8"
SEPT30 = "5f1b02a"
ARMS: dict[str, tuple[str, list[str]]] = {
    "rel": (RELEASE, []),
    "rel-no-cae": (RELEASE, ["--no-check-after-edit"]),
    "rel-no-tail": (RELEASE, ["--no-executor-tail"]),
    "rel-no-rpc": (RELEASE, ["--no-review-presupplied-checks"]),
    "sep30": (SEPT30, []),
}
# Arms added from the environment, `ISO_ARM_<name>=<sha> [flags...]`, for a fix branch whose
# commit is not known when this file is written (DESIGN-LOG §120).
for _key, _value in os.environ.items():
    if _key.startswith("ISO_ARM_"):
        _sha, *_flags = _value.split()
        ARMS[_key[len("ISO_ARM_") :].lower().replace("_", "-")] = (_sha, _flags)


def sh(cmd: list[str], cwd: Path | None = None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False, **kw)


def build(sha: str) -> Path:
    """agent/, bench/ and the lockfile of one commit, synced: the harness and nothing else."""
    tree = harness_tree(sha)
    if (tree / ".venv" / "bin" / "harness").exists():
        return tree
    tree.mkdir(parents=True, exist_ok=True)
    paths = ["agent", "bench", "pyproject.toml", "uv.lock", "README.md", "LICENSE"]
    listed = sh(["git", "-C", str(REPO), "ls-tree", "--name-only", sha]).stdout.split()
    paths += [p for p in ("THIRD_PARTY_NOTICES.md",) if p in listed]
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", sha, *paths],
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    (tree / "COMMIT").write_text(sh(["git", "-C", str(REPO), "rev-parse", sha]).stdout)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    r = sh(["uv", "sync", "--frozen", "-q"], cwd=tree, env=env)
    if r.returncode:
        raise SystemExit(f"uv sync failed in {tree}: {r.stderr[-500:]}")
    return tree


def make_workspace(ws: Path, d: dict) -> None:
    """A fresh repository holding exactly the task's base tree, with its own venv."""
    ws.mkdir(parents=True)
    git_dir = str(DATA / "django.git")
    archive = subprocess.run(
        ["git", "--git-dir", git_dir, "archive", "--format=tar", d["base"]],
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(ws)], input=archive.stdout, check=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "base",
        "GIT_AUTHOR_EMAIL": "base@localhost",
        "GIT_COMMITTER_NAME": "base",
        "GIT_COMMITTER_EMAIL": "base@localhost",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    }
    sh(["git", "init", "-q", "-b", "main"], ws)
    (ws / ".git" / "info" / "exclude").write_text(".venv/\n*.egg-info/\n__pycache__/\n")
    sh(["git", "add", "-A"], ws)
    sh(["git", "commit", "-q", "-m", "base", "--no-verify"], ws, env=env)
    venv_env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    sh(["uv", "venv", "-q", "-p", d.get("python") or "3.8", ".venv"], ws, env=venv_env)
    r = sh(
        ["uv", "pip", "install", "-q", *DEPS, "-e", "."],
        ws,
        env={**venv_env, "VIRTUAL_ENV": str(ws / ".venv")},
    )
    if r.returncode:
        raise RuntimeError(f"install failed: {r.stderr[-300:]}")


def workspace_problem(ws: Path, d: dict) -> str | None:
    """Why this workspace may not be used, or None: §118's checked reset."""
    tree = sh(["git", "rev-parse", "HEAD^{tree}"], ws).stdout.strip()
    base_tree = sh(
        ["git", "--git-dir", str(DATA / "django.git"), "rev-parse", f"{d['base']}^{{tree}}"]
    ).stdout.strip()
    if not tree or tree != base_tree:
        return f"HEAD tree {tree[:12]} is not the base tree {base_tree[:12]}"
    status = sh(["git", "status", "--porcelain"], ws).stdout.strip()
    if status:
        return f"status not clean: {status[:200]}"
    commits = sh(["git", "rev-list", "--all", "--count"], ws).stdout.strip()
    if commits != "1":
        return f"{commits} commits reachable, expected 1"
    return None


def profile(allowed_rw: list[Path], allowed_ro: list[Path]) -> str:
    """sandbox-exec rules: the home folder denied, then the run's own paths allowed (the
    last matching rule wins). File metadata stays readable, so paths still resolve."""

    def sub(p: Path) -> str:
        return f'(subpath "{p}")'

    lines = [
        "(version 1)",
        "(allow default)",
        f"(deny file-read-data file-read-xattr file-write* {sub(HOME)})",
        "(deny file-read-data file-read-xattr file-write* "
        '(subpath "/private/tmp/claude-501") (subpath "/tmp/claude-501"))',
        "(allow file-read-data file-read-xattr "
        + " ".join(sub(HOME / p) for p in HOME_READ)
        + " "
        + " ".join(sub(p) for p in allowed_ro)
        + ")",
        "(allow file-write* " + " ".join(sub(HOME / p) for p in HOME_WRITE) + ")",
        "(allow file-read-data file-read-xattr file-write* "
        + " ".join(sub(p) for p in allowed_rw)
        + ")",
    ]
    return "\n".join(lines) + "\n"


def harness_argv(tree: Path, goal: str, ws: Path, traces: Path, flags: list[str]) -> list[str]:
    return [
        str(tree / ".venv" / "bin" / "harness"),
        "run",
        goal,
        "-w",
        str(ws),
        "--all-mercury",
        "-y",
        "--max-usd",
        MAX_USD_A_RUN,
        "--no-memory",
        "--no-clarify",
        "--no-ask",
        *[arg for pat in WEB_DENY for arg in ("--web-deny", pat)],
        "--no-animate",
        "--test-command",
        TEST_COMMAND,
        "--test-label-root",
        "tests",
        "--trace-dir",
        str(traces),
        *flags,
    ]


def trace_stats(traces: Path) -> tuple[int, float, float]:
    """(model calls, cost, wall minutes) from the run's trace, as swe.score reads them."""
    events = [
        json.loads(line)
        for t in traces.glob("run-*.jsonl")
        for line in t.read_text().splitlines()
        if line.strip()
    ]
    calls = [e for e in events if e.get("kind") == "llm_call"]
    wall = (max(e["ts"] for e in events) - min(e["ts"] for e in events)) / 60 if events else 0
    return len(calls), sum(e.get("cost_usd") or 0 for e in calls), wall


_lock = threading.Lock()


def record(prefix: str, row: dict) -> None:
    with _lock:
        with (REPO / "results" / f"{prefix}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        with RESULTS_TXT.open("a") as f:
            f.write(f"{row['round']}\t{row['arm']}\t{row['line']}\n")


def run_job(prefix: str, rep: int, task: str, arm: str) -> dict:
    sha, flags = ARMS[arm]
    tree = harness_tree(sha)
    round_name = f"{prefix}-r{rep}"
    arm_dir = RUNS / f"{round_name}-{arm}"
    ws, traces, tmp = arm_dir / task, arm_dir / "traces" / task, arm_dir / "tmp" / task
    d = inst(task)
    row = {"round": round_name, "arm": arm, "task": task, "rep": rep, "sha": sha}
    if ws.exists():
        # A run killed with its driver (resume after a stop): its folders are set aside, not
        # reused and not deleted, and the run starts again from a fresh workspace.
        stamp = time.strftime("%H%M%S")
        for p in (ws, traces, tmp, arm_dir / f"{task}.log"):
            if p.exists():
                p.rename(p.with_name(f"{p.name}.abandoned-{stamp}"))
    make_workspace(ws, d)
    problem = workspace_problem(ws, d)
    if problem:
        row.update(status="not started", line=f"{task} NOT STARTED: {problem}")
        return row
    traces.mkdir(parents=True)
    tmp.mkdir(parents=True)
    sb = arm_dir / f"{task}.sb"
    sb.write_text(profile([ws, traces, tmp], [tree]))
    goal = (DATA / f"goal-{task}.txt").read_text()
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    started = time.time()
    with (arm_dir / f"{task}.log").open("w") as log:
        proc = runlimits.start(
            ["/usr/bin/sandbox-exec", "-f", str(sb), *harness_argv(tree, goal, ws, traces, flags)],
            cwd=ws,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        finished = runlimits.wait(proc, time.monotonic() + RUN_TIMEOUT_S, log)
        log.write(f"\n[harness exited {proc.returncode}]\n")
    seconds = time.time() - started
    ok, note = score_state(ws, d)
    calls, cost, wall = trace_stats(traces)
    diff = sh(["git", "diff", "--shortstat", "--", "django"], ws).stdout.strip()
    status = "ran" if finished else f"killed at {RUN_TIMEOUT_S:.0f}s"
    line = (
        f"{task} {d.get('difficulty', '?'):<15} {'RESOLVED' if ok else 'not     '} "
        f"{wall:5.1f} min {calls:4d} calls ${cost:.3f} | {diff} | {note} | {status} {seconds:.0f}s"
    )
    row.update(
        status=status, resolved=ok, calls=calls, cost=round(cost, 5), wall_min=round(wall, 2),
        seconds=round(seconds), note=note[:300], line=line,
    )  # fmt: skip
    return row


def jobs(ids: list[str], arms: list[str], repeats: int, seed: int) -> list[tuple[int, str, str]]:
    """Interleaved: every repeat runs every task once per arm, the arm order rotated by
    repeat and task so no arm always goes first, the tasks shuffled within a repeat. An id
    written `9296:3` runs only in the first 3 repeats."""
    rng = random.Random(seed)
    caps = {i.split(":")[0]: int(i.split(":")[1]) if ":" in i else repeats for i in ids}
    out = []
    for rep in range(1, repeats + 1):
        tasks = [t for t, cap in caps.items() if rep <= cap]
        rng.shuffle(tasks)
        for j, task in enumerate(tasks):
            k = (rep + j) % len(arms)
            out += [(rep, task, arm) for arm in arms[k:] + arms[:k]]
    return out


def balance() -> float | None:
    from round import balance as bal

    return bal()


def run(args: argparse.Namespace) -> None:
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known {list(ARMS)}")
    for arm in arms:
        build(ARMS[arm][0])
    ids = args.ids.split(",")
    todo = jobs(ids, arms, args.repeats, args.seed)
    done = set()
    out = REPO / "results" / f"{args.prefix}.jsonl"
    if out.exists():  # resume: a run already recorded is not run again
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["rep"], r["task"], r["arm"]))
            args.spent += r.get("cost") or 0.0
    todo = [j for j in todo if j not in done]
    print(
        f"balance ${balance()}; {len(todo)} runs to do; spent so far ${args.spent:.2f}", flush=True
    )
    state: dict[str, float] = {"spent": args.spent, "stopped": 0.0}
    queue = list(todo)

    def worker() -> None:
        while True:
            with _lock:
                if not queue or state["stopped"]:
                    return
                if state["spent"] >= args.max_spend:
                    state["stopped"] = 1.0
                    print(f"STOP: spend ${state['spent']:.2f} reached the cap", flush=True)
                    return
                rep, task, arm = queue.pop(0)
            try:
                row = run_job(args.prefix, rep, task, arm)
            except Exception as exc:  # one run's error must not end the others
                error = f"{task} ERROR {type(exc).__name__}: {exc}"[:300]
                row = {"round": f"{args.prefix}-r{rep}", "arm": arm, "task": task, "rep": rep,
                       "status": "error", "line": error}  # fmt: skip
            record(args.prefix, row)
            with _lock:
                state["spent"] += float(row.get("cost") or 0.0)
            print(f"[{arm} r{rep}] {row['line']}  (spent ${state['spent']:.2f})", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.parallel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"balance ${balance()}; measured spend ${state['spent']:.2f}", flush=True)


def table(prefix: str) -> None:
    rows = [json.loads(x) for x in (REPO / "results" / f"{prefix}.jsonl").read_text().splitlines()]
    by: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by.setdefault((r["arm"], r["task"]), []).append(r)
    arms = sorted({r["arm"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    print("arm".ljust(14), *(t.rjust(7) for t in tasks), "  total  calls     $/run")
    for arm in arms:
        cells, res, n, calls, cost = [], 0, 0, 0.0, 0.0
        for t in tasks:
            rs = [r for r in by.get((arm, t), []) if "resolved" in r]
            k = sum(bool(r["resolved"]) for r in rs)
            cells.append(f"{k}/{len(rs)}".rjust(7))
            res, n = res + k, n + len(rs)
            calls += sum(r["calls"] for r in rs)
            cost += sum(r["cost"] for r in rs)
        print(
            arm.ljust(14),
            *cells,
            f"{res:>3}/{n:<3} {calls / max(n, 1):6.1f}  {cost / max(n, 1):8.4f}",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("sha")
    r = sub.add_parser("run")
    r.add_argument("prefix")
    r.add_argument("--ids", required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--max-spend", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=120)
    r.add_argument("--spent", type=float, default=0.0, help=argparse.SUPPRESS)
    t = sub.add_parser("table")
    t.add_argument("prefix")
    a = ap.parse_args()
    if a.cmd == "build":
        print(build(a.sha))
    elif a.cmd == "run":
        run(a)
    else:
        table(a.prefix)


if __name__ == "__main__":
    main()
