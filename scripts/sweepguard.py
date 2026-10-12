"""Guards for a benchmark sweep: a spend and stall watchdog, and the scout check (DESIGN-LOG §96.2).

  python sweepguard.py watch --pid PID --state FILE --total-usd 10 --brave-max 2600 \
      --brave-dir DIR ... --watch-dir DIR ... [--stall-min 20] [--every 60]
  python sweepguard.py scout ARM KIND DIR [DIR ...] --est-usd X [--max-fail-frac 0.1]

`watch` polls every `--every` seconds and kills the sweep's whole process tree (every run
under it, `runlimits.kill_tree`) when:
- OpenRouter spend since the current stage began exceeds that stage's cap. The sweep writes
  "STAGE CAP_USD" lines to `--state`, and the last line is the current stage.
- Spend since the watchdog started exceeds `--total-usd`.
- Brave searches (web_search calls in the traces under `--brave-dir`) exceed `--brave-max`.
- Nothing under `--watch-dir` has been written for `--stall-min` minutes.
The reason goes to STATE.stopped and to stdout. It exits when the process group is gone.

`scout` classifies one arm's scout runs and prints CONTINUE or DROP with the reason. A run
fails when the time limit killed it, when the harness never reached its `totals` trace event,
or when its log shows a crash. KIND is how runs are laid out: `bench` (a results .jsonl),
`harness` (a round directory of <id>.log and traces/<id>/run-*.jsonl), or `mini` (mini's
trajectories). Exit 0 for CONTINUE, 3 for DROP.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runlimits  # noqa: E402


def api_key() -> str:
    """The OpenRouter key, from the environment or the Keychain item the harness uses. Never
    printed, never on a command line."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    r = subprocess.run(
        ["security", "find-generic-password", "-a", "harness", "-s", "OpenRouter", "-w"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


_CRASH = re.compile(r"^(Traceback \(most recent call last\)|error: [A-Z]\w+)", re.MULTILINE)
# The harness's own report of a run that stopped on an error, a provider failure included
# (`FAILED: LLMError: provider error: ...`, §89.2). Retries that later succeeded print
# "N attempts over Ts" and are not failures.
_FAILED = re.compile(r"^FAILED: (\w+)|The run did not finish", re.MULTILINE)


def balance() -> float | None:
    """The OpenRouter balance in dollars, or None when it cannot be read. The key never prints."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {api_key()}"},
        )
        data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
        return float(data["total_credits"]) - float(data["total_usage"])
    except Exception:
        return None


def search_counts(dirs: list[Path]) -> tuple[int, int]:
    """(web_search calls, searches that answered). A search the backend refused (quota, rate
    limit, a bad key) returns `error: ...` and writes no `web_search` event, so the difference
    is the failed searches. GAIA would otherwise run on with dead search and look finished."""
    calls = answered = 0
    for d in dirs:
        for f in d.rglob("run-*.jsonl"):
            with f.open(errors="ignore") as fh:
                for line in fh:
                    if '"kind": "tool_call"' in line and '"web_search"' in line:
                        calls += 1
                    elif '"kind": "web_search"' in line:
                        answered += 1
    return calls, answered


_SKIP_DIRS = frozenset(
    {".venv", ".git", "django", "tests", "docs", "js_tests", "extras", "node_modules", "mini"}
)


def newest_write(dirs: list[Path]) -> float:
    latest = 0.0
    for d in dirs:
        if not d.exists():
            continue
        for root, subdirs, files in os.walk(d):
            # Logs and traces are what a live run keeps writing; a django checkout's own tree
            # is tens of thousands of files and would make every poll slow.
            subdirs[:] = [x for x in subdirs if x not in _SKIP_DIRS]
            for name in files:
                try:
                    latest = max(latest, os.stat(os.path.join(root, name)).st_mtime)
                except OSError:
                    continue
    return latest


def idle_seconds(newest: float, started: float, now: float) -> float:
    """Seconds since the sweep last wrote anything, counted from the watchdog's start at most:
    the first smoke was stopped on its first poll because the run folders' newest file was
    195 minutes old, from before the sweep began (§96.2)."""
    return now - max(newest, started)


def current_stage(state: Path) -> tuple[str, float] | None:
    try:
        lines = [x.split() for x in state.read_text().splitlines() if x.strip()]
    except OSError:
        return None
    if not lines or len(lines[-1]) < 2:
        return None
    return lines[-1][0], float(lines[-1][1])


@dataclass
class Limits:
    total_usd: float
    brave_max: int
    stall_s: float
    # Failed searches before the sweep stops: more than this many, and more than this share.
    search_fail_min: int = 30
    search_fail_share: float = 0.15
    # Polls in a row with no readable balance before the sweep is stopped: a cap that cannot
    # be checked is no cap. A first version swallowed a NameError here and read "unreadable".
    unreadable_max: int = 5


def verdict(
    stage: tuple[str, float] | None,
    stage_spend: float | None,
    total_spend: float | None,
    searches: int,
    idle_s: float,
    limits: Limits,
    unreadable: int = 0,
    search_failed: int = 0,
) -> str:
    """The reason to stop the sweep now, or "" to let it run."""
    if unreadable >= limits.unreadable_max:
        return f"the OpenRouter balance was unreadable for {unreadable} polls in a row"
    if stage and stage_spend is not None and stage_spend > stage[1]:
        return f"stage {stage[0]} spent ${stage_spend:.2f}, over its ${stage[1]:.2f} cap"
    if total_spend is not None and total_spend > limits.total_usd:
        return f"the sweep spent ${total_spend:.2f}, over its ${limits.total_usd:.2f} cap"
    if searches > limits.brave_max:
        return f"{searches} Brave searches, over the cap of {limits.brave_max}"
    if search_failed > limits.search_fail_min and search_failed > limits.search_fail_share * max(
        searches, 1
    ):
        return f"{search_failed} of {searches} searches failed: the search backend is refusing"
    if idle_s > limits.stall_s:
        return f"nothing written for {idle_s / 60:.0f} min: stalled"
    return ""


def watch(a: argparse.Namespace) -> None:
    limits = Limits(a.total_usd, a.brave_max, a.stall_min * 60)
    state = Path(a.state)
    started = time.time()
    start = balance()
    if start is None:
        sys.exit("cannot read the OpenRouter balance: not starting a watchdog with no spend cap")
    stage_name, stage_start = None, start
    unreadable = 0
    while True:
        try:
            os.kill(a.pid, 0)
        except ProcessLookupError:
            print("sweep ended; watchdog done", flush=True)
            return
        stage = current_stage(state)
        now = balance()
        unreadable = unreadable + 1 if now is None else 0
        if stage and stage[0] != stage_name:
            stage_name, stage_start = stage[0], now
        stage_spend = (stage_start - now) if (now is not None and stage_start is not None) else None
        total_spend = (start - now) if (now is not None and start is not None) else None
        idle = idle_seconds(newest_write([Path(d) for d in a.watch_dir]), started, time.time())
        calls, answered = search_counts([Path(d) for d in a.brave_dir])
        reason = verdict(
            stage, stage_spend, total_spend, calls, idle, limits, unreadable, calls - answered
        )
        if reason:
            Path(f"{state}.stopped").write_text(reason + "\n")
            print(f"STOPPING THE SWEEP: {reason}", flush=True)
            runlimits.kill_tree(a.pid)
            return
        time.sleep(a.every)


# -- the scout check ---------------------------------------------------------------------


@dataclass
class Run:
    name: str
    failed: str  # "" when the run finished cleanly, else why it did not
    cost: float


def harness_runs(root: Path) -> list[Run]:
    """One Run per <id>.log in a round directory with traces/<id>/run-*.jsonl."""
    runs = []
    for log in sorted(root.glob("*.log")):
        if log.name.startswith("mini-"):
            continue
        rid = log.stem
        text = log.read_text(errors="ignore")
        traces = sorted((root / "traces" / rid).glob("run-*.jsonl"))
        totals = None
        if traces:
            for line in traces[-1].read_text(errors="ignore").splitlines():
                if '"kind": "totals"' in line:
                    totals = json.loads(line)
        if runlimits.KILLED_MARK in text:
            failed = "killed at the time limit"
        elif m := _FAILED.search(text):
            failed = f"stopped on an error ({m.group(1) or 'did not finish'})"
        elif _CRASH.search(text):
            failed = "crashed"
        elif totals is None:
            failed = "never finished (no totals event)"
        else:
            failed = ""
        runs.append(Run(rid, failed, float((totals or {}).get("cost_usd") or 0.0)))
    return runs


def mini_runs(root: Path) -> list[Run]:
    runs = []
    for log in sorted(root.glob("mini-*.log")):
        rid = log.stem.removeprefix("mini-")
        text = log.read_text(errors="ignore")
        traj = root / "mini" / f"{rid}.traj.json"
        info = {}
        if traj.exists():
            try:
                info = json.loads(traj.read_text()).get("info") or {}
            except json.JSONDecodeError:
                info = {}
        status = str(info.get("exit_status") or "")
        cost = float((info.get("model_stats") or {}).get("instance_cost") or 0.0)
        if runlimits.KILLED_MARK in text:
            failed = "killed at the time limit"
        elif not traj.exists():
            failed = "no trajectory"
        elif status != "Submitted":
            failed = f"exit status {status or 'missing'}"
        else:
            failed = ""
        runs.append(Run(rid, failed, cost))
    return runs


def bench_runs(path: Path) -> list[Run]:
    runs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        failed = ""
        if r.get("timed_out"):
            failed = "killed at the time limit"
        elif r.get("agent_error"):
            failed = f"agent error: {str(r['agent_error'])[:60]}"
        runs.append(
            Run(f"{r.get('task')}#{r.get('repeat')}", failed, float(r.get("cost_usd") or 0))
        )
    return runs


def scout_verdict(runs: list[Run], est_usd: float, max_fail_frac: float) -> tuple[bool, str]:
    """(continue?, why). Continue when at most max_fail_frac of the runs failed (at least one
    failure is always allowed) and the mean cost a run is at most twice the estimate."""
    if not runs:
        return False, "no scout runs found"
    failed = [r for r in runs if r.failed]
    allowed = max(1, int(len(runs) * max_fail_frac))
    mean = sum(r.cost for r in runs) / len(runs)
    detail = f"{len(runs)} runs, {len(failed)} failed (allowed {allowed}), ${mean:.4f} a run"
    if len(failed) > allowed:
        whys = sorted({r.failed for r in failed})
        return False, f"{detail}: too many failures ({'; '.join(whys)})"
    if mean > 2 * est_usd:
        return False, f"{detail}: over twice the ${est_usd:.4f} estimate"
    return True, detail


def scout(a: argparse.Namespace) -> None:
    runs: list[Run] = []
    for d in a.dirs:
        p = Path(d)
        runs += {"bench": bench_runs, "harness": harness_runs, "mini": mini_runs}[a.kind](p)
    ok, why = scout_verdict(runs, a.est_usd, a.max_fail_frac)
    print(f"{a.arm}: {'CONTINUE' if ok else 'DROP'} - {why}", flush=True)
    for r in runs:
        if r.failed:
            print(f"    {r.name}: {r.failed}")
    sys.exit(0 if ok else 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--state", required=True)
    w.add_argument("--total-usd", type=float, required=True)
    w.add_argument("--brave-max", type=int, default=2600)
    w.add_argument("--brave-dir", action="append", default=[])
    w.add_argument("--watch-dir", action="append", default=[])
    w.add_argument("--stall-min", type=float, default=20)
    w.add_argument("--every", type=float, default=60)
    s = sub.add_parser("scout")
    s.add_argument("arm")
    s.add_argument("kind", choices=["bench", "harness", "mini"])
    s.add_argument("dirs", nargs="+")
    s.add_argument("--est-usd", type=float, required=True)
    s.add_argument("--max-fail-frac", type=float, default=0.1)
    a = ap.parse_args()
    watch(a) if a.cmd == "watch" else scout(a)


if __name__ == "__main__":
    main()
