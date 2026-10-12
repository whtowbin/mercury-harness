"""One round of DESIGN-LOG §94 and §109: every arm on every django task, interleaved per task.

  python3 round.py ROUND ID [ID ...] [--arms a,b,c] [--parallel 3]

For each task the arms run one after another on one shared worktree (ROUND/<id>, from
`swe.py setup`), reset to the base commit between arms, so the interleaving is per task and
the disk cost is one checkout per task. Each arm gets its own round directory
(ROUND-<arm>/<id>, a symlink to the shared worktree) so logs, traces and trajectories never
mix. Tasks run `--parallel` at a time; a run is killed at RUN_TIMEOUT_S.

Every score line is appended to results/s94-harness-compare.txt as
`ROUND<TAB>arm<TAB>score line`, and the OpenRouter balance is logged before and after.
Nothing here writes to the harness repository except that results file.

§109 adds per-arm caps and stop rules, so a scout needs no one watching:
- CAPS gives each arm a wall cap and a spend cap a problem. The harness and mini enforce the
  spend cap themselves (--max-usd, -l); the others are read from their logs afterwards. A run
  over either cap is scored as it stands and its line gets `stopped: time` or `stopped: cost`.
- The guard (`Tally`) stops an arm for the round after 2 crashes, timeouts or provider
  failures, or after 4 real tasks with none resolved at over $0.15 or 10 minutes a task. Stopped
  arms are logged as `ROUND<TAB>arm<TAB>STOPPED ...` and skipped for the rest of the round.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # scripts/: runlimits
import runlimits  # noqa: E402  scripts/runlimits.py (DESIGN-LOG §96.2)
from mini import api_key  # noqa: E402
from swe import REPO, RUNS  # noqa: E402

PY = sys.executable
RUN_TIMEOUT_S = float(os.environ.get("ROUND_RUN_TIMEOUT_S", "1800"))
RESULTS = REPO / "results" / "s94-harness-compare.txt"
GLM = "z-ai/glm-5.3-flash"
MERCURY = "inception/mercury-2.5"
AGENT_PY = "agent.py"  # scripts/agents through one driver (§109)
# arm -> (driver, extra environment). The driver's `run` and `score` take (ROUND, ID).
ARMS: dict[str, tuple[str, dict[str, str]]] = {
    "mini": ("mini.py", {}),
    "mercury-only": ("swe.py", {"SWE_MODELS": "--all-mercury"}),
    # §76's arm: GLM plans, Mercury executes and reviews; §77's latency sort for GLM.
    "glm-plan": (
        "swe.py",
        {
            "SWE_MODELS": f"--planner {GLM} --executor {MERCURY} --reviewer {MERCURY} "
            "--provider-sort latency"
        },
    ),
    "aider": ("aider.py", {}),
    "mini-haiku": ("mini.py", {"MINI_MODEL": "openrouter/anthropic/claude-haiku-4.5"}),
    # §109 rows. GLM in every phase; the other agents with GLM; Goose and Codex CLI with Mercury.
    "glm-only": (
        "swe.py",
        {
            "SWE_MODELS": f"--planner {GLM} --executor {GLM} --reviewer {GLM} "
            "--provider-sort latency"
        },
    ),
    "mini-glm": ("mini.py", {"MINI_MODEL": f"openrouter/{GLM}"}),
    "aider-glm": ("aider.py", {"AIDER_MODEL": f"openrouter/{GLM}"}),
    "goose": (AGENT_PY, {"AGENT": "goose", "AGENT_MODEL": MERCURY}),
    "codex": (AGENT_PY, {"AGENT": "codex", "AGENT_MODEL": MERCURY}),
}
# arm -> (wall cap a problem in seconds, spend cap a problem in dollars), §109's table: 90th
# percentiles of §94's Mercury arms were $0.08-0.12 and 400-530 s; GLM costs about twice.
CAPS: dict[str, tuple[float, float]] = {
    "mini": (900, 0.25),
    "mercury-only": (900, 0.25),
    "glm-plan": (900, 0.50),
    "glm-only": (900, 0.50),
    "mini-glm": (900, 0.50),
    "aider": (900, 2.0),
    "aider-glm": (900, 0.50),
    "mini-haiku": (900, 2.0),
    "goose": (900, 0.25),
    "codex": (900, 0.25),
}
# How each driver learns its arm's spend cap (the time cap is this script's kill timer).
CAP_ENV = {"swe.py": "SWE_MAX_USD", "mini.py": "MINI_COST_LIMIT", "aider.py": "", AGENT_PY: ""}
CRASH_WORDS = ("crashed", "no trajectory", "no log", "no session", "no turn", "killed", "failed")
_COST = re.compile(r"\$([0-9]+\.[0-9]+)")


class Tally:
    """§109's stop rules, one per arm per round."""

    def __init__(self) -> None:
        self.tasks = 0
        self.resolved = 0
        self.failures = 0
        self.spend = 0.0
        self.seconds = 0.0
        self.stopped = ""

    def add(self, line: str, seconds: float) -> None:
        self.tasks += 1
        self.seconds += seconds
        self.resolved += "RESOLVED" in line
        m = _COST.search(line)
        self.spend += float(m.group(1)) if m else 0.0
        # A run that resolved the task is not a failure even if it ended at a cap (§109.5).
        self.failures += "RESOLVED" not in line and any(w in line for w in CRASH_WORDS)
        if self.failures >= 2:
            self.stopped = f"{self.failures} failures in {self.tasks} tasks"
        elif self.tasks >= 4 and self.resolved == 0:
            per_task_usd, per_task_min = self.spend / self.tasks, self.seconds / self.tasks / 60
            if per_task_usd > 0.15 or per_task_min > 10:
                self.stopped = (
                    f"0/{self.tasks} resolved at ${per_task_usd:.2f} "
                    f"and {per_task_min:.1f} min a task"
                )


_tallies: dict[str, Tally] = {}
_lock = threading.RLock()  # re-entered: run_arm logs while holding it (§109.5)


def balance() -> float | None:
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {api_key()}"}
        )
        d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
        return round(d["total_credits"] - d["total_usage"], 2)
    except Exception:
        return None


def log(round_name: str, arm: str, line: str) -> None:
    with _lock:
        RESULTS.parent.mkdir(exist_ok=True)
        with RESULTS.open("a") as f:
            f.write(f"{round_name}\t{arm}\t{line.rstrip()}\n")


def reset(ws: Path) -> None:
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=ws, check=False)
    subprocess.run(["git", "clean", "-fdq"], cwd=ws, check=False)  # keeps ignored .venv


def arm_root(round_name: str, arm: str, i: str, shared: Path) -> Path:
    root = RUNS / f"{round_name}-{arm}"
    root.mkdir(exist_ok=True)
    link = root / i
    if not link.exists():
        link.symlink_to(shared)
    return root


def flag_caps(arm: str, line: str, seconds: float, finished: bool) -> str:
    wall_cap, usd_cap = CAPS[arm]
    m = _COST.search(line)
    flags = []
    if not finished or seconds > wall_cap:
        flags.append("stopped: time")
    if m and float(m.group(1)) > usd_cap:
        flags.append("stopped: cost")
    return " ".join(flags)


def done_pairs(round_name: str) -> set[tuple[str, str]]:
    """(arm, task) pairs this round already scored, so a restarted round resumes (§109.5).

    Read under the lock and tolerant of a malformed line: an unlocked read raced another thread's
    half-written line and raised, which ended that task's remaining arms (§109.5)."""
    pairs: set[tuple[str, str]] = set()
    with _lock:
        text = RESULTS.read_text() if RESULTS.exists() else ""
    for raw in text.splitlines():
        parts = raw.split("\t", 2)
        if len(parts) == 3 and parts[0] == round_name and parts[2].split(" ")[0].isdigit():
            pairs.add((parts[1], parts[2].split(" ")[0]))
    return pairs


def run_arm(round_name: str, arm: str, i: str, shared: Path) -> None:
    driver, extra = ARMS[arm]
    if (arm, i) in done_pairs(round_name):
        return
    with _lock:
        tally = _tallies.setdefault(arm, Tally())
        if tally.stopped:
            log(round_name, arm, f"{i} skipped (arm stopped: {tally.stopped})")
            return
    root = arm_root(round_name, arm, i, shared)
    wall_cap, usd_cap = CAPS[arm]
    env = {**os.environ, **extra}
    if CAP_ENV[driver]:
        env[CAP_ENV[driver]] = str(usd_cap)
    env.setdefault("AGENT_TIMEOUT_S", str(wall_cap))
    env.setdefault("MINI_TIMEOUT_S", str(wall_cap))
    reset(shared)
    started = time.time()
    # The arm's wall cap, enforced on the whole process group: the old subprocess.run timeout
    # killed the driver and left the harness it had started running (DESIGN-LOG §96.2).
    timeout = min(RUN_TIMEOUT_S, wall_cap)
    finished = runlimits.run([PY, str(HERE / driver), "run", root.name, i], timeout, env=env)
    seconds = time.time() - started
    status = "ran" if finished else f"killed at {timeout:.0f}s"
    scored = subprocess.run(
        [PY, str(HERE / driver), "score", root.name, i],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    line = (scored.stdout.strip() or scored.stderr.strip()[-300:]) + f" | {status} {seconds:.0f}s"
    flags = flag_caps(arm, line, seconds, finished)
    if flags:
        line += f" | {flags}"
    log(round_name, arm, line)
    print(f"[{arm}] {line}", flush=True)
    with _lock:
        tally.add(line, seconds)
        if tally.stopped:
            log(round_name, arm, f"STOPPED after {i}: {tally.stopped}; spend ${tally.spend:.2f}")
            print(f"[{arm}] STOPPED: {tally.stopped}", flush=True)
    reset(shared)


def run_task(round_name: str, i: str, arms: list[str]) -> None:
    shared = RUNS / round_name / i
    subprocess.run([PY, str(HERE / "swe.py"), "setup", round_name, i], check=False)
    if not (shared / ".venv").exists():
        log(round_name, "setup", f"{i} setup failed")
        return
    for arm in arms:
        try:
            run_arm(round_name, arm, i, shared)
        except Exception as exc:  # one arm's error must not end the task's other arms (§109.5)
            log(round_name, arm, f"{i} ERROR {type(exc).__name__}: {exc}"[:300])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("round")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    arms = a.arms.split(",")
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        sys.exit(f"unknown arms: {unknown}; known: {list(ARMS)}")
    (RUNS / a.round).mkdir(parents=True, exist_ok=True)
    log(a.round, "balance", f"start ${balance()} arms={arms} ids={a.ids}")
    sem = threading.Semaphore(a.parallel)

    def worker(i: str) -> None:
        with sem:
            run_task(a.round, i, arms)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in a.ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(a.round, "balance", f"end ${balance()}")


if __name__ == "__main__":
    main()
