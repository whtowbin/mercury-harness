"""mini-swe-agent on the django tasks swe.py runs: the same model, a different harness.

  python3 mini.py run ROUND ID [ID ...]     one mini run per task in ROUND/<id> (swe.py setup first)
  python3 mini.py score ROUND ID [ID ...]   swe.py's score line, from mini's trajectory

Pre-registered in DESIGN-LOG §94. Workspaces come from `swe.py setup ROUND ID` (a django
worktree at the task's base commit with its own .venv); scoring is swe.py's `score_state`
(FAIL_TO_PASS and PASS_TO_PASS, the workspace restored after), so both harnesses are scored
by one function. Nothing here writes to the harness repository.

Environment: MINI_MODEL (default openrouter/inception/mercury-2.5), MINI_COST_LIMIT (default
2.0, the harness's --max-usd), MINI_REASONING (default high, the harness's executor effort),
MINI_TIMEOUT_S (default 1800). The OpenRouter key is read from the Keychain item the harness
uses (`-s OpenRouter -a harness`) or OPENROUTER_API_KEY.

The web: the harness runs these tasks with --web-deny on django's fix sources (§80). mini has
no such switch, so its commands run behind a dead proxy (127.0.0.1:9): pip, curl and git all
honour it, and the venv is already prepared. The model call itself is made by mini's own
process, which does not see the proxy variables.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: runlimits
import runlimits
from agents import mini as mini_agent
from agents import sandboxed
from swe import DATA, RUNS, inst, score_state, sh

MODEL = os.environ.get("MINI_MODEL", "openrouter/inception/mercury-2.5")
COST_LIMIT = os.environ.get("MINI_COST_LIMIT", "2.0")
REASONING = os.environ.get("MINI_REASONING", "high")
# About 1.5x the slowest mini run on record (9.6 min, §94); DESIGN-LOG §96.2.
TIMEOUT_S = float(os.environ.get("MINI_TIMEOUT_S", "900"))
MINI_VERSION = "2.4.6"  # pinned: the comparison names one harness version
DEAD_PROXY = "http://127.0.0.1:9"


def api_key() -> str:
    env = os.environ.get("OPENROUTER_API_KEY", "")
    if env:
        return env
    r = subprocess.run(
        ["security", "find-generic-password", "-a", "harness", "-s", "OpenRouter", "-w"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def traj_path(root: Path, i: str) -> Path:
    return root / "mini" / f"{i}.traj.json"


def command(root: Path, i: str, ws: Path) -> list[str]:
    goal = (DATA / f"goal-{i}.txt").read_text()
    venv_bin = ws / ".venv" / "bin"
    path = f"{venv_bin}:{os.environ.get('PATH', '')}"
    overrides = [
        f"environment.cwd={ws}",
        f"environment.env.PATH={path}",
        f"environment.env.VIRTUAL_ENV={ws / '.venv'}",
        *(
            f"environment.env.{k}={DEAD_PROXY}"
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
        ),
        f"model.model_kwargs.reasoning_effort={REASONING}",
    ]
    args = ["uvx", "--from", f"mini-swe-agent=={MINI_VERSION}", "mini", "-m", MODEL, "-t", goal]
    args += ["-y", "--exit-immediately", "-l", COST_LIMIT, "-o", str(traj_path(root, i))]
    args += ["-c", "mini.yaml"]
    for o in overrides:
        args += ["-c", o]
    return args


def run(root: Path, ids: list[str]) -> None:
    key = api_key()
    if not key:
        sys.exit("no OpenRouter key: Keychain item OpenRouter/harness or OPENROUTER_API_KEY")
    procs = []
    for i in ids:
        ws = root / i
        if not (ws / ".venv").exists():
            sys.exit(f"{i}: run `swe.py setup {root.name} {i}` first")
        log = open(root / f"mini-{i}.log", "w")  # noqa: SIM115 - held until wait()
        extra = {
            "MSWEA_COST_TRACKING": "default",
            "MSWEA_CONFIGURED": "true",  # skip the first-run wizard
            "MSWEA_MODEL_NAME": MODEL,
            "MSWEA_SILENT_STARTUP": "1",
        }
        # Sandboxed like scripts/agents (TODO public release item 7); one folder per run, since
        # the runs start together. DATA/django.git: git inside the worktree reads it.
        traj_path(root, i).parent.mkdir(parents=True, exist_ok=True)
        argv, env = sandboxed(
            mini_agent, command(root, i, ws), ws, root / "sandbox" / f"mini-{i}", key, extra,
            ro=[DATA / "django.git"], rw=[traj_path(root, i).parent],
        )  # fmt: skip
        procs.append(
            (i, runlimits.start(argv, cwd=ws, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        )
    # The whole process group: p.kill() stopped only uvx and left mini's own python running.
    deadline = time.monotonic() + float(TIMEOUT_S)
    for i, p, log in procs:
        if not runlimits.wait(p, deadline, log):
            print(i, "killed after", TIMEOUT_S, "s")


def score(root: Path, i: str) -> None:
    d = inst(i)
    ws = root / i
    ok, note = score_state(ws, d)
    diff = sh(["git", "diff", "--shortstat", "--", "django"], ws).stdout.strip()
    calls = cost = 0.0
    status = "no trajectory"
    p = traj_path(root, i)
    if p.exists():
        t = json.loads(p.read_text())
        stats = t.get("info", {}).get("model_stats", {})
        calls, cost = stats.get("api_calls", 0), stats.get("instance_cost", 0.0)
        status = t.get("info", {}).get("exit_status", "?")
    print(
        f"{i} {d.get('difficulty', '?'):<15} {'RESOLVED' if ok else 'not     '} mini "
        f"{int(calls):4d} calls ${cost:.3f} {status} | {diff} | {note}"
    )


if __name__ == "__main__":
    cmd, round_name, *ids = sys.argv[1:]
    root = RUNS / round_name
    root.mkdir(exist_ok=True)
    if cmd == "run":
        run(root, ids)
    elif cmd == "score":
        for i in ids:
            score(root, i)
    else:
        sys.exit(__doc__)
