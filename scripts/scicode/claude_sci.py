"""Claude Code on the SciCode problems sci.py sets up, on the user's subscription (§127).

  python3 sci.py setup ROUND dev 7 29 ...
  python3 claude_sci.py run ROUND MODEL [PID ...]       # MODEL: haiku | sonnet | opus
  python3 sci.py score ROUND > ROUND.out
  python3 claude_sci.py stats ROUND

Each problem gets a fresh workspace outside the home folder (Claude Code's own sandbox denies
the home folder; see scripts/agents/claude.py), with its own virtualenv; `solution.py` is
copied back to the round's folder for sci.py's scorer. One problem at a time: the weekly
allowance guard is checked before each, and a sweep stops when it is spent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import runlimits  # noqa: E402
from agents import ENV_KEEP, ENV_KEEP_PREFIXES, load  # noqa: E402
from sci import AGENT_DEPS, RUNS  # noqa: E402

ROOT = Path("/private/tmp/s126-claude")
TIMEOUT_S = runlimits.env_seconds("AGENT_TIMEOUT_S", 900)


def run_one(root: Path, pid: str, model: str) -> None:
    agent = load("claude")
    if reason := agent.allowance_stop():
        print(pid, "SKIPPED:", reason, flush=True)
        return
    ws = ROOT / root.name / pid
    out = root / "agent-out" / pid
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    (ws / "README.md").write_text((root / pid / "README.md").read_text())
    base = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(["uv", "venv", "-q", "-p", "3.11", ".venv"], cwd=ws, env=base, check=True)
    venv_env = {**base, "VIRTUAL_ENV": str(ws / ".venv")}
    subprocess.run(
        ["uv", "pip", "install", "-q", *AGENT_DEPS, "pytest"], cwd=ws, env=venv_env, check=True
    )
    goal = (root / f"{pid}.goal").read_text()
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith(ENV_KEEP_PREFIXES)}
    os.environ["AGENT_MODEL"] = model  # agents/claude.py reads it for the no-thinking variant
    env.update(agent.env(ws, out))
    with open(out / "stdout.log", "w") as log:
        finished = runlimits.run(
            agent.argv(goal, ws, out, model, "high"), TIMEOUT_S, log=log,
            cwd=ws, env=env, stdin=subprocess.DEVNULL,
        )  # fmt: skip
    if (ws / "solution.py").exists():
        shutil.copy2(ws / "solution.py", root / pid / "solution.py")
    agent.record_usage(agent.stats(out).extra)
    print(pid, "done" if finished else f"killed after {TIMEOUT_S:.0f}s", flush=True)


def stats(root: Path) -> None:
    agent = load("claude")
    total = 0.0
    for p in sorted(root.glob("*.goal")):
        s = agent.stats(root / "agent-out" / p.stem)
        total += s.cost
        print(f"{p.stem:>4} claude {s.calls} turns ${s.cost:.4f} (list price) {s.status}")
    print(f"total ${total:.4f} at list price; not billed (subscription)")


if __name__ == "__main__":
    cmd, round_name, *rest = sys.argv[1:]
    root = RUNS / round_name
    if cmd == "run":
        model, *ids = rest
        for pid in ids or sorted(p.stem for p in root.glob("*.goal")):
            run_one(root, pid, model)
    elif cmd == "stats":
        stats(root)
    else:
        sys.exit(__doc__)
