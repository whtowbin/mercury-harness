"""Any agent from scripts/agents on the SciCode problems sci.py sets up (DESIGN-LOG §109).

  python3 sci.py setup ROUND dev
  AGENT=mini [AGENT_MODEL=...] python3 agents_sci.py run ROUND [PID ...]
  python3 sci.py score ROUND > ROUND.out          # solved and sub-steps, by SciCode's scorer
  AGENT=mini python3 agents_sci.py stats ROUND     # the agent's calls and cost per problem

The agent gets the same goal text and the same workspace (`.venv` with numpy and scipy) the
harness gets, and writes solution.py; sci.py's scorer reads only that file. Its calls and cost
column reads harness traces, so this prints the agent's own.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import runlimits  # noqa: E402
from agents import api_key, load, sandboxed  # noqa: E402
from sci import PARALLEL, RUNS  # noqa: E402

AGENT = os.environ.get("AGENT", "mini")
MODEL = os.environ.get("AGENT_MODEL", "openrouter/inception/mercury-2.5")
REASONING = os.environ.get("AGENT_REASONING", "high")
TIMEOUT_S = runlimits.env_seconds("AGENT_TIMEOUT_S", 900)


def run_one(root: Path, pid: str, key: str) -> None:
    agent = load(AGENT)
    ws = root / pid
    out = root / "agent-out" / pid
    out.mkdir(parents=True, exist_ok=True)
    goal = (root / f"{pid}.goal").read_text()
    argv, env = sandboxed(
        agent, agent.argv(goal, ws, out, MODEL, REASONING), ws, out, key, agent.env(ws, out)
    )
    with open(out / "stdout.log", "w") as log:
        finished = runlimits.run(
            argv, TIMEOUT_S, log=log,
            cwd=ws, env=env, stdin=subprocess.DEVNULL,  # else Codex waits on stdin
        )  # fmt: skip
    print(pid, "done" if finished else f"killed after {TIMEOUT_S:.0f}s", flush=True)


def run(root: Path, ids: list[str]) -> None:
    key = api_key()
    if not key:
        sys.exit("no OpenRouter key")
    ids = ids or sorted(p.stem for p in root.glob("*.goal"))
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        list(pool.map(lambda pid: run_one(root, pid, key), ids))


def stats(root: Path) -> None:
    agent = load(AGENT)
    total = 0.0
    for p in sorted(root.glob("*.goal")):
        s = agent.stats(root / "agent-out" / p.stem)
        total += s.cost
        calls = s.calls if s.calls is not None else f"{s.extra.get('steps', 0)}*"
        print(f"{p.stem:>4} {AGENT} {calls} calls ${s.cost:.4f} {s.status}")
    print(f"total ${total:.4f}")


if __name__ == "__main__":
    cmd, round_name, *ids = sys.argv[1:]
    root = RUNS / round_name
    if cmd == "run":
        run(root, ids)
    elif cmd == "stats":
        stats(root)
    else:
        sys.exit(__doc__)
