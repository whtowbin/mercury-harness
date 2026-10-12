"""Any agent from scripts/agents on the django tasks swe.py runs (DESIGN-LOG §109).

  AGENT=goose [AGENT_MODEL=...] python3 agent.py run ROUND ID [ID ...]
  AGENT=goose python3 agent.py score ROUND ID [ID ...]

Workspaces come from `swe.py setup`; scoring is swe.py's score_state, so every agent is scored by
one function. Per-run output lands in ROUND/out/<id>/ (stdout.log plus the agent's own files).
Environment: AGENT (mini|aider|goose|codex), AGENT_MODEL (default openrouter/inception/mercury-2.5),
AGENT_REASONING (default high), AGENT_TIMEOUT_S (default 900).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import runlimits  # noqa: E402
from agents import api_key, load, sandboxed  # noqa: E402
from swe import DATA, RUNS, inst, score_state, sh  # noqa: E402

AGENT = os.environ.get("AGENT", "mini")
MODEL = os.environ.get("AGENT_MODEL", "openrouter/inception/mercury-2.5")
REASONING = os.environ.get("AGENT_REASONING", "high")
TIMEOUT_S = runlimits.env_seconds("AGENT_TIMEOUT_S", 900)
WEB_DENY_TEXT = ("github.com/django", "djangoproject.com", "swe-bench", "swebench")


def out_dir(root: Path, i: str) -> Path:
    d = root / "out" / i
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(root: Path, ids: list[str]) -> None:
    key = api_key()
    if not key:
        sys.exit("no OpenRouter key")
    agent = load(AGENT)
    for i in ids:
        ws = root / i
        if not (ws / ".venv").exists():
            sys.exit(f"{i}: run `swe.py setup {root.name} {i}` first")
        out = out_dir(root, i)
        goal = (DATA / f"goal-{i}.txt").read_text()
        # The workspace is a worktree of DATA/django.git, so git inside it reads that repository;
        # the rest of DATA (goals, test patches) stays unreadable. The history itself is still
        # visible to the agent: iso.py's one-commit repositories are the fix for that (§118).
        argv, env = sandboxed(
            agent, agent.argv(goal, ws, out, MODEL, REASONING), ws, out, key,
            agent.env(ws, out), ro=[DATA / "django.git"],
        )  # fmt: skip
        with open(out / "stdout.log", "w") as log:
            finished = runlimits.run(
                argv, TIMEOUT_S, log=log,
                cwd=ws, env=env, stdin=subprocess.DEVNULL,  # else Codex waits on stdin
            )  # fmt: skip
        if not finished:
            print(i, "killed after", TIMEOUT_S, "s", flush=True)


def score(root: Path, i: str) -> None:
    d = inst(i)
    ws = root / i
    ok, note = score_state(ws, d)
    diff = sh(["git", "diff", "--shortstat", "--", "django"], ws).stdout.strip()
    s = load(AGENT).stats(out_dir(root, i))
    calls = f"{s.calls:4d}" if s.calls is not None else f"{s.extra.get('steps', 0):3d}*"
    print(
        f"{i} {d.get('difficulty', '?'):<15} {'RESOLVED' if ok else 'not     '} {AGENT} "
        f"{calls} calls ${s.cost:.3f} {s.status} | {diff} | {note}"
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
