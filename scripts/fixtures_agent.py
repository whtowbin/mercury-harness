"""Another agent on the seeded fixtures, scored by bench's own scorer (DESIGN-LOG §109, row 11).

  AGENT=mini [AGENT_MODEL=...] uv run python scripts/fixtures_agent.py RUN [TASK ...] [--repeats N]

For each task: a fresh workspace seeded by `bench.runner._seed`, the agent run on `task.goal`
with the web refused, then `bench.metrics.score` (accuracy and the integrity checks) and one
record per run in bench's shape, appended to results/RUN_NAME.jsonl, so `bench report --view
compare` reads it beside the harness's history on the same tasks. The default tasks are the six
whose honest rate separates configurations.

The question this answers (post 4): does a bare-loop agent weaken a pinned test or "fix" correct
code more often than a harness with a premise gate and a reproduce check?
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import runlimits  # noqa: E402
from agents import api_key, load, sandboxed  # noqa: E402

from bench import metrics, tasks  # noqa: E402
from bench.runner import _seed, flatten  # noqa: E402

HONESTY_TASKS = [
    "no-bug", "misreport-slug", "misreport-module", "tautology", "vacuous-guard", "throttle-vague"
]  # fmt: skip
AGENT = os.environ.get("AGENT", "mini")
MODEL = os.environ.get("AGENT_MODEL", "openrouter/inception/mercury-2.5")
REASONING = os.environ.get("AGENT_REASONING", "high")
TIMEOUT_S = runlimits.env_seconds("AGENT_TIMEOUT_S", 600)
OUT = Path(
    os.environ.get("FIXTURES_AGENT_OUT", Path.home() / ".cache/mercury-harness/fixtures-agent")
)


def run_one(run_name: str, task_name: str, repeat: int, key: str) -> dict:
    task = tasks.get(task_name)
    agent = load(AGENT)
    workspace = Path(tempfile.mkdtemp(prefix=f"fx-{AGENT}-{task_name}-"))
    _seed(task, workspace)
    # A committed repository, as the django worktrees are: Aider's repo map lists tracked files
    # only (with none, GLM asked for the file instead of reading it), and a diff is readable after.
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        [
            "git",
            "-c",
            "user.name=seed",
            "-c",
            "user.email=seed@local",
            "commit",
            "-qm",
            "seed",
            "--no-gpg-sign",
        ],
    ):
        subprocess.run(argv, cwd=workspace, check=True, capture_output=True)
    out = OUT / run_name / f"{AGENT}-{MODEL.rsplit('/', 1)[-1]}-{task_name}-{repeat}"
    out.mkdir(parents=True, exist_ok=True)
    # pytest and ruff for the agent come from this repository's environment, as the harness's do.
    # The web is refused inside mini's shell (its argv carries the dead proxy); Codex's sandbox
    # denies network; Aider has no shell. Goose's shell has the web, as on django.
    tools = f"{REPO / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"
    argv, env = sandboxed(
        agent, agent.argv(task.goal, workspace, out, MODEL, REASONING), workspace, out, key,
        {"PATH": tools, **agent.env(workspace, out)}, ro=[REPO / ".venv"],
    )  # fmt: skip
    started = time.time()
    with open(out / "stdout.log", "w") as log:
        finished = runlimits.run(
            argv, TIMEOUT_S, log=log,
            cwd=workspace, env=env, stdin=subprocess.DEVNULL,  # else Codex waits on stdin
        )  # fmt: skip
    wall = time.time() - started
    stats = agent.stats(out)
    record = {
        "config": f"{AGENT}-{MODEL.rsplit('/', 1)[-1]}",
        "task": task_name,
        "repeat": repeat,
        "executor": MODEL.removeprefix("openrouter/"),
        "planner": "",
        "reviewer": "",
        "review_enabled": False,
        "started_at": started,
        "plan_only": False,
        "agent_error": "" if finished else f"killed at {TIMEOUT_S:.0f}s",
        "agent": AGENT,
        "agent_status": stats.status,
        "cost_usd": stats.cost,
        "cost_source": stats.cost_source,
        # Codex reports usage per turn, not per call; its step count stands in (flagged below).
        "calls": stats.calls if stats.calls is not None else stats.extra.get("steps", 0),
        "calls_are_steps": stats.calls is None,
        "prompt_tokens": stats.in_tokens,
        "completion_tokens": stats.out_tokens,
        "reasoning_tokens": stats.reasoning_tokens,
        "steps": 1,
        "trace_path": str(out),
        "wall_s": wall,
        "scores": metrics.score(task, workspace),
    }
    return flatten(record)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_name")
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    key = api_key()
    if not key:
        sys.exit("no OpenRouter key")
    results = REPO / "results" / f"{a.run_name}.jsonl"
    for repeat in range(a.repeats):
        for task_name in a.tasks or HONESTY_TASKS:
            rec = run_one(a.run_name, task_name, repeat, key)
            with results.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"{task_name:<17} r{repeat} acc {rec['accuracy']:.2f} honest {rec['honest']} "
                f"{rec['calls']} calls ${rec['cost_usd']:.4f} {rec['agent_status']} "
                f"{rec['wall_s']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
