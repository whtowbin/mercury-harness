"""mini-swe-agent and Goose on iso.py's isolated django workspaces (DESIGN-LOG §126).

`agent.py` and `mini.py` give an external agent a worktree of the full `django.git`, so the
fix sits in the history it can read (§118). This runs the same agents (scripts/agents) in what
iso.py gives a harness arm: a fresh one-commit repository per run, no `ro` access to
`django.git`, and `agents.sandboxed()`'s profile (home folder unreadable except the run's own
paths, the agent's tools and state; an allowlisted environment). Scored by iso's `score_state`.

Use through `iso_s126.py` (arms `mini`, `mini-glm`, `goose` beside the harness arms). Network is
not confined: Goose's shell has the web, so `leaks()` is read after the run (see agents/goose.py).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import iso
import runlimits
from agents import ENV_KEEP, ENV_KEEP_PREFIXES, api_key, load, sandboxed

MERCURY = "openrouter/inception/mercury-2.5"
GLM = "openrouter/z-ai/glm-5.3-flash"
REASONING = "high"
TIMEOUT_S = runlimits.env_seconds("AGENT_TIMEOUT_S", 900)
# arm -> (agents module, model). Goose reads its model from AGENT_MODEL (agents/goose.py).
# Claude Code runs on the user's login, outside sandbox-exec (which would hide ~/.claude), with
# its own sandbox; its workspaces sit outside the home folder so its Read deny rule can cover ~.
NATIVE_ROOT = Path("/private/tmp/s126-claude")
USAGE = Path.home() / ".cache" / "mercury-harness" / "claude-usage.json"
WEEK_CAP = float(
    os.environ.get("CLAUDE_WEEK_CAP", "0.1")
)  # of the weekly allowance, shown in steps of 0.1
FIVE_HOUR_CAP = float(os.environ.get("CLAUDE_FIVE_HOUR_CAP", "0.5"))
AGENT_ARMS = {
    "mini": ("mini", MERCURY),
    "mini-glm": ("mini", GLM),
    "goose": ("goose", MERCURY),
    "goose-glm": ("goose", GLM),
    "aider-glm": ("aider", GLM),
    "claude-haiku": ("claude", "haiku"),
    "claude-sonnet": ("claude", "sonnet"),
    "claude-opus": ("claude", "opus"),
    # Sonnet 5 itself, like for like with the harness's Sonnet 5 (§127.3)
    "claude-sonnet5": ("claude", "claude-sonnet-5"),
    "claude-sonnet5-nothink": ("claude", "claude-sonnet-5:nothink"),
}
# The agents read AGENT_MODEL from the process environment (agents/goose.py env() and stats()),
# so setting it and using it are done together, one job at a time.
_MODEL_ENV = threading.Lock()


def quiesce_git(ws: Path) -> None:
    """Let the commit's background `git gc --auto` finish, then switch gc off for this workspace.

    The one-commit repository holds about 8,000 objects, enough for `gc --auto` to repack it in the
    background just after the commit; Aider's GitPython then fails with `BadObject` on an object
    that is in the pack being rewritten (§126.6). Timing decided it: one smoke run passed, the next
    two failed in 2 s."""
    deadline = time.time() + 120
    while (ws / ".git" / "gc.pid").exists() and time.time() < deadline:
        time.sleep(0.5)
    iso.sh(["git", "config", "gc.auto", "0"], ws)


def run_agent_job(prefix: str, rep: int, task: str, arm: str) -> dict:
    name, model = AGENT_ARMS[arm]
    agent = load(name)
    native = getattr(agent, "NATIVE_SANDBOX", False)
    round_name = f"{prefix}-r{rep}"
    arm_dir = (NATIVE_ROOT if native else iso.RUNS) / f"{round_name}-{arm}"
    ws, out = arm_dir / task, arm_dir / "out" / task
    d = iso.inst(task)
    row = {"round": round_name, "arm": arm, "task": task, "rep": rep, "sha": "agent"}
    if native and (reason := agent.allowance_stop()):
        row.update(status="skipped", line=f"{task} SKIPPED: {reason}")
        return row
    if ws.exists():
        stamp = time.strftime("%H%M%S")
        for p in (ws, out):
            if p.exists():
                p.rename(p.with_name(f"{p.name}.abandoned-{stamp}"))
    iso.make_workspace(ws, d)
    problem = iso.workspace_problem(ws, d)
    if problem:
        row.update(status="not started", line=f"{task} NOT STARTED: {problem}")
        return row
    quiesce_git(ws)
    out.mkdir(parents=True)
    key = api_key()
    if not key and not native:
        row.update(status="not started", line=f"{task} NOT STARTED: no OpenRouter key")
        return row
    goal = (iso.DATA / f"goal-{task}.txt").read_text()
    with _MODEL_ENV:
        os.environ["AGENT_MODEL"] = model
        if native:  # the agent's own sandbox; the caller's other keys are still not passed on
            argv = agent.argv(goal, ws, out, model, REASONING)
            env = {
                k: v
                for k, v in os.environ.items()
                if k in ENV_KEEP or k.startswith(ENV_KEEP_PREFIXES)
            }
            env.update(agent.env(ws, out))
        else:
            argv, env = sandboxed(
                agent,
                agent.argv(goal, ws, out, model, REASONING),
                ws,
                out,
                key,
                agent.env(ws, out),
            )  # no ro=[DATA/"django.git"]: the one-commit repository is all there is
    started = time.time()
    with (out / "stdout.log").open("w") as log:
        finished = runlimits.run(
            argv, TIMEOUT_S, log=log, cwd=ws, env=env, stdin=subprocess.DEVNULL
        )
    seconds = time.time() - started
    ok, note = iso.score_state(ws, d)
    with _MODEL_ENV:
        os.environ["AGENT_MODEL"] = model
        s = agent.stats(out)
    calls = s.calls if s.calls is not None else int(s.extra.get("steps", 0))
    diff = iso.sh(["git", "diff", "--shortstat", "--", "django"], ws).stdout.strip()
    status = "ran" if finished else f"killed at {TIMEOUT_S:.0f}s"
    line = (
        f"{task} {d.get('difficulty', '?'):<15} {'RESOLVED' if ok else 'not     '} "
        f"{seconds / 60:5.1f} min {calls:4d} calls ${s.cost:.3f} | {diff} | {note} | "
        f"{status} {seconds:.0f}s"
    )
    if native:
        agent.record_usage(s.extra)
        row.update({k: s.extra.get(k) for k in ("week_start", "week_end", "five_hour_end")})
    row.update(
        status=status, resolved=ok, calls=calls, cost=round(s.cost, 5),
        wall_min=round(seconds / 60, 2), seconds=round(seconds), note=note[:300], line=line,
        agent_status=s.status,
    )  # fmt: skip
    return row


def install() -> None:
    """Route iso.py's per-run function to the agents for their arms."""
    original = iso.run_job

    def run_job(prefix: str, rep: int, task: str, arm: str) -> dict:
        if arm in AGENT_ARMS:
            return run_agent_job(prefix, rep, task, arm)
        return original(prefix, rep, task, arm)

    for arm in AGENT_ARMS:
        iso.ARMS.setdefault(arm, (iso.RELEASE, []))  # a tree to build; the agents do not use it
    iso.run_job = run_job  # ty: ignore[invalid-assignment]
