"""mini-swe-agent 2.4.6: one bash tool, its default mini.yaml prompt (DESIGN-LOG §94, §109)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents import DEAD_PROXY, PROXY_VARS, Stats

VERSION = "2.4.6"
COST_LIMIT = os.environ.get("MINI_COST_LIMIT", "2.0")


def argv(goal: str, ws: Path, out: Path, model: str, reasoning: str) -> list[str]:
    venv_bin = ws / ".venv" / "bin"
    overrides = [
        f"environment.cwd={ws}",
        f"environment.env.PATH={venv_bin}:{os.environ.get('PATH', '')}",
        f"environment.env.VIRTUAL_ENV={ws / '.venv'}",
        *(f"environment.env.{k}={DEAD_PROXY}" for k in PROXY_VARS),
        f"model.model_kwargs.reasoning_effort={reasoning}",
    ]
    args = ["uvx", "--from", f"mini-swe-agent=={VERSION}", "mini", "-m", model, "-t", goal]
    args += ["-y", "--exit-immediately", "-l", COST_LIMIT, "-o", str(out / "traj.json")]
    args += ["-c", "mini.yaml"]
    for o in overrides:
        args += ["-c", o]
    return args


def env(ws: Path, out: Path) -> dict[str, str]:
    return {
        "MSWEA_COST_TRACKING": "default",
        "MSWEA_CONFIGURED": "true",
        "MSWEA_SILENT_STARTUP": "1",
    }


def stats(out: Path) -> Stats:
    p = out / "traj.json"
    if not p.exists():
        return Stats(status="no trajectory")
    t = json.loads(p.read_text())
    info = t.get("info", {})
    calls = cost = in_t = out_t = reason = 0
    for m in t.get("messages", []):
        u = (m.get("extra") or {}).get("response", {}).get("usage")
        if u:
            calls += 1
            cost += float(u.get("cost") or 0.0)  # OpenRouter's own figure per call
            in_t += int(u.get("prompt_tokens") or 0)
            out_t += int(u.get("completion_tokens") or 0)
            reason += int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    if not calls:  # older trajectories: fall back to mini's own totals
        ms = info.get("model_stats", {})
        calls, cost = int(ms.get("api_calls", 0)), float(ms.get("instance_cost", 0.0))
    return Stats(calls, cost, in_t, out_t, reason, str(info.get("exit_status", "?")), "reported")
