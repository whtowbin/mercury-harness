"""mini-swe-agent 2.4.6: one bash tool, its default mini.yaml prompt (DESIGN-LOG §94, §109)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents import DEAD_PROXY, PRICES, PROXY_VARS, Stats

VERSION = "2.4.6"
COST_LIMIT = os.environ.get("MINI_COST_LIMIT", "2.0")


# uvx resolves mini's dependencies afresh: litellm 1.106.0 (resolved 20 Oct) imports `mcp`, which
# nothing installs, and every run died on its first call. 1.104.0 is what §94-§126 ran on.
LITELLM = "1.104.0"
# Models the pinned litellm has no price for (Mercury 2.5 and 3). mini refuses to run a model
# it cannot cost, and ignoring that would switch off its spend limit, so their prices go in a
# registry file written beside the run's output (readable inside the sandbox).
UNMAPPED = {
    "openrouter/inception/mercury-3": "mercury-3",
    "openrouter/inception/mercury-2.5": "mercury-2.5",  # missing too (s131-mini: 0 of 30 ran)
}


def _registry(model: str, out: Path) -> Path | None:
    if model not in UNMAPPED:
        return None
    pin, pcached, pout = PRICES[UNMAPPED[model]]
    entry = {
        "input_cost_per_token": pin / 1e6,
        "output_cost_per_token": pout / 1e6,
        "cache_read_input_token_cost": (pcached if pcached is not None else pin) / 1e6,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "supports_function_calling": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "litellm_registry.json"
    path.write_text(json.dumps({model: entry}))
    return path


def argv(goal: str, ws: Path, out: Path, model: str, reasoning: str) -> list[str]:
    venv_bin = ws / ".venv" / "bin"
    overrides = [
        f"environment.cwd={ws}",
        f"environment.env.PATH={venv_bin}:{os.environ.get('PATH', '')}",
        f"environment.env.VIRTUAL_ENV={ws / '.venv'}",
        *(f"environment.env.{k}={DEAD_PROXY}" for k in PROXY_VARS),
        f"model.model_kwargs.reasoning_effort={reasoning}",
    ]
    registry = _registry(model, out)
    if registry:
        overrides.append(f"model.litellm_model_registry={registry}")
    args = ["uvx", "--from", f"mini-swe-agent=={VERSION}", "--with", f"litellm=={LITELLM}"]
    args += ["mini", "-m", model, "-t", goal]
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
