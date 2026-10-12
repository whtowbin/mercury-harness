"""Codex CLI 0.160.0 (@openai/codex) through OpenRouter's Responses API, non-interactive, in the
workspace-write sandbox, which denies network to the commands it runs (DESIGN-LOG §109). Usage is
read from the `turn.completed` events of its JSON stream; cost is tokens times list price, cached
input at the cached price."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents import Stats, price

VERSION = "0.160.0"
PROVIDER = (
    'model_providers.openrouter={ name = "OpenRouter", base_url = "https://openrouter.ai/api/v1", '
    'env_key = "OPENROUTER_API_KEY", wire_api = "responses" }'
)


def argv(goal: str, ws: Path, out: Path, model: str, reasoning: str) -> list[str]:
    return [
        # Resolved: the sandbox refuses a writable root that is a symlink (round.py's layout).
        "codex", "exec", "--skip-git-repo-check", "--ephemeral", "-C", str(ws.resolve()),
        "--sandbox", "workspace-write", "-m", model.removeprefix("openrouter/"),
        "-c", 'model_provider="openrouter"', "-c", PROVIDER,
        "-c", f'model_reasoning_effort="{reasoning}"',
        # No web: on 9296 Codex answered "based on my web search" instead of editing (§109.2).
        "-c", 'web_search="disabled"',
        "--json", "-o", str(out / "last_message.txt"), goal,
    ]  # fmt: skip


def env(ws: Path, out: Path) -> dict[str, str]:
    path = f"{ws / '.venv' / 'bin'}:/opt/homebrew/bin:{os.environ.get('PATH', '')}"
    return {"PATH": path}


def stats(out: Path) -> Stats:
    log = out / "stdout.log"
    if not log.exists():
        return Stats(status="no log")
    in_t = out_t = reason = cached = 0
    turns = steps = 0
    errors: list[str] = []
    model = __import__("os").environ.get("AGENT_MODEL") or "inception/mercury-2.5"
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "turn.completed":
            u = ev.get("usage") or {}
            turns += 1
            in_t += int(u.get("input_tokens") or 0)
            cached += int(u.get("cached_input_tokens") or 0)
            out_t += int(u.get("output_tokens") or 0)
            reason += int(u.get("reasoning_output_tokens") or 0)
        elif ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") in ("command_execution", "agent_message", "file_change"):
                steps += 1
            if item.get("type") == "error":
                errors.append(str(item.get("message", ""))[:120])
    status = "done" if turns else ("crashed" if errors else "no turn")
    cost = price(model, in_t, out_t, cached)
    s = Stats(None, cost, in_t, out_t, reason, status, "tokens x list price")
    s.extra = {"steps": steps, "errors": errors, "cached_tokens": cached}
    return s
