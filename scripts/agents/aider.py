"""Aider 0.86.2: architect/editor, no shell (DESIGN-LOG §94, §109). Runs from its own venv;
see scripts/swebench/aider.py's docstring for the one-time install."""

from __future__ import annotations

import os
import re
from pathlib import Path

from agents import Stats

VERSION = "0.86.2"
VENV = Path(os.environ.get("AIDER_VENV", Path.home() / ".cache" / "mercury-harness" / "aider-venv"))
_COST = re.compile(r"Cost: \$([0-9.]+) message, \$([0-9.]+) session")
_OWN_FRAME = re.compile(r'File "[^"]*/aider/[^"]*\.py"')
_TOKENS = re.compile(r"Tokens: ([0-9.]+)(k?) sent, ([0-9.]+)(k?) received")


def argv(goal: str, ws: Path, out: Path, model: str, reasoning: str) -> list[str]:
    goal_file = out / "goal.txt"
    goal_file.write_text(goal)
    return [
        str(VENV / "bin" / "aider"),
        "--model", model, "--reasoning-effort", reasoning,
        "--message-file", str(goal_file),
        "--yes-always", "--no-auto-commits", "--no-gitignore", "--no-detect-urls",
        "--no-show-model-warnings", "--analytics-disable", "--no-check-update",
        "--no-stream", "--no-pretty",
    ]  # fmt: skip


def env(ws: Path, out: Path) -> dict[str, str]:
    return {}


def stats(out: Path) -> Stats:
    log = out / "stdout.log"
    if not log.exists():
        return Stats(status="no log")
    text = log.read_text(errors="replace")
    costs = _COST.findall(text)
    tokens = _TOKENS.findall(text)
    sent = int(sum(float(a) * (1000 if ka else 1) for a, ka, _, _ in tokens))
    received = int(sum(float(b) * (1000 if kb else 1) for _, _, b, kb in tokens))
    # A traceback in Aider's own frames is a crash; one it quotes from running the model's test
    # file ("Fix any errors below") is the agent working.
    crashed = _OWN_FRAME.search(text) is not None
    status = "crashed" if crashed else ("edited" if "Applied edit" in text else "no edit")
    return Stats(
        len(tokens), float(costs[-1][1]) if costs else 0.0, sent, received, 0, status, "reported"
    )
