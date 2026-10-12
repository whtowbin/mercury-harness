"""Claude Code itself as a benchmark agent, on the user's subscription (DESIGN-LOG §127).

`claude -p` headless, with only the six file and shell tools (no connectors, skills, messaging
or web tools), settings from the run's own file (`--setting-sources project`, so the user's
hooks, plugins and MCP servers do not load), and Claude Code's own sandbox: the home folder
unreadable and the network on an allowlist that names only the loopback address. The run
needs the user's login, so it is *not* wrapped in `sandbox-exec` (which would hide
`~/.claude` from it); `NATIVE_SANDBOX` tells the drivers so, and its workspace sits outside
the home folder. Checked on 6 Oct: a shell command in it could not read a file in the home
folder, `~/.claude` or `~/Documents`, and a `curl` to a host off the list got
`X-Proxy-Error: blocked-by-allowlist` (only in `dontAsk` mode: under `bypassPermissions` the
allowlist did nothing).

Cost: the subscription is not billed per token; `total_cost_usd` is the list-price equivalent.
What a run uses of the weekly allowance is read from the stream's `rate_limit_event`.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

from agents import Stats

CLAUDE_BIN = shutil.which("claude") or "claude"  # absolute: the run's PATH does not hold it
USAGE = Path.home() / ".cache" / "mercury-harness" / "claude-usage.json"
# The weekly figure moves in steps of 0.1, so a cap of 0.1 stops a sweep once it has used
# somewhere between 0.05 and 0.15 of the week, well inside the 0.3 the user allowed.
WEEK_CAP = float(os.environ.get("CLAUDE_WEEK_CAP", "0.1"))
FIVE_HOUR_CAP = float(os.environ.get("CLAUDE_FIVE_HOUR_CAP", "0.5"))
_LOCK = threading.Lock()
NATIVE_SANDBOX = True  # not under sandbox-exec: Claude Code's own sandbox and permissions
MAX_TURNS = os.environ.get("CLAUDE_MAX_TURNS", "60")
MAX_USD = os.environ.get("CLAUDE_MAX_USD", "2.0")  # list-price cap a run, via --max-budget-usd
TOOLS = "Bash,Edit,Glob,Grep,Read,Write"
SETTINGS = {
    "permissions": {
        "defaultMode": "dontAsk",
        "allow": ["Bash", "Edit", "Glob", "Grep", "Read", "Write"],
        "deny": ["WebFetch", "WebSearch", "Read(~/**)", "Edit(~/**)", "Write(~/**)"],
    },
    "sandbox": {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "filesystem": {
            "denyRead": ["~/"],
            # the venv's interpreter and uv's cache live under the home folder
            "allowRead": [".", "~/.local/share/uv", "~/.cache/uv"],
            "allowWrite": ["."],
        },
        "network": {"allowedDomains": ["127.0.0.1"]},
    },
}


NO_THINKING = (
    ":nothink"  # a model name ending so runs with thinking switched off (MAX_THINKING_TOKENS=0)
)


def argv(goal: str, ws: Path, out: Path, model: str, reasoning: str) -> list[str]:
    model = model.removesuffix(NO_THINKING)
    settings = out / "settings.json"
    settings.write_text(json.dumps(SETTINGS, indent=1))
    return [
        CLAUDE_BIN, "-p", goal, "--model", model, "--settings", str(settings),
        "--setting-sources", "project", "--tools", TOOLS, "--strict-mcp-config",
        "--disable-slash-commands", "--output-format", "stream-json", "--verbose",
        "--no-session-persistence", "--max-turns", MAX_TURNS, "--max-budget-usd", MAX_USD,
    ]  # fmt: skip


def env(ws: Path, out: Path) -> dict[str, str]:
    """The workspace's own virtualenv first on PATH and no other: a caller started with `uv run`
    carries the repository's .venv/bin, which the sandbox refuses (the first smoke run's
    `python` died on it)."""
    env = {
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "PATH": f"{ws / '.venv' / 'bin'}:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
        "VIRTUAL_ENV": str(ws / ".venv"),
    }
    if os.environ.get("AGENT_MODEL", "").endswith(NO_THINKING):  # the drivers set it per job
        env["MAX_THINKING_TOKENS"] = "0"
    return env


def utilization(log: Path) -> dict[str, float | None]:
    """The five-hour and seven-day allowance fractions in a run's first and last
    `rate_limit_event` (None when the log has none)."""
    seen = []
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if '"rate_limit_event"' in line:
                try:
                    w = json.loads(line)["rate_limit_info"]["unifiedWindows"]
                except (ValueError, KeyError):
                    continue
                seen.append((w["five_hour"]["utilization"], w["seven_day"]["utilization"]))
    if not seen:
        return {
            "five_hour_start": None,
            "five_hour_end": None,
            "week_start": None,
            "week_end": None,
        }
    return {
        "five_hour_start": seen[0][0], "five_hour_end": seen[-1][0],
        "week_start": seen[0][1], "week_end": seen[-1][1],
    }  # fmt: skip


def stats(out: Path) -> Stats:
    log = out / "stdout.log"
    if not log.exists():
        return Stats(status="no log")
    result = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("{") and '"type":"result"' in line.replace('": ', '":'):
            try:
                result = json.loads(line)
            except ValueError:
                continue
    if result is None:
        return Stats(status="no result", extra=utilization(log))
    usage = result.get("usage") or {}
    return Stats(
        calls=int(result.get("num_turns") or 0),
        cost=float(result.get("total_cost_usd") or 0.0),
        in_tokens=int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0),
        out_tokens=int(usage.get("output_tokens") or 0),
        status=str(result.get("subtype") or result.get("terminal_reason") or "?"),
        cost_source="list-price equivalent, not billed",
        extra=utilization(log),
    )


def allowance_stop() -> str | None:
    """Why no further run should start: this sweep's share of the weekly allowance is spent,
    or the five-hour window is more than half used (it would lock the user out)."""
    if not USAGE.exists():
        return "no usage baseline (run scripts/claude_usage.py baseline)"
    u = json.loads(USAGE.read_text())
    spent = u["week"] - u["baseline_week"]
    if spent >= WEEK_CAP - 1e-9:
        return f"weekly allowance cap reached ({spent:.2f} of {WEEK_CAP})"
    if u["five_hour"] >= FIVE_HOUR_CAP:
        return f"five-hour window at {u['five_hour']:.2f}"
    return None


def record_usage(extra: dict) -> None:
    """Keep the latest allowance readings from a run's stream for the next run's guard."""
    if extra.get("week_end") is None or not USAGE.exists():
        return
    with _LOCK:
        u = json.loads(USAGE.read_text())
        u["week"] = max(u["week"], extra["week_end"])
        u["five_hour"] = extra["five_hour_end"]
        USAGE.write_text(json.dumps(u))
