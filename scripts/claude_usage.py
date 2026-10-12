"""The weekly and five-hour allowance of the user's Claude login, read from Claude Code (§127).

  uv run python scripts/claude_usage.py          # print the current readings (one Haiku call)
  uv run python scripts/claude_usage.py baseline # also store them as the sweep's starting point

`claude -p` streams a `rate_limit_event` whose `unifiedWindows` hold each window's utilization
as a fraction. This makes the smallest call there is to read them, with the benchmark's
restrictions (no tools, no user settings), so the reading costs a few thousand tokens.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

USAGE = Path.home() / ".cache" / "mercury-harness" / "claude-usage.json"


def read() -> tuple[float, float]:
    r = subprocess.run(
        ["claude", "-p", "ok", "--model", "haiku", "--tools", "", "--setting-sources", "project",
         "--strict-mcp-config", "--disable-slash-commands", "--output-format", "stream-json",
         "--verbose", "--no-session-persistence", "--max-turns", "1"],
        capture_output=True, text=True, timeout=120, check=False, cwd="/private/tmp",
    )  # fmt: skip
    last = None
    for line in r.stdout.splitlines():
        if '"rate_limit_event"' in line:
            last = json.loads(line)["rate_limit_info"]["unifiedWindows"]
    if last is None:
        raise SystemExit("no rate_limit_event in Claude Code's output; is it logged in?")
    return last["five_hour"]["utilization"], last["seven_day"]["utilization"]


if __name__ == "__main__":
    five, week = read()
    print(f"five-hour window {five:.3f}, weekly {week:.3f} (fractions of the allowance)")
    if sys.argv[1:] == ["baseline"]:
        USAGE.write_text(json.dumps({"baseline_week": week, "week": week, "five_hour": five}))
        print(f"stored as the baseline in {USAGE}")
