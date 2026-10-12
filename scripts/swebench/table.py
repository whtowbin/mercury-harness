"""One round of results/s94-harness-compare.txt as a table: arm by task, with cost, wall and calls.

  python3 table.py ROUND [ROUND ...]

Y resolved, . not, s skipped after the arm was stopped; a T or C suffix marks a problem
stopped at its wall or spend cap (§109).
"""

import collections
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "s94-harness-compare.txt"
_COST = re.compile(r"\$([0-9.]+)")
_SECS = re.compile(r"(?:ran|killed at \d+s) (\d+)s")
_CALLS = re.compile(r"(\d+)\*? calls")


def table(rnd: str) -> str:
    stops: list[str] = []
    rows: dict[str, dict[str, str]] = collections.defaultdict(dict)
    cost: dict[str, float] = collections.defaultdict(float)
    secs: dict[str, float] = collections.defaultdict(float)
    calls: dict[str, int] = collections.defaultdict(int)
    for raw in RESULTS.read_text().splitlines():
        parts = raw.split("\t", 2)
        if len(parts) != 3 or parts[0] != rnd or parts[1] in ("balance", "setup"):
            continue
        _, arm, line = parts
        tid = line.split()[0]
        if not tid.isdigit():  # a STOPPED line: shown as a note under the table
            stops.append(f"{arm}: {line}")
            continue
        if "skipped" in line:  # the arm was stopped before this task
            rows[arm][tid] = "s"
            continue
        m, s, k = _COST.search(line), _SECS.search(line), _CALLS.search(line)
        flag = "T" if "stopped: time" in line else ("C" if "stopped: cost" in line else "")
        rows[arm][tid] = ("Y" if "RESOLVED" in line else ".") + flag
        cost[arm] += float(m.group(1)) if m else 0.0
        secs[arm] += int(s.group(1)) if s else 0
        calls[arm] += int(k.group(1)) if k else 0
    tasks = sorted({t for r in rows.values() for t in r}, key=int)
    out = [
        f"{'arm':<13}"
        + "".join(f"{t:>7}" for t in tasks)
        + f"{'res':>5}{'usd':>8}{'min':>7}{'calls':>7}"
    ]
    for arm, r in rows.items():
        won = sum(v.startswith("Y") for v in r.values())
        out.append(
            f"{arm:<13}"
            + "".join(f"{r.get(t, '-'):>7}" for t in tasks)
            + f"{won:>5}{cost[arm]:>8.3f}{secs[arm] / 60:>7.1f}{calls[arm]:>7}"
        )
    out.append(f"total ${sum(cost.values()):.2f}")
    out.extend(f"  {s}" for s in stops)
    return "\n".join(out)


if __name__ == "__main__":
    for rnd in sys.argv[1:] or sys.exit(__doc__):
        print(f"== {rnd}\n{table(rnd)}")
