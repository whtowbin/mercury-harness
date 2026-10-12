"""Why failing SciCode sub-steps failed: re-run each round's scoring scripts and tally the
final error line (AssertionError = wrong numbers; anything else = the code did not run).

Usage: SCI_DATA=... python classify.py RUN_DIR [RUN_DIR ...]  (run dirs hold scoring/)
"""

import collections
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("SCI_DATA", REPO / "traces" / "scicode-data"))
P = str(DATA / "score-env" / "bin" / "python")
arms = sys.argv[1:]
tally = collections.Counter()
examples = collections.defaultdict(list)
for arm in arms:
    for s in sorted(pathlib.Path(arm, "scoring").glob("*/step_*.py")):
        if s.name == "step_44.3.py":
            continue  # the 10-minute simulation; its failures are timeouts, counted apart
        try:
            r = subprocess.run(
                [P, s.name], cwd=s.parent, capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            tally["timeout"] += 1
            continue
        if r.returncode == 0:
            continue
        last = (r.stderr.strip().splitlines() or ["?"])[-1]
        kind = last.split(":")[0].strip()
        tally[kind] += 1
        if kind != "AssertionError":
            examples[kind].append(f"{arm}/{s.parent.name}/{s.name}: {last[:140]}")
print(dict(tally))
for v in examples.values():
    for e in v[:4]:
        print(" ", e)
