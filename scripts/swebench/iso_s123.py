"""§123's django scout: iso.py (on master since 9a2d8e4) with arms that set Settings
fields the CLI has no flag for, through scripts/s123_launch.py.

  python iso_s123.py run s123-dj --ids 11848,11964 --arms rel,nbh,res --repeats 3 --max-spend 0.8
  python iso_s123.py run s125-plan --ids 11951,11964,11790 --arms p10,p05 --repeats 12

`p10` and `p05` are plan-only (`--stop-after plan`) with Mercury's temperature unset (1.0) and 0.5
through OpenRouter (§125); they skip the hidden-test scoring, which a plan-only run cannot pass.

`rel` is the control (343bca8's defaults, as §120). `nbh` sets preload_neighbours and
executor_hypothesis; `res` is --research-first.
"""

import shutil
import sys
from pathlib import Path

import iso

LAUNCH = Path(__file__).resolve().parents[1] / "s123_launch.py"
iso.ARMS["nbh"] = (
    iso.RELEASE,
    ["--s123-set", "preload_neighbours=1,executor_hypothesis=1"],
)
iso.ARMS["res"] = (iso.RELEASE, ["--research-first"])
iso.ARMS["p10"] = (iso.RELEASE, ["--stop-after", "plan"])
iso.ARMS["p05"] = (iso.RELEASE, ["--stop-after", "plan", "--mercury-temperature", "0.5"])

_build, _argv = iso.build, iso.harness_argv


def build(sha: str) -> Path:
    tree = _build(sha)
    shutil.copy2(LAUNCH, tree / "s123_launch.py")  # the sandbox may read the tree, not scripts/
    return tree


def harness_argv(tree: Path, goal: str, ws: Path, traces: Path, flags: list[str]) -> list[str]:
    argv = _argv(tree, goal, ws, traces, flags)
    if "--s123-set" in flags:
        argv[0:1] = [str(tree / ".venv" / "bin" / "python"), str(tree / "s123_launch.py")]
    return argv


PLAN_ONLY_ARMS = {"p10", "p05"}
_score = iso.score_state


def score_state(ws, d):
    """Plan-only arms have no edit to score: skip the hidden tests (minutes of django suite)."""
    if ws.parent.name.rsplit("-", 1)[-1] in PLAN_ONLY_ARMS:  # <prefix>-r<k>-<arm>/<task>
        return False, "plan-only"
    return _score(ws, d)


iso.build = build  # ty: ignore[invalid-assignment]
iso.harness_argv = harness_argv  # ty: ignore[invalid-assignment]
iso.score_state = score_state  # ty: ignore[invalid-assignment]

if __name__ == "__main__":
    sys.argv[0] = "iso_s123.py"
    iso.main()
