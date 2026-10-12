"""Aider on the django tasks swe.py runs: the same model, the architect/editor harness.

  python3 aider.py run ROUND ID [ID ...]     one Aider run per task in ROUND/<id> (setup first)
  python3 aider.py score ROUND ID [ID ...]   swe.py's score line, cost and calls from Aider's log

Part of DESIGN-LOG §94. Aider gets the issue text as one message and exits; it has no shell
tool, so it edits from its repo map and what the model asks to see, and cannot run django's
tests or fetch the web (`--no-detect-urls`). That is the harness under test, not a handicap to
correct. Auto-commits are off so `git diff -- django` scores the edit; `.aider*` files land
in the workspace and are ignored by the scorer.

Environment: AIDER_MODEL (default openrouter/inception/mercury-2.5), AIDER_REASONING (default
high), AIDER_TIMEOUT_S (default 1800), AIDER_EDIT_FORMAT (unset: Aider picks for the model).
The OpenRouter key comes from the Keychain item the harness uses, or OPENROUTER_API_KEY.

Aider runs from its own virtualenv (AIDER_VENV, default ~/.cache/mercury-harness/aider-venv),
because its lock-pinned scipy 1.15.3 wheel does not load on this macOS (a dyld error in
`_propack`) and `uvx` refuses to override the pin. Build it once:

  uv venv --python 3.12 ~/.cache/mercury-harness/aider-venv
  VIRTUAL_ENV=~/.cache/mercury-harness/aider-venv uv pip install aider-chat==0.86.2 "scipy>=1.16"
"""

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: agents
from agents import aider as aider_agent
from agents import sandboxed
from mini import api_key
from swe import DATA, RUNS, inst, score_state, sh

MODEL = os.environ.get("AIDER_MODEL", "openrouter/inception/mercury-2.5")
REASONING = os.environ.get("AIDER_REASONING", "high")
TIMEOUT_S = float(os.environ.get("AIDER_TIMEOUT_S", "1800"))
EDIT_FORMAT = os.environ.get("AIDER_EDIT_FORMAT", "")
AIDER_VERSION = "0.86.2"  # pinned: the comparison names one harness version
VENV = Path(os.environ.get("AIDER_VENV", Path.home() / ".cache" / "mercury-harness" / "aider-venv"))
_COST = re.compile(r"Cost: \$([0-9.]+) message, \$([0-9.]+) session")


def log_path(root: Path, i: str) -> Path:
    return root / f"aider-{i}.log"


def command(i: str) -> list[str]:
    args = [
        str(VENV / "bin" / "aider"),
        "--model", MODEL, "--reasoning-effort", REASONING,
        "--message-file", str(DATA / f"goal-{i}.txt"),
        "--yes-always", "--no-auto-commits", "--no-gitignore", "--no-detect-urls",
        "--no-show-model-warnings", "--analytics-disable", "--no-check-update",
        "--no-stream", "--no-pretty",
    ]  # fmt: skip
    if EDIT_FORMAT:
        args += ["--edit-format", EDIT_FORMAT]
    return args


def run(root: Path, ids: list[str]) -> None:
    key = api_key()
    if not key:
        sys.exit("no OpenRouter key: Keychain item OpenRouter/harness or OPENROUTER_API_KEY")
    procs = []
    for i in ids:
        ws = root / i
        if not (ws / ".venv").exists():
            sys.exit(f"{i}: run `swe.py setup {root.name} {i}` first")
        if not (VENV / "bin" / "aider").exists():
            sys.exit(f"no Aider at {VENV}: see the module docstring for the one-time install")
        log = open(log_path(root, i), "w")  # noqa: SIM115 - held until wait()
        # Sandboxed like scripts/agents (TODO public release item 7). DATA/django.git: git inside
        # the worktree reads it. aider_agent.VENV is this module's VENV unless AIDER_VENV differs.
        argv, env = sandboxed(
            aider_agent, command(i), ws, root / "sandbox" / f"aider-{i}", key, {},
            ro=[DATA / "django.git", VENV],
        )  # fmt: skip
        # stdin closed: Aider's crash handler asks "open a GitHub issue? (Y/n)" and would
        # otherwise hold the run until the timeout (the 9296 smoke run lost 900s to it).
        p = subprocess.Popen(
            argv,
            cwd=ws,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append((i, p))
    for i, p in procs:
        try:
            p.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            print(i, "killed after", TIMEOUT_S, "s")


def stats(log: Path) -> tuple[int, float]:
    """Model calls and session cost from Aider's own accounting lines."""
    if not log.exists():
        return 0, 0.0
    text = log.read_text(errors="replace")
    costs = _COST.findall(text)
    calls = len(re.findall(r"^Tokens: ", text, re.M))
    return calls, float(costs[-1][1]) if costs else 0.0


def score(root: Path, i: str) -> None:
    d = inst(i)
    ws = root / i
    ok, note = score_state(ws, d)
    diff = sh(["git", "diff", "--shortstat", "--", "django"], ws).stdout.strip()
    calls, cost = stats(log_path(root, i))
    print(
        f"{i} {d.get('difficulty', '?'):<15} {'RESOLVED' if ok else 'not     '} aider "
        f"{calls:4d} calls ${cost:.3f} | {diff} | {note}"
    )


if __name__ == "__main__":
    cmd, round_name, *ids = sys.argv[1:]
    root = RUNS / round_name
    root.mkdir(exist_ok=True)
    if cmd == "run":
        run(root, ids)
    elif cmd == "score":
        for i in ids:
            score(root, i)
    else:
        sys.exit(__doc__)
