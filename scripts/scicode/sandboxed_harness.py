"""The harness CLI under sandbox-exec, for sci.py (DESIGN-LOG §126): `SCI_HARNESS` points here.

  SCI_TREE=~/.cache/mercury-harness/iso-harness-323fa64 \
  SCI_HARNESS=scripts/scicode/sandboxed_harness.py python sci.py run ROUND ...

sci.py runs the harness straight from the repository, which can read the whole home folder.
This wrapper runs a frozen tree's `harness` with the same profile iso.py gives a django run
(swebench/iso.py `profile`): the home folder unreadable and unwritable except the run's own
workspace and trace folder (read from `-w` and `--trace-dir`), the tree's code, and the uv, git
and keychain files a run needs; TMPDIR is a folder beside the trace folder; and the environment
is the allowlist of scripts/agents (the caller's other API keys are not passed on).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "swebench"), str(HERE.parent)]
import iso  # noqa: E402  # ty: ignore[unresolved-import]
from agents import ENV_KEEP, ENV_KEEP_PREFIXES  # noqa: E402


def clean_path(tree: Path) -> str:
    """PATH with the frozen tree's tools first and no other virtualenv's: a caller started with
    `uv run` carries the repository's own .venv/bin, which the sandbox (rightly) does not let a
    run execute (the first fixtures smoke died on `uv run pytest` resolving there)."""
    rest = [p for p in os.environ.get("PATH", "").split(":") if "/.venv/bin" not in p]
    return ":".join([str(tree / ".venv" / "bin"), *rest])


def command(args: list[str], tree: Path) -> tuple[list[str], dict[str, str]]:
    """The sandbox-exec argv and environment for the harness run `args` (as sci.py passes them)."""
    ws = Path(args[args.index("-w") + 1]).resolve()
    traces = Path(args[args.index("--trace-dir") + 1]).resolve()
    tmp = traces.parent.parent / "tmp" / ws.name
    traces.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    sb = ws.parent / f"{ws.name}.sb"
    sb.write_text(iso.profile([ws, traces, tmp], [tree]))
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith(ENV_KEEP_PREFIXES)}
    env.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp), PATH=clean_path(tree))
    # uvx makes tool environments under uv's tools folder: a folder of the run's own, not the user's
    tools = traces.parent.parent / "uv-tools" / ws.name
    env.update(UV_TOOL_DIR=str(tools), UV_TOOL_BIN_DIR=str(tools / "bin"))
    harness = tree / ".venv" / "bin" / "harness"
    return ["/usr/bin/sandbox-exec", "-f", str(sb), str(harness), *args], env


def main() -> None:
    tree = Path(os.environ["SCI_TREE"]).expanduser().resolve()
    argv, env = command(sys.argv[1:], tree)
    os.execve(argv[0], argv, env)


if __name__ == "__main__":
    main()
