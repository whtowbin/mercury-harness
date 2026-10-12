"""`bench run` for a frozen harness tree under sandbox-exec (DESIGN-LOG §126).

  python scripts/sandboxed_bench.py TREE RUNDIR [bench run flags ...]
  python scripts/sandboxed_bench.py ~/.cache/mercury-harness/iso-harness-323fa64 \
      ~/.cache/mercury-harness/s126-fixtures --configs mercury-only,mercury-noreview \
      --repeats 3 --parallel 3 --timeout 600

Runs `TREE/.venv/bin/python -m bench.cli run --out RUNDIR/results.jsonl --trace-dir
RUNDIR/traces ...` with iso.py's profile: the home folder unreadable and unwritable except RUNDIR
(results, traces, TMPDIR and the runs' workspaces), the tree's code, and the uv, git and keychain
files a run needs; the environment is scripts/agents' allowlist. The model can still read the
tree it runs from (the fixtures' hidden tests are in it), which an unsandboxed run could too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "swebench"), str(HERE)]
import iso  # noqa: E402  # ty: ignore[unresolved-import]
from agents import ENV_KEEP, ENV_KEEP_PREFIXES  # noqa: E402


def clean_path(tree: Path) -> str:
    """PATH with the frozen tree's tools first and no other virtualenv's: a caller started with
    `uv run` carries the repository's own .venv/bin, which the sandbox (rightly) does not let a
    run execute (the first fixtures smoke died on `uv run pytest` resolving there)."""
    rest = [p for p in os.environ.get("PATH", "").split(":") if "/.venv/bin" not in p]
    return ":".join([str(tree / ".venv" / "bin"), *rest])


def command(tree: Path, rundir: Path, bench_args: list[str]) -> tuple[list[str], dict[str, str]]:
    tmp = rundir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    sb = rundir / "sandbox.sb"
    sb.write_text(iso.profile([rundir], [tree]))
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith(ENV_KEEP_PREFIXES)}
    env.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp), PATH=clean_path(tree))
    # uvx makes tool environments under uv's tools folder: a folder of the run's own, not the user's
    env.update(UV_TOOL_DIR=str(rundir / "uv-tools"), UV_TOOL_BIN_DIR=str(rundir / "uv-tools-bin"))
    py = tree / ".venv" / "bin" / "python"
    sub = bench_args[0] if bench_args and not bench_args[0].startswith("-") else "run"
    rest = bench_args[1:] if sub != "run" or bench_args[:1] == ["run"] else bench_args
    argv = [
        "/usr/bin/sandbox-exec", "-f", str(sb), str(py), "-m", "bench.cli", sub,
        "--out", str(rundir / "results.jsonl"), "--trace-dir", str(rundir / "traces"), *rest,
    ]  # fmt: skip
    return argv, env


def main() -> None:
    tree, rundir = (
        Path(sys.argv[1]).expanduser().resolve(),
        Path(sys.argv[2]).expanduser().resolve(),
    )
    argv, env = command(tree, rundir, sys.argv[3:])
    os.chdir(tree)
    os.execve(argv[0], argv, env)


if __name__ == "__main__":
    main()
