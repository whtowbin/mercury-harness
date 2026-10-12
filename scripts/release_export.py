#!/usr/bin/env python3
"""Export the files that ship into a fresh git repo of one commit, and check that export.

  uv run python scripts/release_export.py export DEST [--message MSG] [--verify]
  uv run python scripts/release_export.py verify DEST
  uv run python scripts/release_export.py assets OUTDIR --parquet CALLS.parquet [--tag v0.1.0]

`export` copies every file of `release_check.shipped_files(ROOT)` (tracked by git, minus
LEFT_OUT) into DEST, keeping relative paths and file modes, then runs `git init -b main`,
`git add -A` and `git commit` there. The private history stays behind. It refuses (exit 2) a
DEST that exists and is not empty. The commit uses the caller's git identity.

`verify` (or `export --verify`) runs these steps inside DEST, in order, one PASS/FAIL line
each with the elapsed seconds and the tail of the output on a failure. It continues after a
failure and exits 1 if any step failed:

  release_check tree --skip gaia   (GAIA metadata lives outside the repo, so that check is
                                    not run here; run it against the source tree)
  release_check dist, hygiene refs, uv sync, ruff check, ruff format --check, ty check,
  the fast tests, then a build whose wheel is installed into a fresh venv and
  `harness --help` and `bench --help` run from there.

`assets` builds the GitHub release assets (decided 10 Oct: the data ships with the release,
not in the tree): the results files (release_check.ASSETS) as one tarball, the per-call metrics
parquet (scripts/release_call_metrics.py), and DATA.md describing both, generated from the files
themselves. It runs `release_check data` over them, exits 1 on any finding, and prints the
`gh release` command; it never uploads.

This tool never pushes and never adds a git remote: publishing the export is a manual step.
Exit status: 0 clean, 1 a verify step or an asset check failed, 2 it could not run. Stdlib
only, apart from duckdb (a dev dependency) for `assets`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_check as rc  # noqa: E402

DEFAULT_MESSAGE = "mercury-harness v0.1.0"
TAIL_LINES = 25

# (name, argv) run inside the export. The install step needs temp dirs, so it is a function.
Step = tuple[str, Sequence[str]]
STEPS: list[Step] = [
    (
        "release_check tree (gaia skipped)",
        ["uv", "run", "python", "scripts/release_check.py", "tree", "--skip", "gaia"],
    ),
    ("release_check dist", ["uv", "run", "python", "scripts/release_check.py", "dist"]),
    ("hygiene refs", ["uv", "run", "python", "scripts/hygiene.py", "refs"]),
    # Every extra: ty checks the optional imports too (typesafe-sdk, textual, duckdb).
    ("uv sync", ["uv", "sync", "--all-extras"]),
    ("ruff check", ["uv", "run", "ruff", "check"]),
    ("ruff format --check", ["uv", "run", "ruff", "format", "--check"]),
    ("ty check", ["uv", "run", "ty", "check"]),
    ("fast tests", ["uv", "run", "pytest", "-q", "-n", "auto", "-m", "not slow"]),
]


# --- running things ---------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    """The caller's environment minus an active venv: `uv run` in the export must use its own."""
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def run(argv: Sequence[str], cwd: Path) -> tuple[int, str]:
    """Run a command with output captured; (exit code, stdout and stderr together)."""
    try:
        p = subprocess.run(
            list(argv), cwd=cwd, env=_clean_env(), capture_output=True, text=True, check=False
        )
    except OSError as e:
        return 127, f"{argv[0]}: {e}"
    return p.returncode, p.stdout + p.stderr


def tail(text: str, lines: int = TAIL_LINES) -> str:
    return "\n".join(text.rstrip().splitlines()[-lines:])


def install_step(dest: Path) -> tuple[bool, str]:
    """Build the wheel, install it into a fresh venv and run the console scripts from there."""
    with tempfile.TemporaryDirectory() as tmp:
        out, venv = Path(tmp) / "dist", Path(tmp) / "venv"
        code, log = run(["uv", "build", "--wheel", "--out-dir", str(out)], dest)
        wheels = sorted(out.glob("*.whl"))
        if code or not wheels:
            return False, log or "uv build made no wheel"
        for argv in (
            ["uv", "venv", str(venv)],
            ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), str(wheels[0])],
            [str(venv / "bin" / "harness"), "--help"],
            [str(venv / "bin" / "bench"), "--help"],
        ):
            code, log = run(argv, dest)
            if code:
                return False, f"$ {' '.join(argv)}\n{log}"
    return True, ""


def verify(
    dest: Path,
    steps: Sequence[Step] = STEPS,
    install: Callable[[Path], tuple[bool, str]] | None = install_step,
    out: Callable[[str], None] = print,
) -> int:
    """Every step in order, a PASS/FAIL line each; 1 if any failed."""
    results: list[bool] = []

    def record(name: str, ok: bool, started: float, log: str) -> None:
        out(f"{'PASS' if ok else 'FAIL'}  {time.monotonic() - started:6.1f}s  {name}")
        if not ok and log.strip():
            out("\n".join("    " + line for line in tail(log).splitlines()))
        results.append(ok)

    for name, argv in steps:
        started = time.monotonic()
        code, log = run(argv, dest)
        record(name, code == 0, started, log)
    if install is not None:
        started = time.monotonic()
        ok, log = install(dest)
        record("build, fresh venv, harness and bench --help", ok, started, log)
    failed = results.count(False)
    out(
        f"{failed} of {len(results)} step(s) failed"
        if failed
        else f"all {len(results)} steps passed"
    )
    return 1 if failed else 0


# --- export -----------------------------------------------------------------------------


def copy_files(root: Path, files: Sequence[str], dest: Path) -> int:
    """Copy each file to the same relative path under dest, keeping its mode; total bytes."""
    total = 0
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
        total += target.stat().st_size
    return total


def git_commit(dest: Path, message: str) -> str:
    """A new repo on `main` holding everything in dest as one commit; its hash."""
    for argv in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", message],
    ):
        subprocess.run(argv, cwd=dest, check=True, capture_output=True, text=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def export(root: Path, dest: Path, message: str, out: Callable[[str], None] = print) -> int:
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        out(f"refusing: {dest} exists and is not empty")
        return 2
    files = rc.shipped_files(root)
    dest.mkdir(parents=True, exist_ok=True)
    size = copy_files(root, files, dest)
    try:
        commit = git_commit(dest, message)
    except subprocess.CalledProcessError as e:
        out(f"git failed: {' '.join(e.cmd)}\n{tail(e.stderr or '')}")
        return 2
    out(f"exported {len(files)} files, {size} bytes, to {dest}")
    out(f"commit {commit} on main (no remote added, nothing pushed)")
    return 0


REPO_SLUG = "whtowbin/mercury-harness"


def data_md(tag: str, parquet: Path, results: Sequence[str]) -> str:
    """What the assets hold, read from the files so it cannot drift from them."""
    import duckdb  # dev dependency, as for release_check data

    rel = duckdb.read_parquet(str(parquet))
    columns = "\n".join(f"| `{c}` | {t} |" for c, t in zip(rel.columns, rel.types, strict=True))
    counts = rel.aggregate("suite, count(*) AS calls, count(DISTINCT run) AS runs", "suite")
    suites = "\n".join(f"| {s} | {c:,} | {r:,} |" for s, c, r in counts.order("suite").fetchall())
    return f"""# mercury-harness {tag}: data

Two files, for checking the numbers in the design log and the posts. GAIA appears as aggregate
scores only (in the repository's docs), never per question. Both files passed the release
checks (scripts/release_check.py data): no home path, identity, key, private project name or
GAIA content.

## mercury-harness-results-{tag}.tar.gz

{len(results)} per-run results files under `results/`, as `bench` wrote them: one JSON object
per run (configuration, task, outcome, cost, tokens, timings), plus a few text summaries of
sweeps. They are records, not traces, but they do carry short text: error messages and
tracebacks from failed runs, test names from scoring notes, trace file names, and a few short
model answers from probes. Extract it in a clone of the repository and the commands in README.md
and docs/benchmarking.md work as written, for example
`uv run bench report "results/*.jsonl" --view history`.

## mercury-harness-calls-{tag}.parquet

One row per model call (or failed attempt) in the django, SciCode and seeded-task runs, built by
`scripts/release_call_metrics.py` from the local traces. Rows per suite:

| suite | calls | runs |
|---|---|---|
{suites}

| column | type |
|---|---|
{columns}

Numbers and labels only: no prompt, reply, tool argument, path or error text. `phase` is the
harness phase without its step id; `failure_cause` is a category, never the error text.
Load it with DuckDB:

    duckdb -c "select suite, model, sum(cost_usd)
               from 'mercury-harness-calls-{tag}.parquet' group by all"
"""


def build_assets(root: Path, out: Path, parquet: Path, tag: str) -> int:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print(f"refusing: {out} exists and is not empty")
        return 2
    if not parquet.is_file():
        print(f"no such parquet: {parquet} (build it with scripts/release_call_metrics.py)")
        return 2
    results = rc.asset_files(root)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out / f"mercury-harness-results-{tag}.tar.gz", "w:gz") as tar:
        for rel in results:
            tar.add(root / rel, arcname=rel)
    calls = out / f"mercury-harness-calls-{tag}.parquet"
    shutil.copy2(parquet, calls)
    (out / "DATA.md").write_text(data_md(tag, calls, results), encoding="utf-8")
    rules, skipped = rc._rules_for(root, set())
    found = rc.scan_data([calls, *(root / p for p in results)], rules)
    rc.report(found, limit=15)
    for s in skipped:
        print("not run:", s)
    if found or skipped:
        return 1
    print(f"assets in {out}. To publish (after the repo and tag exist):")
    print(f"  gh release create {tag} {out}/* --repo {REPO_SLUG} --title 'mercury-harness {tag}'")
    return 0


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="copy the shipped files into DEST as one git commit")
    ex.add_argument("dest", type=Path)
    ex.add_argument("--message", default=DEFAULT_MESSAGE, help="commit message")
    ex.add_argument("--verify", action="store_true", help="then run the verify steps in DEST")
    ve = sub.add_parser("verify", help="run the verify steps in an existing export")
    ve.add_argument("dest", type=Path)
    asx = sub.add_parser("assets", help="build the GitHub release assets (results, parquet)")
    asx.add_argument("out", type=Path)
    asx.add_argument("--parquet", type=Path, required=True)
    asx.add_argument("--tag", default="v0.1.0")
    args = ap.parse_args(argv)

    if args.cmd == "assets":
        return build_assets(root, args.out.resolve(), args.parquet.expanduser(), args.tag)
    if args.cmd == "verify":
        if not args.dest.is_dir():
            print(f"no such export: {args.dest}")
            return 2
        return verify(args.dest.resolve())
    code = export(root, args.dest.resolve(), args.message)
    if code == 0 and args.verify:
        return verify(args.dest.resolve())
    return code


if __name__ == "__main__":
    sys.exit(main())
