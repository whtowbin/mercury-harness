#!/usr/bin/env python3
"""Deterministic repo-hygiene checks, so nobody re-derives them by hand.

Subcommands (each exits non-zero when it finds something worth blocking on):

    docs             CLAUDE.md / rules files within their line budget; DESIGN-LOG
                     index <-> section headings; TUNABLES.md rows
                     point at constants that still exist (--fix refreshes the
                     recorded line numbers, which rot on every edit).
    refs             Backticked file paths in living docs that no longer exist.
    test-deletions   Staged diff removes more `def test_` functions than it adds.
    stale            Report-only: tracked files nothing references and nobody
                     has touched lately, plus oversized gitignored dirs.
                     With --every N it runs at most once per N days and prints
                     nothing when clean, so a SessionStart hook can call it.

Stdlib only, no model calls. Pure functions take text/paths so the tests can
feed them fixtures.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# History files legitimately mention things that no longer exist.
HISTORY_DOCS = {"DESIGN-LOG.md", "DESIGN-LOG-ARCHIVE.md"}
# Docs that are meant to describe the present, so a dead path in them is a bug.
LIVE_DOC_GLOBS = [
    "CLAUDE.md",
    ".claude/rules/*.md",
    "README.md",
    "TUNABLES.md",
    "TASKS.md",
    "docs/*.md",
]
# Claude Code's own guidance: past ~200 lines a CLAUDE.md file costs context and
# lowers adherence. Held under this so growth is noticed before it is a problem.
INSTRUCTION_LINE_BUDGET = 150
PATH_EXTS = ("py", "md", "toml", "yaml", "yml", "json", "jsonl", "sh")
# Named in docs but legitimately absent from the repo: files the harness creates
# inside a *workspace* at runtime, and one CLAUDE.md says to create on demand.
NOT_TRACKED_BY_DESIGN = {"memory.md", "scratch.md", "DESIGN-LOG-ARCHIVE.md"}
IGNORED_DIR_PREFIXES = ("traces/", "results/", "dist/", "profiling/", ".harness/")
# Gitignored dirs that grow without bound; report when they pass this size.
BLOAT_LIMIT_MB = {"traces": 200, "results": 50, "dist": 5}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout


def tracked_files() -> list[str]:
    return [f for f in _git("ls-files").splitlines() if f]


# --- docs -------------------------------------------------------------------


def index_drift(design_log: str) -> tuple[list[str], list[str]]:
    """(sections with no index row, index rows with no section)."""
    heads = set(re.findall(r"^## (\d+)[.\s]", design_log, re.M))
    rows = set(re.findall(r"^\| (\d+) \|", design_log, re.M))
    key = lambda n: int(n)  # noqa: E731
    return sorted(heads - rows, key=key), sorted(rows - heads, key=key)


_TUNABLE_ROW = re.compile(r"^\| `([^`]+)`.*?\| `([\w/.-]+\.py):(\d+)` \|", re.M)


def tunable_problems(tunables: str, read) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """(missing constants, {name: (recorded_line, actual_line)} for drifted rows).

    `read(path)` returns that file's text, so tests need no real files."""
    missing: list[str] = []
    drift: dict[str, tuple[int, int]] = {}
    for label, path, line in _TUNABLE_ROW.findall(tunables):
        name = label.split()[0]
        lines = read(path).splitlines()
        hits = [i + 1 for i, text in enumerate(lines) if re.match(rf"\s*{re.escape(name)}\b", text)]
        if not hits:
            missing.append(f"{name} (no longer in {path})")
        elif int(line) not in hits:
            drift[name] = (int(line), hits[0])
    return missing, drift


def over_budget(line_counts: dict[str, int], budget: int = INSTRUCTION_LINE_BUDGET) -> list[str]:
    """Instruction files longer than `budget` lines, as printable problems."""
    return [
        f"{name} is {n} lines (budget {budget})" for name, n in line_counts.items() if n > budget
    ]


def cmd_docs(fix: bool) -> int:
    problems: list[str] = []
    files = [ROOT / "CLAUDE.md", *sorted((ROOT / ".claude" / "rules").glob("*.md"))]
    counts = {
        str(f.relative_to(ROOT)): len(f.read_text().splitlines()) for f in files if f.exists()
    }
    problems += over_budget(counts)
    log = (ROOT / "DESIGN-LOG.md").read_text()
    no_row, no_section = index_drift(log)
    problems += [f"DESIGN-LOG §{n} has no index row" for n in no_row]
    problems += [f"DESIGN-LOG index row {n} has no section" for n in no_section]

    tun_path = ROOT / "TUNABLES.md"
    tun = tun_path.read_text()
    missing, drift = tunable_problems(tun, lambda p: (ROOT / p).read_text())
    problems += [f"TUNABLES: {m}" for m in missing]
    if drift and fix:
        for name, (old, new) in drift.items():
            tun = re.sub(
                rf"(^\| `{re.escape(name)}\b.*?\.py:){old}(` \|)", rf"\g<1>{new}\2", tun, flags=re.M
            )
        tun_path.write_text(tun)
        print(f"refreshed {len(drift)} TUNABLES line number(s): {', '.join(drift)}")
        return 1  # files changed; re-stage, like any fixing hook
    problems += [f"TUNABLES: {n} recorded at line {o}, now {a}" for n, (o, a) in drift.items()]
    for p in problems:
        print(p)
    return 1 if problems else 0


# --- refs -------------------------------------------------------------------


def dead_refs(text: str, exists) -> list[str]:
    """Backticked file paths in `text` for which `exists(path)` is false."""
    dead = []
    for token in sorted(set(re.findall(r"`([^`\s]+)`", text))):
        token = token.split(":", 1)[0] if re.search(r"\.\w+:\d+", token) else token
        if not re.fullmatch(rf"[\w./-]+\.(?:{'|'.join(PATH_EXTS)})", token):
            continue
        if token.startswith((*IGNORED_DIR_PREFIXES, "~", "/")) or "*" in token:
            continue
        if token in NOT_TRACKED_BY_DESIGN:
            continue
        if not exists(token):
            dead.append(token)
    return dead


def cmd_refs() -> int:
    files = set(tracked_files())
    basenames = {Path(f).name for f in files}
    exists = lambda p: p in files or (("/" not in p) and p in basenames)  # noqa: E731
    found = 0
    for pattern in LIVE_DOC_GLOBS:
        for doc in sorted(ROOT.glob(pattern)):
            # Only tracked docs: names resolve against tracked files, so an untracked doc that
            # another session has not committed yet names its own untracked files and failed
            # every other session's commit (2026-09-30, a handoff note naming its post draft).
            if doc.name in HISTORY_DOCS or str(doc.relative_to(ROOT)) not in files:
                continue
            for token in dead_refs(doc.read_text(), exists):
                print(f"{doc.relative_to(ROOT)}: `{token}` does not exist")
                found += 1
    return 1 if found else 0


# --- test-deletions ---------------------------------------------------------


def test_delta(diff: str) -> tuple[int, int]:
    """(test functions removed, test functions added) in a unified diff."""
    removed = added = 0
    for line in diff.splitlines():
        if re.match(r"-\s*(async\s+)?def test_\w*\(", line):
            removed += 1
        elif re.match(r"\+\s*(async\s+)?def test_\w*\(", line):
            added += 1
    return removed, added


def cmd_test_deletions() -> int:
    diff = _git("diff", "--cached", "-U0", "--", "tests", "*test_*.py")
    removed, added = test_delta(diff)
    if removed > added:
        print(
            f"staged diff removes {removed} test function(s) and adds {added}. "
            "If deliberate (a test replaced or split), rename-detect it by adding the "
            "replacement in the same commit; otherwise restore the test."
        )
        return 1
    return 0


# --- stale ------------------------------------------------------------------


def unreferenced(files: list[str], read) -> list[str]:
    """Tracked non-code root files whose name no other tracked text file mentions."""
    texts = {f: read(f) for f in files if f.endswith((".md", ".py", ".toml", ".yaml", ".json"))}
    out = []
    for f in files:
        if "/" in f or f in {"LICENSE", "uv.lock", ".gitignore"} or f.startswith("."):
            continue
        name = Path(f).name
        if name in {"README.md", "CLAUDE.md", "pyproject.toml"}:
            continue
        if not any(name in t for other, t in texts.items() if other != f):
            out.append(f)
    return out


def cmd_stale(every: int | None, days: int) -> int:
    stamp = ROOT / ".git" / "hygiene-stamp"
    if every is not None and stamp.exists() and time.time() - stamp.stat().st_mtime < every * 86400:
        return 0
    files = tracked_files()
    findings: list[str] = []
    read = lambda f: (ROOT / f).read_text(errors="ignore")  # noqa: E731
    cutoff = time.time() - days * 86400
    for f in unreferenced(files, read):
        last = _git("log", "-1", "--format=%ct", "--", f).strip()
        if last and int(last) < cutoff:
            findings.append(f"{f}: referenced nowhere and untouched for {days}+ days")
    for name, limit in BLOAT_LIMIT_MB.items():
        path = ROOT / name
        if path.is_dir():
            mb = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
            if mb > limit:
                findings.append(f"{name}/ is {mb:.0f}MB (limit {limit}MB): prune old runs")
    if every is not None:
        stamp.touch()
    if findings:
        print("Repo hygiene (report only; ask before deleting anything):")
        for f in findings:
            print(f"  - {f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("docs")
    d.add_argument("--fix", action="store_true", help="refresh TUNABLES line numbers")
    sub.add_parser("refs")
    sub.add_parser("test-deletions")
    s = sub.add_parser("stale")
    s.add_argument("--every", type=int, help="run at most once per N days, silent when clean")
    s.add_argument("--days", type=int, default=30)
    args = ap.parse_args(argv)
    if args.cmd == "docs":
        return cmd_docs(args.fix)
    if args.cmd == "refs":
        return cmd_refs()
    if args.cmd == "test-deletions":
        return cmd_test_deletions()
    return cmd_stale(args.every, args.days)


if __name__ == "__main__":
    sys.exit(main())
