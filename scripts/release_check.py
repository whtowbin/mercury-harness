#!/usr/bin/env python3
"""The release gate: every check that must pass before this repo or a build of it goes public.

  uv run python scripts/release_check.py tree [--untracked] [--rule NAME ...] [--skip gaia]
  uv run python scripts/release_check.py dist
  uv run python scripts/release_check.py words | copies | all
  uv run python scripts/release_check.py data [PARQUET ...]
  uv run python scripts/release_check.py files [--assets] [--untracked]

`tree` scans the files that ship (tracked by git, minus LEFT_OUT; `--untracked` adds untracked,
non-ignored files, which may be committed later) with these rules:

  home       this machine's home directory
  identity   the author's names, git author, email and host names (this machine's Identity
             plus pyproject's authors), outside the files that credit the author
  secret     API keys and tokens (release_traces.SECRET_PATTERNS, emails aside)
  session    parallel-session labels (ai-agent-xx)
  project    the author's private app projects by name
  left-out   a reference to a file that does not ship: a dead link in the public repo
  gaia       GAIA task ids, answers of 12+ characters and 8-word runs of question text
             (GAIA's terms forbid resharing outside a gated repo)
  junk       .DS_Store, __pycache__ and compiled files

`dist` builds the sdist and the wheel into a temporary folder, then checks that no member is a
left-out, hidden or junk file, and runs the privacy rules over every member's text.

`words` and `copies` read the author's local data (app-run traces, reference projects in
caches); they are in scripts/release_local.py. `all` runs tree, words, copies and dist.

`data [FILE...]` checks released data files, the named ones plus the results that ship as a
release asset (ASSETS): a parquet's columns against the allow-list of the
script that builds it (release_call_metrics.py), every distinct text value for being a label
(64 characters or fewer, no spaces), and those values and any other file's text against the
privacy and GAIA rules.

`files` prints the exported tree's file list (`--assets`: the release-asset files), one per
line, for the export step. `all` also runs `data` over the asset files.

Exit status: 1 on any finding; 2 when a check could not run (GAIA metadata missing) and was
not skipped with `--skip`. A skipped check is printed, never silent. Stdlib, plus duckdb (a dev
dependency) to read GAIA's parquet metadata.

Why a tool: on 7 Oct two model-run audits of the same tree disagreed and each was partly wrong
(docs/plans is left out; DESIGN-LOG §118 has the earlier contamination check).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_traces as rt  # noqa: E402

# What stays private. One list, used by every check here and by the export.
LEFT_OUT = (
    "TODO.md",
    "TODO_local.md",
    "TASKS.md",
    "CLAUDE.md",
    ".claude/*",
    "docs/reviews/*",
    "docs/archive/*",
    "docs/posts/*",
    "docs/handoff-*",
    "docs/plans/*",
    "docs/mercury-tool-call-fix-plan*",
    "release-prep/*",
    # Local diffusion experiments on the author's dissertation text.
    "scripts/dllm_local/*",
    "results/dllm-local-*",
    # Clarify replays of the interactive app runs: their goals are the author's own words.
    "results/s92*-clarify-*",
    # Blog tooling: each writes into docs/posts/, which does not ship.
    "scripts/analysis/post_figures.py",
    "scripts/analysis/post_tables.py",
    "scripts/analysis/weak_strong_figures.py",
    "scripts/analysis/gold_overlap.py",
    "scripts/build_posts_page.py",
    "scripts/build_posts_page.template.html",
    "scripts/blog_export.py",
    "scripts/analysis/post1_new_figures.py",
)

# What ships as a GitHub release asset, not in the exported tree (decided 10 Oct): the per-run
# results. `data` checks them; a doc citing one is not a dead link, since the asset holds it.
ASSETS = ("results/*",)

# Files where a rule's hit is expected, with the reason. Globs, matched like LEFT_OUT.
ALLOW: dict[str, dict[str, str]] = {
    "identity": {
        "LICENSE": "copyright line",
        "pyproject.toml": "authors",
        "docs/conf.py": "Sphinx author and copyright",
        "README.md": "byline",
        "THIRD_PARTY_NOTICES.md": "copyright line",
        "PKG-INFO": "the sdist's metadata: authors and README",
        "*.dist-info/METADATA": "the wheel's metadata: authors and README",
        "*.dist-info/licenses/*": "the license",
    },
    "project": {
        "scripts/release_traces.py": "its denylist must name the projects it refuses",
        "tests/test_release_traces.py": "tests that denylist",
        "tests/test_redact_runs.py": "tests that personal listings are blanked",
    },
    "left-out": {
        "scripts/hygiene.py": "checks CLAUDE.md and TASKS.md by name",
        "tests/test_hygiene.py": "tests those checks",
        "scripts/release_traces.py": "names the clarify replays it refuses",
        "tests/test_release_traces.py": "tests that refusal",
        "tests/test_release_export.py": "left-out files as fixtures the export must skip",
    },
    "secret": {
        "tests/test_release_traces.py": "fake keys the scrubber must refuse",
        "tests/test_redact_runs.py": "fake keys the scrubber must refuse",
        "tests/test_setup.py": "fake keys for the key check",
        "tests/test_keystore.py": "a fake key for the keyring round trip",
    },
}
# This tool and its tests name everything they look for, under every rule.
SELF = ("scripts/release_check.py", "scripts/release_local.py", "tests/test_release_check.py")

# Text the identity rule ignores before matching: the public repo's own URL credits the author
# on purpose, and a home path is the home rule's finding (one report per line, not two).
REPO_URL = r"(?:github\.com/)?[\w-]+/mercury-harness\b"
PROJECT_NAMES = ("jumpin-plumber", "Flappy-bird2", "flappy-tdd", "mario_clone", "Mine Sweeper")
SESSION = re.compile(r"\bai-agent-[0-9a-f]{2}\b")
JUNK = re.compile(r"(^|/)(\.DS_Store|__pycache__/.*|.*\.py[co])$")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".parquet", ".gz", ".zip"}
GAIA_METADATA = Path(
    os.environ.get("GAIA_DATA", Path.home() / ".cache" / "mercury-harness" / "gaia")
) / ("2023/validation/metadata.parquet")
GAIA_ANSWER_MIN = 12  # shorter answers ("3", "Paris") match ordinary text
GAIA_SHINGLE = 8


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    text: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.text}"


@dataclass(frozen=True)
class Gaia:
    """What of GAIA's validation set must not appear: ids, long answers, question text."""

    ids: frozenset[str] = frozenset()
    answers: tuple[str, ...] = ()
    shingles: frozenset[tuple[str, ...]] = frozenset()

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, str, str]]) -> Gaia:
        ids, answers, shingles = set(), set(), set()
        for task_id, question, answer in rows:
            ids.add(task_id)
            answer = (answer or "").strip()
            if len(answer) >= GAIA_ANSWER_MIN and re.search(r"[A-Za-z]", answer):
                answers.add(answer)
            words = _words(question or "")
            grams = (tuple(words[i : i + GAIA_SHINGLE]) for i in range(len(words) - 7))
            shingles |= {g for g in grams if _distinctive(g)}
        return cls(frozenset(ids), tuple(sorted(answers)), frozenset(shingles))


@dataclass
class Rules:
    """Line rules (a name and a pattern) plus the GAIA check, which looks across lines."""

    line: dict[str, re.Pattern[str]] = field(default_factory=dict)
    ignore: dict[str, re.Pattern[str]] = field(default_factory=dict)
    gaia: Gaia | None = None


# --- what ships -------------------------------------------------------------------------


def left_out(path: str, patterns: Iterable[str] = LEFT_OUT) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def allowed(rule: str, path: str) -> bool:
    return path in SELF or left_out(path, ALLOW.get(rule, {}))


def _git(root: Path, *args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


def _candidates(root: Path, untracked: bool) -> list[str]:
    files = _git(root, "ls-files", "-z")
    if untracked:
        files += _git(root, "ls-files", "-z", "--others", "--exclude-standard")
    return sorted(p for p in set(files) if not left_out(p) and (root / p).is_file())


def shipped_files(root: Path, untracked: bool = False) -> list[str]:
    """The files of the exported tree."""
    return [p for p in _candidates(root, untracked) if not left_out(p, ASSETS)]


def asset_files(root: Path, untracked: bool = False) -> list[str]:
    """The files that ship as a release asset."""
    return [p for p in _candidates(root, untracked) if left_out(p, ASSETS)]


# --- rules ------------------------------------------------------------------------------


def author_names(pyproject: str) -> list[str]:
    """Each word of 4+ letters in pyproject's author names: a first name or a surname alone
    is as identifying as the full name."""
    authors = tomllib.loads(pyproject).get("project", {}).get("authors", [])
    return sorted({w for a in authors for w in re.findall(r"[A-Za-z]{4,}", a.get("name", ""))})


def _alternation(values: Iterable[str]) -> str:
    """release_traces._word over many values, its boundaries hoisted out of the alternation:
    a lookbehind inside every branch made this search 20x slower."""
    alts = "|".join(re.escape(v) for v in sorted(set(values), key=len, reverse=True))
    return rf"(?:(?<![A-Za-z0-9])|(?<=\\[ntr]))(?:{alts})(?![A-Za-z0-9])"


def _ref_pattern(glob: str) -> str:
    """A reference to a left-out path: a glob's literal prefix, or a file's base name."""
    if "*" in glob:
        return re.escape(glob.split("*", 1)[0])
    return re.escape(glob.rsplit("/", 1)[-1]) + r"\b"


def _refs(globs: Iterable[str]) -> re.Pattern[str]:
    """Not after `/`, `~` or `}`: `~/.claude/projects` is Claude Code's folder, not this repo's."""
    return re.compile(r"(?<![\w/~}.-])(?:" + "|".join(_ref_pattern(g) for g in globs) + ")")


def build_rules(ident: rt.Identity, extra_names: Iterable[str] = (), gaia: Gaia | None = None):
    who = [*ident.names, *extra_names, *ident.emails, ident.user]
    who += [h for h in ident.hosts if not rt.generic_host(h)]
    secrets = [p.pattern for n, p in rt.SECRET_PATTERNS.items() if n != "email"]
    line = {
        "home": re.compile(re.escape(str(ident.home)) + r"(?=/|\b)"),
        "identity": re.compile(_alternation(who), re.I),
        "secret": re.compile("|".join(f"(?:{s})" for s in secrets)),
        "session": SESSION,
        "project": re.compile("|".join(re.escape(n) for n in PROJECT_NAMES), re.I),
        "left-out": _refs(LEFT_OUT),
    }
    ignore = {"identity": re.compile(f"{REPO_URL}|{line['home'].pattern}")}
    return Rules(line=line, ignore=ignore, gaia=gaia)


# --- scanning ---------------------------------------------------------------------------


def _distinctive(gram: tuple[str, ...]) -> bool:
    """A run of question words worth matching: "a a b a b a b a" (a question about a string)
    also occurs in an unrelated test."""
    return len(set(gram)) >= 6 and sum(len(w) >= 4 for w in gram) >= 3


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _blank(m: re.Match[str]) -> str:
    return " " * len(m.group())


def _line_hits(path: str, text: str, name: str, rules: Rules) -> list[Finding]:
    """One finding per matching line. Whole-text search, since most files match nothing (a
    search per line cost 13 s over the tree). Ignored text is blanked, keeping offsets."""
    ignore = rules.ignore.get(name)
    subject = ignore.sub(_blank, text) if ignore else text
    found: dict[int, Finding] = {}
    for m in rules.line[name].finditer(subject):
        line = text.count("\n", 0, m.start()) + 1
        if line not in found:
            lo = max(m.start() - 40, text.rfind("\n", 0, m.start()) + 1)
            hi = text.find("\n", m.end())
            hi = m.end() + 40 if hi == -1 else min(hi, m.end() + 40)
            found[line] = Finding(name, path, line, text[lo:hi].strip())
    return list(found.values())


def scan_text(path: str, text: str, rules: Rules, only: set[str] | None = None) -> list[Finding]:
    found: list[Finding] = []
    for name in rules.line:
        if (only is None or name in only) and not allowed(name, path):
            found += _line_hits(path, text, name, rules)
    if rules.gaia and (only is None or "gaia" in only) and not allowed("gaia", path):
        found += _gaia_hits(path, text, rules.gaia)
    return found


def _gaia_hits(path: str, text: str, gaia: Gaia) -> list[Finding]:
    found = [
        Finding("gaia", path, text.count("\n", 0, m.start()) + 1, f"task id {m.group()}")
        for m in re.finditer(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", text
        )
        if m.group() in gaia.ids
    ]
    lower = text.lower()
    for answer in gaia.answers:  # substring search: a regex alternation of them took 4 s
        if (at := lower.find(answer.lower())) != -1:
            found.append(Finding("gaia", path, text.count("\n", 0, at) + 1, f"answer {answer!r}"))
    words = _words(text)
    for i in range(len(words) - GAIA_SHINGLE + 1):
        gram = tuple(words[i : i + GAIA_SHINGLE])
        if gram in gaia.shingles:
            found.append(Finding("gaia", path, 0, "question text: " + " ".join(gram)))
            break  # one per file is enough to act on
    return found


def read_text(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_tree(
    root: Path, files: list[str], rules: Rules, only: set[str] | None = None
) -> list[Finding]:
    found = [Finding("junk", p, 0, "should not be committed") for p in files if JUNK.search(p)]
    if only is not None and "junk" not in only:
        found = []
    for p in files:
        text = read_text(root / p)
        if text is not None:
            found += scan_text(p, text, rules, only)
    return found


# --- dist -------------------------------------------------------------------------------


def member_problems(name: str, shipped: set[str]) -> str | None:
    """Why a built archive must not hold this member, or None. Everything in it is a shipped
    file or build metadata; that catches what git does not track, like `.remember/`."""
    if left_out(name):
        return "left out of the release"
    if JUNK.search(name):
        return "junk"
    if name in shipped or name == "PKG-INFO" or re.match(r"[^/]+\.dist-info/", name):
        return None
    return "not a shipped file (untracked or ignored)"


def _members(archive: Path) -> dict[str, bytes]:
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as z:
            return {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    with tarfile.open(archive) as t:
        out = {}
        for m in t.getmembers():
            f = t.extractfile(m) if m.isfile() else None
            if f is not None:
                out[m.name.split("/", 1)[1] if "/" in m.name else m.name] = f.read()
        return out


def scan_dist(root: Path, rules: Rules) -> list[Finding]:
    privacy = {"home", "identity", "secret", "session", "project", "gaia"}
    shipped = set(shipped_files(root))
    found: list[Finding] = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["uv", "build", "--sdist", "--wheel", "--out-dir", tmp],
            cwd=root,
            check=True,
            capture_output=True,
        )
        archives = sorted([*Path(tmp).glob("*.whl"), *Path(tmp).glob("*.tar.gz")])
        if len(archives) != 2:
            made = [a.name for a in archives]
            raise RuntimeError(f"uv build made {made}, not one wheel and one sdist")
        for archive in archives:
            for name, data in _members(archive).items():
                label = f"{archive.name}!{name}"
                if why := member_problems(name, shipped):
                    found.append(Finding("dist", label, 0, why))
                    continue
                if b"\0" not in data[:8192]:
                    text = data.decode("utf-8", errors="replace")
                    hits = scan_text(name, text, rules, privacy)
                    found += [Finding(f.rule, label, f.line, f.text) for f in hits]
    return found


# --- data -------------------------------------------------------------------------------

# The columns a released parquet may hold besides release_call_metrics.CALL_FIELDS: where a row
# came from and what kind of event it was. A new column is a release decision, so the check
# fails on one it does not know.
DATA_ID_COLUMNS = {"suite", "sweep", "task", "run", "ts", "kind", "failure_cause"}
# A label: a model, phase, folder or category name. Anything longer, or with spaces, is text.
LABEL = re.compile(r"[\w.:/@+-]{0,64}")
# OpenRouter's provider names have spaces ("Amazon Bedrock"): words of a label, up to four.
SPACED_LABEL = re.compile(r"(?=.{0,64}$)[\w.:/@+-]+(?: [\w.:/@+-]+){0,4}")
SPACED_LABEL_COLUMNS = {"provider"}
PRIVACY_RULES = {"home", "identity", "secret", "session", "project", "gaia"}


def parquet_findings(path: Path, rules: Rules, allowed_columns: set[str]) -> list[Finding]:
    import duckdb  # dev dependency

    name = path.name
    rel = duckdb.read_parquet(str(path))
    found = [
        Finding("data", name, 0, f"column {c!r} is not on the release allow-list")
        for c in rel.columns
        if c not in allowed_columns
    ]
    texts = [c for c, t in zip(rel.columns, rel.types, strict=True) if str(t) == "VARCHAR"]
    lines = []
    for column in texts:
        values = rel.select(column).distinct().fetchall()
        for (value,) in values:
            label = SPACED_LABEL if column in SPACED_LABEL_COLUMNS else LABEL
            if value is not None and not label.fullmatch(value):
                found.append(Finding("data", name, 0, f"{column}: not a label: {value[:80]!r}"))
            lines.append(f"{column}={value}")
    hits = scan_text(name, "\n".join(lines), rules, PRIVACY_RULES)
    return found + [Finding(f.rule, name, 0, f.text) for f in hits]


def scan_data(paths: list[Path], rules: Rules) -> list[Finding]:
    """Released data files: a parquet by its columns and distinct text values, anything else as
    text. Allowed parquet columns come from the script that builds it."""
    from release_call_metrics import CALL_FIELDS

    allowed_columns = DATA_ID_COLUMNS | set(CALL_FIELDS)
    found: list[Finding] = []
    for path in paths:
        if path.suffix == ".parquet":
            found += parquet_findings(path, rules, allowed_columns)
        elif (text := read_text(path)) is not None:
            found += scan_text(path.name, text, rules, PRIVACY_RULES)
    return found


# --- reporting --------------------------------------------------------------------------


def load_gaia(path: Path) -> Gaia | None:
    if not path.exists():
        return None
    import duckdb  # dev dependency

    rows = duckdb.sql(
        f'select task_id, "Question", "Final answer" from read_parquet(\'{path}\')'
    ).fetchall()
    return Gaia.from_rows(rows)


def report(found: list[Finding], limit: int, out: Callable[[str], None] = print) -> None:
    by_rule: dict[str, list[Finding]] = {}
    for f in found:
        by_rule.setdefault(f.rule, []).append(f)
    for rule, items in sorted(by_rule.items()):
        files = len({f.path for f in items})
        out(f"== {rule}: {len(items)} in {files} file(s)")
        for f in items[:limit]:
            out(f"  {f}")
        if len(items) > limit:
            out(f"  ... and {len(items) - limit} more (--limit to see them)")
    out(f"{len(found)} finding(s)" if found else "clean")


def _rules_for(root: Path, skip: set[str]) -> tuple[Rules, list[str]]:
    skipped = []
    gaia = None
    if "gaia" not in skip:
        gaia = load_gaia(GAIA_METADATA)
        if gaia is None:
            skipped.append(f"gaia: no metadata at {GAIA_METADATA} (set GAIA_DATA, or --skip gaia)")
    names = author_names((root / "pyproject.toml").read_text())
    return build_rules(rt.this_machine(), names, gaia), skipped


def _run(cmd: str, root: Path, args: argparse.Namespace) -> tuple[list[Finding], list[str]]:
    """One check's findings and the parts of it that could not run."""
    files = shipped_files(root, args.untracked)
    if cmd in ("words", "copies"):
        import release_local  # reads the author's local data; see its docstring

        check = release_local.check_words if cmd == "words" else release_local.check_copies
        found, skipped = check(root, files)
        return [f for f in found if not allowed(f.rule, f.path)], skipped
    rules, skipped = _rules_for(root, set(args.skip))
    if cmd == "data":  # the files named, plus the results that ship as a release asset
        named = [Path(p).expanduser() for p in args.paths]
        return scan_data([*named, *(root / p for p in asset_files(root))], rules), skipped
    if cmd == "dist":
        return scan_dist(root, rules), skipped
    print(f"{len(files)} shipped files")
    return scan_tree(root, files, rules, set(args.rule) if args.rule else None), skipped


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["tree", "dist", "words", "copies", "data", "all", "files"])
    ap.add_argument("paths", nargs="*", help="data: released data files besides the results")
    ap.add_argument("--assets", action="store_true", help="files: list the release-asset files")
    ap.add_argument("--untracked", action="store_true", help="also untracked, non-ignored files")
    ap.add_argument("--rule", action="append", help="tree: run only these rules (repeatable)")
    ap.add_argument("--skip", action="append", default=[], help="skip a check (gaia)")
    ap.add_argument("--limit", type=int, default=15, help="findings shown per rule")
    args = ap.parse_args(argv)

    if args.cmd == "files":
        listing = asset_files if args.assets else shipped_files
        print("\n".join(listing(root, args.untracked)))
        return 0
    found: list[Finding] = []
    skipped: list[str] = []
    for cmd in ("tree", "words", "copies", "dist", "data") if args.cmd == "all" else (args.cmd,):
        more, missing = _run(cmd, root, args)
        found += more
        skipped += [m for m in missing if m not in skipped]
    report(found, args.limit)
    for s in skipped:
        print("not run:", s)
    if found:
        return 1
    return 2 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
