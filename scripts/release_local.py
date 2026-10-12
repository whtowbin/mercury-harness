"""Release checks that read the author's local data, not the repo: run through release_check.py.

  uv run python scripts/release_check.py words    # the author's typed text, quoted in a file
  uv run python scripts/release_check.py copies   # runs of lines shared with projects read

`words` collects what the author typed into the harness (goals, clarify and ask_user answers,
of 4+ words) from the interactive app runs' traces (APP_RUN_TRACES), drops any phrase that is
also a benchmark fixture's text (bench/), and reports every shipped file that quotes one.

`copies` finds each reference project (REFERENCES: agents compared against, skills read) by
package name in uv's and Claude Code's caches, and reports every shipped .py/.md file sharing
a run of COPY_RUN non-trivial lines with one. The three credited adaptations
(THIRD_PARTY_NOTICES.md) are paraphrased and do not match.

A source that cannot be found is reported as not run, never as clean: the one-off copy check
this replaces (release-prep/copycheck.py) had silently compared against nothing for three of
its ten references once uv's cache hashes changed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from release_check import Finding, read_text

# Where interactive app runs keep their traces: a project folder under home, and this repo's
# own early app runs. Benchmark traces live in subfolders and are not read.
APP_RUN_TRACES = ("~/*/traces/run-*.jsonl", "traces/run-*.jsonl")
TYPED = {"run_start": "goal", "clarify_answered": "answer", "ask_user_answered": "answer"}
TYPED |= {"counter_answered": "answer"}
MIN_WORDS = 4  # "yes", "fix those" are everyone's words

# Each reference: globs tried in order, newest match wins. `~` is home; others are repo-relative.
_UV = "~/.cache/uv/archive-v0/*"
REFERENCES: dict[str, tuple[str, ...]] = {
    "mini-swe-agent (MIT)": (
        f"{_UV}/lib/python*/site-packages/minisweagent",
        f"{_UV}/minisweagent",
    ),
    "aider (Apache-2.0)": (
        "~/.cache/mercury-harness/aider-venv/lib/python*/site-packages/aider",
        f"{_UV}/lib/python*/site-packages/aider",
    ),
    "smolagents (Apache-2.0)": (f"{_UV}/lib/python*/site-packages/smolagents", f"{_UV}/smolagents"),
    "swebench (MIT)": (f"{_UV}/lib/python*/site-packages/swebench", f"{_UV}/swebench"),
    "SciCode (Apache-2.0)": ("traces/scicode-data/repo",),
    "astral skills (MIT)": ("~/.claude/plugins/cache/astral-sh",),
    "mattpocock skills (MIT)": (
        "~/.claude/plugins/cache/claude-plugins-official/mattpocock-skills",
    ),
    "anthropic skills": ("~/.claude/plugins/cache/anthropic-agent-skills",),
    "claude-plugins-official": ("~/.claude/plugins/cache/claude-plugins-official",),
    "user skills": ("~/.claude/skills",),
}
COPY_RUN = 6
_TRIVIAL = re.compile(r"[\W_]+|return|pass|else:|try:|finally:|\)|\]|\}|\"\"\"|'''")


# --- words ------------------------------------------------------------------------------


def _expand(pattern: str, root: Path) -> list[str]:
    path = os.path.expanduser(pattern) if pattern.startswith("~") else str(root / pattern)
    return sorted(glob.glob(path))


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def typed_texts(trace_files: Iterable[str]) -> set[str]:
    """Every goal and answer of MIN_WORDS+ words, normalised (lower case, single spaces)."""
    found: set[str] = set()
    for f in trace_files:
        with open(f, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"kind"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                field = TYPED.get(event.get("kind", ""))
                text = _norm(str(event.get(field) or "")) if field else ""
                if len(text.split()) >= MIN_WORDS:
                    found.add(text)
    return found


def quoted(path: str, text: str, phrases: Iterable[str]) -> list[Finding]:
    norm = _norm(text)
    return [Finding("words", path, 0, f"quotes {p[:80]!r}") for p in sorted(phrases) if p in norm]


def check_words(root: Path, files: list[str]) -> tuple[list[Finding], list[str]]:
    traces = [f for pattern in APP_RUN_TRACES for f in _expand(pattern, root)]
    if not traces:
        return [], [f"words: no app-run traces at {', '.join(APP_RUN_TRACES)}"]
    phrases = typed_texts(traces)
    fixtures = " ".join(
        _norm(t) for p in files if p.startswith("bench/") and (t := read_text(root / p))
    )
    phrases = {p for p in phrases if p not in fixtures}
    found: list[Finding] = []
    for p in files:
        if (text := read_text(root / p)) is not None:
            found += quoted(p, text, phrases)
    return found, []


# --- copies -----------------------------------------------------------------------------


def code_lines(text: str, python: bool) -> list[tuple[int, str]]:
    """Lines worth comparing: no blanks, no comments in Python, nothing shorter than 12
    characters, no bare punctuation or keywords."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        s = " ".join(line.split())
        if not s or (python and s.startswith("#")):
            continue
        if len(s) >= 12 and not _TRIVIAL.fullmatch(s):
            out.append((n, s))
    return out


def runs(lines: list[tuple[int, str]], k: int = COPY_RUN) -> Iterator[tuple[str, int]]:
    for i in range(len(lines) - k + 1):
        block = "\n".join(s for _, s in lines[i : i + k])
        yield hashlib.md5(block.encode()).hexdigest(), lines[i][0]


def _source_files(folder: str) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", ".git"}]
        for name in filenames:
            if name.endswith((".py", ".md")):
                yield Path(dirpath) / name


def reference_index(refs: dict[str, str]) -> dict[str, tuple[str, str, int]]:
    index: dict[str, tuple[str, str, int]] = {}
    for name, folder in refs.items():
        for path in _source_files(folder):
            text = read_text(path)
            if text is None:
                continue
            for digest, line in runs(code_lines(text, path.suffix == ".py")):
                index.setdefault(digest, (name, str(path), line))
    return index


def find_references(root: Path) -> tuple[dict[str, str], list[str]]:
    found, missing = {}, []
    for name, patterns in REFERENCES.items():
        matches = [m for p in patterns for m in _expand(p, root) if os.path.isdir(m)]
        if matches:
            found[name] = max(matches, key=os.path.getmtime)
        else:
            missing.append(f"copies: {name} not found (looked in {', '.join(patterns)})")
    return found, missing


def shared_runs(path: str, text: str, index: dict[str, tuple[str, str, int]]) -> list[Finding]:
    by_ref: dict[str, list[tuple[int, str, int]]] = {}
    for digest, line in runs(code_lines(text, path.endswith(".py"))):
        if digest in index:
            name, ref_path, ref_line = index[digest]
            by_ref.setdefault(name, []).append((line, ref_path, ref_line))
    home = os.path.expanduser("~")
    return [
        Finding(
            "copies",
            path,
            hits[0][0],
            f"{len(hits)} shared {COPY_RUN}-line runs with {name}, first "
            f"{hits[0][1].replace(home, '~')}:{hits[0][2]}",
        )
        for name, hits in by_ref.items()
    ]


def check_copies(root: Path, files: list[str]) -> tuple[list[Finding], list[str]]:
    refs, missing = find_references(root)
    index = reference_index(refs)
    found: list[Finding] = []
    for p in files:
        if p.endswith((".py", ".md")) and (text := read_text(root / p)) is not None:
            found += shared_runs(p, text, index)
    return found, missing
