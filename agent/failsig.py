"""What a pytest failure says, boiled down: which test, where, and the error.

Two callers need the same reading of `--tb=short` output. The tool loop's
test watcher (runner.py) reads one `run_tests` result to notice a failure
that edits are not changing. The baseline (phases/finalcheck.py) reads the
whole pre-run suite so the planner is told *why* tests are red, not only
their names: the baseline run already had the full traceback in hand and kept
just the node ids, so a plan for "fix the failing tests" was written blind to
the assertion, the file, and whether the suite even collected.

Deterministic and model-free. The parts that a regex cannot read (a non-pytest
runner, a traceback with no workspace frame) come back empty rather than
guessed at.
"""

from __future__ import annotations

import re

FAILED_ID_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_COUNT_RE = re.compile(r"(\d+) (failed|errors?)\b")
FRAME_RE = re.compile(r"^(?P<file>[^\s:]+\.py):(?P<line>\d+): in (?P<func>\S+)", re.MULTILINE)
ERROR_LINE_RE = re.compile(r"^E\s+(?P<msg>.+)$", re.MULTILINE)
# `____ test_name ____` heads each failure's section; collection errors read
# `____ ERROR collecting tests/x.py ____`.
_SECTION_RE = re.compile(r"^_{2,}\s+(?P<name>.+?)\s+_{2,}$", re.MULTILINE)

# The summary line: counts, then the timing pytest always ends it with —
# "1 failed, 2 passed in 0.03s", "no tests ran in 0.01s", optionally inside
# "====" banners. Requiring the timing is what keeps a traceback line such as
# `E  AssertionError: 3 failed to connect` from standing in for it.
_SUMMARY = re.compile(
    r"^=*\s*(?P<body>(?:\d+ \w+(?:, \d+ \w+)*|no tests ran) in [\d.]+s(?: \([\d:]+\))?)\s*=*$",
    re.IGNORECASE | re.MULTILINE,
)

MAX_DIGEST_FAILURES = 5
MAX_DIGEST_CHARS = 1_500


def pytest_summary(output: str, exit_code: int) -> str:
    """pytest's real summary is its *last* banner.

    Earlier banners are section headers — `==== ERRORS ====`, `==== FAILURES ====` —
    and matching one of those reports "ERRORS" for a run that actually passed.

    The one implementation: `run_tests` and the final gate each had their own,
    and they were not equal — the gate's scanned lines for any digit plus any of
    four keywords, so a traceback line such as `E  AssertionError: 3 failed to
    connect` could stand in for the summary. This one requires the counts *and*
    the trailing timing.
    """
    matches = [m.group("body").strip() for m in _SUMMARY.finditer(output or "")]
    return matches[-1] if matches else f"pytest exited {exit_code}"


def failed_count(text: str | None) -> int | None:
    """How many tests a pytest summary says went wrong, or None when it does
    not say.

    Errors count alongside failures: a collection error reports `1 error` and
    no failures at all, and reading that as zero says the suite improved at the
    exact moment an edit stopped it running. None rather than 0 for an
    unreadable summary, for the same reason — "cannot tell" must not be
    mistaken for "nothing wrong".
    """
    # Summed, not the first match: "2 failed, 1 error" is three tests gone
    # wrong. Reading only the first count under-reported every run that had
    # both (found building agent/testcmd.py, DESIGN-LOG §70).
    counts = [int(m.group(1)) for m in _COUNT_RE.finditer(_summary_part(text or ""))]
    if counts:
        return sum(counts)
    return 0 if "passed" in (text or "") else None


def _summary_part(text: str) -> str:
    """pytest's last summary banner when there is one, else the whole text.

    Summing counts over a whole output would also add up the "N failed" a
    traceback or a test name happens to contain; the banner is the one place the
    counts are pytest's own. Without a banner (a bare "1 failed" string, as the
    callers pass) the text is all there is.
    """
    banners = [m.group("body") for m in _SUMMARY.finditer(text)]
    return banners[-1] if banners else text


def failure_details(text: str) -> tuple[frozenset[str], int, str]:
    """(failing test ids, failed count, "file:line in func -- error") from one
    run_tests result. The deepest frame outside a test file is where the
    failure actually surfaced in the code being written; the test's own
    frame only says which assertion tripped."""
    ids = frozenset(FAILED_ID_RE.findall(text))
    count = failed_count(text) or 0
    frames = list(FRAME_RE.finditer(text))
    # pytest prints workspace frames relative and library frames absolute, so an
    # absolute path is the stdlib or site-packages: a collection error's deepest
    # frame is importlib, which says nothing about the project.
    local = [f for f in frames if not f.group("file").startswith("/")]
    code = [f for f in local if not f.group("file").rsplit("/", 1)[-1].startswith("test_")]
    frame = (code or local or frames or [None])[-1]
    where = ""
    if frame is not None:
        where = f"{frame.group('file')}:{frame.group('line')} in {frame.group('func')}"
        err = _error_after(text, frame.end())
        if err:
            where += f" -- {err[:160]}"
    return ids, count, where


def _error_after(text: str, pos: int) -> str:
    match = ERROR_LINE_RE.search(text, pos)
    return match.group("msg") if match else ""


def failure_digest(output: str, limit: int = MAX_DIGEST_FAILURES) -> list[tuple[str, str]]:
    """(test name, "where -- error") per failing section of a whole-suite run.

    A collection error has no test frame, only `E   ImportError: ...` lines, so
    those are read directly: that first error line is the whole diagnosis.
    """
    output = output or ""
    heads = list(_SECTION_RE.finditer(output))
    digest: list[tuple[str, str]] = []
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(output)
        section = output[head.end() : end]
        name = head.group("name")
        _, _, where = failure_details(section)
        if not where:
            error = _error_after(section, 0)
            where = error[:160]
        if where:
            digest.append((name, where))
        if len(digest) >= limit:
            break
    return digest


def format_digest(digest: list[tuple[str, str]], total: int = 0) -> str:
    """The digest as lines for a prompt, capped so it cannot dominate one."""
    lines = [f"  - {name}: {where}" for name, where in digest]
    text = "\n".join(lines)
    if len(text) > MAX_DIGEST_CHARS:
        text = text[:MAX_DIGEST_CHARS].rsplit("\n", 1)[0]
    if total > len(digest):
        text += f"\n  (+{total - len(digest)} more)"
    return text
