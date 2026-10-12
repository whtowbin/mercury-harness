"""Adapt a project's own test command to look like pytest.

The harness's test tooling assumes pytest everywhere: `agent/failsig.py`'s
regexes and `agent/phases/finalcheck.py`'s `failing_tests` all read pytest's
`--tb=short` output and its `FAILED x::y` / summary-banner conventions. Some
projects do not run pytest at all — django's own `tests/runtests.py
--verbosity 2` prints Python `unittest` verbose output instead, one line per
test plus a closing `FAIL:`/`ERROR:` section per failure. Found on SWE-bench's
django tasks (DESIGN-LOG §69).

Rather than teach every parser a second grammar, this module runs the
project's own command unchanged and APPENDS a pytest-shaped summary block to
its output: a `____ <id> ____` section with converted traceback frames and a
final `E   <exception>` line per failure, `FAILED`/`ERROR` lines, and a
summary banner. Every existing parser keeps working unchanged because it
already reads the *last* matching banner/section in the text, not the first.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

# A bare id line: unittest prints `test_x (package.module.Class)` alone, with
# no trailing " ... ok", when the test has a docstring — the result for that
# test then rides on the *next* printed line instead.
_ID_FULL_RE = re.compile(r"^(?P<name>\S+) \((?P<qual>[\w.]+)\)$")

# The closing `FAIL: test_x (Class)` / `ERROR: test_x (Class)` header that
# opens each failure's traceback section. Authoritative over whatever the
# per-line pass recorded: a subTest failure prints no result of its own on
# its line (see module docstring on `parse_unittest`), so only this header
# says the test actually failed.
_HEADER_RE = re.compile(
    r"^={10,}\n(?P<kind>FAIL|ERROR): (?P<name>\S+) \((?P<qual>[\w.]+)\).*\n-{10,}\n",
    re.MULTILINE,
)

# unittest's traceback frame: `  File "<path>", line <N>, in <func>`.
_FILE_LINE_RE = re.compile(
    r'^  File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)$',
    re.MULTILINE,
)

# The separator right before the final `Ran N tests in ...s` line — the
# closing section's body never runs past this point.
_RAN_RE = re.compile(r"^-{10,}\nRan \d+ tests?", re.MULTILINE)

# unittest's own closing banner, `FAILED (errors=1)` / `FAILED (failures=1,
# errors=2)`. It happens to satisfy `FAILED_ID_RE` too (`FAILED` then
# whitespace then a space-free token) and would otherwise show up as a bogus
# failing-test id called "(errors=1)" once the real appended block is
# concatenated after it. A colon breaks that false match while leaving the
# line's own words untouched (`_HEADER_RE`'s `FAIL:`/`ERROR:` sections
# already use the same colon convention, so this doesn't invent new syntax).
_RUNNER_BANNER_RE = re.compile(r"^FAILED (\([^)\n]*\))$", re.MULTILINE)

# Status words unittest prints verbatim after " ... "; "skipped" additionally
# carries a quoted reason (`skipped 'reason'`), matched by prefix instead of
# by an exact word.
_STATUS_WORDS = {
    "ok": "ok",
    "FAIL": "FAIL",
    "ERROR": "ERROR",
    "expected failure": "expected failure",
    "unexpected success": "unexpected success",
}

_LOADER_FAILURE_PREFIX = "unittest.loader._FailedTest."

# unittest's own closing lines, which it prints at every verbosity: `Ran 7 tests
# in 0.2s`, then `OK`, `OK (skipped=1)` or `FAILED (failures=1, errors=2)`. Without
# `-v` (django's runner at its default verbosity) there are no per-test lines,
# only dots, and the counts come from here: the first live runs read "no tests
# ran" for suites that had run and failed (DESIGN-LOG §70).
_RAN_COUNT_RE = re.compile(r"^Ran (\d+) tests? in", re.MULTILINE)
_OUTCOME_RE = re.compile(r"^(?:OK|FAILED)(?: \((?P<detail>[^)\n]*)\))?\s*$", re.MULTILINE)


def build_argv(template: str, targets: list[str]) -> list[str]:
    """Turn a runner template (e.g. `"python tests/runtests.py {targets}"`)
    plus the targets to run into an argv list.

    A `{targets}` token is replaced in place (dropped entirely for no
    targets, so a bare rerun of the whole suite is still expressible);
    without one, targets are appended, matching how most test runners take
    positional test selectors.
    """
    tokens = shlex.split(template)
    if "{targets}" in tokens:
        expanded: list[str] = []
        for token in tokens:
            if token == "{targets}":
                expanded.extend(targets)
            else:
                expanded.append(token)
        return expanded
    return [*tokens, *targets]


def _canonical_id(name: str, qual: str) -> str:
    """`test_x (package.module.Class)` -> `package.module.Class.test_x` —
    space-free, so it satisfies `FAILED_ID_RE`'s `\\S+` capture.

    Python's own unittest changed what it prints in the parens across the
    versions this harness runs on: older ones print just the class
    (`pagination.tests.PaginationTests`), newer ones print the fully
    qualified test id, method name and all
    (`test_sample.SampleTests.test_docstring`). Appending the name again onto
    an already-qualified id would double it, so when the parenthesised part
    already ends with the name, it *is* the canonical id.
    """
    if qual.endswith(f".{name}"):
        return qual
    return f"{qual}.{name}"


def _canonical_status(word: str) -> str | None:
    word = word.strip()
    if word in _STATUS_WORDS:
        return _STATUS_WORDS[word]
    if word.startswith("skipped"):
        return "skipped"
    return None


def _parse_lines(output: str) -> dict[str, str]:
    """The per-line reading of unittest's verbose output, before closing
    headers get the last word.

    A subTest failure prints no result on its own line — in
    `django_subtest_fail.txt` the line reads `test_parsing_rfc850 (...) ...
    test_parsing_year_less_than_70 (...) ... ok`, where the `ok` belongs to
    the *second* test named on the line, and the first is left unresolved
    here (its real status comes only from the closing `FAIL:` section).
    """
    statuses: dict[str, str] = {}
    pending: str | None = None
    for line in output.splitlines():
        if " ... " not in line:
            match = _ID_FULL_RE.match(line.strip())
            pending = _canonical_id(match.group("name"), match.group("qual")) if match else pending
            continue
        parts = line.split(" ... ")
        status = _canonical_status(parts[-1])
        if status is None:
            pending = None
            continue
        if pending is not None:
            statuses[pending] = status
            pending = None
            continue
        match = _ID_FULL_RE.match(parts[-2].strip())
        if match:
            statuses[_canonical_id(match.group("name"), match.group("qual"))] = status
    return statuses


def _closing_sections(output: str) -> dict[str, tuple[str, str]]:
    """id -> (FAIL|ERROR, body text) for each closing section, in the order
    the headers appear."""
    headers = list(_HEADER_RE.finditer(output))
    ran = _RAN_RE.search(output)
    end_default = ran.start() if ran else len(output)
    sections: dict[str, tuple[str, str]] = {}
    for index, head in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else end_default
        body = output[head.end() : end]
        tid = _canonical_id(head.group("name"), head.group("qual"))
        sections[tid] = (head.group("kind"), body)
    return sections


def parse_unittest(output: str) -> dict[str, str]:
    """Map each test id to one of `ok`, `FAIL`, `ERROR`, `skipped`,
    `expected failure`, `unexpected success`.

    The per-line pass is overridden by the closing section's `FAIL:`/`ERROR:`
    headers, which are authoritative (see `_parse_lines`).
    """
    statuses = _parse_lines(output)
    for tid, (kind, _body) in _closing_sections(output).items():
        statuses[tid] = kind
    return statuses


def _relativize(path: str, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path


def _convert_frames(body: str, root: Path) -> list[str]:
    """unittest's `  File "<path>", line N, in <func>` -> pytest's short
    `<path>:<N>: in <func>`, workspace paths made relative so
    `failure_details` can still tell a library frame (left absolute) from
    project code."""
    return [
        f"{_relativize(match.group('path'), root)}:{match.group('line')}: in {match.group('func')}"
        for match in _FILE_LINE_RE.finditer(body)
    ]


def _final_exception_line(body: str) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _bucket_counts(statuses: dict[str, str]) -> dict[str, int]:
    """Fold unittest's six statuses into the four the summary banner and
    `failed_count` know about. `expected failure` is a successful outcome
    (it ran fine) so it joins `passed`; `unexpected success` is the one
    outcome unittest itself fails the run over, so it joins `failed`. Neither
    is in the spec's own banner vocabulary, so this mapping is a judgment
    call, not something a fixture pins down."""
    counts = {"failed": 0, "errors": 0, "passed": 0, "skipped": 0}
    for status in statuses.values():
        if status in ("ok", "expected failure"):
            counts["passed"] += 1
        elif status == "skipped":
            counts["skipped"] += 1
        elif status in ("FAIL", "unexpected success"):
            counts["failed"] += 1
        elif status == "ERROR":
            counts["errors"] += 1
    return counts


def _runner_counts(output: str) -> dict[str, int] | None:
    """failed/errors/passed/skipped from unittest's closing `Ran N tests` and
    `OK`/`FAILED (...)` lines, or None when it printed neither."""
    ran = list(_RAN_COUNT_RE.finditer(output))
    if not ran:
        return None
    total = int(ran[-1].group(1))
    detail = {}
    outcomes = list(_OUTCOME_RE.finditer(output, ran[-1].end()))
    if outcomes and outcomes[-1].group("detail"):
        for part in outcomes[-1].group("detail").split(","):
            key, _, value = part.strip().partition("=")
            if value.strip().isdigit():
                detail[key.strip()] = int(value)
    failed = detail.get("failures", 0) + detail.get("unexpected successes", 0)
    errors = detail.get("errors", 0)
    skipped = detail.get("skipped", 0)
    return {
        "failed": failed,
        "errors": errors,
        "passed": max(0, total - failed - errors - skipped),
        "skipped": skipped,
    }


def _summary_banner(counts: dict[str, int], elapsed_s: float) -> str:
    parts = []
    if counts["failed"]:
        parts.append(f"{counts['failed']} failed")
    if counts["errors"]:
        parts.append(f"{counts['errors']} {'error' if counts['errors'] == 1 else 'errors'}")
    if counts["passed"]:
        parts.append(f"{counts['passed']} passed")
    if counts["skipped"]:
        parts.append(f"{counts['skipped']} skipped")
    body = ", ".join(parts) if parts else "no tests ran"
    return f"==== {body} in {elapsed_s:.2f}s ===="


def _derive_exit_code(
    statuses: dict[str, str], counts: dict[str, int], output: str, exit_code: int
) -> int:
    """pytest's convention: 0 clean, 1 some failed/errored, 5 nothing ran,
    2 the runner itself never got to run anything asked for.

    A `unittest.loader._FailedTest` entry means an import blew up before any
    real test collected — exit 2 regardless of how that one pseudo-test's
    status reads, since nothing that was actually asked for ran
    (`django_bad_label.txt`). When nothing at all parsed, a traceback or
    Import/ModuleNotFoundError in the text means the runner crashed before
    printing any result (exit 2); otherwise the caller's own exit code is the
    only other signal available, and a nonzero one still says something
    stopped it (exit 2), leaving a clean nonzero-test empty run as the
    remaining case, "no tests ran" (exit 5).
    """
    if any(tid.startswith(_LOADER_FAILURE_PREFIX) for tid in statuses):
        return 2
    if sum(counts.values()) and not statuses:
        # Non-verbose output: no per-test lines, but the runner's own tally.
        return 1 if counts["failed"] or counts["errors"] else 0
    if not statuses:
        crashed = (
            "Traceback (most recent call last):" in output
            or "ImportError" in output
            or "ModuleNotFoundError" in output
        )
        return 2 if crashed or exit_code != 0 else 5
    return 1 if counts["failed"] or counts["errors"] else 0


def _failure_block(tid: str, kind: str, body: str, root: Path) -> tuple[str, str]:
    """(pytest-shaped section text, `FAILED`/`ERROR` summary line) for one
    closing section."""
    frames = _convert_frames(body, root)
    final_line = _final_exception_line(body)
    section = "\n".join([f"____ {tid} ____", *frames, f"E   {final_line}"])
    marker = "ERROR" if kind == "ERROR" else "FAILED"
    return section, f"{marker} {tid} - {final_line}"


def pytest_shaped(
    output: str, exit_code: int, elapsed_s: float, root: str | Path
) -> tuple[int, str]:
    """Run a non-pytest test command's raw output through, largely unchanged,
    and append the pytest-shaped block `failsig.py` and
    `finalcheck.failing_tests` read: one section per failure, `FAILED`/
    `ERROR` lines, and a summary banner. Returns `(derived_exit_code,
    output_with_block_appended)`.

    The one edit to the runner's own text is `_RUNNER_BANNER_RE`'s colon —
    see its comment for why leaving that byte alone would misinform every
    downstream parser rather than just this one.
    """
    root_path = Path(root)
    statuses = parse_unittest(output)
    sections = _closing_sections(output)

    blocks: list[str] = []
    failed_lines: list[str] = []
    for tid, (kind, body) in sections.items():
        section, failed_line = _failure_block(tid, kind, body, root_path)
        blocks.append(section)
        failed_lines.append(failed_line)

    counts = _bucket_counts(statuses)
    # The runner's own tally wins when it accounts for more tests than the
    # per-test lines do, which is every run without `-v`.
    tally = _runner_counts(output)
    if tally is not None and sum(tally.values()) > sum(counts.values()):
        counts = tally
    banner = _summary_banner(counts, elapsed_s)
    appended = "\n\n".join([*blocks, *failed_lines, banner])
    safe_output = _RUNNER_BANNER_RE.sub(r"FAILED: \1", output)
    new_output = safe_output.rstrip("\n") + "\n\n" + appended + "\n"
    return _derive_exit_code(statuses, counts, output, exit_code), new_output
