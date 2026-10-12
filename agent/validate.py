"""Code validation: AST parse, then Ruff, then ty.

The point of this module is the feedback loop. A model that learns its edit broke the
file in the *same tool result* as the edit can fix it on the next turn; one that finds
out three steps later usually cannot. So the cheap checks run automatically after every
write, and the expensive one is a tool the model calls deliberately.

Cost ordering, measured on this project:
  ast.parse   ~1ms    always on, catches syntax breakage
  ruff        ~15ms   on by default, catches undefined names and unused imports
  ty          ~1s     opt-in, catches signature and type disagreements across files
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import toolchain

MAX_DIAGNOSTICS = 12


@dataclass
class Diagnostic:
    file: str
    line: int
    column: int
    code: str
    message: str
    fix: str = ""

    def render(self) -> str:
        where = f"{self.file}:{self.line}:{self.column}"
        body = f"{where}: {self.code} {self.message}"
        return f"{body}\n    fix: {self.fix}" if self.fix else body


# Ruff rules that indicate real breakage (pyflakes, syntax, bugbear, bandit), as against
# style. after_write surfaces only these, and a lint verify blocks only on these (§110). A
# letter then a digit, so SIM, SLF and SLOT (style) are not read as bandit's S.
_SERIOUS_RULE = re.compile(r"^(F|E9|B|S)\d")


def is_serious(d: Diagnostic) -> bool:
    return bool(_SERIOUS_RULE.match(d.code))


def syntax_check(path: Path, source: str | None = None) -> Diagnostic | None:
    """Parse a Python file. This is the check that must never be skipped."""
    if path.suffix != ".py":
        return None
    try:
        text = source if source is not None else path.read_text(encoding="utf-8")
    except OSError as exc:
        return Diagnostic(str(path), 0, 0, "IO", str(exc))
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        line = exc.lineno or 0
        context = ""
        lines = text.splitlines()
        if 0 < line <= len(lines):
            context = f"\n    {line}\t{lines[line - 1]}"
            if exc.offset and exc.offset > 0:
                context += "\n    " + " " * (len(str(line)) + 1) + " " * (exc.offset - 1) + "^"
        return Diagnostic(str(path), line, exc.offset or 0, "SyntaxError", f"{exc.msg}{context}")
    return None


def ruff_check(
    root: Path, paths: list[str], timeout: int = 60, options: list[str] | None = None
) -> list[Diagnostic]:
    """Lint with Ruff, parsed from its JSON output. `options` are extra Ruff
    arguments (a `--select`, say) applied as given."""
    if not paths:
        return []
    code, out, err = toolchain.run(
        "ruff",
        ["check", "--output-format", "json", "--force-exclude", *(options or []), *paths],
        root,
        timeout=timeout,
    )
    return _ruff_result(root, code, out, err)


def ruff_check_text(
    root: Path, rel_path: str, source: str, options: list[str] | None = None, timeout: int = 60
) -> list[Diagnostic]:
    """Ruff on `source` as if it were `rel_path`, through stdin, so the project's
    own configuration for that path applies: how a file read before an edit."""
    argv = toolchain.resolve("ruff", str(root))
    if not argv:
        return []
    try:
        proc = subprocess.run(
            [
                *argv,
                "check",
                "--output-format",
                "json",
                *(options or []),
                "--stdin-filename",
                rel_path,
                "-",
            ],
            cwd=root,
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=toolchain.workspace_env(root),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return _ruff_result(root, proc.returncode, proc.stdout, proc.stderr)


def _ruff_result(root: Path, code: int, out: str, err: str) -> list[Diagnostic]:
    if code == 127:
        return []
    if not out.strip():
        if code not in (0, 1) and err.strip():
            return [Diagnostic("", 0, 0, "ruff", err.strip()[:400])]
        return []
    try:
        items = json.loads(out)
    except json.JSONDecodeError:
        return []
    diagnostics: list[Diagnostic] = []
    for item in items:
        location = item.get("location") or {}
        fix = (item.get("fix") or {}).get("message") or ""
        diagnostics.append(
            Diagnostic(
                file=_relative(root, item.get("filename", "")),
                line=int(location.get("row", 0) or 0),
                column=int(location.get("column", 0) or 0),
                code=str(item.get("code") or "RUFF"),
                message=str(item.get("message") or ""),
                fix=fix,
            )
        )
    return diagnostics


def ty_check(root: Path, paths: list[str] | None = None, timeout: int = 180) -> list[Diagnostic]:
    """Type check with ty. Slower, and it reads the whole project to resolve imports."""
    args = ["check", "--output-format", "concise"]
    if paths:
        args.extend(paths)
    code, out, err = toolchain.run("ty", args, root, timeout=timeout)
    if code == 127:
        return []
    if code == 124:
        return [Diagnostic("", 0, 0, "ty", err or "ty timed out")]
    diagnostics: list[Diagnostic] = []
    for line in (out or "").splitlines():
        parsed = _parse_concise(root, line)
        if parsed:
            diagnostics.append(parsed)
    return diagnostics


# Enough for a handful of errors with their definition sites; past that the
# model should fix some and re-check rather than read a wall of them.
TY_CONTEXT_CHAR_CAP = 4_000


def ty_context(root: Path, paths: list[str] | None = None, timeout: int = 180) -> str:
    """ty's full diagnostics, for the model to read rather than for gating.

    The concise format this module parses gives a location and a message. The
    full format adds the line itself and, crucially, an "info: Function defined
    here" pointer at the *other* end of the mismatch:

        error[invalid-argument-type]: Argument to function `greet` is incorrect
         --> mod.py:7:16
        info: Function defined here
         --> mod.py:1:5

    That second location is the half of a cross-file defect the fixer cannot
    see from its own file — the same gap review.py's `files` field exists to
    close — and it was being thrown away because concise does not carry it.

    Only called when concise already found errors, so the extra ty invocation
    lands on a path that is about to spend a fix round anyway, never on the
    common path where the types are fine.
    """
    args = ["check", "--output-format", "full"]
    if paths:
        args.extend(paths)
    code, out, err = toolchain.run("ty", args, root, timeout=timeout)
    if code in (124, 127):
        return ""
    text = (out or err or "").strip()
    if len(text) > TY_CONTEXT_CHAR_CAP:
        text = text[:TY_CONTEXT_CHAR_CAP] + "\n... (truncated; fix these and re-check)"
    return text


# ty's concise line, as ty actually emits it:
#     mod.py:7:16: error[invalid-argument-type] Argument to `greet` is incorrect
# The location comes first and the severity second. This module had it the
# other way round — "error[rule] path:line:col: message" — since the initial
# commit, so _parse_concise matched nothing and ty_check returned an empty
# list for every project it was ever pointed at. Across the recorded traces,
# all 46 final checks reported type_errors: 0 and all 42 check_code(types=true)
# calls reported "ty: 0 issue(s)", on codebases that did have type errors.
# A gate that cannot fail is not a gate.
_CONCISE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<severity>error|warning|info)(?:\[(?P<code>[^\]]+)\])?\s+(?P<message>.*)$"
)
# The shape this module used to expect. Kept because it costs one regex and
# means a future ty that reverts the order does not silently zero the gate again.
_CONCISE_LEGACY = re.compile(
    r"^(?P<severity>error|warning|info)(?:\[(?P<code>[^\]]+)\])?\s+"
    r"(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<message>.*)$"
)


def _parse_concise(root: Path, line: str) -> Diagnostic | None:
    """One `ty --output-format concise` line, in either field order."""
    line = line.strip()
    if not line or line.startswith(("Found", "All checks", "warning: ", "error: ")):
        return None
    match = _CONCISE.match(line) or _CONCISE_LEGACY.match(line)
    if not match:
        return None
    return Diagnostic(
        file=_relative(root, match["file"]),
        line=int(match["line"]),
        column=int(match["col"]),
        code=match["code"] or match["severity"],
        message=match["message"].strip(),
    )


def _relative(root: Path, filename: str) -> str:
    if not filename:
        return ""
    try:
        return str(Path(filename).resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return filename


def render(diagnostics: list[Diagnostic], limit: int = MAX_DIAGNOSTICS) -> str:
    if not diagnostics:
        return ""
    shown = diagnostics[:limit]
    body = "\n".join(d.render() for d in shown)
    if len(diagnostics) > limit:
        body += f"\n... and {len(diagnostics) - limit} more"
    return body


def after_write(root: Path, rel_path: str, source: str | None = None, lint: bool = True) -> str:
    """The automatic post-edit check. Returns "" when the file is clean.

    Syntax errors short-circuit the lint pass: Ruff on an unparseable file only
    restates the same error in a less useful form.
    """
    path = root / rel_path
    if path.suffix != ".py":
        return ""
    broken = syntax_check(path, source)
    if broken:
        return (
            "\n\n*** SYNTAX ERROR — you broke this file, fix it before doing "
            f"anything else ***\n{broken.render()}"
        )
    if not lint:
        return ""
    diagnostics = ruff_check(root, [str(path)])
    # Style nits are noise mid-step; surface the ones that indicate real breakage.
    serious = [d for d in diagnostics if is_serious(d)]
    if not serious:
        return ""
    return "\n\nruff:\n" + render(serious, limit=6)
