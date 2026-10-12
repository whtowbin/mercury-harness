"""ty's output, parsed as ty actually writes it.

This module's parser expected `error[rule] path:line:col: message` since the
initial commit. ty emits `path:line:col: error[rule] message` — location
first, severity second — so _parse_concise matched nothing and ty_check
returned an empty list for every project it was ever pointed at. Across the
recorded traces all 46 final checks reported type_errors: 0 and all 42
check_code(types=true) calls reported "ty: 0 issue(s)", on code that had type
errors. The strings below are copied from real ty 0.0.80 output.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent import validate
from agent.validate import _parse_concise

ROOT = Path("/proj")


# -- the format ty actually emits ----------------------------------------


def test_the_real_concise_format_is_parsed():
    d = _parse_concise(
        ROOT,
        "mod.py:7:16: error[invalid-argument-type] Argument to "
        "function `greet` is incorrect: Expected `str`, found `Literal[42]`",
    )
    assert d is not None
    assert (d.file, d.line, d.column) == ("mod.py", 7, 16)
    assert d.code == "invalid-argument-type"
    assert "Expected `str`" in d.message


def test_a_warning_is_parsed_too():
    d = _parse_concise(ROOT, "a/b.py:12:3: warning[possibly-unbound] `x` may be unbound")
    assert d is not None and d.code == "possibly-unbound" and d.line == 12


def test_a_severity_without_a_rule_still_parses():
    d = _parse_concise(ROOT, "mod.py:1:1: error Something went wrong")
    assert d is not None and d.code == "error"


def test_a_path_with_directories_is_parsed():
    d = _parse_concise(ROOT, "src/pkg/mod.py:3:9: error[bad-return-type] nope")
    assert d is not None and d.file == "src/pkg/mod.py"


# -- the order this module used to assume, kept as a fallback -------------


def test_the_legacy_order_still_parses():
    """So a future ty that reverts the order does not silently zero the gate."""
    d = _parse_concise(ROOT, "error[bad-return-type] mod.py:3:9: nope")
    assert d is not None and d.file == "mod.py" and d.line == 3


# -- lines that are not diagnostics --------------------------------------


def test_summary_and_status_lines_are_ignored():
    for line in (
        "Found 3 diagnostics",
        "All checks passed!",
        "",
        "warning: `x` is deprecated",
        "error: could not open file",
    ):
        assert _parse_concise(ROOT, line) is None


# -- end to end against the real tool ------------------------------------


@pytest.mark.skipif(not (shutil.which("ty") or shutil.which("uvx")), reason="ty is not available")
def test_ty_check_finds_a_real_cross_file_error(tmp_path):
    (tmp_path / "mod.py").write_text("def greet(name: str) -> str:\n    return name\n")
    (tmp_path / "caller.py").write_text("from mod import greet\n\ngreet('a', 'b')\n")
    found = validate.ty_check(tmp_path)
    assert found, "ty reports this error on the command line; the parser must see it too"
    assert any("greet" in d.message for d in found)


@pytest.mark.skipif(not (shutil.which("ty") or shutil.which("uvx")), reason="ty is not available")
def test_ty_check_agrees_with_the_command_line(tmp_path):
    """The regression that matters: parsed count == what ty printed."""
    (tmp_path / "m.py").write_text("def f(x: int) -> int:\n    return x\n\nf('a')\nf('b')\n")
    argv = ["ty"] if shutil.which("ty") else ["uvx", "ty"]
    raw = subprocess.run(
        [*argv, "check", "--output-format", "concise", "m.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    printed = sum(1 for line in raw.stdout.splitlines() if "error[" in line or "warning[" in line)
    assert len(validate.ty_check(tmp_path, ["m.py"])) == printed


@pytest.mark.skipif(not (shutil.which("ty") or shutil.which("uvx")), reason="ty is not available")
def test_the_full_context_names_where_the_other_side_is_defined(tmp_path):
    (tmp_path / "mod.py").write_text("def greet(name: str) -> str:\n    return name\n")
    (tmp_path / "caller.py").write_text("from mod import greet\n\ngreet('a', 'b')\n")
    context = validate.ty_context(tmp_path, ["caller.py"])
    assert "mod.py" in context, "the fixer needs the other half of the mismatch"
    assert "caller.py" in context
