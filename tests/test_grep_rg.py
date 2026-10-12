"""grep: the ripgrep fast path, and the Python fallback it must never regress.

No test here assumes rg is or isn't installed on the machine running the
suite: the ones that specifically exercise the fallback force it (by making
`toolchain.run_capturing` behave as if rg were absent, or by using a pattern
rg's engine cannot parse), so they hold regardless of environment.
"""

from __future__ import annotations

import pytest

from agent import toolchain
from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.fs import grep, write_file
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _no_rg(monkeypatch):
    """Simulate rg not being on PATH: `_rg_grep` sees a 127 and returns None,
    same as it would if the binary genuinely were not installed."""
    real = toolchain.run_capturing

    def fake(argv, **kwargs):
        if argv[0] == "rg":
            return toolchain.ProcResult(127, "", "rg: command not found", 0.0)
        return real(argv, **kwargs)

    monkeypatch.setattr(toolchain, "run_capturing", fake)


# -- basic correctness, whichever path answers ----------------------------


def test_grep_finds_a_literal_match(ctx):
    write_file(ctx, "a.py", "def target_function():\n    return 1\n")
    result = grep(ctx, "target_function")
    assert "a.py:1:" in result


def test_grep_reports_no_matches(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    assert "no matches" in grep(ctx, "zzz_not_present_zzz")


def test_grep_respects_glob_filter(ctx):
    write_file(ctx, "a.py", "needle = 1\n")
    write_file(ctx, "a.txt", "needle = 1\n")
    result = grep(ctx, "needle", glob="*.py")
    assert "a.py" in result and "a.txt" not in result


def test_grep_is_case_insensitive_when_asked(ctx):
    write_file(ctx, "a.py", "NeedleValue = 1\n")
    assert "no matches" in grep(ctx, "needlevalue")
    assert "a.py" in grep(ctx, "needlevalue", ignore_case=True)


def test_grep_bad_python_regex_is_rejected_before_any_search(ctx):
    write_file(ctx, "a.py", "x = 1\n")
    result = grep(ctx, "(unclosed")
    assert result.startswith("error: bad regex")


def test_grep_ignores_denylisted_directories(ctx):
    (ctx.workspace.root / "__pycache__").mkdir()
    (ctx.workspace.root / "__pycache__" / "a.py").write_text("needle = 1\n")
    write_file(ctx, "b.py", "needle = 1\n")
    result = grep(ctx, "needle")
    assert "b.py" in result
    assert "__pycache__" not in result


def test_grep_caps_results_and_says_so(ctx):
    """Spread across files, not lines within one: a single file's own match
    count is capped silently by construction (see _grep_one_file / rg's own
    --max-count), so only enough *files* each contributing a match exercises
    the cap this test is actually for."""
    for i in range(10):
        write_file(ctx, f"f{i}.py", f"needle_{i} = {i}\n")
    result = grep(ctx, "needle_", max_results=5)
    assert "suppressed" in result


# -- the fallback path specifically ----------------------------------------


def test_grep_falls_back_to_python_when_rg_is_missing(ctx, monkeypatch):
    _no_rg(monkeypatch)
    write_file(ctx, "a.py", "def target_function():\n    return 1\n")
    result = grep(ctx, "target_function")
    assert "a.py:1:" in result


def test_grep_fallback_respects_glob_and_case(ctx, monkeypatch):
    _no_rg(monkeypatch)
    write_file(ctx, "a.py", "NeedleValue = 1\n")
    write_file(ctx, "a.txt", "NeedleValue = 1\n")
    result = grep(ctx, "needlevalue", glob="*.py", ignore_case=True)
    assert "a.py" in result and "a.txt" not in result


def test_a_pattern_rgs_engine_rejects_still_works_via_fallback(ctx):
    """Backreferences are valid Python `re` (this tool's documented contract)
    but rg's Rust regex engine rejects them outright. `_rg_grep` must
    recognise that rejection and fall back rather than reporting a false
    'no matches' for a pattern it never actually evaluated."""
    write_file(ctx, "a.py", "abab\nabcd\n")
    # (\w\w)\1 : two identical two-character groups back to back — "abab"
    # matches (group="ab", repeated), "abcd" does not.
    result = grep(ctx, r"(\w\w)\1")
    assert "a.py:1:" in result
    assert "a.py:2:" not in result
