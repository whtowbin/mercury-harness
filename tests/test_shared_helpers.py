"""The helpers extracted from code that had grown several copies of each.

These tests exist mostly to keep the copies from coming back: each one names
a behaviour that previously differed between two call sites doing the same
job.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.phases.packing import batch_to_budget, fit_to_budget, load_files
from agent.text import clip, numbered, truncate
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import IGNORED_DIRS, Workspace, exclude_globs, is_ignored, walk_files


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- line numbering -------------------------------------------------------


def test_numbering_starts_at_one():
    assert numbered("a\nb") == "1\ta\n2\tb"


def test_numbering_can_start_elsewhere_for_a_windowed_read():
    assert numbered("x\ny", start=10) == "10\tx\n11\ty"


def test_numbering_an_empty_string_is_empty():
    assert numbered("") == ""


def test_numbering_keeps_blank_lines():
    assert numbered("a\n\nb") == "1\ta\n2\t\n3\tb"


# -- clip vs truncate -----------------------------------------------------


def test_clip_leaves_short_text_alone():
    assert clip("short", 20) == "short"


def test_clip_marks_the_cut_without_reporting_a_count():
    """The distinction from truncate(): a status line must not grow a
    "+N chars elided" tail longer than the thing it is summarizing."""
    clipped = clip("x" * 50, 10)
    assert clipped.endswith("…")
    assert "chars" not in clipped
    assert len(clipped) <= 11


def test_truncate_still_reports_what_it_dropped():
    assert "chars elided" in truncate("x" * 50, 10)


# -- one ignore-dir set ---------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "traces",
    ],
)
def test_the_usual_suspects_are_all_ignored(name, tmp_path):
    assert is_ignored(tmp_path / name / "f.py")


def test_a_normal_path_is_not_ignored(tmp_path):
    assert not is_ignored(tmp_path / "src" / "mod.py")


def test_ignoring_matches_at_any_depth(tmp_path):
    assert is_ignored(tmp_path / "src" / ".venv" / "lib" / "mod.py")


def test_caches_that_one_copy_used_to_miss_are_covered():
    """glob_files scanned .pytest_cache and .mypy_cache because its own copy
    of the skip set had drifted; workspace_size_mb counted traces/."""
    assert {".pytest_cache", ".mypy_cache", "traces"} <= set(IGNORED_DIRS)


# -- one pruned traversal, instead of six copies of unpruned-then-filter --


def test_walk_files_prunes_an_ignored_directory_rather_than_filtering_it(tmp_path):
    """The actual point, not just the end result: a huge .venv must never be
    opened at all. Proven by planting a file the walk would have had to
    *read* the directory to find, then asserting it never shows up -- the
    same observable result an unpruned-then-filtered walk would also
    produce, but for a much more expensive reason."""
    (tmp_path / "a.py").write_text("x\n")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "dep.py").write_text("y\n")
    found = {str(p.relative_to(tmp_path)) for p in walk_files(tmp_path)}
    assert found == {"a.py"}


def test_walk_files_skip_hidden_also_prunes_dot_directories(tmp_path):
    (tmp_path / "a.py").write_text("x\n")
    hidden = tmp_path / ".config"
    hidden.mkdir()
    (hidden / "b.py").write_text("y\n")
    without_skip = {p.name for p in walk_files(tmp_path)}
    with_skip = {p.name for p in walk_files(tmp_path, skip_hidden=True)}
    assert without_skip == {"a.py", "b.py"}
    assert with_skip == {"a.py"}


def test_walk_files_on_a_nonexistent_root_is_empty_not_an_error(tmp_path):
    assert list(walk_files(tmp_path / "nope")) == []


def test_walk_files_on_a_file_not_a_directory_is_empty_not_an_error(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x\n")
    assert list(walk_files(f)) == []


def test_walk_files_does_not_hang_on_a_symlink_cycle(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "loop").symlink_to(d)
    # The only assertion that matters is that this line returns at all;
    # os.walk's default followlinks=False is what keeps it from recursing
    # into its own symlink forever.
    assert list(walk_files(tmp_path)) == []


def test_exclude_globs_uses_the_given_flag_name(tmp_path):
    """rg spells this flag --glob (singular); ast-grep spells it --globs
    (plural). One list of directories, two flag names -- the parameter this
    exists for, not two near-identical list comprehensions in two files."""
    rg_style = exclude_globs("--glob")
    astgrep_style = exclude_globs("--globs")
    assert all(a.startswith("--glob=!") for a in rg_style)
    assert all(a.startswith("--globs=!") for a in astgrep_style)
    assert len(rg_style) == len(IGNORED_DIRS)


# -- loading files into a prompt -----------------------------------------


def test_loading_numbers_the_lines(ctx):
    (ctx.workspace.root / "a.py").write_text("one\ntwo\n")
    loaded, missing = load_files(ctx, ["a.py"], 1000)
    assert missing == []
    assert loaded[0].body == "1\tone\n2\ttwo"


def test_a_missing_file_is_reported_not_raised(ctx):
    loaded, missing = load_files(ctx, ["nope.py"], 1000)
    assert loaded == []
    assert missing == ["nope.py"]


def test_a_path_escaping_the_workspace_is_reported_as_missing(ctx):
    loaded, missing = load_files(ctx, ["../outside.py"], 1000)
    assert loaded == []
    assert missing == ["../outside.py"]


def test_a_directory_is_reported_as_missing(ctx):
    (ctx.workspace.root / "pkg").mkdir()
    _, missing = load_files(ctx, ["pkg"], 1000)
    assert missing == ["pkg"]


def test_an_oversized_file_is_capped_and_flagged(ctx):
    (ctx.workspace.root / "big.py").write_text("x" * 5000)
    loaded, _ = load_files(ctx, ["big.py"], per_file_cap=100)
    assert loaded[0].truncated
    assert len(loaded[0].body) < 5000


def test_a_file_under_the_cap_is_not_flagged_as_truncated(ctx):
    (ctx.workspace.root / "small.py").write_text("x" * 50)
    loaded, _ = load_files(ctx, ["small.py"], per_file_cap=100)
    assert not loaded[0].truncated


# -- fitting into one budget (the execute phase's strategy) --------------


def test_everything_that_fits_is_included(ctx):
    (ctx.workspace.root / "a.py").write_text("x\n")
    loaded, _ = load_files(ctx, ["a.py"], 1000)
    included, skipped = fit_to_budget(loaded, 1000)
    assert len(included) == 1
    assert skipped == []


def test_what_does_not_fit_is_skipped_not_dropped_silently(ctx):
    (ctx.workspace.root / "big.py").write_text("y\n" * 4000)
    loaded, _ = load_files(ctx, ["big.py"], 100_000)
    included, skipped = fit_to_budget(loaded, 5)
    assert included == []
    assert len(skipped) == 1


def test_smallest_files_come_first_so_a_budget_buys_the_most_files(ctx):
    (ctx.workspace.root / "big.py").write_text("y\n" * 500)
    (ctx.workspace.root / "small.py").write_text("x\n")
    loaded, _ = load_files(ctx, ["big.py", "small.py"], 100_000)
    included, _ = fit_to_budget(loaded, 40, smallest_first=True)
    assert [f.rel for f in included] == ["small.py"]


# -- batching to a budget (the review phase's strategy) ------------------


def test_batching_keeps_everything(ctx):
    for name in ("a.py", "b.py"):
        (ctx.workspace.root / name).write_text("x\n" * 200)
    loaded, _ = load_files(ctx, ["a.py", "b.py"], 100_000)
    batches = batch_to_budget(loaded, 50)
    assert sum(len(b) for b in batches) == 2


def test_a_file_larger_than_the_whole_budget_becomes_its_own_batch(ctx):
    (ctx.workspace.root / "huge.py").write_text("x\n" * 5000)
    loaded, _ = load_files(ctx, ["huge.py"], 100_000)
    assert len(batch_to_budget(loaded, 10)) == 1


def test_small_files_share_one_batch(ctx):
    for name in ("a.py", "b.py", "c.py"):
        (ctx.workspace.root / name).write_text("x\n")
    loaded, _ = load_files(ctx, ["a.py", "b.py", "c.py"], 100_000)
    assert len(batch_to_budget(loaded, 100_000)) == 1


def test_nothing_loaded_means_no_batches():
    assert batch_to_budget([], 100) == []


# -- ToolContext.try_read ------------------------------------------------


def test_try_read_returns_contents(ctx):
    (ctx.workspace.root / "a.py").write_text("hello")
    assert ctx.try_read("a.py") == "hello"


def test_try_read_returns_none_for_anything_unreadable(ctx):
    assert ctx.try_read("missing.py") is None
    assert ctx.try_read("../escape.py") is None
