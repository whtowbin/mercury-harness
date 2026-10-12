"""Preloading enough of a file that the model does not have to page it in.

The cliff: a workspace grew from ~950 lines to 1,732 (game.py 20KB, its tests
37KB), both files crossed PRELOAD_CHAR_CAP, and the model went from having them
in its prompt to reading them in pieces. 88 of 125 read_file calls in a traced
session carried an offset — most of them literally `offset=401`, read_file's
old default limit — and 45 of the 63 in the execute phase were of files the
step had *declared*, which is exactly what the preload exists to cover. Runs
went from 73s to 114s.

The trade is measured. TTFT by prompt size over 944 traced calls: 1.22s at
10-15k tokens, 1.41s at 15-20k, 1.64s at 20-30k — so +15,000 tokens of prefill
costs about 0.4s, against ~1.3s for one avoided round trip. And the content
reaches the context either way: paging only decides whether it arrives in one
prefill or over several trips.
"""

from __future__ import annotations

from agent.config import Budget
from agent.llm import approx_tokens
from agent.phases.execute import PRELOAD_CHAR_CAP, _preload
from agent.text import numbered
from agent.tools.fs import DEFAULT_LINE_LIMIT

# A real module and a real test file, at the sizes that triggered this.
MODULE_LINES = 610
TESTS_LINES = 1_122


# The real files average 33 characters a line; a fixture with longer lines
# would be a different, larger test than the one that failed.
CHARS_PER_LINE = 33


def _source(lines: int) -> str:
    out = []
    for i in range(lines):
        body = f"    value_{i} = compute({i})"
        out.append(body.ljust(CHARS_PER_LINE - 1)[: CHARS_PER_LINE - 1] + "\n")
    return "".join(out)


def _ctx(tmp_path):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    root = tmp_path / "ws"
    (root / "tests").mkdir(parents=True)
    (root / "game.py").write_text(_source(MODULE_LINES))
    (root / "tests" / "test_game.py").write_text(_source(TESTS_LINES))
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def test_a_real_module_and_its_tests_both_arrive_whole(tmp_path):
    ctx = _ctx(tmp_path)
    budget = Budget().step_budget() * 2 // 3
    out = _preload(ctx, ["game.py", "tests/test_game.py"], budget)
    assert "truncated" not in out, "a 610-line module must not be cut in half"
    assert "not preloaded" not in out, "its 1122-line test file must not be dropped"


def test_the_per_file_cap_holds_a_thousand_line_file(tmp_path):
    ctx = _ctx(tmp_path)
    assert len((ctx.workspace.root / "tests" / "test_game.py").read_text()) < PRELOAD_CHAR_CAP


def test_the_budget_covers_the_numbered_size_not_the_raw_size(tmp_path):
    """Line numbers are added before the budget is applied and cost ~11% more,
    which is why a budget sized against raw bytes silently dropped a file."""
    ctx = _ctx(tmp_path)
    raw = sum(
        approx_tokens((ctx.workspace.root / p).read_text())
        for p in ("game.py", "tests/test_game.py")
    )
    marked = sum(
        approx_tokens(numbered((ctx.workspace.root / p).read_text()))
        for p in ("game.py", "tests/test_game.py")
    )
    assert marked > raw
    assert Budget().step_budget() * 2 // 3 > marked


def test_read_file_reads_most_files_whole_by_default():
    """`offset=401` appears all over the traces: the model hitting the old
    400-line default and coming back for the rest at ~1.3s a time."""
    assert DEFAULT_LINE_LIMIT >= TESTS_LINES


def test_a_file_too_big_for_the_cap_is_still_truncated_not_dropped(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.workspace.root / "huge.py").write_text(_source(40_000))
    out = _preload(ctx, ["huge.py"], Budget().step_budget() * 2 // 3)
    assert "truncated" in out
    assert "read_file with an offset" in out, "and it must say how to get the rest"


# -- the cliff must announce itself next time -----------------------------


def test_a_step_whose_files_do_not_fit_says_so(tmp_path, capsys):
    """The failure this whole change is about was invisible: the budget
    quietly stopped holding its input and the runs just got slower."""
    from agent.trace import Trace

    ctx = _ctx(tmp_path)
    said: list[str] = []
    ctx.trace.say = said.append  # type: ignore[method-assign]
    _preload(ctx, ["game.py", "tests/test_game.py"], 2_000)
    assert any("preload budget" in line for line in said), said
    assert any("round trip apiece" in line for line in said)
    assert isinstance(ctx.trace, Trace)


def test_a_step_whose_files_fit_says_nothing(tmp_path):
    ctx = _ctx(tmp_path)
    said: list[str] = []
    ctx.trace.say = said.append  # type: ignore[method-assign]
    _preload(ctx, ["game.py", "tests/test_game.py"], Budget().step_budget() * 2 // 3)
    assert said == []


# -- budgets scale with the context window --------------------------------


def test_the_step_budget_scales_with_the_context_window():
    """Both sides constructed explicitly: a bare Budget() reads
    MODEL_CONTEXT_TOKENS from the ambient environment, so comparing against
    one makes the test depend on whoever's shell is running it."""
    from agent.config import Budget as B

    assert (
        B(model_context_tokens=256_000).step_budget() > B(model_context_tokens=64_000).step_budget()
    )


def test_an_explicit_budget_is_used_exactly_including_a_smaller_one():
    """An earlier version floored these at the default, which turned every
    lowering into a no-op: --step-budget 8000 on a 128k window silently became
    32,000, and MODEL_CONTEXT_TOKENS=8000 changed nothing at all."""
    from agent.config import Budget as B

    assert B(step_prompt_token_budget=8_000, model_context_tokens=128_000).step_budget() == 8_000
    assert (
        B(review_prompt_token_budget=20_000, reviewer_context_tokens=200_000).review_budget()
        == 20_000
    )


def test_a_small_context_window_actually_shrinks_the_step_budget():
    from agent.config import Budget as B

    assert B(model_context_tokens=64_000).step_budget() == 16_000


def test_review_chunks_are_sized_from_the_reviewers_window_not_the_executors():
    """The chunks go to reviewer_model. Sizing them from a 1M-context executor
    overflows a Sonnet reviewer — which is the hybrid config.py recommends."""
    from agent.config import Budget as B

    hybrid = B(model_context_tokens=1_000_000, reviewer_context_tokens=200_000)
    assert hybrid.review_budget() < 200_000
    assert hybrid.review_budget() != hybrid.step_budget() * 2


def test_a_malformed_context_env_value_does_not_kill_the_cli(monkeypatch):
    """It is read at class-definition time, so a typo took down every command
    with a raw ValueError before it could say what was wrong."""
    from agent.config import _env_int

    for bad in ("128k", "", "   ", "not-a-number", "-5", "0"):
        monkeypatch.setenv("SOME_CONTEXT_VAR", bad)
        assert _env_int("SOME_CONTEXT_VAR", 128_000) == 128_000
    monkeypatch.setenv("SOME_CONTEXT_VAR", "64_000")
    assert _env_int("SOME_CONTEXT_VAR", 128_000) == 64_000


def test_the_char_cap_and_the_budget_are_reported_separately(tmp_path):
    """One is a per-file cap no budget flag can change; the other is the token
    budget those flags do change. Conflating them advised the wrong knob."""
    import agent.phases.execute as ex

    ctx = _ctx(tmp_path)
    said: list[str] = []
    ctx.trace.say = said.append  # type: ignore[method-assign]

    original = ex.PRELOAD_CHAR_CAP
    ex.PRELOAD_CHAR_CAP = 2_000
    try:
        _preload(ctx, ["game.py"], 100_000)
    finally:
        ex.PRELOAD_CHAR_CAP = original
    assert any("per-file preload cap" in line and "PRELOAD_CHAR_CAP" in line for line in said), said
    assert not any("--step-budget" in line for line in said), (
        "the char cap is not fixed by a budget flag"
    )
