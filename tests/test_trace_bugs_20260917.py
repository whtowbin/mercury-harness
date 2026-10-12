"""The two complaints from the 2026-09-17 cpx_barometer trace.

That run spent 359 of its 548 seconds (66%) printing nothing at all, opening
with 50.9 seconds of silence between the goal being entered and the first
visible line. It also kept failing on the environment: a step whose verify
command was `uv run python -c 'import xgboost; import lightgbm'` failed
because the approval layer read the semicolons inside that quoted Python
string as shell command chaining, and refused to run it.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver, Denied, is_compound

# -- shell quoting is honoured, so one command is not read as several ------


@pytest.mark.parametrize(
    "command",
    [
        # The traced failure, and the shape it belongs to.
        "uv run python -c 'import xgboost; import lightgbm'",
        'uv run python -c "import x; import y"',
        "echo '$(whoami)'",  # single quotes make substitution literal
        "uv run pytest -q",
        "python -c 'x=1'",
    ],
)
def test_a_quoted_separator_is_not_command_chaining(command):
    assert not is_compound(command)


@pytest.mark.parametrize(
    "command",
    [
        "grep -r foo . | head -20",
        "pytest && ruff check .",
        "ls; rm -rf /tmp/x",
        "echo $(whoami)",
        'echo "$(whoami)"',  # double quotes still expand substitutions
        "echo `id`",
        "ls & rm x",
        "uv run python -c 'import a; import b' && rm x",
        "python -c 'x=1; rm -rf /tmp/x",  # unterminated: stays fully visible
    ],
)
def test_real_chaining_is_still_refused(command):
    assert is_compound(command)


def test_the_traced_verify_command_now_runs_unattended():
    """It was not the tool call alone: the same string was the step's own
    `verify`, so the step failed for being unverifiable rather than for being
    wrong, and the executor then spent all fourteen of its iterations trying
    to get the check to run."""
    approver = Approver(auto_approve=False, unattended=True)
    approver.check_bash("uv run python -c 'import xgboost; import lightgbm'")


def test_an_unattended_run_still_refuses_a_real_chain():
    """The security property is unchanged; only the reading of quotes moved."""
    approver = Approver(auto_approve=False, unattended=True)
    with pytest.raises(Denied):
        approver.check_bash("uv run pytest -q && rm -rf /tmp/x")


# -- the pre-run baselines overlap the questions they used to block ------


def test_the_baseline_runs_while_clarify_asks_its_questions(tmp_path, monkeypatch):
    """Running the suite is the longest thing a run does before it speaks --
    49.3 seconds in one traced run, against 0.87s for the clarifying
    questions that followed it and 24s of the person answering them. All of
    that was dead CPU, because the baseline had already finished by then.

    It is still finished before anything reads it; only the waiting moved.
    """
    import threading

    from agent import loop as loop_module

    started = threading.Event()
    release = threading.Event()

    def slow_baseline(ctx, trace, settings=None, cancel=None):
        started.set()
        release.wait(timeout=5)
        return None

    monkeypatch.setattr(loop_module, "collect_baseline", slow_baseline)
    monkeypatch.setattr(loop_module, "collect_type_baseline", lambda ctx, trace: None)

    trace = _trace(tmp_path)
    handle = loop_module._start_baselines(_ctx(tmp_path, trace), trace, _settings(tmp_path))
    # The caller is free while the suite is still running: that is the point.
    assert started.wait(timeout=5), "the baseline should start immediately"
    assert not release.is_set(), "and the caller should not be blocked on it"
    release.set()
    assert handle.tests is None
    assert handle.types is None


def test_a_baseline_failure_is_not_silently_swallowed(tmp_path, monkeypatch):
    """A baseline that quietly became None would make the gate treat every
    pre-existing failure as this run's fault -- the exact confusion the
    baseline exists to prevent. It must surface, not degrade."""
    from agent import loop as loop_module

    def boom(ctx, trace, settings=None, cancel=None):
        raise RuntimeError("pytest could not start")

    monkeypatch.setattr(loop_module, "collect_baseline", boom)
    monkeypatch.setattr(loop_module, "collect_type_baseline", lambda ctx, trace: None)
    trace = _trace(tmp_path)
    handle = loop_module._start_baselines(_ctx(tmp_path, trace), trace, _settings(tmp_path))
    with pytest.raises(RuntimeError, match="pytest could not start"):
        _ = handle.tests


def test_nothing_is_started_when_the_gate_is_off(tmp_path):
    """--no-final-check means no gate, so there is nothing to baseline for."""
    from agent import loop as loop_module
    from agent.config import Settings

    trace = _trace(tmp_path)
    settings = Settings(workspace=str(tmp_path), goal="g", final_check=False)
    handle = loop_module._start_baselines(_ctx(tmp_path, trace), trace, settings)
    assert handle.tests is None and handle.types is None


def _trace(tmp_path):
    from agent.trace import Trace

    return Trace(str(tmp_path / "t"), echo=False)


def _settings(tmp_path):
    from agent.config import Settings

    return Settings(workspace=str(tmp_path), goal="g")


def _ctx(tmp_path, trace):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.workspace import Workspace

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ToolContext(workspace=Workspace(ws), approver=Approver(auto_approve=True), trace=trace)
