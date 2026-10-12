"""The edit tools' trace events carry the size and gist of each change.

The TUI shows each edit as "Edit throttle.py  +2 -1" with its first added
line beneath. It cannot work that out from the tool_call event, whose
arguments are clipped to 200 characters, so the edit tools say it
themselves. Run against real files in a temporary workspace.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.tools import ToolContext
from agent.tools.fs import _change_summary, edit_file, write_file
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx_and_events(tmp_path):
    events: list[dict] = []
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False, on_event=events.append),
    )
    return ctx, events, tmp_path


def test_an_edit_event_counts_lines_and_names_the_first_added_line(ctx_and_events):
    ctx, events, root = ctx_and_events
    (root / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    result = edit_file(ctx, "a.py", "y = 2\n", "y = 20\nw = 4\n")
    assert not result.startswith("error"), result
    event = next(e for e in events if e["kind"] == "edit_file")
    assert (event["added"], event["removed"], event["preview"]) == (2, 1, "y = 20")
    assert (root / "a.py").read_text() == "x = 1\ny = 20\nw = 4\nz = 3\n"


def test_a_write_event_counts_lines(ctx_and_events):
    ctx, events, _root = ctx_and_events
    write_file(ctx, "b.py", "a = 1\nb = 2\nc = 3\n")
    event = next(e for e in events if e["kind"] == "write_file")
    assert event["lines"] == 3
    assert event["overwrote"] is False


def test_a_pure_deletion_has_no_preview():
    assert _change_summary("a\nb\nc", "a\nc") == {"added": 0, "removed": 1, "preview": ""}


def test_a_large_replacement_is_summarised_quickly():
    """unified_diff, not ndiff: ndiff's character-level matching is
    quadratic, and this runs inside a tool call."""
    import time

    old = "\n".join(f"line {i} = {i}" for i in range(5000))
    new = old.replace("line 2500 = 2500", "changed = 1")
    started = time.perf_counter()
    summary = _change_summary(old, new)
    assert time.perf_counter() - started < 0.5
    assert summary == {"added": 1, "removed": 1, "preview": "changed = 1"}
