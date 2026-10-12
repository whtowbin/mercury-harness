"""editcalls: the one list of tools that edit the workspace, and what each call edits.

edit_blocks was missing from the after-edit check's own list and from the premise gate's, so
§105.2's and §106's edit_blocks arms ran without both (DESIGN-LOG §106.1). These tests fail when
a new execution tool is left unclassified, and check every edit tool against real files: each
call's reported paths are the files that actually changed.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.editcalls import CHANGE_TOOLS, EDIT_TOOLS, edited_paths, string_edits, written_text
from agent.llm import Reply, Usage
from agent.runner import AFTER_EDIT_NOTE, PremiseGate, _batching_nudge, run_tool_loop
from agent.tools import ToolContext, execution_registry, with_edit_blocks
from agent.trace import Trace
from agent.workspace import Workspace

# Execution tools that may change files but are not edits the gates and checks reason about,
# each for a stated reason. A new tool must land in CHANGE_TOOLS or here.
NOT_EDITS = {
    # Shell and interpreter: arbitrary effects, not attributable to a path.
    "run_bash": "arbitrary command",
    "run_python": "arbitrary code",
    "trace_python": "runs code",
    # Version control records changes; it does not make them.
    "git_add": "staging",
    "git_commit": "recording",
    # A rename moves content without changing it; no gate inspects one.
    "move_file": "rename",
    # Harness memory, outside the workspace's source.
    "remember": "memory store",
}
READ_ONLY = {
    "ask_user",
    "audit_previous_runs",
    "check_code",
    "check_complexity",
    "fetch_url",
    "finish_step",
    "git_diff",
    "git_log",
    "glob_files",
    "grep",
    "lint_project",
    "list_dir",
    "read_file",
    "read_files",
    "read_skill",
    "recall",
    "run_tests",
    "show_changes",
    "test_project",
    "validate_data_file",
    "web_search",
}


def test_every_execution_tool_is_classified():
    names = set(execution_registry().names())
    unclassified = names - CHANGE_TOOLS - set(NOT_EDITS) - READ_ONLY
    assert not unclassified, f"classify these in agent/editcalls.py or this test: {unclassified}"
    assert names >= EDIT_TOOLS  # no stale names (the old list carried replace_symbol)


SOURCE = "def add(a, b):\n    return a - b\n\nSPEED = 3\n"

CALLS = {
    "write_file": {"path": "new.py", "content": "X = 1\n"},
    "edit_file": {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"},
    "edit_files": {
        "edits": [
            {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"},
            {"path": "other.py", "old_string": "Y = 1", "new_string": "Y = 2"},
        ]
    },
    "batch_edits_to_file": {
        "path": "calc.py",
        "edits": [
            {"old_string": "a - b", "new_string": "a + b"},
            {"old_string": "SPEED = 3", "new_string": "SPEED = 5"},
        ],
    },
    "edit_blocks": {
        "blocks": "calc.py\n<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n"
        ">>>>>>> REPLACE\nmade.py\n<<<<<<< SEARCH\n=======\nZ = 1\n>>>>>>> REPLACE\n"
    },
}


@pytest.mark.parametrize("name", sorted(EDIT_TOOLS))
def test_each_edit_tool_changes_exactly_the_files_it_reports(tmp_path, name):
    (tmp_path / "calc.py").write_text(SOURCE)
    (tmp_path / "other.py").write_text("Y = 1\n")
    before = {p.name: p.read_text() for p in tmp_path.iterdir()}
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    tool = with_edit_blocks(execution_registry()).get(name) or execution_registry().get(name)
    result = tool.fn(ctx, **CALLS[name])
    assert not str(result).startswith("error"), result
    after = {p.name: p.read_text() for p in tmp_path.iterdir() if p.is_file()}
    changed = {n for n in after if after[n] != before.get(n)}
    assert set(edited_paths(name, CALLS[name])) == changed


def test_string_edits_reads_edit_blocks_and_marks_a_created_file():
    edits = string_edits("edit_blocks", CALLS["edit_blocks"])
    assert [(e["path"], e["create"]) for e in edits] == [("calc.py", False), ("made.py", True)]
    assert edits[0]["old_string"] == "    return a - b"
    assert string_edits("edit_blocks", {"blocks": "no markers"}) == []
    assert written_text("edit_blocks", CALLS["edit_blocks"])[1] == ("made.py", "Z = 1")


# -- the after-edit check, through the real loop ------------------------------


def _call(name: str, **args) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Scripted:
    def __init__(self, turns):
        self.turns = list(turns)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return Reply(content="", tool_calls=self.turns.pop(0), usage=Usage(10, 5, 0.0, 0.1))


def test_an_edit_blocks_turn_gets_the_after_edit_check(tmp_path):
    (tmp_path / "calc.py").write_text(SOURCE)
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    blocks = _call("edit_blocks", blocks=CALLS["edit_blocks"]["blocks"])
    finish = _call("finish_step", summary="done", verified="true")
    client = Scripted([[blocks], [finish]])
    run_tool_loop(
        client=client,
        model="fake",
        phase="execute:fix-add",
        system="s",
        user="u",
        registry=with_edit_blocks(execution_registry()),
        ctx=ctx,
        trace=trace,
        max_iters=2,
        check_after_edit=True,
    )
    seen = client.sent[1][-1]["content"]
    assert AFTER_EDIT_NOTE.strip() in seen and "PASS" in seen
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [e["kind"] for e in events].count("after_edit_check") == 1


# -- the premise gate ---------------------------------------------------------

PINNED = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def test_the_premise_gate_holds_an_edit_blocks_call_that_deletes_a_pinned_test(tmp_path):
    (tmp_path / "calc.py").write_text(SOURCE)
    (tmp_path / "test_calc.py").write_text(PINNED)
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    removal = (
        "test_calc.py\n<<<<<<< SEARCH\ndef test_add():\n    assert add(2, 3) == 5\n=======\n"
        ">>>>>>> REPLACE\n"
    )
    held = PremiseGate(ctx).hold(_call("edit_blocks", blocks=removal))
    assert held is not None and "test_add" in held


def test_the_batching_nudge_names_the_batch_tool_the_kit_has():
    assert "edit_files" in _batching_nudge(["edit_file"], ["edit_file"])
    nudge = _batching_nudge(["edit_file"], ["edit_file"], edit_blocks=True)
    assert "edit_blocks" in nudge and "edit_files" not in nudge
