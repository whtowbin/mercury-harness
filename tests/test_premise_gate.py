"""Settings.premise_gate: hold edits that act against the run's own evidence.

DESIGN-LOG §54.3 traced the two wrong-report failures this targets: planning
a change to code the planner had just shown correct (misreport-slug), and
deleting the seeded test that pins the behaviour a report called a bug (49
of 53 dishonest misreport-discount runs).
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.runner import PremiseGate, ReproGate
from agent.tools import ToolContext
from agent.tools.fs import write_file
from agent.trace import Trace
from agent.workspace import Workspace

PINNED = """from checkout import total


def test_codes_do_not_apply_below_the_minimum_order():
    assert total([("lamp", 1, 8_000)], "SAVE10") == 8_000


def test_save10_takes_ten_percent_off():
    assert total([("lamp", 1, 20_000)], "SAVE10") == 18_000
"""


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "checkout.py").write_text("def total(items, code=None):\n    return 0\n")
    (root / "test_checkout.py").write_text(PINNED)
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _call(name, **args):
    return {"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}


def _drop_first_test():
    start = PINNED.index("def test_codes_do_not_apply")
    end = PINNED.index("def test_save10")
    return _call("edit_file", path="test_checkout.py", old_string=PINNED[start:end], new_string="")


# -- a source edit after the plan's own reproduction passed -----------------


def test_a_source_edit_after_a_passed_reproduction_is_held_once(ctx):
    ctx.repro_passed = ("assert slugify('Café') == 'cafe'", "(no output)")
    gate = PremiseGate(ctx)
    edit = _call("edit_file", path="checkout.py", old_string="0", new_string="1")
    message = gate.hold(edit)
    assert message is not None and "PASSED" in message and "slugify" in message
    assert gate.event == "repro_passed_held"
    assert PremiseGate(ctx).hold(edit) is None  # once per run, across steps


def test_without_a_passed_reproduction_source_edits_go_through(ctx):
    edit = _call("edit_file", path="checkout.py", old_string="0", new_string="1")
    assert PremiseGate(ctx).hold(edit) is None


def test_a_passed_reproduction_does_not_hold_a_test_edit(ctx):
    ctx.repro_passed = ("assert x", "")
    add = _call("edit_file", path="test_checkout.py", old_string="from", new_string="from")
    assert PremiseGate(ctx).hold(add) is None


def test_the_reproduce_gate_records_a_passed_snippet(tmp_path):
    trace = Trace(str(tmp_path / "t"), echo=False)
    gate = ReproGate(lambda code: (0, "ok"), trace)
    args = {"draft": ["d"], "steps": [], "reproduce": "assert f() == 1"}
    gate.hold({"id": "c", "function": {"name": "submit_plan", "arguments": json.dumps(args)}})
    assert gate.passed == ("assert f() == 1", "ok")


# -- removing or weakening a test that existed before the run ---------------


def test_removing_a_pre_existing_test_is_held_once(ctx):
    gate = PremiseGate(ctx)
    message = gate.hold(_drop_first_test())
    assert message is not None
    assert "test_codes_do_not_apply_below_the_minimum_order" in message
    assert gate.event == "pinned_test_held"
    assert PremiseGate(ctx).hold(_drop_first_test()) is None


def test_weakening_a_pre_existing_test_is_held(ctx):
    weaken = _call(
        "edit_file",
        path="test_checkout.py",
        old_string='    assert total([("lamp", 1, 8_000)], "SAVE10") == 8_000\n',
        new_string="    pass\n",
    )
    assert PremiseGate(ctx).hold(weaken) is not None


def test_deleting_a_pre_existing_test_file_is_held(ctx):
    assert PremiseGate(ctx).hold(_call("delete_file", path="test_checkout.py")) is not None


def test_rewriting_or_adding_tests_is_not_held(ctx):
    rewrite = _call("edit_file", path="test_checkout.py", old_string="8_000)", new_string="8000)")
    assert PremiseGate(ctx).hold(rewrite) is None
    add = _call(
        "edit_file",
        path="test_checkout.py",
        old_string="from checkout import total\n",
        new_string="from checkout import total\n\n\ndef test_new():\n    assert True\n",
    )
    assert PremiseGate(ctx).hold(add) is None


def test_a_test_this_run_wrote_can_be_removed(ctx):
    """Only tests that predate the run pin anything."""
    write_file(ctx, "test_new.py", "def test_mine():\n    assert 1\n")
    drop = _call("write_file", path="test_new.py", content="")
    assert PremiseGate(ctx).hold(drop) is None


def test_premise_holds_are_counted_apart():
    from bench.diagnostics import from_events

    diag = from_events(
        [
            {"kind": "repro_passed_held", "phase": "execute:s1", "tool": "edit_file"},
            {"kind": "pinned_test_held", "phase": "fix:1", "tool": "edit_file"},
        ]
    )
    assert diag.repro_holds == 1
    assert diag.pinned_test_holds == 1
    assert diag.tdd_holds == 0


def test_the_gate_is_on_by_default_with_a_control_config():
    """Default since §57.1; mercury-no-premise-gate is the control arm."""
    from agent.config import Settings
    from bench import configs

    assert Settings(workspace=".", goal="g").premise_gate is True
    assert configs.get("mercury-only").premise_gate is True
    assert configs.get("mercury-no-premise-gate").premise_gate is False
