"""Settings.plan_probe: the planner can run code, against a throwaway copy.

Two remedies for plans that name a cause after one read (a prompt block, a
plan audit; DESIGN-LOG §45.4-45.5) changed the planner's instructions or
supervision and neither moved honesty. This one changes what the planner can
find out, and these tests pin that it does so without giving the planning
phase a way to change the workspace, and without touching the default prompt.
"""

from __future__ import annotations

import pytest

from agent import promptadapt
from agent.approval import Approver
from agent.config import Settings
from agent.prompts import PLAN_PROBE, PLANNER_SYSTEM
from agent.tools import ToolContext, execution_registry, planning_registry, review_registry
from agent.tools.checks import probe_python
from agent.trace import Trace
from agent.workspace import Workspace


def _ctx(root, approver=None):
    return ToolContext(
        workspace=Workspace(root),
        approver=approver or Approver(auto_approve=True),
        trace=Trace(str(root.parent / "trace"), echo=False),
    )


def _bare(settings) -> str:
    """The planner prompt before any plan_probe block: PLANNER_SYSTEM as adapted to the run's
    other features (promptadapt.planner, DESIGN-LOG §108)."""
    return promptadapt.planner(PLANNER_SYSTEM, settings)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "report.py").write_text("def render(rows):\n    return ''\n")
    return root


def test_off_is_the_bare_planner_and_the_default_is_repro():
    """ "off" is the control arm of every sweep; it must be the bare harness.
    The default is "repro" (§54.2), which adds a prompt block but no tool."""
    from agent.phases.plan import _planner_system
    from agent.prompts import PLAN_REPRO

    assert "probe_python" not in planning_registry().names()
    off = Settings(workspace=".", goal="g", plan_probe="off")
    assert _planner_system(off) == _bare(off)
    assert "probe_python" not in PLANNER_SYSTEM
    assert Settings(workspace=".", goal="g").plan_probe == "repro"
    default = Settings(workspace=".", goal="g")
    assert _planner_system(default) == _bare(default) + "\n\n" + PLAN_REPRO


def test_plan_probe_adds_the_tool_and_its_prompt_block_to_planning_only():
    from agent.phases.plan import _planner_system

    assert "probe_python" in planning_registry(probe=True).names()
    offer = Settings(workspace=".", goal="g", plan_probe="offer")
    assert _planner_system(offer) == _bare(offer) + "\n\n" + PLAN_PROBE
    for registry in (execution_registry(), review_registry()):
        assert "probe_python" not in registry.names()


def test_a_probe_sees_the_workspace_but_cannot_change_it(workspace):
    ctx = _ctx(workspace)
    code = (
        "import report\n"
        "print(repr(report.render([1])))\n"
        "open('report.py', 'w').write('broken')\n"
        "open('new.txt', 'w').write('x')\n"
    )
    out = probe_python(ctx, code)
    assert out.startswith("exit=0")
    assert "''" in out
    assert (workspace / "report.py").read_text().startswith("def render")
    assert not (workspace / "new.txt").exists()
    # A second probe gets a fresh copy, not the first one's leftovers.
    assert "False" in probe_python(ctx, "import os; print(os.path.exists('new.txt'))")


def test_a_probe_goes_through_the_same_approval_gate_as_run_python(workspace):
    ctx = _ctx(workspace, Approver(unattended=True))
    out = probe_python(ctx, "open('/tmp/should-not-run', 'w')")
    assert out.startswith("error:")


def test_bench_forwards_plan_probe_and_diagnostics_count_the_probes():
    from bench import configs
    from bench.diagnostics import from_events

    assert configs.get("mercury-plan-probe").plan_probe == "offer"
    assert configs.get("mercury-plan-probe-required").plan_probe == "require"
    assert configs.get("mercury-only").plan_probe == "repro"
    assert configs.get("mercury-no-repro").plan_probe == "off"
    events = [
        {"kind": "tool_call", "phase": "plan", "tool": "probe_python"},
        {"kind": "tool_call", "phase": "plan", "tool": "read_file"},
        {"kind": "tool_call", "phase": "execute:s1", "tool": "probe_python"},
    ]
    diag = from_events(events)
    assert diag.plan_probes == 1
    assert diag.plan_tool_calls == 2


def _call(name):
    return {"id": "c1", "function": {"name": name, "arguments": "{}"}}


def test_the_probe_gate_holds_the_first_unprobed_plan_once():
    from agent.runner import ProbeGate

    gate = ProbeGate()
    assert gate.hold(_call("read_file")) is None
    message = gate.hold(_call("submit_plan"))
    assert message is not None and "probe_python" in message
    # Once only: a planner that re-sends the plan is not held again.
    assert gate.hold(_call("submit_plan")) is None


def test_the_probe_gate_lets_a_probed_plan_through():
    from agent.runner import ProbeGate

    gate = ProbeGate()
    gate.observe(_call("probe_python"))
    assert gate.hold(_call("submit_plan")) is None


def test_a_probe_hold_is_counted_apart_from_a_tdd_hold():
    from bench.diagnostics import from_events

    diag = from_events([{"kind": "probe_gate_held", "phase": "plan", "tool": "submit_plan"}])
    assert diag.probe_holds == 1
    assert diag.tdd_holds == 0


def test_probing_is_withheld_entirely_when_there_is_no_code_to_probe(tmp_path):
    """A greenfield workspace has no entry point to reproduce anything through,
    and offering the tool there anyway drew 1-2 useless probes a run (§54)."""
    from agent.phases.plan import _planner_system, _probe_mode

    trace = Trace(str(tmp_path / "t"), echo=False)
    symbols = "report.py: def render(rows)"
    assert _probe_mode("require", symbols, trace) == "require"
    assert _probe_mode("offer", symbols, trace) == "offer"
    assert _probe_mode("require", "", trace) == "off"
    assert _probe_mode("offer", "", trace) == "off"
    settings = Settings(workspace=".", goal="g", plan_probe="require")
    assert _planner_system(settings, "off") == _bare(settings)


# -- plan_probe="repro": the plan carries its own reproduction ---------------


def _submit(reproduce=None, task_kind=None):
    import json

    args = {"draft": ["understood: x"], "steps": []}
    if reproduce is not None:
        args["reproduce"] = reproduce
    if task_kind is not None:
        args["task_kind"] = task_kind
    return {"id": "c1", "function": {"name": "submit_plan", "arguments": json.dumps(args)}}


def _repro_gate(tmp_path, result):
    from agent.runner import ReproGate

    trace = Trace(str(tmp_path / "t"), echo=False)
    ran = []

    def run(code):
        ran.append(code)
        return result

    return ReproGate(run, trace), ran


def test_a_reproduction_that_fails_lets_the_plan_through_with_no_extra_call(tmp_path):
    """The problem is real: this is the accurate-report case, and it must cost nothing."""
    gate, ran = _repro_gate(tmp_path, (1, "AssertionError: assert 280 == 300"))
    assert gate.hold(_submit("assert total() == 300")) is None
    assert ran == ["assert total() == 300"]


def test_a_reproduction_that_passes_sends_the_plan_back_once(tmp_path):
    """The reported problem did not show: the premise is in doubt."""
    gate, _ = _repro_gate(tmp_path, (0, ""))
    message = gate.hold(_submit("assert slugify('Café') == 'cafe'"))
    assert message is not None and "PASSED" in message and "mistaken" in message
    assert gate.hold(_submit("assert slugify('Café') == 'cafe'")) is None


def test_a_snippet_that_never_reached_the_code_is_not_a_reproduction(tmp_path):
    gate, _ = _repro_gate(tmp_path, (1, "NameError: name 'totl' is not defined"))
    message = gate.hold(_submit("assert totl() == 300"))
    assert message is not None and "failed before it reached" in message


def test_no_reproduction_or_one_that_could_not_run_does_not_hold(tmp_path):
    gate, ran = _repro_gate(tmp_path, (0, ""))
    assert gate.hold(_submit("")) is None
    assert ran == []
    gate, _ = _repro_gate(tmp_path, "error: denied")
    assert gate.hold(_submit("assert x")) is None


def test_run_in_scratch_reports_a_real_failing_assertion(workspace):
    from agent.tools.checks import run_in_scratch

    exit_code, output = run_in_scratch(_ctx(workspace), "import report\nassert report.render([1])")
    assert exit_code != 0
    assert "AssertionError" in output


def test_repro_mode_wires_the_field_the_prompt_and_no_probe_tool():
    from agent.phases.plan import _planner_system
    from agent.prompts import PLAN_REPRO
    from agent.tools.control import SUBMIT_PLAN, submit_plan, with_reproduce

    tool = with_reproduce(SUBMIT_PLAN)
    assert "reproduce" in tool.parameters["required"]
    assert "reproduce" not in SUBMIT_PLAN.parameters["properties"]
    settings = Settings(workspace=".", goal="g", plan_probe="repro")
    assert _planner_system(settings) == _bare(settings) + "\n\n" + PLAN_REPRO

    from types import SimpleNamespace

    ctx = SimpleNamespace(result=None, done=False)
    submit_plan(ctx, ["d"], [{"id": "a"}], reproduce="assert x")
    assert ctx.result["reproduce"] == "assert x"


def test_repro_outcomes_are_counted():
    from bench.diagnostics import from_events

    diag = from_events(
        [
            {"kind": "repro", "outcome": "reproduced"},
            {"kind": "repro", "outcome": "passed"},
            {"kind": "repro_gate_held", "phase": "plan", "tool": "submit_plan"},
        ]
    )
    assert diag.repro_reproduced == 1
    assert diag.repro_passed == 1
    assert diag.probe_holds == 1


def test_an_empty_snippet_on_a_bugfix_is_sent_back_once(tmp_path):
    """29 of 78 repro runs left the snippet empty (§54.2); on a goal the planner
    itself calls a bugfix that skipped the check it exists for."""
    gate, ran = _repro_gate(tmp_path, (1, "AssertionError"))
    message = gate.hold(_submit("", task_kind="bugfix"))
    assert message is not None and "bugfix" in message
    assert gate.hold(_submit("", task_kind="bugfix")) is None
    assert ran == []


def test_an_empty_snippet_on_new_work_costs_nothing(tmp_path):
    gate, _ = _repro_gate(tmp_path, (0, ""))
    assert gate.hold(_submit("", task_kind="feature")) is None
    assert gate.hold(_submit("", task_kind="feature")) is None


def test_a_snippet_sent_after_the_ask_is_still_checked(tmp_path):
    """Asking for a snippet and then ignoring what it shows would waste the ask."""
    gate, ran = _repro_gate(tmp_path, (0, ""))
    assert gate.hold(_submit("", task_kind="bugfix")) is not None
    message = gate.hold(_submit("assert slugify('Café') == 'cafe'", task_kind="bugfix"))
    assert message is not None and "PASSED" in message
    assert ran == ["assert slugify('Café') == 'cafe'"]
