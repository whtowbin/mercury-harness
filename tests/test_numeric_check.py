"""The numeric self-check (agent/numericcheck.py, DESIGN-LOG §81)."""

from __future__ import annotations

import json

from agent.config import Settings
from agent.numericcheck import EXECUTOR_NUMERIC_CHECK, NumericCheckGate
from agent.phases.execute import _executor_system, _step_gate
from agent.runner import GateChain, PremiseGate


def _call(name: str, **args) -> dict:
    return {"id": "c", "function": {"name": name, "arguments": json.dumps(args)}}


def test_finishing_right_after_writing_python_is_held_once():
    gate = NumericCheckGate()
    gate.observe(_call("write_file", path="solution.py", content="def f(): ..."))
    held = gate.hold(_call("finish_step", summary="done"))
    assert held and "run_python" in held
    assert gate.hold(_call("finish_step", summary="done")) is None  # once


def test_running_the_code_after_the_last_edit_satisfies_it():
    gate = NumericCheckGate()
    gate.observe(_call("edit_file", path="solution.py", old="a", new="b"))
    gate.observe(_call("run_python", code="from solution import f; print(f(0))"))
    assert gate.hold(_call("finish_step", summary="done")) is None


def test_an_edit_after_the_run_needs_another_run():
    gate = NumericCheckGate()
    gate.observe(_call("run_python", code="print(1)"))
    gate.observe(_call("edit_file", path="solution.py", old="a", new="b"))
    assert gate.hold(_call("finish_step", summary="done")) is not None


def test_a_step_that_wrote_no_python_is_not_held():
    gate = NumericCheckGate()
    gate.observe(_call("write_file", path="README.md", content="notes"))
    assert gate.hold(_call("finish_step", summary="done")) is None


def test_off_by_default_and_the_default_executor_prompt_is_unchanged():
    off = Settings(workspace=".", goal="g")
    on = Settings(workspace=".", goal="g", numeric_check=True)
    assert EXECUTOR_NUMERIC_CHECK not in _executor_system(off)
    assert _executor_system(on).endswith(EXECUTOR_NUMERIC_CHECK)
    # Off: exactly the premise gate every earlier run had, no chain around it.
    assert type(_step_gate(off, ctx=None)) is PremiseGate
    assert isinstance(_step_gate(on, ctx=None), GateChain)
