"""The red-first gate: hold a step's first non-test source edit until a test has run.

Built because the executor prompt asks for write-test, watch-it-fail, fix and
1,285 traced steps that edited source did it 13% of the time. A nudge after the
edit cannot raise that number (the edit has already happened), so the gate acts
before it, once per step, and lets the model re-send the same call straight away.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.phases.execute import Step, _executor_system, _tdd_tests
from agent.prompts import EXECUTOR_ORDER_LEGACY, EXECUTOR_ORDER_TDD, EXECUTOR_SYSTEM
from agent.runner import _TddGate, run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


def call(name, **args):
    return {
        "id": f"c{abs(hash((name, json.dumps(args, sort_keys=True)))) % 100000}",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def edit_source(path="pricing.py"):
    return call("edit_file", path=path, old_string="x = 1", new_string="x = 2")


# -- the gate on its own ---------------------------------------------------


def test_the_first_source_edit_is_held_with_a_message_that_says_what_to_do():
    gate = _TddGate(["test_boundary"])
    message = gate.hold(edit_source())
    assert message and "NOT applied" in message and "run_tests" in message
    assert "test_boundary" in message
    assert "Re-send the same edit call unchanged" in message


def test_it_holds_only_once_per_step():
    gate = _TddGate([])
    assert gate.hold(edit_source()) is not None
    assert gate.hold(edit_source()) is None


def test_editing_a_test_file_is_never_held():
    gate = _TddGate([])
    assert gate.hold(edit_source("test_pricing.py")) is None
    assert gate.hold(edit_source("tests/test_pricing.py")) is None


def test_non_python_files_are_never_held():
    assert _TddGate([]).hold(edit_source("README.md")) is None
    assert _TddGate([]).hold(call("write_file", path="config.toml", content="x")) is None


def test_reads_and_test_runs_are_never_held():
    gate = _TddGate([])
    assert gate.hold(call("read_file", path="pricing.py")) is None
    assert gate.hold(call("run_tests")) is None


def test_write_file_of_source_is_held_too():
    assert _TddGate([]).hold(call("write_file", path="pricing.py", content="x = 1")) is not None


def test_a_batched_edit_touching_source_is_held_even_alongside_a_test_edit():
    gate = _TddGate([])
    batch = call(
        "edit_files",
        edits=[
            {"path": "test_pricing.py", "old_string": "a", "new_string": "b"},
            {"path": "pricing.py", "old_string": "a", "new_string": "b"},
        ],
    )
    assert gate.hold(batch) is not None


def test_a_batched_edit_of_only_tests_is_not_held():
    batch = call("edit_files", edits=[{"path": "test_a.py", "old_string": "a", "new_string": "b"}])
    assert _TddGate([]).hold(batch) is None


def test_running_the_tests_lifts_the_gate():
    gate = _TddGate([])
    gate.observe(call("run_tests"))
    assert gate.hold(edit_source()) is None


def test_running_pytest_through_the_shell_lifts_it_too():
    gate = _TddGate([])
    gate.observe(call("run_bash", command="uv run pytest -q test_pricing.py"))
    assert gate.hold(edit_source()) is None


def test_an_unrelated_shell_command_does_not():
    gate = _TddGate([])
    gate.observe(call("run_bash", command="ls"))
    assert gate.hold(edit_source()) is not None


def test_unparseable_arguments_do_not_crash_the_gate():
    bad = {"id": "x", "function": {"name": "edit_file", "arguments": "{not json"}}
    assert _TddGate([]).hold(bad) is None


# -- through the real tool loop --------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "pricing.py").write_text("x = 1\n")
    (tmp_path / "test_pricing.py").write_text(
        "from pricing import x\n\ndef test_x():\n    assert x == 2\n"
    )
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def reply(text, *calls):
    return Reply(content=text, tool_calls=list(calls), usage=Usage(10, 5, 0.0, 0.1))


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    ):
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def run(ctx, replies, tdd_tests):
    messages = run_tool_loop(
        client=Scripted(replies),
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=12,
        tdd_tests=tdd_tests,
    )[1]
    ctx.trace._fh.flush()
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    return messages, events


FINISH = reply("done", call("finish_step", summary="done", verified="false"))


def test_a_held_edit_changes_nothing_and_says_so_in_its_tool_result(ctx):
    messages, events = run(ctx, [reply("edit", edit_source()), FINISH], ["test_x"])
    assert (ctx.workspace.root / "pricing.py").read_text() == "x = 1\n"
    held = next(m for m in messages if m["role"] == "tool" and "NOT applied" in m["content"])
    assert held["name"] == "edit_file"
    assert any(e["kind"] == "tdd_gate_held" for e in events)


def test_the_same_edit_resent_after_a_test_run_is_applied(ctx):
    _, events = run(
        ctx,
        [
            reply("edit", edit_source()),
            reply("test", call("run_tests", path="test_pricing.py")),
            reply("edit again", edit_source()),
            FINISH,
        ],
        ["test_x"],
    )
    assert (ctx.workspace.root / "pricing.py").read_text() == "x = 2\n"
    assert sum(1 for e in events if e["kind"] == "tdd_gate_held") == 1


def test_resending_immediately_is_allowed_because_it_holds_only_once(ctx):
    """The escape for a step with nothing a test can observe."""
    run(ctx, [reply("edit", edit_source()), reply("edit again", edit_source()), FINISH], ["test_x"])
    assert (ctx.workspace.root / "pricing.py").read_text() == "x = 2\n"


def test_editing_the_test_first_is_applied_and_a_later_source_edit_is_still_held(ctx):
    """Writing the test is right; not running it before the fix is the miss."""
    edit_test = call("edit_file", path="test_pricing.py", old_string="== 2", new_string="== 3")
    messages, _ = run(
        ctx, [reply("test", edit_test), reply("src", edit_source()), FINISH], ["test_x"]
    )
    assert "== 3" in (ctx.workspace.root / "test_pricing.py").read_text()
    assert (ctx.workspace.root / "pricing.py").read_text() == "x = 1\n"
    assert any("NOT applied" in m["content"] for m in messages if m["role"] == "tool")


def test_with_the_gate_off_the_edit_is_applied_at_once(ctx):
    _, events = run(ctx, [reply("edit", edit_source()), FINISH], None)
    assert (ctx.workspace.root / "pricing.py").read_text() == "x = 2\n"
    assert not any(e["kind"] == "tdd_gate_held" for e in events)


# -- when the gate applies -------------------------------------------------


def _step(tests):
    return Step(id="s", title="t", intent="i", acceptance="a", tests=tests)


def test_the_gate_needs_the_setting_a_first_attempt_and_a_planned_test():
    on, off = SimpleNamespace(tdd_gate=True), SimpleNamespace(tdd_gate=False)
    assert _tdd_tests(on, _step(["test_a"]), 1) == ["test_a"]
    assert _tdd_tests(off, _step(["test_a"]), 1) is None
    assert _tdd_tests(on, _step(["test_a"]), 2) is None  # a retry follows a failure
    assert _tdd_tests(on, _step([]), 1) is None  # nothing planned to write
    assert _tdd_tests(SimpleNamespace(), _step(["test_a"]), 1) is None


# -- the prompt arm --------------------------------------------------------


def test_the_default_executor_order_is_tdd_first_and_legacy_is_the_old_prompt():
    """tdd_first is the default since DESIGN-LOG §51, both from Settings and for
    a caller that sets nothing; "legacy" must stay byte-for-byte the prompt every
    earlier sweep used, or those sweeps lose their control arm."""
    from agent.config import Settings

    assert EXECUTOR_ORDER_TDD in _executor_system(SimpleNamespace())
    # With the defaults the order's test stages are adapted to the harness's own test run
    # after an edit (check_after_edit, DESIGN-LOG §104): the order is the same, the wording
    # no longer asks for the run_tests call the harness makes.
    default = _executor_system(Settings(workspace=".", goal="x"))
    assert EXECUTOR_ORDER_TDD.splitlines()[0] in default
    assert "Confirm the new test fails for the reason you expect: the harness runs" in default
    assert _executor_system(SimpleNamespace(executor_order="legacy")) == EXECUTOR_SYSTEM


def test_tdd_first_replaces_the_batch_everything_guidance_rather_than_adding_to_it():
    text = _executor_system(SimpleNamespace(executor_order="tdd_first"))
    assert EXECUTOR_ORDER_TDD in text and EXECUTOR_ORDER_LEGACY not in text
    assert "apply them together" not in text
    assert "Work in this order" in text


def test_tdd_first_leaves_the_rest_of_the_prompt_intact():
    text = _executor_system(SimpleNamespace(executor_order="tdd_first"))
    assert text.replace(EXECUTOR_ORDER_TDD, EXECUTOR_ORDER_LEGACY) == EXECUTOR_SYSTEM


def test_the_legacy_guidance_is_what_the_default_prompt_contains():
    assert EXECUTOR_ORDER_LEGACY in EXECUTOR_SYSTEM
    assert "apply them together" in EXECUTOR_ORDER_LEGACY


def test_tdd_first_still_lets_a_step_with_nothing_to_test_skip_the_test_stages():
    assert "skips stages 2, 3 and" in EXECUTOR_ORDER_TDD
