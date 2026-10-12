"""Not losing a diagnosis, and not stopping before the fix is made.

Every test here corresponds to something that actually went wrong in
traces/run-20260910-204717: the model worked out why two tests failed, ran out
of iterations before it could act, and the harness discarded the analysis and
reported that nothing had happened.
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.phases.finalcheck import BaselineTests, FinalCheck, failing_tests
from agent.runner import _salvage, run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(
        self,
        model,
        messages,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
        reasoning_effort=None,
    ) -> Reply:
        self.seen.append(list(messages))
        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.0, 0.1))
        return self.replies.pop(0)


def call(name, **args):
    return {
        "id": f"c{abs(hash(json.dumps(args, sort_keys=True))) % 10000}",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def thinking(text, *tool_calls):
    return Reply(content=text, tool_calls=list(tool_calls), usage=Usage(10, 5, 0.0, 0.1))


# -- salvaging an analysis the loop would otherwise discard ---------------


def test_salvage_picks_the_last_substantive_assistant_text():
    messages = [
        {"role": "assistant", "content": "early musing that is long enough to keep"},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "the real diagnosis, also long enough to keep"},
    ]
    assert "real diagnosis" in _salvage(messages)


def test_salvage_ignores_the_empty_content_of_a_tool_only_turn():
    """A tool-call turn carries content="", which must not be mistaken for the
    model's last word — the real analysis is the turn before it."""
    messages = [
        {
            "role": "assistant",
            "content": "a genuine analysis worth keeping, long enough to be real prose",
        },
        {"role": "assistant", "content": "", "tool_calls": [call("read_file", path="a")]},
    ]
    assert "genuine analysis" in _salvage(messages)


def test_salvage_skips_throwaway_chatter_too_short_to_be_a_diagnosis():
    messages = [{"role": "assistant", "content": "Let me look."}]
    assert _salvage(messages) == ""


def test_salvage_returns_nothing_when_the_model_only_made_tool_calls():
    messages = [{"role": "assistant", "content": "", "tool_calls": [call("x")]}]
    assert _salvage(messages) == ""


def test_an_exhausted_loop_keeps_the_models_analysis(ctx):
    """The headline bug: a fix phase diagnosed two failing tests correctly,
    hit its iteration cap, and the harness threw the whole thing away."""
    diagnosis = (
        "The test expects bird.y to include GRAVITY but update() only adds "
        "velocity, so it returns 192 instead of 192.8."
    )
    client = ScriptedClient([thinking(diagnosis, call("read_file", path="a.py")) for _ in range(4)])
    result, _ = run_tool_loop(
        client=client,
        model="fake",
        phase="fix:1",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
        salvage=True,
    )
    assert result is not None
    assert "GRAVITY" in result["summary"]
    assert result["incomplete"] is True
    assert result["verified"] is False


def test_a_salvaged_result_is_marked_unconfirmed(ctx):
    client = ScriptedClient(
        [
            thinking("a long enough analysis to be salvaged here", call("read_file", path="a.py"))
            for _ in range(4)
        ]
    )
    result, _ = run_tool_loop(
        client=client,
        model="fake",
        phase="fix:1",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
        salvage=True,
    )
    assert "INCOMPLETE" in result["notes"]


def test_salvage_is_off_unless_asked_for(ctx):
    """Plan and review expect their own result shapes; a summary-shaped
    salvage must not be handed to them."""
    client = ScriptedClient(
        [
            thinking(
                "some long analysis text that would be salvaged", call("read_file", path="a.py")
            )
            for _ in range(4)
        ]
    )
    result, _ = run_tool_loop(
        client=client,
        model="fake",
        phase="plan",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
    )
    assert result is None


def test_a_finished_loop_returns_the_real_result_not_a_salvage(ctx):
    client = ScriptedClient(
        [
            thinking(
                "some analysis text long enough to salvage",
                call("finish_step", summary="actually done"),
            )
        ]
    )
    result, _ = run_tool_loop(
        client=client,
        model="fake",
        phase="execute:s",
        system="s",
        user="u",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=3,
        salvage=True,
    )
    assert result["summary"] == "actually done"
    assert not result.get("incomplete")


# -- pre-existing failures are not this run's fault ----------------------


def test_failing_tests_are_parsed_from_pytest_output():
    output = (
        "FAILED tests/test_a.py::test_one - assert 1 == 2\n"
        "FAILED tests/test_b.py::test_two - boom\n"
        "ERROR tests/test_c.py\n"
    )
    assert failing_tests(output) == [
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
        "tests/test_c.py",
    ]


def test_no_failures_parses_to_an_empty_list():
    assert failing_tests("5 passed in 0.1s") == []


def test_a_pre_existing_failure_does_not_fail_the_run():
    check = FinalCheck(
        tests_ran=True,
        tests_passed=False,
        baseline_known=True,
        pre_existing_failures=["tests/test_flappy.py::test_pipes"],
        new_failures=[],
    )
    assert check.ok


def test_a_newly_broken_test_does_fail_the_run():
    check = FinalCheck(
        tests_ran=True,
        tests_passed=False,
        baseline_known=True,
        pre_existing_failures=[],
        new_failures=["tests/test_x.py::test_y"],
    )
    assert not check.ok


def test_without_a_baseline_any_failure_counts():
    check = FinalCheck(tests_ran=True, tests_passed=False, baseline_known=False)
    assert not check.ok


def test_the_finding_names_the_new_failures_and_excuses_the_old_ones():
    check = FinalCheck(
        tests_ran=True,
        tests_passed=False,
        baseline_known=True,
        new_failures=["tests/test_x.py::test_y"],
        pre_existing_failures=["tests/test_old.py::test_z"],
        tests_output="boom",
    )
    finding = check.findings()[0]
    assert "tests/test_x.py::test_y" in finding["issue"]
    assert "tests/test_old.py::test_z" in finding["fix"]
    assert finding["file"] == "tests/test_x.py"


def test_a_pre_existing_failure_produces_no_finding_to_fix():
    check = FinalCheck(
        tests_ran=True,
        tests_passed=False,
        baseline_known=True,
        pre_existing_failures=["tests/test_old.py::test_z"],
    )
    assert check.findings() == []


# -- the failure signature, which stops a pointless repeat ---------------


def test_the_same_failures_produce_the_same_signature():
    a = FinalCheck(baseline_known=True, new_failures=["t::a", "t::b"])
    b = FinalCheck(baseline_known=True, new_failures=["t::b", "t::a"])
    assert a.signature() == b.signature()


def test_different_failures_produce_different_signatures():
    a = FinalCheck(baseline_known=True, new_failures=["t::a"])
    b = FinalCheck(baseline_known=True, new_failures=["t::c"])
    assert a.signature() != b.signature()


def test_a_verify_failure_is_part_of_the_signature():
    a = FinalCheck(verify_failures=[{"step": "s1"}])
    b = FinalCheck(verify_failures=[{"step": "s2"}])
    assert a.signature() != b.signature()


# -- baseline plumbing ---------------------------------------------------


def test_a_baseline_records_what_was_already_red():
    baseline = BaselineTests(failures=["tests/test_a.py::test_one"])
    assert baseline.failures == ["tests/test_a.py::test_one"]
    assert not baseline.collection_broken
