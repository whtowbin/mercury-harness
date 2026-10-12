"""Executor prompt arms for Mercury's recency weighting (DESIGN-LOG §82).

`executor_batch_last` moves the batching rule to the end of the system prompt;
`executor_tail_reminder` sends the step's goal, the batching rule and the turns
left after the last message on each call, without keeping it in the history.
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.phases.execute import _executor_system
from agent.phases.plan import Step
from agent.prompts import EXECUTOR_BATCH_LAST, EXECUTOR_BATCH_RULE
from agent.runner import run_tool_loop
from agent.steptail import step_tail
from agent.tools import ToolContext, execution_registry
from agent.trace import Trace
from agent.workspace import Workspace
from bench import configs


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def settings(**flags):
    return SimpleNamespace(executor_order="tdd_first", **flags)


# -- batch_last ---------------------------------------------------------------


def test_the_default_prompt_is_unchanged():
    system = _executor_system(settings())
    assert EXECUTOR_BATCH_RULE in system and EXECUTOR_BATCH_LAST not in system


def test_batch_last_moves_the_rule_to_the_very_end():
    system = _executor_system(settings(executor_batch_last=True))
    assert EXECUTOR_BATCH_RULE not in system
    assert system.endswith(EXECUTOR_BATCH_LAST)
    # Only the one bullet moved: the rest of the prompt is the default, in order.
    default = _executor_system(settings())
    assert system == default.replace(EXECUTOR_BATCH_RULE, "") + EXECUTOR_BATCH_LAST


# -- tail reminder ---------------------------------------------------------------

STEP = Step(
    id="fix-eviction", title="Evict the oldest entry", intent="i", acceptance="test_evict passes"
)


def test_the_tail_names_the_step_and_counts_turns():
    text = step_tail(STEP, iteration=3, max_iters=14)
    assert "[fix-eviction] Evict the oldest entry" in text
    assert "Done when: test_evict passes" in text
    assert "Turns left: 11 of 14" in text
    assert "last turns" not in text
    assert "this is one of your last turns" in step_tail(STEP, iteration=12, max_iters=14)


def call(name, **args):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class Recording:
    """Returns scripted replies and keeps a copy of every message list it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return self.replies.pop(0)


def test_the_tail_ends_every_call_after_the_first_and_never_enters_the_history(ctx):
    usage = Usage(10, 5, 0.0, 0.1)
    client = Recording(
        [
            Reply(content="", tool_calls=[call("read_file", path="a.py")], usage=usage),
            Reply(content="", tool_calls=[call("read_file", path="a.py")], usage=usage),
            Reply(
                content="",
                tool_calls=[call("finish_step", summary="s", verified="false")],
                usage=usage,
            ),
        ]
    )
    _, history = run_tool_loop(
        client=client,
        model="fake",
        phase="execute:s",
        system="sys",
        user="the step",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
        tail=lambda i, n, m: f"TAIL {i}/{n}",
    )
    first, second, third = client.sent
    assert first[-1]["content"] == "the step"  # turn 0 already ends with the step
    assert second[-1] == {"role": "user", "content": "TAIL 1/6"}
    assert third[-1] == {"role": "user", "content": "TAIL 2/6"}
    # Each call's prefix is the previous call's history, unchanged: the cache holds.
    assert second[:-1] == first + [m for m in history[len(first) : len(second) - 1]]
    assert not any("TAIL" in str(m.get("content")) for m in history)


def test_without_a_tail_the_messages_are_the_history(ctx):
    usage = Usage(10, 5, 0.0, 0.1)
    client = Recording(
        [
            Reply(content="", tool_calls=[call("read_file", path="a.py")], usage=usage),
            Reply(
                content="",
                tool_calls=[call("finish_step", summary="s", verified="false")],
                usage=usage,
            ),
        ]
    )
    run_tool_loop(
        client=client,
        model="fake",
        phase="execute:s",
        system="sys",
        user="the step",
        registry=execution_registry(),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=6,
    )
    assert client.sent[1][-1]["role"] == "tool"


# -- bench arms ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("mercury-inception-batchlast", "executor_batch_last"),
        ("mercury-inception-tail", "executor_tail_reminder"),
        ("mercury-inception-t05", "mercury_temperature"),
    ],
)
def test_each_arm_differs_from_mercury_inception_in_one_field(name, field):
    control = dataclasses.asdict(configs.CONFIGS["mercury-inception"])
    arm = dataclasses.asdict(configs.CONFIGS[name])
    differing = {k for k in control if control[k] != arm[k]} - {"name", "description"}
    assert differing == {field}
