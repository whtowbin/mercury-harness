"""Plan gates on the planner's last word (DESIGN-LOG §88.3).

A planner that runs out of turns is forced to submit_plan once (runner._last_word),
and that call used to skip every gate: half of django's research-first plans ended
there, unseen by the research, call-site and findings gates. `last_word_gates`
chooses: off (as before), turn (a hold gets one more forced turn), notes (the plan
goes through and the hold's message becomes a finding).
"""

from __future__ import annotations

import json

import pytest

from agent.approval import Approver
from agent.llm import Reply, Usage
from agent.runner import run_tool_loop
from agent.tools import ToolContext, planning_registry
from agent.trace import Trace
from agent.workspace import Workspace

STEP = {"id": "s", "title": "t", "intent": "i", "acceptance": "a", "files": ["a.py"]}
HOLD = "[harness] Plan NOT submitted: no decisions."


def call(name: str, **args) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def submit(findings: list[str]) -> dict:
    return call("submit_plan", draft=["d"], steps=[STEP], findings=findings)


class HoldsSubmit:
    """Holds every submit_plan whose findings are empty, like DecisionsGate but not once-only,
    so the test can see that a second last-word turn is never held."""

    event = "test_gate_held"

    def __init__(self) -> None:
        self.holds = 0

    def observe(self, call: dict) -> None:
        return None

    def hold(self, call: dict) -> str | None:
        fn = call.get("function") or {}
        if fn.get("name") != "submit_plan":
            return None
        if json.loads(fn.get("arguments") or "{}").get("findings"):
            return None
        self.holds += 1
        return HOLD


class Scripted:
    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return Reply(content="", tool_calls=[self.replies.pop(0)], usage=Usage(10, 5, 0.0, 0.1))


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def exhaust(ctx, replies, mode):
    """One exploring turn (max_iters=1), then the last word."""
    gate = HoldsSubmit()
    client = Scripted(replies)
    result, _ = run_tool_loop(
        client=client,
        model="fake",
        phase="plan",
        system="s",
        user="u",
        registry=planning_registry("terse"),
        ctx=ctx,
        trace=ctx.trace,
        max_iters=1,
        salvage=True,
        gate=gate,
        last_word_gates=mode,
    )
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    return result, client, gate, events


def test_off_skips_the_gate_as_before(ctx):
    result, client, gate, events = exhaust(ctx, [call("read_file", path="a.py"), submit([])], "off")
    assert gate.holds == 0 and len(client.sent) == 2
    assert result["steps"][0]["id"] == "s"
    assert not [e for e in events if e["kind"] == "last_word_gate"]


def test_turn_gives_a_held_last_word_one_more_forced_turn(ctx):
    result, client, _gate, events = exhaust(
        ctx,
        [call("read_file", path="a.py"), submit([]), submit(["decision: radians because pi/4"])],
        "turn",
    )
    assert len(client.sent) == 3  # explore, last word, one extra turn
    assert any(HOLD in str(m.get("content")) for m in client.sent[2])
    assert result["findings"] == ["decision: radians because pi/4"]
    assert [e["kind"] for e in events if e["kind"].startswith("last_word")] == [
        "last_word",
        "last_word_gate",
        "last_word_extra_turn",
    ]


def test_the_extra_turn_is_never_held_so_a_plan_always_comes_out(ctx):
    result, client, gate, _ = exhaust(
        ctx, [call("read_file", path="a.py"), submit([]), submit([])], "turn"
    )
    assert gate.holds == 1 and len(client.sent) == 3
    assert result["steps"][0]["id"] == "s"  # no one-step fallback


def test_notes_lets_the_plan_through_and_hands_the_hold_to_the_executor(ctx):
    result, client, _gate, _ = exhaust(ctx, [call("read_file", path="a.py"), submit([])], "notes")
    assert len(client.sent) == 2
    assert result["steps"][0]["id"] == "s"
    assert result["findings"] == ["planner check not addressed before the plan was forced: " + HOLD]


def test_a_plan_the_gate_accepts_is_untouched_in_every_mode(ctx):
    for mode in ("turn", "notes"):
        result, client, _gate, _ = exhaust(
            ctx, [call("read_file", path="a.py"), submit(["decision: x because y"])], mode
        )
        assert len(client.sent) == 2 and result["findings"] == ["decision: x because y"]
