"""Mercury's hidden reasoning, and the process counters added beside it.

Mercury 2.5 returns its reasoning as a token count and never as text
(DESIGN-LOG §53), so the count is the only view of how hard a call thought.
These tests pin that it survives from the provider's usage block to the
trace, the run totals and the bench diagnostics.
"""

from __future__ import annotations

import json

from agent.llm import OpenRouterClient, Usage, _usage_from
from agent.trace import Trace
from bench import diagnostics

# -- reasoning tokens: provider -> Usage -> trace -> totals ----------------------


def test_usage_reads_reasoning_tokens():
    usage = _usage_from(
        {"completion_tokens": 1214, "completion_tokens_details": {"reasoning_tokens": 1151}},
        latency=1.0,
        ttft=0.1,
    )
    assert usage.reasoning_tokens == 1151


def test_usage_without_details_reports_zero_reasoning():
    assert _usage_from({"completion_tokens": 7}, latency=1.0, ttft=0.1).reasoning_tokens == 0


def test_a_streamed_usage_chunk_carries_reasoning_tokens(monkeypatch):
    """The shape OpenRouter actually sent for mercury-2.5 at effort=high."""
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "391"}}]}),
        "data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 22,
                    "completion_tokens": 361,
                    "completion_tokens_details": {"reasoning_tokens": 351},
                },
            }
        ),
        "data: [DONE]",
    ]

    class Resp:
        status_code = 200
        text = ""

        def __init__(self):
            self.headers: dict[str, str] = {}

        def iter_lines(self, decode_unicode=False):
            yield from lines

        def close(self):
            pass

    client = OpenRouterClient("k", stream=True)
    monkeypatch.setattr(client.session, "post", lambda *a, **k: Resp())
    reply = client.chat("m", [{"role": "user", "content": "hi"}])
    assert reply.usage.reasoning_tokens == 351


def test_the_llm_call_event_and_totals_carry_reasoning_tokens(tmp_path):
    trace = Trace(str(tmp_path), echo=False)
    usage = Usage(prompt_tokens=10, completion_tokens=100, reasoning_tokens=90)
    trace.llm_call("execute:s1", "mercury", usage)
    trace.llm_call("execute:s1", "mercury", usage)
    trace._fh.flush()
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [e["reasoning_tokens"] for e in events if e["kind"] == "llm_call"] == [90, 90]
    assert trace.totals.reasoning_tokens == 180
    assert trace.totals.by_model["mercury"]["reasoning_tokens"] == 180


# -- diagnostics ----------------------------------------------------------------


def ev(kind: str, **fields) -> dict:
    return {"kind": kind, **fields}


def test_edit_misses_count_only_rejected_edits():
    events = [
        ev("tool_call_failed", tool="edit_file", result="error: old_string not found"),
        ev("tool_call_failed", tool="edit_files", result="error: 0/2 applied"),
        ev("tool_call_failed", tool="run_bash", result="error: denied"),
    ]
    result = diagnostics.from_events(events)
    assert result.edit_misses == 2
    # Still counted as tool errors too: the old column must not change meaning.
    assert result.tool_errors == 3


def test_a_rerun_with_no_edit_between_is_counted():
    events = [
        ev("run_tests", summary="1 failed, 2 passed", exit_code=1),
        ev("run_tests", summary="1 failed, 2 passed", exit_code=1),
        ev("tool_call", tool="edit_file", phase="execute:s1"),
        ev("run_tests", summary="3 passed", exit_code=0),
    ]
    assert diagnostics.from_events(events).reruns_without_edit == 1


def test_a_rerun_after_a_green_suite_is_not_counted():
    events = [
        ev("run_tests", summary="3 passed", exit_code=0),
        ev("run_tests", summary="3 passed", exit_code=0),
    ]
    assert diagnostics.from_events(events).reruns_without_edit == 0


def test_reasoning_tokens_are_split_by_phase():
    events = [
        ev("llm_call", phase="plan", reasoning_tokens=500),
        ev("llm_call", phase="execute:s1", reasoning_tokens=300),
        ev("llm_call", phase="fix:f1", reasoning_tokens=50),
        ev("llm_call", phase="review", reasoning_tokens=200),
        ev("llm_call", phase="suggest", reasoning_tokens=999),
        ev("llm_call", phase="execute:s2"),  # an older trace: no field at all
    ]
    result = diagnostics.from_events(events)
    assert result.plan_reasoning_tokens == 500
    assert result.execute_reasoning_tokens == 350
    assert result.review_reasoning_tokens == 200
