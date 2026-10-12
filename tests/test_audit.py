from __future__ import annotations

import json

from agent import audit
from agent.llm import Reply, Usage


class _FakeClient:
    """Records the prompt it was called with and returns a fixed reply —
    the model call itself is what a real test must never run for real, since
    the whole point of this test suite is checking the deterministic
    prompt-building and parsing logic around it, not GLM's actual output."""

    def __init__(self, content: str, cost_usd: float = 0.001):
        self.content = content
        self.cost_usd = cost_usd
        self.calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return Reply(content=self.content, usage=Usage(100, 50, self.cost_usd, 0.5))


def _events(**overrides):
    base = [
        {"ts": 1.0, "kind": "run_start", "goal": "build a game", "workspace": "/tmp/x"},
        {"ts": 2.0, "kind": "plan", "steps": [{"id": "s1"}]},
        {"ts": 3.0, "kind": "log", "message": "-- step 1/1: s1", "detail": False},
        {"ts": 4.0, "kind": "log", "message": "internal detail", "detail": True},
        {
            "ts": 5.0,
            "kind": "loop_exhausted",
            "phase": "execute:s1",
            "iterations": 14,
            "done": False,
        },
        {
            "ts": 6.0,
            "kind": "step_verify",
            "step": "s1",
            "command": "pytest",
            "passed": False,
            "output": "1 failed",
        },
        {"ts": 7.0, "kind": "final_check", "ok": False},
        {"ts": 8.0, "kind": "totals", "cost_usd": 0.05, "calls": 12},
    ]
    return base


# -- build_audit_prompt --------------------------------------------------


def test_prompt_includes_the_trace_summary():
    prompt = audit.build_audit_prompt(_events())
    assert '"goal": "build a game"' in prompt


def test_prompt_includes_narration_but_skips_detail_logs():
    prompt = audit.build_audit_prompt(_events())
    assert "step 1/1" in prompt
    assert "internal detail" not in prompt


def test_prompt_includes_a_failed_step_verify():
    prompt = audit.build_audit_prompt(_events())
    assert "step_verify FAILED" in prompt
    assert "1 failed" in prompt


def test_prompt_includes_loop_exhausted_events():
    prompt = audit.build_audit_prompt(_events())
    assert "loop_exhausted" in prompt


# -- parse_findings -------------------------------------------------------


def test_parses_plain_json_list():
    text = '[{"category": "model_weakness", "summary": "x", "evidence": "y"}]'
    findings = audit.parse_findings(text)
    assert findings == [{"category": "model_weakness", "summary": "x", "evidence": "y"}]


def test_parses_json_wrapped_in_a_code_fence():
    text = '```json\n[{"summary": "x"}]\n```'
    assert audit.parse_findings(text) == [{"summary": "x"}]


def test_parses_json_with_stray_prose_around_it():
    text = 'Sure, here is the list:\n[{"summary": "x"}]\nHope that helps!'
    assert audit.parse_findings(text) == [{"summary": "x"}]


def test_empty_list_is_valid():
    assert audit.parse_findings("[]") == []


def test_unparseable_text_returns_empty_list_not_an_error():
    assert audit.parse_findings("not json at all") == []


def test_a_non_list_top_level_value_is_rejected():
    assert audit.parse_findings('{"summary": "x"}') == []


def test_findings_missing_a_summary_field_are_dropped():
    text = '[{"category": "other"}, {"summary": "kept"}]'
    assert audit.parse_findings(text) == [{"summary": "kept"}]


# -- audit_trace ------------------------------------------------------------


def test_audit_trace_returns_findings_model_and_cost():
    client = _FakeClient('[{"summary": "found something"}]', cost_usd=0.0021)
    result = audit.audit_trace(_events(), client, model="z-ai/glm-5.3-flash")
    assert result["findings"] == [{"summary": "found something"}]
    assert result["model"] == "z-ai/glm-5.3-flash"
    assert result["cost_usd"] == 0.0021


def test_audit_trace_calls_the_client_with_the_requested_model():
    client = _FakeClient("[]")
    audit.audit_trace(_events(), client, model="some/model")
    assert client.calls[0]["model"] == "some/model"


def test_audit_trace_survives_an_unparseable_reply():
    client = _FakeClient("I cannot help with that.")
    result = audit.audit_trace(_events(), client)
    assert result["findings"] == []
    assert result["raw"] == "I cannot help with that."


# -- audit_traces -----------------------------------------------------------


def test_audit_traces_skips_passing_runs_by_default(tmp_path):
    failed = _events()
    passed = [e for e in _events() if e["kind"] != "final_check"] + [
        {"ts": 7.0, "kind": "final_check", "ok": True}
    ]
    (tmp_path / "run-a.jsonl").write_text("\n".join(json.dumps(e) for e in failed))
    (tmp_path / "run-b.jsonl").write_text("\n".join(json.dumps(e) for e in passed))

    client = _FakeClient("[]")
    results = audit.audit_traces(tmp_path, client)
    assert [r["trace"] for r in results] == ["run-a.jsonl"]


def test_audit_traces_all_runs_includes_passing_ones(tmp_path):
    failed = _events()
    passed = [e for e in _events() if e["kind"] != "final_check"] + [
        {"ts": 7.0, "kind": "final_check", "ok": True}
    ]
    (tmp_path / "run-a.jsonl").write_text("\n".join(json.dumps(e) for e in failed))
    (tmp_path / "run-b.jsonl").write_text("\n".join(json.dumps(e) for e in passed))

    client = _FakeClient("[]")
    results = audit.audit_traces(tmp_path, client, failed_only=False)
    assert {r["trace"] for r in results} == {"run-a.jsonl", "run-b.jsonl"}


def test_audit_traces_with_no_matching_runs_returns_empty(tmp_path):
    passed = [e for e in _events() if e["kind"] != "final_check"] + [
        {"ts": 7.0, "kind": "final_check", "ok": True}
    ]
    (tmp_path / "run-a.jsonl").write_text("\n".join(json.dumps(e) for e in passed))

    client = _FakeClient("[]")
    results = audit.audit_traces(tmp_path, client)
    assert results == []


def test_audit_traces_preserves_outcome_order_under_concurrency(tmp_path):
    for name in ("run-a.jsonl", "run-b.jsonl", "run-c.jsonl"):
        (tmp_path / name).write_text("\n".join(json.dumps(e) for e in _events()))

    client = _FakeClient("[]")
    results = audit.audit_traces(tmp_path, client, concurrency=3)
    assert [r["trace"] for r in results] == ["run-a.jsonl", "run-b.jsonl", "run-c.jsonl"]


class _FlakyClient:
    """Raises on the first call, succeeds on the rest — one provider stall
    among several traces, the shape hit in practice (a 180s streaming
    deadline on the audit call itself)."""

    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("request failed after 7 attempts or 420s: stalled")
        return Reply(content="[]", usage=Usage(100, 50, 0.001, 0.5))


def test_audit_traces_one_flaky_call_does_not_discard_the_others(tmp_path):
    for name in ("run-a.jsonl", "run-b.jsonl", "run-c.jsonl"):
        (tmp_path / name).write_text("\n".join(json.dumps(e) for e in _events()))

    results = audit.audit_traces(tmp_path, _FlakyClient(), concurrency=1)
    assert [r["trace"] for r in results] == ["run-a.jsonl", "run-b.jsonl", "run-c.jsonl"]
    failed = [r for r in results if r.get("error")]
    ok = [r for r in results if not r.get("error")]
    assert len(failed) == 1
    assert "stalled" in failed[0]["error"]
    assert failed[0]["cost_usd"] == 0.0
    assert failed[0]["findings"] == []
    assert len(ok) == 2


# -- reasoning_effort --------------------------------------------------------


def test_audit_trace_defaults_to_low_reasoning_effort():
    client = _FakeClient("[]")
    audit.audit_trace(_events(), client)
    assert client.calls[0]["reasoning_effort"] == "low"


def test_audit_trace_reasoning_effort_is_overridable():
    client = _FakeClient("[]")
    audit.audit_trace(_events(), client, reasoning_effort="high")
    assert client.calls[0]["reasoning_effort"] == "high"
