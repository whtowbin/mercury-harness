"""A zero-token refusal fails the call and the run, not the plan (DESIGN-LOG §127.9).

Sonnet 5.5 refused every plan call with HTTP 200, `finish_reason: content_filter`, no content,
no tool call and zero tokens; the harness read that as a reply without a tool call and fell back
to a one-step plan. Against a real local server that answers the way OpenRouter did (the user's
rule for I/O), in both the plain and the streamed wire format the harness uses."""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent import llm
from agent.approval import Approver
from agent.config import Settings
from agent.phases.plan import make_plan
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace

REFUSAL = {
    "provider": "Azure",
    "choices": [
        {
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "content_filter",
            "native_finish_reason": "refusal",
        }
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0},
}
EMPTY_STOP = {
    "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0},
}


def _serve(answer):
    """Answers every request with `answer` (a reply, or a function of the request body that
    returns one), as SSE when the body asks to stream."""
    bodies: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            reply = answer(body) if callable(answer) else answer
            if body.get("stream"):
                choice = reply["choices"][0]
                delta = dict(choice["message"])
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [
                        {"index": i, **call} for i, call in enumerate(delta["tool_calls"])
                    ]
                chunk = {
                    "provider": reply.get("provider", ""),
                    "choices": [
                        {
                            "delta": delta,
                            "finish_reason": choice["finish_reason"],
                            "native_finish_reason": choice.get("native_finish_reason"),
                        }
                    ],
                    "usage": reply["usage"],
                }
                data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                data, content_type = json.dumps(reply).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", bodies, server.shutdown


def _client(url, stream):
    return llm.OpenRouterClient(api_key="x", base_url=url, stream=stream, max_retries=1)


@pytest.mark.parametrize("stream", [False, True])
def test_a_zero_token_content_filter_reply_raises_and_is_not_retried(stream):
    url, bodies, stop = _serve(REFUSAL)
    try:
        with pytest.raises(llm.ModelRefused, match="refused the request") as caught:
            _client(url, stream).chat(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        stop()
    assert len(bodies) == 1
    assert caught.value.finish_reason == "content_filter"
    assert isinstance(caught.value, llm.LLMError)


@pytest.mark.parametrize("stream", [False, True])
def test_an_ordinary_empty_reply_is_still_a_reply(stream):
    """Mercury's case: an empty answer without a tool call stays a valid reply, for the
    planner's one-step fallback and the tool loop's stall nudge to handle."""
    url, _, stop = _serve(EMPTY_STOP)
    try:
        reply = _client(url, stream).chat(model="m", messages=[{"role": "user", "content": "hi"}])
    finally:
        stop()
    assert reply.finish_reason == "stop" and not reply.tool_calls


def _settings(tmp_path, model="m"):
    return Settings(
        workspace=str(tmp_path),
        quiet=True,
        trace_dir=str(tmp_path / "t"),
        goal="fix a.py",
        planner_model=model,
    )


def _plan_with(url, tmp_path, client=None, model="m"):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    try:
        plan = make_plan(client or _client(url, True), _settings(tmp_path, model), ctx, trace)
        return plan, trace
    finally:
        trace.close()


def _kinds(trace):
    return [json.loads(line)["kind"] for line in trace.path.read_text().splitlines()]


def test_the_planner_does_not_fall_back_to_one_step_on_a_refusal(tmp_path):
    url, bodies, stop = _serve(REFUSAL)
    try:
        with pytest.raises(llm.ModelRefused):
            _plan_with(url, tmp_path)
    finally:
        stop()
    trace_files = list((tmp_path / "t").glob("run-*.jsonl"))
    kinds = [json.loads(line)["kind"] for f in trace_files for line in f.read_text().splitlines()]
    assert "model_refused" in kinds
    assert "plan_fallback" not in kinds and "stalled" not in kinds
    assert len(bodies) == 2  # the request, then one retry without the reasoning wording


def test_the_planner_still_falls_back_on_an_ordinary_empty_reply(tmp_path):
    url, _, stop = _serve(EMPTY_STOP)
    try:
        plan, trace = _plan_with(url, tmp_path)
    finally:
        stop()
    assert [s.id for s in plan.steps] == ["whole-goal"]
    assert "plan_fallback" in _kinds(trace)


# -- a model that refuses the reasoning hand-off wording (§127.11) ---------

CLAUSE = "after your chain-of-draft reasoning"
STEP = {"id": "s", "title": "fix", "intent": "make x 2", "acceptance": "x is 2", "files": ["a.py"]}
PLAN_CALL = {
    "id": "c1",
    "type": "function",
    "function": {
        "name": "submit_plan",
        "arguments": json.dumps({"draft": ["understood: fix a.py"], "steps": [STEP]}),
    },
}
PLANNED = {
    "provider": "Claude Platform on AWS",
    "choices": [
        {
            "message": {"role": "assistant", "content": "", "tool_calls": [PLAN_CALL]},
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
}


def _submit_description(body: dict) -> str:
    return next(
        t["function"]["description"]
        for t in body.get("tools") or []
        if t["function"]["name"] == "submit_plan"
    )


def _refuses_the_clause(body: dict) -> dict:
    """Sonnet 5.5 as measured: refuses while submit_plan's description carries the clause."""
    return REFUSAL if CLAUSE in _submit_description(body) else PLANNED


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_plan_call_is_retried_without_the_clause_and_the_model_is_remembered(
    tmp_path, stream
):
    url, bodies, stop = _serve(_refuses_the_clause)
    try:
        client = _client(url, stream)
        plan, trace = _plan_with(url, tmp_path, client=client)
        _plan_with(url, tmp_path / "again", client=client)
    finally:
        stop()
    assert [s.id for s in plan.steps] == ["s"]
    assert "plan_fallback" not in _kinds(trace) and "model_refused" not in _kinds(trace)
    # One refusal, one retry without the clause, and the second plan sent without it at once.
    assert [CLAUSE in _submit_description(b) for b in bodies] == [True, False, False]
    assert _submit_description(bodies[1]) == "Submit the finished plan. Call exactly once."
    assert client.refuses_reasoning_wording == {"m"}


def test_a_refusal_that_survives_the_rewording_still_fails_the_run(tmp_path):
    url, bodies, stop = _serve(REFUSAL)
    try:
        with pytest.raises(llm.ModelRefused):
            _plan_with(url, tmp_path)
    finally:
        stop()
    assert [CLAUSE in _submit_description(b) for b in bodies] == [True, False]


def test_a_model_that_never_refused_is_sent_the_planner_tools_unchanged(tmp_path):
    """Mercury's and GLM's planner request stays byte-for-byte what it was before the
    rewording existed (§127.10.1: dropping the clause for everyone moved Mercury's plans):
    the wire body carries exactly the tools the plan phase handed the client, clause included."""
    from agent.tools.control import SUBMIT_PLAN

    url, bodies, stop = _serve(PLANNED)
    client = _client(url, True)
    handed: list[list[dict]] = []
    chat = client.chat

    def spy(model, messages, tools=None, **kwargs):
        handed.append(json.loads(json.dumps(tools)))
        return chat(model, messages, tools=tools, **kwargs)

    client.chat = spy
    try:
        _plan_with(url, tmp_path, client=client, model="inception/mercury-2.5")
    finally:
        stop()
    assert SUBMIT_PLAN.description == (
        "Submit the finished plan. Call exactly once, after your chain-of-draft reasoning."
    )
    sent = handed[0] if not client.strict_tools else llm.strict_schemas(handed[0])
    assert json.dumps(bodies[0]["tools"]) == json.dumps(sent)
    assert CLAUSE in _submit_description(bodies[0])
    assert client.refuses_reasoning_wording == set()


def test_the_rewording_never_mutates_the_callers_schemas():
    described = {"name": "submit_plan", "description": f"x, {CLAUSE}."}
    tools = [{"type": "function", "function": described}]
    plain = llm.without_refused_wording(tools)
    assert plain is not None and plain[0]["function"]["description"] == "x."
    assert CLAUSE in tools[0]["function"]["description"]
    assert llm.without_refused_wording(plain) is None
