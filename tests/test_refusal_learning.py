"""A provider's 400 about a forced tool choice or reasoning off is learned once (DESIGN-LOG §129.4).

Against a real local server, not a mocked post (the user's rule for I/O): the server refuses the
first request the way a model that must reason does, and the test reads the bodies it received."""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent import llm

OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
FORCED = {"type": "function", "function": {"name": "finish_step"}}
TOOLS = [
    {"type": "function", "function": {"name": "finish_step", "parameters": {"type": "object"}}}
]


def _serve(refuse):
    """Answers 400 when refuse(body) is a message, else 200. Returns (url, bodies, stop)."""
    bodies: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            message = refuse(body)
            status, reply = (400, {"error": {"message": message}}) if message else (200, OK)
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", bodies, server.shutdown


def _client(url):
    return llm.OpenRouterClient(api_key="x", base_url=url, stream=False, max_retries=1)


def _ask(client, **kw):
    return client.chat(model="m", messages=[{"role": "user", "content": "hi"}], **kw).content


def refuse_forced(body):
    if isinstance(body.get("tool_choice"), dict):
        return "Thinking may not be enabled when tool_choice forces tool use."


def test_a_refused_forced_tool_choice_is_retried_as_auto_and_remembered():
    url, bodies, stop = _serve(refuse_forced)
    try:
        client = _client(url)
        assert _ask(client, tools=TOOLS, tool_choice=FORCED) == "ok"
        assert _ask(client, tools=TOOLS, tool_choice=FORCED) == "ok"
    finally:
        stop()
    assert [b["tool_choice"] for b in bodies] == [
        FORCED,
        "auto",
        "auto",
    ]  # the second call skips it
    assert client.refuses_forced_choice == {"m"}


def test_refused_reasoning_off_is_retried_at_low_and_remembered():
    def refuse(body):
        if (body.get("reasoning") or {}).get("effort") == "none":
            return "This model requires reasoning to be enabled."

    url, bodies, stop = _serve(refuse)
    try:
        client = _client(url)
        assert _ask(client, reasoning_effort="none") == "ok"
        assert _ask(client, reasoning_effort="none") == "ok"
    finally:
        stop()
    assert [b["reasoning"]["effort"] for b in bodies] == ["none", "low", "low"]


def test_an_unrelated_400_still_raises_and_is_not_retried():
    url, bodies, stop = _serve(lambda body: "context length exceeded")
    try:
        with pytest.raises(llm.LLMError, match="context length"):
            _ask(_client(url), tools=TOOLS, tool_choice=FORCED)
    finally:
        stop()
    assert len(bodies) == 1


def test_a_provider_that_accepts_the_forced_choice_is_untouched():
    url, bodies, stop = _serve(lambda body: None)
    try:
        client = _client(url)
        _ask(client, tools=TOOLS, tool_choice=FORCED)
    finally:
        stop()
    assert bodies[0]["tool_choice"] == FORCED
    assert client.refuses_forced_choice == set()


class ProseThenPlan:
    """A model that refused a forced tool choice: first a prose reply, then the tool call."""

    def __init__(self, tool_calls_first=False):
        self.refuses_forced_choice = {"m"}
        self.sent: list[list[dict]] = []
        self.replies = [[], [_submit_call()]]
        if tool_calls_first:
            self.replies = [[_submit_call()]]

    def chat(self, model, messages, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return llm.Reply(
            content="here is my plan in words",
            tool_calls=self.replies.pop(0),
            usage=llm.Usage(10, 5, 0.0, 0.1),
        )


def _submit_call():
    step = {"id": "s", "title": "t", "intent": "i", "acceptance": "a", "files": ["a.py"]}
    args = {"draft": ["d"], "steps": [step]}
    return {
        "id": "c1",
        "type": "function",
        "function": {"name": "submit_plan", "arguments": json.dumps(args)},
    }


def _last_word(client, tmp_path):
    from agent.approval import Approver
    from agent.runner import _last_word as last_word
    from agent.tools import ToolContext, planning_registry
    from agent.trace import Trace
    from agent.workspace import Workspace

    (tmp_path / "a.py").write_text("x = 1\n")
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    messages = [{"role": "user", "content": "plan it"}]
    last_word(client, "m", messages, planning_registry(), ctx, ctx.trace, "plan", None, None)
    return ctx


def test_a_model_that_refused_the_forced_choice_is_asked_once_more_in_prose(tmp_path):
    client = ProseThenPlan()
    ctx = _last_word(client, tmp_path)
    assert ctx.done and len(client.sent) == 2
    assert "Call submit_plan now" in client.sent[1][-1]["content"]


def test_a_model_that_complies_is_not_asked_twice(tmp_path):
    client = ProseThenPlan(tool_calls_first=True)
    ctx = _last_word(client, tmp_path)
    assert ctx.done and len(client.sent) == 1


def test_both_refusals_in_a_row_are_both_adapted_in_one_call():
    def refuse(body):
        if (body.get("reasoning") or {}).get("effort") == "none":
            return "This model requires reasoning to be enabled."
        if isinstance(body.get("tool_choice"), dict):
            return 'tool_choice: type "tool" and "any" are not supported for this model.'

    url, bodies, stop = _serve(refuse)
    try:
        client = _client(url)
        assert _ask(client, tools=TOOLS, tool_choice=FORCED, reasoning_effort="none") == "ok"
    finally:
        stop()
    assert [(b["tool_choice"], b["reasoning"]["effort"]) for b in bodies] == [
        (FORCED, "none"),
        (FORCED, "low"),
        ("auto", "low"),
    ]
