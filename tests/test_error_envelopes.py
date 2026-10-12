"""Error envelopes named in words: transient ones retried, out-of-quota ones final (§89.3).

Inception sends {"code": "server_error"} inside a 200 body or mid-stream; the int parse turned
it into None and every one killed its run. Against a real local server, not a mocked post.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent import llm

SERVER_ERROR = {
    "error": {
        "message": "The server had an error while processing your request.",
        "type": "server_error",
        "param": None,
        "code": "server_error",
    }
}
QUOTA = {
    "error": {
        "message": "Free tier limit reached. Please upgrade to a paid plan.",
        "type": "free_tier_quota_exceeded",
        "code": "free_tier_quota_exceeded",
    }
}
OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def _serve(bodies: list[tuple[int, dict]]):
    calls: list[int] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status, body = bodies[min(len(calls), len(bodies) - 1)]
            calls.append(status)
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", calls, server.shutdown


def _chat(base: str) -> str:
    client = llm.OpenRouterClient(api_key="x", base_url=base, stream=False, max_retries=2)
    return client.chat(model="m", messages=[{"role": "user", "content": "hi"}]).content


def test_a_server_error_envelope_in_a_200_is_retried_and_the_call_succeeds(monkeypatch):
    monkeypatch.setattr(llm.OpenRouterClient, "_sleep", lambda self, *a, **k: True)
    base, calls, stop = _serve([(200, SERVER_ERROR), (200, OK)])
    try:
        assert _chat(base) == "ok"
    finally:
        stop()
    assert len(calls) == 2


def test_an_out_of_quota_envelope_is_final_and_not_retried(monkeypatch):
    monkeypatch.setattr(llm.OpenRouterClient, "_sleep", lambda self, *a, **k: True)
    base, calls, stop = _serve([(200, QUOTA), (200, OK)])
    try:
        with pytest.raises(llm.LLMError, match="free_tier_quota_exceeded"):
            _chat(base)
    finally:
        stop()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        {"code": "server_error"},
        {"type": "overloaded_error"},
        {"code": "rate_limit_exceeded", "type": "rate_limit_error"},
        {"code": 503},
    ],
)
def test_transient_names_and_codes_are_retryable(error):
    assert isinstance(llm._envelope_error(error), llm._Retryable)


@pytest.mark.parametrize(
    "error",
    [
        {"code": "free_tier_quota_exceeded", "type": "rate_limit_error"},
        {"code": "insufficient_quota"},
        {"code": "invalid_request_error"},
        {"code": 400},
    ],
)
def test_terminal_names_and_other_errors_stay_final(error):
    assert not isinstance(llm._envelope_error(error), llm._Retryable)
