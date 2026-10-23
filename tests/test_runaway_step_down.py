"""§131.12: a Mercury runaway is asked again at low effort, against a real local stream.

Mercury 3 streams no reasoning and, on a hard prompt at medium or high effort, often
reasons to the token cap without answering. Two shapes reach the client: nothing but
keep-alives until the first-content cut, or a finished stream with finish_reason
"length" and no text. The server here answers by the effort it is sent, so each test
sees what the client actually asked for, in order.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from agent import llm

MERCURY = "inception/mercury-3"
MESSAGES = [{"role": "user", "content": "plan it"}]


def _server(runaway: str):
    """`runaway`: "silent" (keep-alives until cut) or "cap" (length stop, no text) at
    medium or high effort; low effort always answers "ok"."""
    efforts: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            effort = (body.get("reasoning") or {}).get("effort", "")
            efforts.append(effort)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            try:
                if effort == "low":
                    send({"choices": [{"delta": {"content": "ok"}}]})
                    send({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                    send({"usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}})
                elif runaway == "silent":
                    for _ in range(40):
                        self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                        self.wfile.flush()
                        time.sleep(0.1)
                else:
                    send({"choices": [{"delta": {}, "finish_reason": "length"}]})
                    send(
                        {
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 16000,
                                "completion_tokens_details": {"reasoning_tokens": 16000},
                                "cost": 0.0056,
                            }
                        }
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}", efforts


@pytest.fixture
def silent():
    server, url, efforts = _server("silent")
    yield url, efforts
    server.shutdown()


@pytest.fixture
def cap():
    server, url, efforts = _server("cap")
    yield url, efforts
    server.shutdown()


def _client(url: str, step_down: bool, max_call_s: float = 8.0) -> llm.OpenRouterClient:
    return llm.OpenRouterClient(
        api_key="x",
        base_url=url,
        timeout=20.0,
        max_call_s=max_call_s,
        max_retries=2,
        stream=True,
        runaway_step_down=step_down,
    )


def test_a_cut_for_no_content_is_asked_again_at_low_effort(silent, monkeypatch):
    monkeypatch.setattr(llm, "DIFFUSION_FIRST_CONTENT_TIMEOUT_S", 0.5)
    url, efforts = silent
    reply = _client(url, True).chat(MERCURY, MESSAGES, reasoning_effort="high")
    assert reply.content == "ok"
    assert efforts == ["high", "low"]
    assert "low effort" in reply.warning


def test_without_the_switch_a_cut_repeats_the_same_request(silent, monkeypatch):
    monkeypatch.setattr(llm, "DIFFUSION_FIRST_CONTENT_TIMEOUT_S", 0.5)
    url, efforts = silent
    with pytest.raises(llm.LLMError, match="no content within"):
        _client(url, False, max_call_s=2.5).chat(MERCURY, MESSAGES, reasoning_effort="high")
    assert efforts and set(efforts) == {"high"}


def test_a_reasoning_only_length_stop_is_asked_again_and_both_calls_are_paid_for(cap):
    url, efforts = cap
    reply = _client(url, True).chat(MERCURY, MESSAGES, reasoning_effort="medium")
    assert reply.content == "ok"
    assert efforts == ["medium", "low"]
    assert reply.usage.completion_tokens == 16005
    assert reply.usage.cost_usd == pytest.approx(0.0066)


def test_low_effort_and_other_models_are_never_stepped(cap):
    url, efforts = cap
    reply = _client(url, True).chat("z-ai/glm-5.3-flash", MESSAGES, reasoning_effort="high")
    assert reply.finish_reason == "length" and efforts == ["high"]
