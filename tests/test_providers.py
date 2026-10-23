"""Mercury through Inception's own API (or Venice), OpenRouter for the rest.

The dialect differences were measured against both live APIs on 2026-09-29
(DESIGN-LOG §85); these tests pin the translation, the cost, the temperature
rule and the routing, and run the real client against a real local server.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from agent import providers as P
from agent.llm import OpenRouterClient, _is_diffusion_model

BODY = {
    "model": "inception/mercury-2.5",
    "messages": [{"role": "user", "content": "hi"}],
    "usage": {"include": True},
    "provider": {"sort": "latency"},
    "reasoning": {"effort": "none"},
    "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
    "tool_choice": "auto",
}


# -- request translation ------------------------------------------------------


def test_openrouter_requests_are_unchanged():
    assert P.OPENROUTER.request(dict(BODY)) == BODY


def test_inception_gets_its_own_model_id_and_reasoning_field():
    out = P.INCEPTION.request(dict(BODY))
    assert out["model"] == "mercury-2.5"
    # The OpenRouter object is silently ignored by Inception (a call then runs at
    # medium), so it must become the top-level field; `none` is its `instant`.
    assert out["reasoning_effort"] == "instant"
    assert "reasoning" not in out and "usage" not in out and "provider" not in out
    assert out["tools"] == BODY["tools"] and out["tool_choice"] == "auto"
    assert "stream_options" not in out


@pytest.mark.parametrize(
    ("ours", "theirs"), [("low", "low"), ("high", "high"), ("minimal", "instant")]
)
def test_inception_effort_names(ours, theirs):
    body = {**BODY, "reasoning": {"effort": ours}}
    assert P.INCEPTION.request(body)["reasoning_effort"] == theirs


def test_a_streamed_inception_request_asks_for_usage():
    out = P.INCEPTION.request({**BODY, "stream": True})
    assert out["stream_options"] == {"include_usage": True}


def test_venice_gets_its_model_id():
    assert P.VENICE.request(dict(BODY))["model"] == "mercury-2-5"


def test_an_unmapped_model_is_sent_as_named():
    assert P.INCEPTION.request({**BODY, "model": "mercury-2.5"})["model"] == "mercury-2.5"


# -- temperature --------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "override", "sent"),
    [
        (0.0, None, None),  # Inception would reset it to 1.0: send nothing, same effect
        (0.2, None, None),
        (0.7, None, 0.7),
        (None, None, None),
        (0.1, 0.5, 0.5),  # an explicit Mercury temperature wins
        (None, 0.9, 0.9),
        (0.2, 0.3, None),  # an override outside the range is not sent either
    ],
)
def test_mercury_temperature(requested, override, sent):
    assert P.effective_temperature("inception/mercury-2.5", requested, override) == sent


def test_other_models_keep_their_temperature():
    assert P.effective_temperature("z-ai/glm-5.3-flash", 0.0, 0.5) == 0.0


def test_mercury_timeouts_apply_to_the_direct_model_id():
    assert _is_diffusion_model("mercury-2.5") and _is_diffusion_model("inception/mercury-2.5")
    assert not _is_diffusion_model("z-ai/glm-5.3-flash")


# -- cost ---------------------------------------------------------------------


def test_cost_from_the_price_table_counts_cached_input_at_its_rate():
    # 1M prompt of which 400k cached, 100k out: 0.6*0.04 + 0.4*0.004 + 0.1*0.15
    cost = P.INCEPTION.cost("inception/mercury-2.5", 1_000_000, 400_000, 100_000)
    assert cost == pytest.approx(0.024 + 0.0016 + 0.015)


def test_mercury_3_is_priced_at_its_launch_rate_on_the_inception_path():
    assert P.INCEPTION.models["inception/mercury-3"] == "mercury-3"
    # 0.6M fresh * 0.10 + 0.4M cached * 0.01 + 0.1M out * 0.35. On OpenRouter the response's
    # own usage.cost is used (the OpenRouter table is empty); this is the direct-API path.
    cost = P.INCEPTION.cost("inception/mercury-3", 1_000_000, 400_000, 100_000)
    assert cost == pytest.approx(0.06 + 0.004 + 0.035)
    assert P.INCEPTION.cost("inception/mercury-3", 1000, 0, 1000) > 0


def test_an_unpriced_model_costs_nothing_rather_than_guessing():
    assert P.INCEPTION.cost("some/other-model", 1000, 0, 1000) == 0.0


# -- routing ------------------------------------------------------------------


class _Recorder:
    def __init__(self, provider):
        self.provider = provider
        self.calls: list[str] = []
        self.closed = False

    def chat(self, model, messages, **kwargs):
        self.calls.append(model)
        return self.provider.name

    def close(self):
        self.closed = True


def test_mercury_goes_to_the_chosen_provider_and_the_rest_to_openrouter():
    built: list[str] = []

    def build(provider):
        built.append(provider.name)
        return _Recorder(provider)

    client = P.RoutingClient("inception", build)
    assert client.chat("inception/mercury-2.5", []) == "inception"
    assert built == ["inception"]  # OpenRouter not built until a model needs it
    assert client.chat("z-ai/glm-5.3-flash", []) == "openrouter"
    assert client.chat("inception/mercury-2.5", []) == "inception"
    assert built == ["inception", "openrouter"]
    client.close()


def test_openrouter_is_the_default_and_builds_a_plain_client():
    class S:
        api_key = "sk-or-test"

    client = P.build_client(S(), stream=False)
    assert isinstance(client, OpenRouterClient) and client.provider is P.OPENROUTER
    client.close()


def test_a_missing_key_says_how_to_store_it_without_shell_history(monkeypatch):
    monkeypatch.setattr(P, "keychain_secret", lambda service, account="harness": "")
    monkeypatch.delenv("INCEPTION_API_KEY", raising=False)

    class S:
        mercury_provider = "inception"

    client = P.build_client(S(), stream=False)
    with pytest.raises(RuntimeError) as err:
        client.chat("inception/mercury-2.5", [])
    message = str(err.value)
    # `harness setup` reads the key hidden and stores it via `security -i`'s stdin, so the
    # advice keeps the key out of shell history, as the `-w`-last command it replaced did.
    assert "harness setup --provider inception" in message
    assert "-w '" not in message
    assert "INCEPTION_API_KEY" in message


def test_an_unknown_provider_is_refused():
    class S:
        mercury_provider = "nowhere"
        api_key = "k"

    with pytest.raises(ValueError, match="nowhere"):
        P.build_client(S())


# -- the real client against a real local server -------------------------------


class _Server:
    """Answers /chat/completions the way Inception does (no cost, a warning),
    streamed or not, and records every request body."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                outer.headers.append(dict(self.headers))
                usage = {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 400},
                    "completion_tokens_details": {"reasoning_tokens": 90},
                }
                warning = "Warning: Temperature has been set to 1."
                if body.get("stream"):
                    chunks = [
                        {"model": "mercury-2.5", "choices": [{"delta": {"content": "hel"}}]},
                        {
                            "model": "mercury-2.5",
                            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
                        },
                        {"model": "mercury-2.5", "choices": [], "usage": usage, "warning": warning},
                    ]
                    payload = (
                        "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    )
                    data = payload.encode()
                    ctype = "text/event-stream"
                else:
                    data = json.dumps(
                        {
                            "model": "mercury-2.5",
                            "choices": [
                                {
                                    "message": {"role": "assistant", "content": "hello"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": usage,
                            "warning": warning,
                        }
                    ).encode()
                    ctype = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join(timeout=5)


@pytest.mark.parametrize("stream", [False, True])
def test_a_real_inception_call_is_translated_priced_and_warned(stream):
    server = _Server()
    with server as base:
        client = OpenRouterClient("inc-key", base_url=base, provider=P.INCEPTION, stream=stream)
        reply = client.chat(
            "inception/mercury-2.5",
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            reasoning_effort="high",
        )
        client.close()
    sent = server.bodies[0]
    assert sent["model"] == "mercury-2.5"
    assert sent["reasoning_effort"] == "high" and "reasoning" not in sent
    assert "temperature" not in sent  # 0.2 would have been reset to 1.0 anyway
    assert "usage" not in sent
    assert ("stream_options" in sent) == stream
    assert "X-Title" not in server.headers[0]  # OpenRouter's attribution stays with OpenRouter
    assert server.headers[0]["Authorization"] == "Bearer inc-key"
    assert reply.content == "hello"
    assert reply.usage.cached_tokens == 400
    assert reply.usage.cost_usd == pytest.approx(
        P.INCEPTION.cost("inception/mercury-2.5", 1000, 400, 100)
    )
    assert reply.warning.startswith("Warning:")
    assert reply.provider == "inception"


def test_a_real_openrouter_call_keeps_its_dialect():
    server = _Server()
    with server as base:
        client = OpenRouterClient("or-key", base_url=base, stream=False, mercury_temperature=0.5)
        client.chat(
            "inception/mercury-2.5",
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            reasoning_effort="low",
        )
        client.close()
    sent = server.bodies[0]
    assert sent["model"] == "inception/mercury-2.5"
    assert sent["reasoning"] == {"effort": "low"} and sent["usage"] == {"include": True}
    assert sent["temperature"] == 0.5
    assert server.headers[0]["X-Title"] == "Mercury dLLM coding harness"


def test_a_client_built_from_default_settings_carries_both_runaway_limits():
    """§95.1: the 16,000-token cap and the 30 s stall bound are Settings defaults, so
    every run's client gets them without a flag; 0 on the command line turns them off."""
    from agent.cli import _unless_given
    from agent.config import Settings

    defaults = Settings(workspace=".")

    class S:
        api_key = "sk-or-test"
        mercury_max_tokens = defaults.mercury_max_tokens
        mercury_content_stall_s = defaults.mercury_content_stall_s

    client = P.build_client(S(), stream=False)
    assert client.mercury_max_tokens == 16_000
    assert client.content_stall_s == 30.0
    client.close()

    class Off(S):
        mercury_max_tokens = _unless_given(0, "mercury_max_tokens")
        mercury_content_stall_s = _unless_given(0.0, "mercury_content_stall_s")

    off = P.build_client(Off(), stream=False)
    assert off.mercury_max_tokens is None and off.content_stall_s is None
    off.close()
