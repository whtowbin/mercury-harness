"""SSE parsing, tool-call accumulation, and prompt caching."""

from __future__ import annotations

import json

import pytest

from agent.llm import (
    MIN_CACHE_TOKENS,
    OpenRouterClient,
    Usage,
    _merge_tool_call,
    _usage_from,
    apply_cache_control,
    supports_caching,
)


class FakeResponse:
    """Stands in for a streamed requests.Response."""

    def __init__(self, lines: list[str] | list[bytes], status_code: int = 200):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = ""
        self._lines: list = list(lines)
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        # Mirrors what `requests` actually does, not what the old call site
        # assumed: decode_unicode=True decodes using resp.encoding, and
        # `requests` defaults that to ISO-8859-1 for any text/* response
        # without an explicit charset -- exactly OpenRouter's SSE stream.
        # Passing bytes straight through regardless of decode_unicode, as an
        # earlier version of this fixture did, could not have caught the bug
        # this simulates: the fallback branch that manually decodes bytes as
        # UTF-8 only runs when it still sees bytes, and decode_unicode=True
        # never leaves it as bytes.
        for line in self._lines:
            if decode_unicode and isinstance(line, bytes):
                yield line.decode("iso-8859-1")
            else:
                yield line

    def close(self):
        self.closed = True


def sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def delta(**fields) -> str:
    return sse({"choices": [{"delta": fields}]})


@pytest.fixture
def patch_post(monkeypatch):
    """Swap requests' post for a canned SSE response."""

    def install(client: OpenRouterClient, response: FakeResponse) -> OpenRouterClient:
        monkeypatch.setattr(client.session, "post", lambda *a, **k: response)
        return client

    return install


def client_returning(patch_post, lines: list[str]) -> OpenRouterClient:
    return patch_post(OpenRouterClient("test-key", stream=True), FakeResponse(lines))


# -- tool call accumulation ----------------------------------------------


def test_merge_assembles_fragmented_arguments():
    partial: dict[int, dict] = {}
    _merge_tool_call(
        partial,
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"pa'},
        },
    )
    _merge_tool_call(partial, {"index": 0, "function": {"arguments": 'th": "a.py"'}})
    _merge_tool_call(partial, {"index": 0, "function": {"arguments": "}"}})
    assert partial[0]["id"] == "call_1"
    assert partial[0]["function"]["name"] == "read_file"
    assert json.loads(partial[0]["function"]["arguments"]) == {"path": "a.py"}


def test_merge_keeps_parallel_calls_separate():
    partial: dict[int, dict] = {}
    _merge_tool_call(
        partial, {"index": 0, "id": "a", "function": {"name": "read_file", "arguments": "{}"}}
    )
    _merge_tool_call(
        partial,
        {"index": 1, "id": "b", "function": {"name": "grep", "arguments": '{"pattern":"x"}'}},
    )
    assert len(partial) == 2
    assert partial[1]["function"]["name"] == "grep"


def test_stream_yields_content_and_tool_calls(patch_post):
    lines = [
        delta(content="thinking"),
        delta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": '{"pa'},
                }
            ]
        ),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 'th": "."}'}}]),
        sse(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
            }
        ),
        "data: [DONE]",
    ]
    reply = client_returning(patch_post, lines).chat("m", [{"role": "user", "content": "hi"}])
    assert reply.content == "thinking"
    assert reply.finish_reason == "tool_calls"
    assert len(reply.tool_calls) == 1
    assert json.loads(reply.tool_calls[0]["function"]["arguments"]) == {"path": "."}
    assert reply.usage.completion_tokens == 20
    assert reply.usage.cost_usd == 0.001


def test_stream_records_time_to_first_token(patch_post):
    lines = [delta(content="a"), delta(content="b"), "data: [DONE]"]
    reply = client_returning(patch_post, lines).chat("m", [{"role": "user", "content": "hi"}])
    assert reply.usage.ttft_s > 0
    assert reply.usage.latency_s >= reply.usage.ttft_s


def test_stream_ignores_keepalive_comments_and_junk(patch_post):
    lines = [
        ": OPENROUTER PROCESSING",
        "",
        "garbage without prefix",
        "data: {not valid json",
        delta(content="ok"),
        "data: [DONE]",
    ]
    reply = client_returning(patch_post, lines).chat("m", [{"role": "user", "content": "hi"}])
    assert reply.content == "ok"


def test_stream_surfaces_a_provider_error(patch_post):
    from agent.llm import LLMError

    lines = [sse({"error": {"message": "model is overloaded"}}), "data: [DONE]"]
    with pytest.raises(LLMError, match="overloaded"):
        client_returning(patch_post, lines).chat("m", [{"role": "user", "content": "hi"}])


def test_stream_closes_the_response(patch_post):
    response = FakeResponse([delta(content="x"), "data: [DONE]"])
    client = patch_post(OpenRouterClient("k", stream=True), response)
    client.chat("m", [{"role": "user", "content": "hi"}])
    assert response.closed


# -- on_call_start / on_call_end, the spinner's hooks -----------------------


def test_on_call_start_and_end_bracket_a_tool_only_reply(patch_post):
    """No content delta ever arrives on a tool-only reply, so on_call_end
    must still fire — from chat()'s own finally, not from inside the
    stream-parsing loop."""
    events: list[str] = []
    lines = [
        delta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": "{}"},
                }
            ]
        ),
        "data: [DONE]",
    ]
    client = patch_post(
        OpenRouterClient(
            "k",
            stream=True,
            on_call_start=lambda: events.append("start"),
            on_call_end=lambda: events.append("end"),
        ),
        FakeResponse(lines),
    )
    client.chat("m", [{"role": "user", "content": "hi"}])
    assert events == ["start", "end"]


def test_on_call_end_fires_early_on_the_first_content_delta(patch_post):
    events: list[str] = []
    lines = [delta(content="a"), delta(content="b"), "data: [DONE]"]
    client = patch_post(
        OpenRouterClient(
            "k",
            stream=True,
            on_call_start=lambda: events.append("start"),
            on_call_end=lambda: events.append("end"),
        ),
        FakeResponse(lines),
    )
    client.chat("m", [{"role": "user", "content": "hi"}])
    # "end" must not be duplicated by chat()'s own finally once it already
    # fired early from the first delta.
    assert events == ["start", "end"]


def test_on_call_end_fires_exactly_once_even_with_many_content_deltas(patch_post):
    events: list[str] = []
    lines = [delta(content=c) for c in "abcde"] + ["data: [DONE]"]
    client = patch_post(
        OpenRouterClient("k", stream=True, on_call_end=lambda: events.append("end")),
        FakeResponse(lines),
    )
    client.chat("m", [{"role": "user", "content": "hi"}])
    assert events == ["end"]


def test_on_call_end_fires_on_a_provider_error_too(patch_post):
    from agent.llm import LLMError

    events: list[str] = []
    lines = [sse({"error": {"message": "model is overloaded"}}), "data: [DONE]"]
    client = patch_post(
        OpenRouterClient("k", stream=True, max_retries=0, on_call_end=lambda: events.append("end")),
        FakeResponse(lines),
    )
    with pytest.raises(LLMError):
        client.chat("m", [{"role": "user", "content": "hi"}])
    assert events == ["end"]


def _boom() -> None:
    raise ValueError("boom")


def test_a_raising_hook_does_not_break_the_call(patch_post):
    """The hooks exist purely to drive a UI indicator; a bug in one must
    never take down the actual model call."""
    lines = [delta(content="ok"), "data: [DONE]"]
    client = patch_post(
        OpenRouterClient("k", stream=True, on_call_start=_boom, on_call_end=_boom),
        FakeResponse(lines),
    )
    reply = client.chat("m", [{"role": "user", "content": "hi"}])
    assert reply.content == "ok"


def test_stream_stuck_on_keepalives_past_the_timeout_is_retried_then_fails(patch_post, monkeypatch):
    """A provider that only ever sends keep-alive comments (never content, never
    [DONE]) must not hold the loop open forever: each keep-alive line resets the
    underlying socket read-timeout, so only a wall-clock deadline independent of
    that per-chunk timeout can catch it."""
    from agent.llm import LLMError

    def forever_keepalives():
        while True:
            yield ": OPENROUTER PROCESSING"

    class InfiniteResponse(FakeResponse):
        def __init__(self):
            super().__init__([])

        def iter_lines(self, decode_unicode: bool = False):
            yield from forever_keepalives()

    client = patch_post(OpenRouterClient("k", stream=True, max_retries=1), InfiniteResponse())
    client.timeout = 0.01
    monkeypatch.setattr(client, "_sleep", lambda *a, **k: None)  # don't actually backoff
    with pytest.raises(LLMError, match="deadline"):
        client.chat("m", [{"role": "user", "content": "hi"}])


def test_on_delta_receives_content_live(patch_post):
    seen: list[str] = []
    client = patch_post(
        OpenRouterClient("k", stream=True, on_delta=seen.append),
        FakeResponse([delta(content="he"), delta(content="llo"), "data: [DONE]"]),
    )
    client.chat("m", [{"role": "user", "content": "hi"}])
    assert "".join(seen) == "hello"


# -- usage metrics --------------------------------------------------------


def test_decode_rate_excludes_prefill():
    usage = Usage(completion_tokens=100, latency_s=2.0, ttft_s=1.0)
    assert usage.decode_s == 1.0
    assert usage.decode_tok_s == 100.0
    assert usage.end_to_end_tok_s == 50.0


def test_decode_rate_is_zero_without_streaming():
    assert Usage(completion_tokens=100, latency_s=2.0, ttft_s=0.0).decode_tok_s == 0.0


def test_usage_reads_cached_prompt_tokens():
    usage = _usage_from(
        {
            "prompt_tokens": 5000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4096},
        },
        latency=1.0,
        ttft=0.1,
    )
    assert usage.cached_tokens == 4096


# -- prompt caching -------------------------------------------------------


def test_caching_applies_only_to_supported_providers():
    assert supports_caching("anthropic/claude-sonnet-5")
    assert not supports_caching("inception/mercury-2.5")


def test_cache_control_marks_the_stable_prefix():
    big = "x" * (MIN_CACHE_TOKENS * 4 + 100)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": big}]
    marked = apply_cache_control(messages, "anthropic/claude-sonnet-5")
    block = marked[1]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"] == big
    # The original list is not mutated.
    assert isinstance(messages[1]["content"], str)


def test_cache_control_left_alone_for_other_providers():
    big = "x" * (MIN_CACHE_TOKENS * 4 + 100)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": big}]
    assert apply_cache_control(messages, "inception/mercury-2.5") == messages


def test_cache_control_skips_a_prefix_too_small_to_cache():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert apply_cache_control(messages, "anthropic/claude-sonnet-5") == messages


def test_non_streaming_path_still_works(patch_post):
    class Plain(FakeResponse):
        # The non-streaming path reads the body through iter_content under a
        # wall-clock deadline rather than resp.json(), because `timeout` only
        # bounds the gap between reads: a body dripped one byte at a time was
        # never idle and so was read for as long as the provider kept sending
        # (measured at 64s against a 3s budget before this changed).
        def iter_content(self, chunk_size=None):
            yield json.dumps(
                {
                    "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
                }
            ).encode()

    client = patch_post(OpenRouterClient("k", stream=False), Plain([]))
    reply = client.chat("m", [{"role": "user", "content": "hi"}])
    assert reply.content == "done"
    assert reply.usage.ttft_s == 0.0


def test_stream_handles_undecoded_bytes(patch_post):
    """iter_lines() (decode_unicode left at its default, False) always
    yields bytes, and the manual fallback decodes them as UTF-8."""
    raw = [line.encode("utf-8") for line in [delta(content="bytes work"), "data: [DONE]"]]
    client = patch_post(OpenRouterClient("k", stream=True), FakeResponse(raw))
    assert client.chat("m", [{"role": "user", "content": "hi"}]).content == "bytes work"


def test_stream_does_not_mojibake_multibyte_utf8(patch_post):
    """A bench sweep found 23 of 24 runs on a task whose fixture contained an
    ellipsis came back with it corrupted -- 'â\x80¦' or worse, double-corrupted
    to 'Ã\x83Â¢Ã\x82Â\x80Ã\x82Â¦' once a later turn re-read the already-broken
    text. The cause: the old call site asked iter_lines to decode_unicode,
    which decodes using resp.encoding -- ISO-8859-1 by default for a text/*
    response with no explicit charset, exactly what OpenRouter's SSE stream
    is -- so every multi-byte UTF-8 character the model wrote came back
    mojibake'd before the manual UTF-8 fallback ever got a chance to run (it
    only fires on bytes, and decode_unicode=True had already produced a str).
    Every model tested (Mercury, GLM, Qwen) hit this equally, because it was
    never a model quirk -- it was every response, corrupted the same way,
    regardless of which model produced the content."""
    # json.dumps defaults to ensure_ascii=True, which \\u-escapes non-ASCII
    # characters into plain ASCII -- exactly what a real API response does
    # NOT do, and exactly why delta()/sse() above cannot reproduce this bug.
    # ensure_ascii=False is required to put genuine multi-byte UTF-8 bytes on
    # the wire, the way OpenRouter's own response actually arrives.
    payload = json.dumps({"choices": [{"delta": {"content": "alpha beta…"}}]}, ensure_ascii=False)
    raw = [("data: " + payload).encode("utf-8"), b"data: [DONE]"]
    client = patch_post(OpenRouterClient("k", stream=True), FakeResponse(raw))
    assert client.chat("m", [{"role": "user", "content": "hi"}]).content == "alpha beta…"


def test_backoff_outlasts_a_per_minute_rate_limit(monkeypatch):
    """A 20s cap gives up inside the window it is waiting for."""
    slept: list[float] = []
    monkeypatch.setattr("agent.llm.time.sleep", slept.append)
    client = OpenRouterClient("k")
    for attempt in range(client.max_retries):
        client._sleep(attempt, None)
    assert sum(slept) > 60.0, f"total backoff {sum(slept):.0f}s is under a minute"


def test_retry_after_header_is_honoured(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("agent.llm.time.sleep", slept.append)
    OpenRouterClient("k")._sleep(0, "45")
    assert slept == [45.0]


# -- transient provider failures -----------------------------------------


def test_a_502_mid_stream_is_retried_not_fatal(patch_post, monkeypatch):
    """Providers drop connections; that should cost a retry, not the run."""
    monkeypatch.setattr("agent.llm.time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def responder(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(
                [
                    sse({"error": {"code": 502, "message": "Network connection lost."}}),
                    "data: [DONE]",
                ]
            )
        return FakeResponse([delta(content="recovered"), "data: [DONE]"])

    client = OpenRouterClient("k", stream=True)
    monkeypatch.setattr(client.session, "post", responder)
    reply = client.chat("m", [{"role": "user", "content": "hi"}])
    assert reply.content == "recovered"
    assert attempts["n"] == 2


def test_a_terminal_provider_error_is_not_retried(patch_post, monkeypatch):
    from agent.llm import LLMError

    monkeypatch.setattr("agent.llm.time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def responder(*a, **k):
        attempts["n"] += 1
        return FakeResponse(
            [sse({"error": {"code": 400, "message": "bad request"}}), "data: [DONE]"]
        )

    client = OpenRouterClient("k", stream=True)
    monkeypatch.setattr(client.session, "post", responder)
    with pytest.raises(LLMError, match="bad request"):
        client.chat("m", [{"role": "user", "content": "hi"}])
    assert attempts["n"] == 1, "a 400 will not succeed on a retry"


def test_an_error_envelope_without_a_code_is_terminal():
    from agent.llm import LLMError, _envelope_error

    assert isinstance(_envelope_error({"message": "who knows"}), LLMError)


# -- which model actually answered -----------------------------------------


def test_a_stream_records_the_model_and_provider_that_served_it(patch_post):
    """A routing layer can serve another snapshot or upstream provider under an
    unchanged requested id; the trace used to record only the requested one."""
    lines = [
        sse(
            {
                "model": "inception/mercury-2.5-20260901",
                "provider": "Inception",
                "choices": [{"delta": {"content": "hi"}}],
            }
        ),
        sse(
            {
                "model": "inception/mercury-2.5-20260901",
                "provider": "Inception",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        ),
        "data: [DONE]",
    ]
    reply = client_returning(patch_post, lines).chat(
        "inception/mercury-2.5", [{"role": "user", "content": "hi"}]
    )
    assert reply.served_model == "inception/mercury-2.5-20260901"
    assert reply.provider == "Inception"


def test_a_provider_object_with_a_name_is_read_too():
    from agent.llm import _provider_name, _served_model

    assert _provider_name({"provider": {"name": "DeepInfra"}}) == "DeepInfra"
    assert _provider_name({"provider": "Together"}) == "Together"
    assert _provider_name({}) == ""
    assert _served_model({"model": "a/b"}) == "a/b"
    assert _served_model({}) == ""


def test_a_provider_that_says_nothing_leaves_the_fields_empty(patch_post):
    lines = [
        delta(content="x"),
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    reply = client_returning(patch_post, lines).chat("m", [{"role": "user", "content": "hi"}])
    assert reply.served_model == "" and reply.provider == ""


def test_the_trace_event_carries_the_served_model(tmp_path):
    import json as _json

    from agent.llm import Reply, Usage
    from agent.trace import Trace

    trace = Trace(str(tmp_path / "t"), echo=False)
    trace.llm_call(
        "plan",
        "inception/mercury-2.5",
        Usage(),
        reply=Reply(content="", served_model="inception/mercury-2.5-x", provider="Inception"),
    )
    event = next(
        _json.loads(line) for line in trace.path.read_text().splitlines() if '"llm_call"' in line
    )
    assert event["served_model"] == "inception/mercury-2.5-x"
    assert event["provider"] == "Inception"
