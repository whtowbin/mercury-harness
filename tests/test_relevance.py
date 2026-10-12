"""agent/relevance.py: one interface, three backends, picked automatically by
what's actually configured -- a Jev key and the optional package, an LLM
client, or (always available) a free keyword heuristic.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent.config import Settings
from agent.llm import Reply, Usage
from agent.relevance import (
    HeuristicJudge,
    LLMJudge,
    _parse_score_array,
    build_judge,
)


def settings_for(**overrides) -> Settings:
    return dataclasses.replace(Settings(workspace=".", goal="g"), **overrides)


# -- HeuristicJudge ----------------------------------------------------------


def test_a_relevant_candidate_scores_higher_than_an_unrelated_one():
    judge = HeuristicJudge()
    scores = judge.score(
        "fix the player jump collision bug",
        ["player.py handles jump and collision", "audio.py plays background music"],
    )
    assert scores[0] > scores[1]


def test_scores_are_always_in_zero_to_one():
    judge = HeuristicJudge()
    scores = judge.score("x y z", ["a b c", "x y z x y z x y z"])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_empty_context_does_not_crash_and_returns_neutral_scores():
    judge = HeuristicJudge()
    scores = judge.score("", ["anything", "something else"])
    assert scores == [0.5, 0.5]


def test_returns_exactly_one_score_per_candidate():
    judge = HeuristicJudge()
    assert len(judge.score("goal", ["a", "b", "c", "d"])) == 4


def test_no_candidates_returns_no_scores():
    assert HeuristicJudge().score("goal", []) == []


# -- LLMJudge: parsing --------------------------------------------------------


def test_parses_a_clean_json_array():
    assert _parse_score_array("[0.9, 0.1, 0.5]", 3) == [0.9, 0.1, 0.5]


def test_parses_an_array_inside_a_markdown_fence():
    assert _parse_score_array("```json\n[0.9, 0.1]\n```", 2) == [0.9, 0.1]


def test_clamps_out_of_range_values():
    assert _parse_score_array("[1.5, -0.5]", 2) == [1.0, 0.0]


def test_wrong_length_is_rejected_not_silently_truncated():
    assert _parse_score_array("[0.9, 0.1]", 3) is None


def test_garbage_is_rejected_not_raised():
    assert _parse_score_array("not json at all", 2) is None


# -- LLMJudge: end to end -----------------------------------------------------


class _FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def chat(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return Reply(content=self.content, usage=Usage(50, 10, 0.0001, 0.2))


def test_llm_judge_returns_the_parsed_scores():
    client = _FakeClient(json.dumps([0.8, 0.2]))
    judge = LLMJudge(client, "some-model")
    assert judge.score("ctx", ["a", "b"]) == [0.8, 0.2]


def test_llm_judge_sends_the_configured_model_and_low_reasoning():
    client = _FakeClient("[0.5]")
    judge = LLMJudge(client, "inception/mercury-2.5")
    judge.score("ctx", ["a"])
    assert client.calls[0]["model"] == "inception/mercury-2.5"
    assert client.calls[0]["reasoning_effort"] == "low"


def test_llm_judge_keeps_everything_on_a_client_exception():
    """A judge that failed to answer must not make the decision for its
    caller by silently discarding everything -- keep is the safer failure."""

    class Blows:
        def chat(self, **kwargs):
            raise RuntimeError("network is down")

    judge = LLMJudge(Blows(), "model")
    assert judge.score("ctx", ["a", "b", "c"]) == [1.0, 1.0, 1.0]


def test_llm_judge_keeps_everything_on_a_malformed_reply():
    judge = LLMJudge(_FakeClient("garbage, not an array"), "model")
    assert judge.score("ctx", ["a", "b"]) == [1.0, 1.0]


def test_llm_judge_chunks_large_candidate_lists():
    """MAX_BATCH caps how many questions go in one call -- a long tool loop
    or a large memory file must not build one request with hundreds of
    candidates just because nothing capped it."""
    from agent.relevance import MAX_BATCH

    client = _FakeClient(json.dumps([0.5] * MAX_BATCH))
    judge = LLMJudge(client, "model")
    candidates = [f"item {i}" for i in range(MAX_BATCH + 5)]
    scores = judge.score("ctx", candidates)
    assert len(scores) == len(candidates)
    assert len(client.calls) == 2  # one full batch, one partial


# -- build_judge: backend selection ------------------------------------------
# These explicitly patch jev_api_key to "" rather than relying on none being
# configured in the ambient environment -- a real Jev key in this machine's
# own Keychain (added once relevance.py went live) must not change what
# these tests assert, or they'd pass or fail depending on whose machine runs
# them rather than on the code.


def test_no_jev_key_and_no_client_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: ""))
    judge = build_judge(settings_for(), client=None)
    assert isinstance(judge, HeuristicJudge)


def test_a_client_with_no_jev_key_gets_the_llm_judge(monkeypatch):
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: ""))
    judge = build_judge(settings_for(), client=_FakeClient("[0.5]"))
    assert isinstance(judge, LLMJudge)
    assert judge.model == settings_for().executor_model


def test_relevance_model_override_is_used_when_set(monkeypatch):
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: ""))
    judge = build_judge(
        settings_for(relevance_model="z-ai/glm-5.3-flash"), client=_FakeClient("[0.5]")
    )
    assert judge.model == "z-ai/glm-5.3-flash"


def test_a_jev_key_with_the_package_unavailable_falls_through(monkeypatch):
    """The package is genuinely optional (see pyproject.toml's `jev` extra):
    a key with no package importable must degrade to the next backend, not
    raise ImportError up to the caller. Forces that path directly (rather
    than relying on typesafe_sdk being absent, which it is not in this
    environment) by making JevJudge's own construction raise ImportError,
    the same failure an uninstalled package would produce."""
    import agent.relevance as relevance_module

    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: "fake-jev-key"))

    def _raise_import_error(*args, **kwargs):
        raise ImportError("typesafe_sdk not installed")

    monkeypatch.setattr(relevance_module, "JevJudge", _raise_import_error)
    judge = build_judge(settings_for(), client=_FakeClient("[0.5]"))
    assert isinstance(judge, LLMJudge)


def test_a_real_jev_key_is_preferred_when_the_package_is_available(monkeypatch):
    """The mirror of the test above: a key plus an importable package must
    select JevJudge, not fall through just because a client was also given."""
    pytest.importorskip("typesafe_sdk", reason="the jev extra is not installed")
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: "fake-jev-key"))
    judge = build_judge(settings_for(), client=_FakeClient("[0.5]"))
    assert type(judge).__name__ == "JevJudge"


# -- relevance_backend: forcing one backend for an A/B comparison ------------


def test_relevance_backend_heuristic_wins_even_with_a_key_and_client(monkeypatch):
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: "fake-jev-key"))
    judge = build_judge(settings_for(relevance_backend="heuristic"), client=_FakeClient("[0.5]"))
    assert isinstance(judge, HeuristicJudge)


def test_relevance_backend_llm_wins_over_a_configured_jev_key(monkeypatch):
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: "fake-jev-key"))
    judge = build_judge(settings_for(relevance_backend="llm"), client=_FakeClient("[0.5]"))
    assert isinstance(judge, LLMJudge)


def test_relevance_backend_jev_is_the_same_as_auto_when_a_key_is_present(monkeypatch):
    pytest.importorskip("typesafe_sdk", reason="the jev extra is not installed")
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: "fake-jev-key"))
    judge = build_judge(settings_for(relevance_backend="jev"), client=_FakeClient("[0.5]"))
    assert type(judge).__name__ == "JevJudge"


def test_relevance_backend_llm_without_a_client_falls_back_not_raises(monkeypatch):
    """A forced backend this run cannot actually provide must degrade, not
    crash a bench job or a real run."""
    monkeypatch.setattr(Settings, "jev_api_key", property(lambda self: ""))
    judge = build_judge(settings_for(relevance_backend="llm"), client=None)
    assert isinstance(judge, HeuristicJudge)


def test_a_judge_returning_too_few_scores_falls_back_instead_of_crashing():
    """`relevant()` indexes the score dict directly, so a judge that returned
    fewer scores than sections used to leave the tail with no entry and raise
    KeyError mid-run. The Protocol promises one score per candidate; the
    judges behind it are an LLM and a third-party service, so the promise is
    not a guarantee and a short reply degrades like any other judge failure."""
    from agent.memory import Section, _score_sections

    class ShortJudge:
        def score(self, context, candidates):
            return [1.0] * (len(candidates) - 1)

    sections = [Section(title=f"t{i}", body="b", index=i) for i in range(3)]
    scores = _score_sections(sections, "goal", {"goal"}, set(), ShortJudge())
    assert set(scores) == {0, 1, 2}


def test_a_judge_returning_one_score_per_section_is_used_as_is():
    from agent.memory import Section, _score_sections

    class GoodJudge:
        def score(self, context, candidates):
            return [0.25 * (i + 1) for i in range(len(candidates))]

    sections = [Section(title=f"t{i}", body="b", index=i) for i in range(3)]
    assert _score_sections(sections, "g", set(), set(), GoodJudge()) == {0: 0.25, 1: 0.5, 2: 0.75}


def test_llm_judge_calls_reach_the_trace_as_llm_calls(tmp_path):
    """These calls (memory recall twice a run, compaction) were model time no trace
    figure counted (DESIGN-LOG §95). The judge hands each answered call to on_call;
    loop.py points that at the run's real Trace, which this uses too."""
    from agent.trace import Trace

    trace = Trace(str(tmp_path), echo=False)
    judge = build_judge(
        settings_for(relevance_backend="llm"),
        _FakeClient("[0.9, 0.1]"),
        on_call=lambda model, reply: trace.llm_call(
            "relevance", model, reply.usage, reply.finish_reason, reply=reply
        ),
    )
    assert isinstance(judge, LLMJudge)
    assert judge.score("parser bug", ["the parser", "the weather"]) == [0.9, 0.1]
    lines = next(tmp_path.glob("run-*.jsonl")).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    calls = [e for e in events if e["kind"] == "llm_call"]
    assert len(calls) == 1 and calls[0]["phase"] == "relevance"
    assert calls[0]["prompt_tokens"] == 50 and calls[0]["completion_tokens"] == 10


def test_a_failing_on_call_does_not_change_the_scores():
    def boom(model, reply):
        raise RuntimeError("trace is gone")

    judge = LLMJudge(_FakeClient("[0.7]"), "m", on_call=boom)
    assert judge.score("x", ["y"]) == [0.7]
