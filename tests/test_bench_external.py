"""bench/external.py: running Aider on the same tasks, scored the same way.

Only the no-network paths are tested here — a real Aider invocation needs a
model call and is exercised by hand, not in the suite. What matters for these
tests is that a missing key or a missing binary produces a record shaped
exactly like a bench.runner one, so bench.report can render both in one
table without special-casing which harness produced a row.
"""

from __future__ import annotations

from bench.external import run_aider
from bench.tasks import get


def test_a_missing_api_key_produces_a_normal_shaped_record(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("bench.external._api_key", lambda: "")
    record = run_aider("z-ai/glm-5.3-flash", get("vacuous-guard"))
    assert record["config"] == "aider-glm-5.3-flash"
    assert record["task"] == "vacuous-guard"
    assert record["agent_error"]
    assert record["accuracy"] == 0.0
    assert record["timed_out"] is False


def test_a_missing_aider_binary_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr("bench.external._api_key", lambda: "sk-or-fake")
    monkeypatch.setattr("bench.external.shutil.which", lambda name: None)
    record = run_aider("qwen/qwen3.7-flash", get("tautology"))
    assert "aider" in record["agent_error"].lower()
    assert record["config"] == "aider-qwen3.7-flash"


def test_the_record_has_every_field_bench_report_reads(monkeypatch):
    """The whole point of reusing runner.flatten: a report row must not care
    which harness produced it."""
    monkeypatch.setattr("bench.external._api_key", lambda: "")
    record = run_aider("z-ai/glm-5.3-flash", get("no-bug"))
    for field in (
        "accuracy",
        "cost_usd",
        "wall_s",
        "delivered",
        "honest",
        "own_tests_ran",
        "own_tests_miss_the_bug",
        "calls",
        "model_latency_s",
        "timed_out",
        "agent_error",
    ):
        assert field in record, f"missing {field}"


def test_config_name_is_derived_from_the_model_id(monkeypatch):
    monkeypatch.setattr("bench.external._api_key", lambda: "")
    record = run_aider("deepseek/deepseek-v4-flash", get("regression"))
    assert record["config"] == "aider-deepseek-v4-flash"
