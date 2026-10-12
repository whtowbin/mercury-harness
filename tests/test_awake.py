"""agent/awake.py: one caffeinate per process, shared by `harness run` and bench."""

from __future__ import annotations

import sys

import pytest

from agent import awake


def test_keep_awake_is_a_noop_off_macos(monkeypatch):
    monkeypatch.setattr(awake.sys, "platform", "linux")
    assert awake.keep_awake() is None


@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is macOS-only")
def test_keep_awake_starts_one_real_caffeinate_per_process():
    first = awake.keep_awake()
    assert first is not None and first.poll() is None
    assert awake.keep_awake() is first  # not a second process per call


def test_harness_run_holds_the_assertion(monkeypatch):
    from typer.testing import CliRunner

    from agent import cli

    calls = []
    monkeypatch.setattr(cli.awake, "keep_awake", lambda: calls.append(1))
    monkeypatch.setattr(
        cli, "run_agent", lambda settings, **_k: {"totals": {"by_model": {}}, "trace_path": "t"}
    )
    result = CliRunner().invoke(cli.app, ["run", "a goal"])
    assert result.exit_code == 0, result.output
    assert calls == [1]
