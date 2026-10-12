"""scripts/agents/claude.py: Claude Code as a restricted benchmark agent (§127). No model call:
the argument list, the settings it writes, the allowance guard and the stream parser."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_saved = list(sys.path)
sys.path += [str(ROOT / "scripts")]
try:
    from agents import claude  # ty: ignore[unresolved-import]
finally:
    sys.path[:] = _saved


def test_the_run_gets_six_tools_no_connectors_and_a_settings_file_of_its_own(tmp_path):
    ws, out = tmp_path / "ws", tmp_path / "out"
    out.mkdir()
    argv = claude.argv("fix it", ws, out, "haiku", "high")
    assert argv[argv.index("--tools") + 1] == "Bash,Edit,Glob,Grep,Read,Write"
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--setting-sources") + 1] == "project"  # not the user's hooks/plugins
    settings = json.loads((out / "settings.json").read_text())
    assert settings["permissions"]["defaultMode"] == "dontAsk"  # bypassPermissions ignored the list
    assert {"WebFetch", "WebSearch", "Read(~/**)"} <= set(settings["permissions"]["deny"])
    assert settings["sandbox"]["enabled"] and settings["sandbox"]["filesystem"]["denyRead"] == [
        "~/"
    ]


def test_path_has_the_workspace_venv_first_and_no_other_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/Users/x/repo/.venv/bin:/usr/bin")
    env = claude.env(tmp_path / "ws", tmp_path / "out")
    assert env["PATH"].startswith(str(tmp_path / "ws" / ".venv" / "bin"))
    assert "/Users/x/repo" not in env["PATH"]


def test_the_allowance_guard_stops_at_the_cap_and_when_the_five_hour_window_is_busy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(claude, "USAGE", tmp_path / "usage.json")
    assert "baseline" in claude.allowance_stop()  # no baseline: no run
    claude.USAGE.write_text(json.dumps({"baseline_week": 0.1, "week": 0.1, "five_hour": 0.0}))
    assert claude.allowance_stop() is None
    claude.record_usage({"week_end": 0.1, "five_hour_end": 0.6})
    assert "five-hour" in claude.allowance_stop()
    claude.record_usage({"week_end": 0.2, "five_hour_end": 0.1})
    assert "weekly allowance cap" in claude.allowance_stop()  # 0.1 spent of 0.1


def test_stats_read_the_result_and_the_first_and_last_allowance_readings(tmp_path):
    ev = lambda w, f: json.dumps(  # noqa: E731
        {"type": "rate_limit_event", "rate_limit_info": {"unifiedWindows": {
            "five_hour": {"utilization": f}, "seven_day": {"utilization": w}}}}
    )  # fmt: skip
    usage = {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5}
    result = {"type": "result", "subtype": "success", "num_turns": 7, "total_cost_usd": 0.12}
    result["usage"] = usage
    lines = [ev(0.1, 0.01), ev(0.2, 0.03), json.dumps(result)]
    (tmp_path / "stdout.log").write_text("\n".join(lines))
    s = claude.stats(tmp_path)
    assert (s.calls, s.cost, s.in_tokens, s.out_tokens, s.status) == (7, 0.12, 100, 5, "success")
    assert (s.extra["week_start"], s.extra["week_end"]) == (0.1, 0.2)


def test_a_nothink_model_name_switches_thinking_off_and_is_stripped_from_the_command(
    tmp_path, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    argv = claude.argv("fix it", tmp_path / "ws", out, "claude-sonnet-5:nothink", "high")
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"
    monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-5:nothink")
    assert claude.env(tmp_path / "ws", out)["MAX_THINKING_TOKENS"] == "0"
    monkeypatch.setenv("AGENT_MODEL", "claude-sonnet-5")
    assert "MAX_THINKING_TOKENS" not in claude.env(tmp_path / "ws", out)
