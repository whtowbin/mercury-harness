"""scripts/agents: every stats parser against a real file in the agent's own format (§109)."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "swebench"))

import agents  # noqa: E402  # ty: ignore[unresolved-import]  (scripts/ is on sys.path above)
from agents import aider, codex, goose, mini  # noqa: E402  # ty: ignore[unresolved-import]


def test_price_uses_openrouter_list_prices():
    assert agents.price("openrouter/inception/mercury-2.5", 1_000_000, 1_000_000) == pytest.approx(
        0.19
    )
    assert agents.price("z-ai/glm-5.3-flash", 2_000_000, 0) == pytest.approx(0.30)
    # Cached input is part of input and costs a tenth for Mercury (§122: Codex cached 94%).
    assert agents.price("inception/mercury-2.5", 1_000_000, 0, 900_000) == pytest.approx(0.0076)
    # No known cached price: cached input is priced as input, never below it.
    assert agents.price("z-ai/glm-5.3-flash", 2_000_000, 0, 1_000_000) == pytest.approx(0.30)
    assert agents.price("inception/mercury-2.5", 100, 0, 500) == pytest.approx(100 * 0.004 / 1e6)
    with pytest.raises(KeyError):
        agents.price("some/unknown-model", 1, 1)


def test_load_names_only_known_agents():
    assert agents.load("mini") is mini
    with pytest.raises(KeyError):
        agents.load("openhands")


def test_mini_stats_sum_openrouter_cost_per_call(tmp_path):
    usage = lambda c, p, o, r: {  # noqa: E731
        "extra": {
            "response": {
                "usage": {
                    "cost": c,
                    "prompt_tokens": p,
                    "completion_tokens": o,
                    "completion_tokens_details": {"reasoning_tokens": r},
                }
            }
        }
    }
    traj = {
        "info": {"exit_status": "Submitted", "model_stats": {"api_calls": 9, "instance_cost": 9}},
        "messages": [{"role": "user"}, usage(0.001, 100, 10, 5), usage(0.002, 200, 20, 0)],
    }
    (tmp_path / "traj.json").write_text(json.dumps(traj))
    s = mini.stats(tmp_path)
    assert (s.calls, s.in_tokens, s.out_tokens, s.reasoning_tokens) == (2, 300, 30, 5)
    assert s.cost == pytest.approx(0.003)  # OpenRouter's figure, not litellm's model_stats
    assert s.status == "Submitted" and s.cost_source == "reported"
    assert mini.stats(tmp_path / "missing").status == "no trajectory"


def test_mini_argv_refuses_the_web_and_caps_cost(tmp_path):
    argv = mini.argv("fix it", tmp_path, tmp_path, "openrouter/x/y", "high")
    assert "-l" in argv and argv[argv.index("-l") + 1] == mini.COST_LIMIT
    assert any(a == f"environment.env.https_proxy={agents.DEAD_PROXY}" for a in argv)
    assert argv[argv.index("-o") + 1] == str(tmp_path / "traj.json")


def test_aider_stats_read_the_session_total_and_edit_status(tmp_path):
    (tmp_path / "stdout.log").write_text(
        "Tokens: 12k sent, 1.5k received. Cost: $0.0012 message, $0.0012 session.\n"
        "Applied edit to foo.py\n"
        "Tokens: 713 sent, 500 received. Cost: $0.0004 message, $0.0016 session.\n"
    )
    s = aider.stats(tmp_path)
    assert (s.calls, s.in_tokens, s.out_tokens) == (2, 12713, 2000)
    assert s.cost == pytest.approx(0.0016) and s.status == "edited"
    # A traceback Aider quotes from the model's own test file is not a crash; one in Aider's is.
    with (tmp_path / "stdout.log").open("a") as f:
        f.write('Traceback (most recent call last):\n  File "/ws/test_x.py", line 10\n')
    assert aider.stats(tmp_path).status == "edited"
    with (tmp_path / "stdout.log").open("a") as f:
        f.write(
            'Traceback (most recent call last):\n  File "/v/site-packages/aider/main.py", line 1\n'
        )
    assert aider.stats(tmp_path).status == "crashed"


def test_codex_stats_sum_turn_usage_and_price_it(tmp_path):
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
        {"type": "item.completed", "item": {"type": "file_change", "changes": []}},
        {"type": "turn.completed", "usage": {"input_tokens": 1000000, "output_tokens": 0,
                                             "cached_input_tokens": 500000}},
        {"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 1000000,
                                             "reasoning_output_tokens": 7}},
    ]  # fmt: skip
    (tmp_path / "stdout.log").write_text("\n".join(json.dumps(e) for e in events) + "\nnoise\n")
    s = codex.stats(tmp_path)
    assert s.calls is None and s.extra["steps"] == 2
    assert (s.in_tokens, s.out_tokens, s.reasoning_tokens) == (1000000, 1000000, 7)
    # 500k uncached at $0.04, 500k cached at $0.004, 1M out at $0.15.
    assert s.cost == pytest.approx(0.172) and s.status == "done"
    assert s.extra["cached_tokens"] == 500000 and s.cost_source == "tokens x list price"


def test_codex_argv_uses_the_responses_wire_api(tmp_path):
    argv = codex.argv("do it", tmp_path, tmp_path, "openrouter/inception/mercury-2.5", "high")
    assert argv[argv.index("-m") + 1] == "inception/mercury-2.5"
    assert any('wire_api = "responses"' in a for a in argv) and "workspace-write" in argv
    assert 'web_search="disabled"' in argv  # §109.2: it answered from a web search on 9296
    assert argv[argv.index("-C") + 1] == str(tmp_path.resolve())  # the sandbox refuses a symlink


def test_goose_stats_read_the_session_store(tmp_path, monkeypatch):
    db = tmp_path / "sessions.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        create table sessions(id text, name text, created_at text,
            accumulated_input_tokens int, accumulated_output_tokens int,
            accumulated_cache_read_tokens int);
        create table messages(session_id text, role text, content_json text);
        """
    )
    out = tmp_path / "s109-x" / "9296" / "out" / "9296"
    out.mkdir(parents=True)
    name = goose.session_name(out)
    con.execute(
        "insert into sessions values ('sid', ?, '2026', 2000000, 1000000, 1000000)", (name,)
    )
    con.executemany(
        "insert into messages values ('sid', ?, ?)",
        [("user", "fix"), ("assistant", "ok"), ("assistant", "see github.com/django/django")],
    )
    con.commit()
    con.close()
    monkeypatch.setattr(goose, "DB", db)
    (out / "stdout.log").write_text("done\n")
    s = goose.stats(out)
    assert (s.calls, s.in_tokens, s.out_tokens) == (2, 2000000, 1000000)
    # 1M uncached at $0.04, 1M cached at $0.004, 1M out at $0.15.
    assert s.cost == pytest.approx(0.194) and s.status == "done"
    assert goose.leaks(out, ("github.com/django",)) == ["see github.com/django/django"]
    assert goose.stats(tmp_path / "nowhere").status == "no session"


def test_round_tally_applies_the_stop_rules():
    import round as rnd  # ty: ignore[unresolved-import]  (scripts/swebench/round.py)

    t = rnd.Tally()
    t.add("9296 ? RESOLVED goose 12 calls $0.040 done | ran 60s", 60)
    assert not t.stopped
    t.add("11790 ? not goose 0 calls $0.000 crashed | ran 5s", 5)
    t.add("11848 ? not goose 0 calls $0.000 | killed at 900s", 900)
    assert "2 failures" in t.stopped

    t = rnd.Tally()
    for i in range(4):
        t.add(f"{i} ? not mini 40 calls $0.200 Submitted | ran 300s", 300)
    assert t.stopped.startswith("0/4 resolved at $0.20")

    assert rnd.flag_caps("goose", "x $0.300 y", 100, True) == "stopped: cost"
    assert rnd.flag_caps("goose", "x $0.010 y", 100, False) == "stopped: time"
    assert rnd.flag_caps("glm-only", "x $0.300 y", 100, True) == ""


def test_every_arm_has_a_cap_and_a_known_driver():
    import round as rnd  # ty: ignore[unresolved-import]

    assert set(rnd.ARMS) == set(rnd.CAPS)
    for driver, _ in rnd.ARMS.values():
        assert (Path(rnd.HERE) / driver).exists(), driver
        assert driver in rnd.CAP_ENV


def test_resolved_at_a_cap_is_not_a_failure_and_a_stop_does_not_deadlock(tmp_path, monkeypatch):
    import threading

    import round as rnd  # ty: ignore[unresolved-import]

    t = rnd.Tally()
    for i in range(3):
        t.add(f"{i} ? RESOLVED glm 20 calls $0.09 | killed at 900s 900s | stopped: time", 900)
    assert not t.stopped  # §109.5: GLM resolved all three at the wall cap

    # A stopped arm logs while holding the round's lock; with a plain Lock that hung the round.
    monkeypatch.setattr(rnd, "RESULTS", tmp_path / "r.txt")
    rnd._tallies["x"] = rnd.Tally()
    rnd._tallies["x"].stopped = "2 failures in 2 tasks"
    done = threading.Event()

    def go():
        rnd.run_arm("rt", "mini", "1", tmp_path)
        done.set()

    monkeypatch.setitem(rnd._tallies, "mini", rnd._tallies["x"])
    threading.Thread(target=go, daemon=True).start()
    assert done.wait(10), "run_arm deadlocked on its own lock"
    assert "skipped (arm stopped" in (tmp_path / "r.txt").read_text()


def test_done_pairs_resumes_a_round(tmp_path, monkeypatch):
    import round as rnd  # ty: ignore[unresolved-import]

    monkeypatch.setattr(rnd, "RESULTS", tmp_path / "r.txt")
    (tmp_path / "r.txt").write_text(
        "c1\tgoose\t9296 ? RESOLVED goose\nc1\tbalance\tstart $1\n"
        "c1\tcodex\t9296 skipped (arm stopped)\nc2\tgoose\t11790 x\n"
    )
    assert rnd.done_pairs("c1") == {("goose", "9296"), ("codex", "9296")}


def test_done_pairs_survives_a_half_written_line(tmp_path, monkeypatch):
    import round as rnd  # ty: ignore[unresolved-import]

    monkeypatch.setattr(rnd, "RESULTS", tmp_path / "r.txt")
    (tmp_path / "r.txt").write_text("c1\tgoose\t9296 ? RESOLVED\nc1\tmini-gl\n\npartial")
    assert rnd.done_pairs("c1") == {("goose", "9296")}


# --- the sandbox every suite launches an agent in (TODO public release item 7) ---

needs_sandbox = pytest.mark.skipif(
    not agents.SANDBOX_EXEC.exists(), reason="sandbox-exec is macOS only"
)


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A stand-in home folder holding a secret, a workspace and an output folder."""
    home = tmp_path / "home"
    ws, out = home / ".cache" / "runs" / "ws", home / ".cache" / "runs" / "out"
    for d in (ws, out, home / "private"):
        d.mkdir(parents=True)
    (home / "private" / "secret.txt").write_text("secret")
    (ws / "code.py").write_text("x = 1\n")
    return home, ws, out, home / "private" / "secret.txt"


def test_sandboxed_wraps_the_command_and_allowlists_the_environment(tmp_path, monkeypatch):
    home, ws, out, _ = _layout(tmp_path)
    monkeypatch.setenv("BRAVE_API_KEY", "should-not-pass")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    argv, env = agents.sandboxed(
        mini, ["mini", "-t", "goal"], ws, out, "or-key", {"MSWEA_CONFIGURED": "true"}, home=home
    )
    assert argv[:3] == [str(agents.SANDBOX_EXEC), "-f", str(out / "sandbox.sb")]
    assert argv[3:] == ["mini", "-t", "goal"]
    assert "BRAVE_API_KEY" not in env
    assert env["OPENROUTER_API_KEY"] == "or-key"
    assert env["MSWEA_CONFIGURED"] == "true"
    assert env["LC_ALL"] == "en_US.UTF-8"
    assert env["TMPDIR"] == str(out / "tmp") and (out / "tmp").is_dir()
    profile = (out / "sandbox.sb").read_text()
    assert (
        f'(deny file-read-data file-read-xattr file-write* (subpath "{home.resolve()}"))' in profile
    )
    assert str((home / "Library/Application Support/mini-swe-agent").resolve()) in profile


def test_sandboxed_allows_the_drivers_own_state(tmp_path, monkeypatch):
    home, ws, out, _ = _layout(tmp_path)
    monkeypatch.setattr(goose, "DB", home / "goose-store" / "sessions.db")
    monkeypatch.setattr(aider, "VENV", home / "aider-venv")
    agents.sandboxed(goose, ["goose"], ws, out, "k", {}, home=home)
    assert str((home / "goose-store").resolve()) in (out / "sandbox.sb").read_text()
    agents.sandboxed(aider, ["aider"], ws, out, "k", {}, home=home)
    assert str((home / "aider-venv").resolve()) in (out / "sandbox.sb").read_text()


@needs_sandbox
def test_the_sandbox_denies_the_home_folder_and_allows_the_run_against_real_sandbox_exec(
    tmp_path, monkeypatch
):
    import subprocess

    home, ws, out, secret = _layout(tmp_path)
    monkeypatch.setenv("BRAVE_API_KEY", "should-not-pass")

    def run(cmd: list[str]) -> subprocess.CompletedProcess:
        argv, env = agents.sandboxed(mini, cmd, ws, out, "or-key", {}, home=home)
        return subprocess.run(argv, env=env, cwd=ws, capture_output=True, text=True)

    assert run(["/bin/cat", str(secret)]).returncode != 0
    assert run(["/bin/sh", "-c", f"echo x > {home / 'private' / 'new.txt'}"]).returncode != 0
    assert not (home / "private" / "new.txt").exists()
    assert run(["/bin/cat", str(ws / "code.py")]).stdout == "x = 1\n"
    assert run(["/bin/sh", "-c", "echo y > edited.txt"]).returncode == 0
    assert (ws / "edited.txt").read_text() == "y\n"
    assert run(["/bin/sh", "-c", f"echo z > {out / 'log.txt'}"]).returncode == 0
    tools = home / ".local" / "share" / "uv" / "tools"  # the user's installed tools: not writable
    tools.mkdir(parents=True)
    assert run(["/bin/sh", "-c", f"mkdir {tools / '.tmp1'}"]).returncode != 0
    assert not (tools / ".tmp1").exists()
    shown = run(["/usr/bin/env"]).stdout
    assert "OPENROUTER_API_KEY=or-key" in shown and "BRAVE_API_KEY" not in shown
    assert f"UV_TOOL_DIR={out / 'uv-tools'}" in shown  # uvx's own folder, inside the run's output
