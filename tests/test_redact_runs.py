"""scripts/redact_runs.py on a real directory tree: a mini trajectory, its console log and a
Codex event stream, with one personal command among commands that must stay readable."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("redact_runs", REPO / "scripts" / "redact_runs.py")
assert _spec is not None and _spec.loader is not None
rr = importlib.util.module_from_spec(_spec)
sys.modules["redact_runs"] = rr
_spec.loader.exec_module(rr)

PRIVATE = "tax return 2024.pdf\nkid birth certificate.pdf\nscan of passport.pdf"


def make_runs(home: Path) -> Path:
    root = home / ".cache" / "mercury-harness"
    run = root / "swebench-runs" / "r1-mini"
    (run / "mini").mkdir(parents=True)
    gold = f"{home}/Projects/AI Agent/traces/swebench-data/django__django-1.json"

    def call(i, cmd):
        return {"id": f"c{i}", "function": {"arguments": json.dumps({"command": cmd})}}

    def reply(i, out):
        return {
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": json.dumps({"returncode": 0, "output": out}),
        }

    traj = {
        "info": {
            "config": {"env": {"PATH": f"{home}/.claude/plugins/cache/secret-plugin/bin:/usr/bin"}}
        },
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "", "tool_calls": [call(1, "ls ~/Documents")]},
            reply(1, PRIVATE) | {"extra": {"raw_output": PRIVATE}},
            {"role": "assistant", "content": "", "tool_calls": [call(2, f"cat '{gold}'")]},
            reply(2, "the gold patch"),
            {"role": "assistant", "content": "", "tool_calls": [call(3, "ls")]},
            reply(3, "django\nsetup.py"),
        ],
    }
    (run / "mini" / "1.traj.json").write_text(json.dumps(traj))
    (run / "mini-1.log").write_text(
        "Building agent config from specs: ['mini.yaml', \n'environment.env.PATH=/opt/bin:"
        + str(home)[:10]
        + "\n"
        + str(home)[10:]
        + "/.claude/plugins/cache/secret-plugin/bin', \n"
        "mini-swe-agent (step 1, $0.00):\n```\nls ~/Documents\n```\n\nTool:\n<returncode>\n0\n"
        "<output>\ntax return 2024.pdf\nkid birth\ncertificate.pdf\nscan of passport.pdf\n\n"
        + "─"
        * 40
        + "\n\nmini-swe-agent (step 2, $0.00):\n```\ncd ~; ls; find / -name x\n```\n\nTool:\n"
        "<returncode>\n-1\n<output>\nflappy-bird\nmario_clone\n\n<exception_info>\ntimed out\n"
        + "─"
        * 40
        + "\n\nmini-swe-agent (step 3, $0.00):\n```\nls\n```\n\nTool:\n<returncode>\n0\n"
        "<output>\ndjango\nsetup.py\n"
    )
    codex = root / "scicode-runs" / "s1-codex" / "agent-out" / "1"
    codex.mkdir(parents=True)
    item = {
        "id": "item_1",
        "type": "command_execution",
        "command": f"/bin/zsh -lc 'find {home} -name numpy'",
    }
    (codex / "stdout.log").write_text(
        json.dumps({"type": "item.started", "item": item | {"aggregated_output": ""}})
        + "\n"
        + json.dumps(
            {"type": "item.completed", "item": item | {"aggregated_output": f"{home}/Other/numpy"}}
        )
        + "\n"
    )
    return root


def test_dry_run_reports_and_writes_nothing_then_apply_blanks_only_personal_outputs(tmp_path):
    home = tmp_path / "Users" / "person"
    root = make_runs(home)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    before = {p: p.read_bytes() for p in files}
    ident = rr.Identity(home=home)

    report = rr.run([root], ident, apply=False)
    assert {Path(p).name for p in report} == {"1.traj.json", "mini-1.log", "stdout.log"}
    assert {p: p.read_bytes() for p in files} == before

    rr.run([root], ident, apply=True)
    traj = (root / "swebench-runs/r1-mini/mini/1.traj.json").read_text()
    assert "passport" not in traj and "Documents" not in traj and rr.MARKER in traj
    assert "the gold patch" in traj and "swebench-data" in traj  # kept for TODO 2.10
    assert "setup.py" in traj and "secret-plugin" not in traj
    log = (root / "swebench-runs/r1-mini/mini-1.log").read_text()
    assert "passport" not in log and "birth" not in log and "Documents" not in log
    assert rr.MARKER in log and "setup.py" in log and "secret-plugin" not in log
    assert "flappy" not in log  # the timed-out listing, blanked by its command
    events = [json.loads(x) for x in (root / "scicode-runs/s1-codex/agent-out/1/stdout.log").open()]
    assert events[1]["item"]["aggregated_output"] == rr.MARKER
    assert all(str(home) not in e["item"]["command"] for e in events)

    assert rr.run([root], ident, apply=True) == {}  # a second pass finds nothing


def test_personal_ignores_kept_folders_and_bitwise_not():
    ident = rr.Identity(home=Path("/Users/person"))
    assert not rr.personal("x = ~mask & y", "", ident)
    assert not rr.personal("ls ~/.cache/uv/archive-v0", "", ident)
    assert not rr.personal("cat '/Users/person/Projects/AI Agent/traces/a.json'", "", ident)
    assert rr.personal("cd ~; ls", "", ident)
    assert rr.personal("ls /Users", "", ident)
    assert rr.personal("env | grep -i path", "", ident)
    assert rr.personal("find / -name x", "/Users/person/Desktop/x\n", ident)
    assert not rr.personal('grep -n "Cache\\|~~~~" docs/releases/3.1.txt', "", ident)
    assert not rr.personal(
        "sed -i 1d f.py", 'sed: 1: "/Users/person/.cac ...": extra characters', ident
    )
    assert rr.personal("ls ~/Projects", "", ident)
    assert rr.personal("ls -la /Users/person/Projects/", "", ident)
    assert not rr.personal("ls ~/.cache/uv", "", ident)  # exactly a kept folder


def test_scrub_keeps_paths_into_kept_folders():
    ident = rr.Identity(home=Path("/Users/person"))
    assert rr.scrub("cat '/Users/person/Projects/AI Agent/traces/a.json'", ident) == (
        "cat '/Users/person/Projects/AI Agent/traces/a.json'"
    )
    assert rr.scrub("ls /Users/person/Documents/tax.pdf ~/Desktop", ident) == (
        "ls <home>/<redacted> <home>/<redacted>"
    )
    assert rr.scrub("ls -la /Users/person/", ident) == "ls -la <home>/"
    assert rr.scrub("x = ~mask", ident) == "x = ~mask"
