"""scripts/release_traces.py on a real directory tree: what is copied, rewritten, refused."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_traces", REPO / "scripts" / "release_traces.py"
)
assert _spec is not None and _spec.loader is not None
rt = importlib.util.module_from_spec(_spec)
sys.modules["release_traces"] = rt  # dataclasses look the module up by name
_spec.loader.exec_module(rt)


def make_corpus(root: Path, home: Path) -> tuple[list[Path], list[Path]]:
    clean = root / "traces" / "s99-clean"
    clean.mkdir(parents=True)
    (clean / "run-1.jsonl").write_text(
        json.dumps({"kind": "run_start", "workspace": f"{home}/agent-sandbox"})
        + "\n"
        + json.dumps(
            {
                "kind": "log",
                "text": "/private/tmp/claude-501/-x/"
                + "0" * 8
                + "-0000-0000-0000-"
                + "0" * 12
                + "/scratchpad/ws",
            }
        )
        + "\n"
    )
    (clean / "run-2.jsonl").write_text(
        json.dumps({"kind": "log", "text": "Authorization: Bearer {KEY}"}) + "\n"
    )
    (clean / "notes.png").write_bytes(b"\x89PNG")
    leaky = root / "traces" / "s98-leaky"
    leaky.mkdir()
    (leaky / "run-1.jsonl").write_text(
        json.dumps({"text": "mail me at someone@example.com"}) + "\n"
    )
    (leaky / "run-2.jsonl").write_text(json.dumps({"text": "key sk-or-v1-" + "a1" * 20}) + "\n")
    (leaky / "run-3.jsonl").write_text(json.dumps({"text": "/Users/someoneelse/project"}) + "\n")
    gaia = root / "traces" / "gaia-scout"
    gaia.mkdir()
    (gaia / "run-1.jsonl").write_text("{}\n")
    results = root / "results"
    results.mkdir()
    (results / "s99-clean.jsonl").write_text(
        json.dumps({"trace_path": "traces/s99-clean/run-1.jsonl", "workspace": f"{home}/w"}) + "\n"
    )
    return [clean, leaky, gaia], [results / "s99-clean.jsonl"]


def test_generic_host_names_are_not_identity():
    """A macOS LocalHostName of "Mac" replaced as a word turned `mac_romanian.py` into
    `<host>_romanian.py` across a whole dataset."""
    assert rt.generic_host("Mac")
    assert rt.generic_host("MacBook-Pro")
    assert not rt.generic_host("Pats-MacBook-Pro")
    assert not rt.generic_host("Pat\u2019s MacBook Pro")  # a macOS ComputerName


def ident_for(home):
    return rt.Identity(
        home=home,
        hosts=("Persons-Mac",),
        names=("PDoe-Smith", "Doe-Smith"),
        emails=("p@example.com",),
    )


def test_release_rewrites_paths_refuses_leaks_and_writes_manifest(tmp_path):
    home = tmp_path / "home" / "person"
    traces, results = make_corpus(tmp_path, home)
    out = tmp_path / "out"
    manifest = rt.release(out, traces, results, ident=ident_for(home))

    copied = {c["path"] for c in manifest.copied}
    assert copied == {
        "traces/s99-clean/run-1.jsonl",
        "traces/s99-clean/run-2.jsonl",
        "traces/s98-leaky/run-3.jsonl",
        "results/s99-clean.jsonl",
    }
    text = (out / "traces/s99-clean/run-1.jsonl").read_text()
    assert "<home>/<redacted>" in text and str(home) not in text and "agent-sandbox" not in text
    assert "<scratch>/ws" in text
    assert "trace_path" in (out / "results/s99-clean.jsonl").read_text()

    refused = {r["path"]: r["reason"] for r in manifest.refused}
    assert refused["traces/s99-clean/notes.png"].startswith("not a text type")
    assert "email" in refused["traces/s98-leaky/run-1.jsonl"]
    assert "openrouter key" in refused["traces/s98-leaky/run-2.jsonl"]
    # Another account's home (a path a model made up) is rewritten, not refused.
    assert "/Users/<someone>/project" in (out / "traces/s98-leaky/run-3.jsonl").read_text()
    assert refused["gaia-scout"] == "denied by name: gaia"
    assert not manifest.ok
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["refused"] == manifest.refused


def test_scan_ignores_placeholders_and_finds_tokens():
    assert rt.scan("Authorization: Bearer {KEY}") == []
    assert rt.scan("Authorization: Bearer $INCEPTION_API_KEY") == []
    assert rt.scan("Bearer " + "x" * 30) == ["bearer token"]
    assert rt.scan("hf_" + "b" * 24) == ["hugging face token"]
    # A decorator in a JSON-encoded diff is not an address; a real one still is.
    assert rt.scan('{"diff": "\\n+@pytest.mark.parametrize\\n@pytest.fixture"}') == []
    assert rt.scan("wrote to some.person+tag@example.co.uk today") == ["email"]


def test_cli_exit_status_reports_refusals(tmp_path):
    home = tmp_path / "home" / "person"
    traces, results = make_corpus(tmp_path, home)
    out = tmp_path / "release"
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "release_traces.py"),
            str(out),
            "--traces",
            str(traces[0]),
            "--results",
            *map(str, results),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1, r.stdout + r.stderr  # the png is refused
    assert "refused traces/s99-clean/notes.png" in r.stdout
    r2 = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "release_traces.py"),
            str(out),
            "--traces",
            str(traces[0]),
        ],
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 2 and "not empty" in r2.stderr


def test_redact_keeps_named_folders_and_cuts_everything_else_under_home(tmp_path):
    home = tmp_path / "Users" / "person"
    ident = ident_for(home)
    text = "\n".join(
        [
            f"{home}/.cache/mercury-harness/swebench-runs/s1/11790/django/forms.py",
            f'cd "{home}/Projects/AI Agent/traces/swebench-data" && cat a.json',
            f"cd {home}/Projects/AI\\ Agent/scripts",
            "ls ~/Documents ~/Downloads; ls -la ~/Projects/other-project",
            f"ls -la {home}/",
            "drwxr-xr-x@  5 person  staff  160 Oct 2 14:31 .",
            "/private/tmp/claude-501/-Users-person-Projects-AI-Agent/abc/tasks/x.output",
            "Author: PDoe-Smith <12345+pdoe-smith@users.noreply.github.com> on Persons-Mac.local",
            "2024 P Doe-Smith Client Copy.pdf, mail p@example.com",
            "cd /Users/runner/work/django",
        ]
    )
    out = rt.redact(text, ident)
    lines = out.splitlines()
    assert lines[0] == "<runs>/swebench-runs/s1/11790/django/forms.py"
    assert lines[1] == 'cd "<repo>/traces/swebench-data" && cat a.json'
    assert lines[2] == "cd <repo>/scripts"
    assert lines[3] == "ls <home>/<redacted> <home>/<redacted>; ls -la <home>/<redacted>"
    assert lines[4] == "ls -la <home>"
    assert lines[5].split()[2] == "user"
    assert lines[6] == "<scratch>"
    assert lines[7] == "Author: <name> <<email>> on <host>.local"
    assert lines[8] == "2024 P <name> Client Copy.pdf, mail <email>"
    assert lines[9] == "cd /Users/<someone>/work/django"
    assert rt.scan(out, ident) == []
    assert rt.scan("a file of Doe-Smith's", ident) == ["identity"]


def test_a_name_after_a_json_escape_is_still_a_word(tmp_path):
    ident = ident_for(tmp_path / "Users" / "person")
    ps_line = '{"output": "date\\nperson   7215 100.0 python runtests.py"}'
    assert rt.scan(ps_line, ident) == ["identity"]
    assert "person" not in rt.redact(ps_line, ident)
    assert "personal" in rt.redact("a personal note", ident)  # inside a word: left alone


@pytest.mark.parametrize(
    "path",
    [
        "traces/Flappy-bird2/run-1.jsonl",
        "jumpin-plumber/traces/run-1.jsonl",
        "mario_clone/traces/run-1.jsonl",
        "Mine Sweeper/traces/run-1.jsonl",
        "results/s921-clarify-replay.jsonl",
    ],
)
def test_app_run_projects_are_refused_by_name(path):
    assert rt.denied(Path(path).parts) is not None
