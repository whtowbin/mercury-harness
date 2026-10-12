"""The release checks that read local data (scripts/release_local.py), against real trace files
and reference folders written to tmp_path."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "release_local", REPO / "scripts" / "release_local.py"
)
assert _spec is not None and _spec.loader is not None
rl = importlib.util.module_from_spec(_spec)
sys.modules["release_local"] = rl
_spec.loader.exec_module(rl)


def write_trace(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_typed_texts_keeps_goals_and_answers_of_four_words(tmp_path):
    trace = tmp_path / "run-1.jsonl"
    write_trace(
        trace,
        [
            {"kind": "run_start", "goal": "Make  the pipes\nmove FASTER please"},
            {"kind": "clarify_answered", "question": "q?", "answer": "yes"},
            {"kind": "ask_user_answered", "answer": "the clouds drift too slowly"},
            {"kind": "log", "message": "a model message of many words here"},
        ],
    )
    assert rl.typed_texts([str(trace)]) == {
        "make the pipes move faster please",
        "the clouds drift too slowly",
    }


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "agent").mkdir(parents=True)
    (root / "bench").mkdir()
    (root / "agent" / "a.py").write_text("# from a run: 'The clouds drift too\n  slowly'\n")
    (root / "agent" / "b.py").write_text("x = 'fix the pipe gap bug'\n")
    (root / "bench" / "tasks.py").write_text("GOAL = 'fix the pipe gap bug'\n")
    return root, ["agent/a.py", "agent/b.py", "bench/tasks.py"]


def test_check_words_reports_quotes_and_drops_fixture_text(tmp_path, monkeypatch):
    root, files = make_repo(tmp_path)
    write_trace(
        root / "traces" / "run-1.jsonl",
        [
            {"kind": "run_start", "goal": "fix the pipe gap bug"},  # also a fixture goal
            {"kind": "ask_user_answered", "answer": "the clouds drift too slowly"},
        ],
    )
    monkeypatch.setattr(rl, "APP_RUN_TRACES", ("traces/run-*.jsonl",))
    found, missing = rl.check_words(root, files)
    assert missing == []
    assert [(f.rule, f.path) for f in found] == [("words", "agent/a.py")]


def test_check_words_without_traces_is_not_run(tmp_path, monkeypatch):
    root, files = make_repo(tmp_path)
    monkeypatch.setattr(rl, "APP_RUN_TRACES", ("traces/run-*.jsonl",))
    found, missing = rl.check_words(root, files)
    assert found == []
    assert missing and missing[0].startswith("words: no app-run traces")


REFERENCE = "\n".join(
    f"def function_number_{i}(argument):\n    return argument * {i}" for i in range(8)
)


def test_check_copies_finds_shared_runs_and_reports_missing_references(tmp_path, monkeypatch):
    root, files = make_repo(tmp_path)
    ref = tmp_path / "cache" / "pkg-1" / "otherpkg"
    ref.mkdir(parents=True)
    (ref / "mod.py").write_text(REFERENCE + "\n")
    (root / "agent" / "copied.py").write_text("import os\n\n" + REFERENCE + "\n")
    (root / "agent" / "own.py").write_text("# comment\n" + "\n".join(REFERENCE.splitlines()[:5]))
    monkeypatch.setattr(
        rl,
        "REFERENCES",
        {"other (MIT)": (str(tmp_path / "cache" / "*" / "otherpkg"),), "gone": ("nowhere/*",)},
    )
    found, missing = rl.check_copies(root, [*files, "agent/copied.py", "agent/own.py"])
    assert [(f.rule, f.path, f.line) for f in found] == [("copies", "agent/copied.py", 3)]
    assert "with other (MIT)" in found[0].text
    assert missing == ["copies: gone not found (looked in nowhere/*)"]


def test_code_lines_skips_comments_blanks_and_trivial_lines():
    text = "# a comment\n\nreturn\n)\nvalue = compute(alpha, beta)\n"
    assert rl.code_lines(text, python=True) == [(5, "value = compute(alpha, beta)")]
