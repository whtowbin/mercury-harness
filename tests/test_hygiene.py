"""The hygiene script's pure checks, fed fixtures rather than the real repo."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "hygiene", Path(__file__).resolve().parent.parent / "scripts" / "hygiene.py"
)
hygiene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hygiene)


def test_index_drift_reports_both_directions():
    log = "| 1 | a |\n| 3 | c |\n\n## 1. A\n\n## 2. B\n"
    assert hygiene.index_drift(log) == (["2"], ["3"])


def test_tunable_drift_and_missing():
    tunables = "| `KEEP` | `a.py:1` |\n| `MOVED` | `a.py:1` |\n| `GONE` | `a.py:1` |\n"
    source = "KEEP = 1\nMOVED = 2\n"
    missing, drift = hygiene.tunable_problems(tunables, lambda _p: source)
    assert missing == ["GONE (no longer in a.py)"]
    assert drift == {"MOVED": (1, 2)}


def test_dead_refs_skips_globs_ignored_dirs_and_allowlist():
    text = "`agent/x.py` `agent/y.py:12` `*.py` `results/r.jsonl` `memory.md` `gone.md`"
    dead = hygiene.dead_refs(text, lambda p: p == "agent/x.py")
    assert dead == ["agent/y.py", "gone.md"]


def test_test_delta_counts_removed_and_added_tests():
    diff = "-def test_a():\n+def test_b():\n+async def test_c():\n-    x = 1\n"
    assert hygiene.test_delta(diff) == (1, 2)


def test_unreferenced_finds_orphan_root_files():
    files = ["notes.md", "used.md", "README.md", "pkg/inner.md"]
    texts = {"notes.md": "", "used.md": "", "README.md": "see used.md", "pkg/inner.md": ""}
    assert hygiene.unreferenced(files, texts.__getitem__) == ["notes.md"]


def test_over_budget_names_only_the_files_past_the_limit():
    counts = {"CLAUDE.md": 40, ".claude/rules/big.md": 151}
    assert hygiene.over_budget(counts, budget=150) == [
        ".claude/rules/big.md is 151 lines (budget 150)"
    ]


def test_refs_ignores_untracked_docs(tmp_path, monkeypatch, capsys):
    """An untracked doc (another session's, not yet committed) is not checked: its names
    resolve against tracked files only, so it would fail every other commit."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "tracked.md").write_text("See `docs/tracked.md`.\n")
    (tmp_path / "docs" / "handoff.md").write_text("See `docs/draft.md`.\n")
    (tmp_path / "docs" / "draft.md").write_text("draft\n")
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    monkeypatch.setattr(hygiene, "tracked_files", lambda: ["docs/tracked.md"])
    assert hygiene.cmd_refs() == 0
    (tmp_path / "docs" / "tracked.md").write_text("See `docs/gone.md`.\n")
    assert hygiene.cmd_refs() == 1
    assert "docs/tracked.md: `docs/gone.md` does not exist" in capsys.readouterr().out
