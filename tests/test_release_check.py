"""The release gate (scripts/release_check.py): each rule against fixture text, the shipped file
list and exit codes against a real git repository, and the dist check against a real `uv build`."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_check", REPO / "scripts" / "release_check.py"
)
assert _spec is not None and _spec.loader is not None
rc = importlib.util.module_from_spec(_spec)
sys.modules["release_check"] = rc  # dataclasses look the module up by name
_spec.loader.exec_module(rc)

IDENT = rc.rt.Identity(
    home=Path("/Users/pdoe"),
    hosts=("Pats-MacBook-Pro", "Mac"),
    names=("PDoe-Smith", "Doe-Smith"),
    emails=("pat@example.org",),
)


def rules(gaia=None):
    return rc.build_rules(IDENT, ["Patricia"], gaia)


def hits(text, path="agent/x.py", gaia=None):
    return [(f.rule, f.line) for f in rc.scan_text(path, text, rules(gaia))]


def test_left_out_matches_files_and_nested_paths():
    assert rc.left_out("TODO.md")
    assert rc.left_out(".claude/skills/bench-ab/SKILL.md")
    assert rc.left_out("docs/posts/figures/a.png")
    assert rc.left_out("results/s921-clarify-replay.jsonl")
    assert not rc.left_out("docs/cli.md")
    assert not rc.left_out("agent/TODO.md.py")
    assert not rc.left_out("results/s126-fixtures.jsonl")


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("trace at /Users/pdoe/.cache/x.jsonl", "home"),
        ("asked by Patricia", "identity"),
        ("doe-smith's laptop", "identity"),
        ("mail pat@example.org", "identity"),
        ("on Pats-MacBook-Pro.local", "identity"),
        ("key sk-or-v1-" + "a" * 32, "secret"),
        ("found by ai-agent-4f", "session"),
        ("the jumpin-plumber run", "project"),
        ("see CLAUDE.md", "left-out"),
        ("see `docs/posts/01.md`", "left-out"),
        ("in .claude/rules/testing.md", "left-out"),
    ],
)
def test_each_rule_finds_its_pattern(text, rule):
    assert hits(text) == [(rule, 1)]


@pytest.mark.parametrize(
    "text",
    [
        "lib/python3.9/encodings/mac_romanian.py",  # "Mac" alone is no one's host name
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "/Users/someoneelse/project",  # another account, not this one
        "Claude Code keeps transcripts under ~/.claude/projects",
        'f"{home}/.claude/plugins"',
        "MYCLAUDE.md and NOTTODO.md",
        "User-Agent: mercury-harness (github.com/doe-smith/mercury-harness)",
        'REPO_SLUG = "doe-smith/mercury-harness"',
        "ai-agent-xyz is not a session label",
    ],
)
def test_benign_text_is_clean(text):
    assert hits(text) == []


def test_allowlisted_file_keeps_its_expected_hits_only():
    assert hits("Copyright 2026 Patricia Doe-Smith", path="LICENSE") == []
    assert hits("Copyright Patricia, found by ai-agent-4f", path="LICENSE") == [("session", 1)]


def test_one_finding_per_line_with_line_numbers():
    text = "clean\nPatricia and Patricia\nclean\nDoe-Smith\n"
    found = rc.scan_text("a.md", text, rules())
    assert [(f.rule, f.line) for f in found] == [("identity", 2), ("identity", 4)]
    assert found[0].text == "Patricia and Patricia"


GAIA_ROWS = [
    (
        "11111111-2222-3333-4444-555555555555",
        "How many studio albums did the fictional band Zorblax release between 1990 and 1999?",
        "Seventeen albums total",
    ),
    ("66666666-7777-8888-9999-000000000000", "a a b a b a b a b a", "3"),
]


def test_gaia_ids_long_answers_and_question_text():
    gaia = rc.Gaia.from_rows(GAIA_ROWS)
    assert gaia.answers == ("Seventeen albums total",)  # "3" would match anything
    assert hits("id 11111111-2222-3333-4444-555555555555", gaia=gaia) == [("gaia", 1)]
    assert hits("x\nit was SEVENTEEN ALBUMS TOTAL", gaia=gaia) == [("gaia", 2)]
    quoted = "Q: how many studio albums did the fictional band Zorblax release"
    assert hits(quoted, gaia=gaia) == [("gaia", 0)]


def test_gaia_ignores_short_answers_and_runs_of_short_words():
    gaia = rc.Gaia.from_rows(GAIA_ROWS)
    assert hits("the answer is 3; the string a a b a b a b a b a", gaia=gaia) == []
    assert hits("id 99999999-2222-3333-4444-555555555555", gaia=gaia) == []


def test_member_problems():
    shipped = {"agent/cli.py", ".python-version"}
    assert rc.member_problems("agent/cli.py", shipped) is None
    assert rc.member_problems(".python-version", shipped) is None
    assert rc.member_problems("PKG-INFO", shipped) is None
    assert rc.member_problems("mercury_harness-0.1.0.dist-info/METADATA", shipped) is None
    assert rc.member_problems(".claude/worktrees/x/agent/cli.py", shipped) == (
        "left out of the release"
    )
    assert rc.member_problems("agent/__pycache__/cli.cpython-314.pyc", shipped) == "junk"
    assert rc.member_problems(".remember/logs/today.md", shipped) == (
        "not a shipped file (untracked or ignored)"
    )


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repository: tracked, left-out, untracked and ignored files."""
    root = tmp_path / "repo"
    (root / "agent").mkdir(parents=True)
    (root / "docs" / "posts").mkdir(parents=True)
    (root / "agent" / "core.py").write_text("x = 1\n")
    (root / "docs" / "posts" / "01.md").write_text("Patricia's post\n")
    (root / "TODO.md").write_text("private\n")
    (root / ".gitignore").write_text("ignored.txt\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nauthors = [{ name = "Patricia Doe-Smith" }]\n'
    )
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (root / "agent" / "new.py").write_text("y = 2\n")
    (root / "ignored.txt").write_text("not seen\n")
    monkeypatch.setattr(rc.rt, "this_machine", lambda: IDENT)
    monkeypatch.setattr(rc, "GAIA_METADATA", tmp_path / "no-gaia.parquet")
    return root


def test_shipped_files_from_git(repo):
    assert rc.shipped_files(repo) == [".gitignore", "agent/core.py", "pyproject.toml"]
    assert "agent/new.py" in rc.shipped_files(repo, untracked=True)
    assert "ignored.txt" not in rc.shipped_files(repo, untracked=True)


def test_exit_codes(repo, capsys):
    assert rc.main(["tree", "--skip", "gaia"], root=repo) == 0
    assert rc.main(["tree"], root=repo) == 2  # a check that could not run is not a pass
    assert "not run: gaia" in capsys.readouterr().out
    (repo / "agent" / "core.py").write_text("# asked by Patricia\n")
    assert rc.main(["tree", "--skip", "gaia"], root=repo) == 1
    assert "agent/core.py:1" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_dist_flags_untracked_files_a_real_build_picks_up(repo):
    """hatchling's sdist takes untracked, non-ignored files: the release's `.remember/` and
    `.claude/worktrees/` reached the sdist this way."""
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'authors = [{ name = "Patricia Doe-Smith" }]\n\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["agent"]\n'
    )
    (repo / ".remember").mkdir()
    (repo / ".remember" / "today.md").write_text("session notes\n")
    found = rc.scan_dist(repo, rules())
    flagged = {f.path.split("!", 1)[1] for f in found if f.rule == "dist"}
    assert ".remember/today.md" in flagged
    assert "agent/new.py" in flagged  # untracked: not a shipped file
    assert "agent/core.py" not in flagged


def write_parquet(path, rows_sql):
    import duckdb

    duckdb.sql(f"COPY ({rows_sql}) TO '{path}' (FORMAT parquet)")


def test_data_passes_labels_and_flags_text_columns_and_leaks(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    good = tmp_path / "good.parquet"
    write_parquet(
        good,
        "SELECT 'django' AS suite, 'p1-dj' AS sweep, '9296' AS task, 'run-1' AS run, "
        "'execute' AS phase, 'inception/mercury-2.5' AS model, 'Amazon Bedrock' AS provider, "
        "12 AS prompt_tokens",
    )
    assert rc.scan_data([good], rules()) == []
    bad = tmp_path / "bad.parquet"
    write_parquet(
        bad,
        "SELECT 'django' AS suite, 'fix the flagpole so it shows' AS phase, "
        "'/Users/pdoe/x' AS sweep, 'Amazon Bedrock' AS model, 'secret words' AS goal",
    )
    found = {(f.rule, f.text.split(":")[0]) for f in rc.scan_data([bad], rules())}
    assert ("data", "column 'goal' is not on the release allow-list") in found
    assert ("data", "phase") in found  # prose, not a label
    assert ("data", "model") in found  # spaces only allowed in provider names
    assert ("home", "sweep=/Users/pdoe/x") in found


def test_data_scans_a_results_file_as_text(tmp_path):
    results = tmp_path / "r.jsonl"
    results.write_text('{"trace_path": "/Users/pdoe/.cache/run-1.jsonl"}\n')
    assert [f.rule for f in rc.scan_data([results], rules())] == ["home"]


def test_results_ship_as_release_assets_not_in_the_tree(repo, capsys):
    (repo / "results").mkdir()
    (repo / "results" / "s1.jsonl").write_text('{"trace_path": "/Users/pdoe/.cache/r.jsonl"}\n')
    git(repo, "add", "results/s1.jsonl")
    assert "results/s1.jsonl" not in rc.shipped_files(repo)
    assert rc.asset_files(repo) == ["results/s1.jsonl"]
    assert rc.main(["tree", "--skip", "gaia"], root=repo) == 0  # not in the tree
    assert rc.main(["data", "--skip", "gaia"], root=repo) == 1  # but checked as data
    assert "s1.jsonl:1" in capsys.readouterr().out
