"""release_export against a real git repo and real subprocesses (no mocks)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_export", REPO / "scripts" / "release_export.py"
)
assert _spec is not None and _spec.loader is not None
rx = importlib.util.module_from_spec(_spec)
sys.modules["release_export"] = rx
_spec.loader.exec_module(rx)
rc = rx.rc

TRACKED = {
    "README.md": "hello\n",
    "pyproject.toml": "[project]\nname='x'\n",
    "src/a.py": "x = 1\n",
    "bin/tool.sh": "#!/bin/sh\necho hi\n",
    "TODO.md": "private\n",
    "docs/posts/x.md": "private\n",
}


@pytest.fixture(autouse=True)
def identity(monkeypatch):
    for k, v in {"NAME": "Test", "EMAIL": "t@example.invalid"}.items():
        monkeypatch.setenv(f"GIT_AUTHOR_{k}", v)
        monkeypatch.setenv(f"GIT_COMMITTER_{k}", v)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "src"
    for rel, text in TRACKED.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    (r / "bin/tool.sh").chmod(0o755)
    (r / ".gitignore").write_text("ignored.txt\n")
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "private history")
    (r / "untracked.txt").write_text("not committed\n")
    (r / "ignored.txt").write_text("ignored\n")
    return r


def test_export_copies_exactly_the_shipped_files(repo, tmp_path, capsys):
    dest = tmp_path / "out"
    assert rx.main(["export", str(dest), "--message", "first"], root=repo) == 0
    shipped = set(rc.shipped_files(repo))
    assert "TODO.md" not in shipped and "bin/tool.sh" in shipped
    copied = {
        p.relative_to(dest).as_posix()
        for p in dest.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(dest).parts
    }
    assert copied == shipped
    assert (dest / "src/a.py").read_text() == "x = 1\n"
    assert os.access(dest / "bin/tool.sh", os.X_OK)
    assert not os.access(dest / "README.md", os.X_OK)
    assert f"exported {len(shipped)} files" in capsys.readouterr().out


def test_export_is_one_commit_of_those_files_and_has_no_remote(repo, tmp_path, capsys):
    dest = tmp_path / "out"
    assert rx.main(["export", str(dest)], root=repo) == 0
    assert git(dest, "rev-list", "--count", "HEAD").strip() == "1"
    assert git(dest, "log", "-1", "--format=%s").strip() == rx.DEFAULT_MESSAGE
    assert git(dest, "branch", "--show-current").strip() == "main"
    assert set(git(dest, "ls-tree", "-r", "--name-only", "HEAD").split()) == set(
        rc.shipped_files(repo)
    )
    assert git(dest, "remote").strip() == ""
    assert git(dest, "rev-parse", "HEAD").strip() in capsys.readouterr().out


def test_export_refuses_a_non_empty_dest(repo, tmp_path, capsys):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "keep.txt").write_text("mine\n")
    assert rx.main(["export", str(dest)], root=repo) == 2
    assert "not empty" in capsys.readouterr().out
    assert [p.name for p in dest.iterdir()] == ["keep.txt"]


def test_export_accepts_an_existing_empty_dest(repo, tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    assert rx.main(["export", str(dest)], root=repo) == 0
    assert (dest / ".git").is_dir()


def test_verify_reports_pass_and_fail_and_continues(tmp_path):
    steps = [
        ("ok step", [sys.executable, "-c", "import sys; sys.exit(0)"]),
        ("bad step", [sys.executable, "-c", "import sys; print('boom detail'); sys.exit(1)"]),
        ("after the failure", [sys.executable, "-c", "pass"]),
        ("missing program", ["definitely-not-a-program-xyz"]),
    ]
    lines: list[str] = []
    code = rx.verify(tmp_path, steps, install=None, out=lines.append)
    text = "\n".join(lines)
    assert code == 1
    assert any(ln.startswith("PASS") and "ok step" in ln for ln in lines)
    assert any(ln.startswith("FAIL") and "bad step" in ln for ln in lines)
    assert any(ln.startswith("PASS") and "after the failure" in ln for ln in lines)
    assert any(ln.startswith("FAIL") and "missing program" in ln for ln in lines)
    assert "boom detail" in text
    assert "2 of 4 step(s) failed" in text


def test_verify_all_pass_and_install_step_result_counts(tmp_path):
    steps = [("ok", [sys.executable, "-c", "pass"])]
    lines: list[str] = []
    assert rx.verify(tmp_path, steps, install=lambda d: (True, ""), out=lines.append) == 0
    assert lines[-1] == "all 2 steps passed"
    lines.clear()
    assert rx.verify(tmp_path, steps, install=lambda d: (False, "no wheel"), out=lines.append) == 1
    assert any("no wheel" in ln for ln in lines)


def test_verify_runs_in_dest(tmp_path):
    (tmp_path / "marker.txt").write_text("here\n")
    steps = [("cwd", [sys.executable, "-c", "import os; assert os.path.exists('marker.txt')"])]
    assert rx.verify(tmp_path, steps, install=None, out=lambda s: None) == 0


def test_verify_subcommand_missing_dest(tmp_path):
    assert rx.main(["verify", str(tmp_path / "nope")]) == 2


def _parquet(path: Path, sql: str) -> Path:
    import duckdb

    duckdb.sql(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
    return path


@pytest.fixture
def data_repo(repo, tmp_path, monkeypatch):
    """The repo with a tracked results file, a calls parquet, and a stand-in GAIA metadata file
    so the GAIA check runs rather than being skipped."""
    (repo / "results").mkdir()
    (repo / "results" / "s1.jsonl").write_text('{"config": "a", "task": "t", "cost_usd": 0.1}\n')
    git(repo, "add", "results/s1.jsonl")
    git(repo, "commit", "-q", "-m", "results")
    gaia = _parquet(
        tmp_path / "gaia.parquet",
        "SELECT '11111111-2222-3333-4444-555555555555' AS task_id, 'q' AS \"Question\", "
        "'a' AS \"Final answer\"",
    )
    monkeypatch.setattr(rc, "GAIA_METADATA", gaia)
    monkeypatch.setattr(
        rc.rt, "this_machine", lambda: rc.rt.Identity(home=Path("/Users/pdoe"), names=("PDoe",))
    )
    calls = _parquet(
        tmp_path / "calls.parquet",
        "SELECT 'django' AS suite, 'p1' AS sweep, '9296' AS task, 'run-1' AS run, "
        "'execute' AS phase, 'inception/mercury-2.5' AS model, 0.01::DOUBLE AS cost_usd",
    )
    return repo, calls


def test_assets_bundle_the_results_parquet_and_a_data_readme(data_repo, tmp_path):
    import tarfile

    repo, calls = data_repo
    out = tmp_path / "assets"
    assert rx.main(["assets", str(out), "--parquet", str(calls), "--tag", "v9"], root=repo) == 0
    with tarfile.open(out / "mercury-harness-results-v9.tar.gz") as tar:
        assert tar.getnames() == ["results/s1.jsonl"]
    assert (out / "mercury-harness-calls-v9.parquet").read_bytes() == calls.read_bytes()
    readme = (out / "DATA.md").read_text()
    assert "| django | 1 | 1 |" in readme and "| `cost_usd` | DOUBLE |" in readme
    assert rx.main(["assets", str(out), "--parquet", str(calls)], root=repo) == 2  # not empty


def test_assets_fail_on_a_leak_in_the_results(data_repo, tmp_path):
    repo, calls = data_repo
    (repo / "results" / "s1.jsonl").write_text('{"trace_path": "/Users/pdoe/.cache/r.jsonl"}\n')
    assert rx.main(["assets", str(tmp_path / "a"), "--parquet", str(calls)], root=repo) == 1
