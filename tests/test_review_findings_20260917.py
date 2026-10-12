"""Regression tests for the findings of the 2026-09-17 code review that were
still open on 2026-09-24 (see DESIGN-LOG §52)."""

import json
import shlex

import pytest

from agent import dataformats, toolchain
from agent.approval import Approver, Denied, and_chain_segments
from agent.llm import Reply, Usage
from agent.memory import MemoryStore
from agent.phases import coverage
from agent.runner import run_tool_loop
from agent.tools import ToolContext, execution_registry
from agent.tools.fs import MAX_READ_BYTES, glob_files, read_file, write_file
from agent.trace import Trace
from agent.workspace import Workspace


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


# -- #7: synthesized validators quote file names ------------------------------


def test_a_data_file_with_a_space_stays_one_argument(monkeypatch):
    monkeypatch.setattr(dataformats.shutil, "which", lambda tool: tool == "yq")
    command = dataformats.build_validator(["my data.json", "x;rm -rf ~.json"])
    assert shlex.split(command) == ["yq", ".", "my data.json", "x;rm -rf ~.json"]


# -- #10: coverage never touches a file of its report's name ------------------


def test_coverage_leaves_a_project_file_of_the_same_name_alone(ctx, monkeypatch):
    own = ctx.workspace.root / coverage.COVERAGE_JSON
    own.write_text("the project's own file")

    def fake_run(tool, args, root, timeout=None):
        out = args[args.index("-o") + 1]
        assert not out.startswith(str(ctx.workspace.root)), "report written into the workspace"
        with open(out, "w") as fh:
            json.dump({"files": {}}, fh)
        return 0, "", ""

    monkeypatch.setattr(toolchain, "run", fake_run)
    assert coverage._run_json(ctx) == {"files": {}}
    assert own.read_text() == "the project's own file"


# -- #12: the spend cap bounds the last word, not only the loop ----------------


def test_no_last_word_is_asked_for_once_the_cap_is_spent(ctx):
    (ctx.workspace.root / "a.py").write_text("x = 1\n")
    calls = []

    class Client:
        def chat(self, model, messages, **kwargs):
            calls.append(1)
            read = {
                "id": f"c{len(calls)}",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})},
            }
            return Reply(content="", tool_calls=[read], usage=Usage(100, 50, 0.2, 0.1))

    run_tool_loop(
        client=Client(), model="m", phase="execute:s", system="s", user="u",
        registry=execution_registry(), ctx=ctx, trace=ctx.trace,
        max_iters=1, max_usd=0.1, salvage=True,
    )  # fmt: skip
    assert len(calls) == 1, "a last-word call was made after the cap was spent"


# -- #17: the glob fallback prunes ignored directories and keeps rglob's matches


def test_the_glob_fallback_never_descends_into_an_ignored_directory(ctx, monkeypatch):
    write_file(ctx, "top.py", "x = 1\n")
    venv = ctx.workspace.root / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "hidden.py").write_text("x = 1\n")
    monkeypatch.setattr(
        toolchain, "run_capturing", lambda argv, **kw: toolchain.ProcResult(127, "", "", 0.0)
    )
    visited = []
    import agent.tools.fs as fs

    original = fs.walk_files

    def spying_walk(root, **kw):
        for path in original(root, **kw):
            visited.append(path)
            yield path

    monkeypatch.setattr(fs, "walk_files", spying_walk)
    assert glob_files(ctx, "**/*.py").splitlines() == ["top.py"]
    assert not any(".venv" in str(p) for p in visited)


# -- #18: a file over the whole-file cap can be read in slices ----------------


def test_an_oversized_file_is_read_in_the_slice_asked_for(ctx):
    line = "x" * 99 + "\n"
    (ctx.workspace.root / "big.log").write_text(line * (MAX_READ_BYTES // len(line) + 50))
    out = read_file(ctx, "big.log", offset=10, limit=3)
    assert not out.startswith("error")
    assert out.count("x" * 99) == 3
    assert "Use offset=13" in out


def test_an_offset_past_the_end_of_an_oversized_file_says_so(ctx):
    line = "y" * 99 + "\n"
    (ctx.workspace.root / "big.log").write_text(line * (MAX_READ_BYTES // len(line) + 50))
    assert "past the end" in read_file(ctx, "big.log", offset=10**9)


# -- #22: one oversized memory section cannot exceed the recall budget ---------


def test_the_newest_section_is_clipped_to_the_budget(ctx):
    store = MemoryStore(ctx.workspace.root)
    store.memory_path.parent.mkdir(parents=True, exist_ok=True)
    store.memory_path.write_text("## an enormous run\n" + "word " * 5_000 + "\n")
    recalled = store.relevant("anything", budget_tokens=300)
    assert "clipped to the memory budget" in recalled
    assert len(recalled.split()) < 600


# -- new: a verify command that is a plain && chain is approved piece by piece --


def test_a_plain_and_chain_splits_into_its_commands():
    assert and_chain_segments("uv run ruff check app.py && yq '.' 'my data.json'") == [
        "uv run ruff check app.py",
        "yq . 'my data.json'",
    ]


@pytest.mark.parametrize(
    "command",
    ["a", "a; b", "a | b", "a || b", "a && b > out", "a && (b)", "a &&", "a && b & c"],
)
def test_anything_else_is_not_an_and_chain(command):
    assert and_chain_segments(command) is None


def test_an_unattended_verify_chain_of_allowed_commands_is_approved(tmp_path):
    root = tmp_path.resolve()
    approver = Approver(auto_approve=True, unattended=True, workspace_root=root)
    command = approver.check_verify("uv run ruff check app.py && uv run pytest -q test_app.py")
    assert command == "uv run ruff check app.py && uv run pytest -q test_app.py"


def test_one_bad_link_still_denies_the_whole_chain_unattended(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path.resolve())
    with pytest.raises(Denied):
        approver.check_verify("uv run pytest -q && cat ~/.ssh/id_rsa")
    with pytest.raises(Denied):
        approver.check_verify("uv run pytest -q; cat ~/.ssh/id_rsa")


# -- follow-ups found while closing the review ---------------------------------


def test_a_symlinked_workspace_root_does_not_deny_every_path(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    approver = Approver(auto_approve=True, unattended=True, workspace_root=link)
    approver.check_bash("find . -name '*.py'")  # raised Denied before the root was resolved


def test_bench_report_accepts_an_absolute_path(tmp_path, capsys):
    from bench.cli import _expand_path

    results = tmp_path / "r.jsonl"
    results.write_text("")
    assert _expand_path(str(results)) == [results]
    assert _expand_path(str(tmp_path / "*.jsonl")) == [results]
