"""scripts/release_suites.py on a real run tree: which files a suite takes, and the card checks."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "release_suites", REPO / "scripts" / "release_suites.py"
)
assert _spec is not None and _spec.loader is not None
rs = importlib.util.module_from_spec(_spec)
sys.modules["release_suites"] = rs
_spec.loader.exec_module(rs)


def test_a_suite_takes_its_rounds_files_and_skips_worktree_links(tmp_path, monkeypatch):
    runs = tmp_path / "swebench-runs"
    arm = runs / "s109-c1-mini-glm"
    (arm / "mini").mkdir(parents=True)
    (arm / "mini" / "11790.traj.json").write_text(json.dumps({"messages": []}))
    (arm / "mini-11790.log").write_text("log\n")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "x.log").write_text("django checkout\n")
    (arm / "11790").symlink_to(worktree)
    (runs / "s107-r1-ct").mkdir()
    (runs / "s107-r1-ct" / "a.log").write_text("not a released round\n")
    monkeypatch.setattr(rs, "RUNS", tmp_path)
    got = {label for _, label in rs.collect(rs.SUITES["swebench"])}
    assert got == {
        "traces/s109-c1-mini-glm/mini/11790.traj.json",
        "traces/s109-c1-mini-glm/mini-11790.log",
    }


def test_the_card_check_allows_the_repo_link_and_ordinary_words_only(tmp_path):
    ident = rs.Identity(home=tmp_path / "Users" / "pdoe")
    assert rs.check_card(f"See {rs.REPO_URL}. Answers scored right.", ident, gaia_free=False) == []
    assert rs.check_card("by pdoe", ident, gaia_free=False) == ["identity"]


# ---- the per-run index, the dropped external logs and the card sections -------------------------

IDENT_HOME = "pdoe"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _harness_trace(path: Path, commit: str = "abc1234") -> None:
    events = [
        {"ts": 1.0, "kind": "run_start", "executor": "inception/mercury-2.5", "planner": "x/y",
         "harness": {"commit": commit, "dirty": False}},
        {"ts": 2.0, "kind": "llm_call", "prompt_tokens": 10},
        {"ts": 3.0, "kind": "totals", "prompt_tokens": 1000, "completion_tokens": 200,
         "cached_tokens": 600, "calls": 7, "cost_usd": 0.02},
    ]  # fmt: skip
    _write(path, "".join(json.dumps(e) + "\n" for e in events))


SWE_ROUNDS = (
    "s94-r1\tmercury-only\t9296 ?               RESOLVED   1.2 min   20 calls $0.011"
    " | 1 file changed, 4 insertions(+) | f2p ['ok'] p2p_not_ok 0/20 | ran 70s\n"
    "s94-r1\tmini\t9296 ?               not      mini   16 calls $0.003 Submitted"
    " | 1 file changed | f2p ['FAIL'] p2p_not_ok 0/20 | ran 27s\n"
    "s94-r1\tbalance\tstart $25.57\n"
)


def _swebench_tree(tmp_path: Path) -> Path:
    """A tiny swebench run tree and results file: one harness arm, one mini arm."""
    runs = tmp_path / "runs"
    harness = runs / "swebench-runs" / "s94-r1-mercury-only"
    _harness_trace(harness / "traces" / "9296" / "run-1.jsonl")
    _write(harness / "9296.log", "console\n")
    mini = runs / "swebench-runs" / "s94-r1-mini"
    traj = {"info": {"config": {"model": {"model_name": "openrouter/z-ai/glm-5.3-flash"}}}}
    _write(mini / "mini" / "9296.traj.json", json.dumps(traj))
    _write(mini / "mini-9296.log", "PATH=/opt/homebrew/bin find / listing\n")
    return runs


def _build_swebench(tmp_path, monkeypatch, out_name="out"):
    runs = _swebench_tree(tmp_path)
    repo = tmp_path / "repo"
    _write(repo / "results" / "s94-harness-compare.txt", SWE_ROUNDS)
    _write(repo / "LICENSE", "Apache\n")
    monkeypatch.setattr(rs, "RUNS", runs)
    monkeypatch.setattr(rs, "REPO", repo)
    monkeypatch.setattr(rs, "GOOSE_DB", tmp_path / "no-goose.db")
    fake = type("G", (), {"questions": staticmethod(lambda: {"deadbeef-0000": {}})})
    monkeypatch.setattr(rs, "_gaia", lambda: fake)
    ident = rs.Identity(home=tmp_path / "Users" / IDENT_HOME)
    out = tmp_path / out_name
    return out, rs.build("swebench", out, ident)


def _index(out: Path) -> dict:
    rows = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    return {(r["run_folder"], r["task_id"]): r for r in rows}


def test_a_swebench_build_indexes_a_harness_run_and_an_external_run(tmp_path, monkeypatch):
    out, manifest = _build_swebench(tmp_path, monkeypatch)
    rows = _index(out)
    assert set(rows) == {("s94-r1-mercury-only", "9296"), ("s94-r1-mini", "9296")}
    assert all(list(r) == rs.INDEX_FIELDS for r in rows.values())
    h = rows["s94-r1-mercury-only", "9296"]
    assert h | {"trace_file": None} == {
        "suite": "swebench", "run_folder": "s94-r1-mercury-only", "agent": "harness",
        "model": "inception/mercury-2.5", "task_id": "9296", "resolved": True,
        "steps_passed": None, "steps_total": None, "cost_usd": 0.011, "calls": 20,
        "prompt_tokens": 1000, "completion_tokens": 200, "cached_tokens": 600, "wall_s": 70.0,
        "trace_file": None, "patch_file": None, "harness_ref": "abc1234", "harness_dirty": False,
    }  # fmt: skip
    assert h["trace_file"] == "traces/s94-r1-mercury-only/traces/9296/run-1.jsonl"
    m = rows["s94-r1-mini", "9296"]
    assert (m["agent"], m["resolved"], m["calls"], m["wall_s"]) == ("mini", False, 16, 27.0)
    assert m["model"] == "z-ai/glm-5.3-flash"  # read from the dropped trajectory
    assert m["trace_file"] is None and m["harness_ref"] is None and m["prompt_tokens"] is None
    assert any(e["path"] == "index.jsonl" for e in manifest.copied)


def test_external_raw_logs_are_left_out_and_listed_in_the_manifest(tmp_path, monkeypatch):
    out, _ = _build_swebench(tmp_path, monkeypatch)
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert "traces/s94-r1-mercury-only/9296.log" in on_disk  # the harness's own log stays
    assert not [p for p in on_disk if "mini" in p.replace("mercury", "")]
    assert "opt/homebrew" not in "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["dropped"] == [
        {"path": "traces/s94-r1-mini/mini-9296.log", "reason": "external-agent raw log (D3)"},
        {"path": "traces/s94-r1-mini/mini/9296.traj.json", "reason": "external-agent raw log (D3)"},
    ]
    assert saved["refused"] == []


def test_the_card_carries_the_schema_loading_and_event_sections(tmp_path, monkeypatch):
    out, _ = _build_swebench(tmp_path, monkeypatch)
    card = (out / "README.md").read_text()
    for heading in ("## Files", "## index.jsonl schema", "## Loading", "## Trace events"):
        assert heading in card
    assert all(f"| `{field}` |" in card for field in rs.INDEX_FIELDS)
    assert "read_json('index.jsonl')" in card and 'load_dataset("json"' in card
    assert "| `run_start` |" in card and "| `llm_call` |" in card
    assert card.startswith("---\n")  # the gate's front matter still leads


def test_the_scicode_index_keeps_steps_and_nulls_what_an_external_agent_did_not_record(
    tmp_path, monkeypatch
):
    runs = tmp_path / "runs"
    _harness_trace(runs / "scicode-runs" / "p1-sci-r1" / "traces" / "1" / "run-1.jsonl")
    _write(runs / "scicode-runs" / "p1-sci-r1" / "1" / "solution.py", "def f(): pass\n")
    _write(runs / "scicode-runs" / "s109-sci-codex" / "1" / "solution.py", "def g(): pass\n")
    _write(runs / "scicode-runs" / "s109-sci-codex" / "agent-out" / "1" / "stdout.log", "x\n")
    scores = "   1 CG   SOLVED steps 1/1    2.5 min    8 calls $0.003\n\nproblems 1/15\n"
    _write(runs / "scicode-runs" / "p1-sci-r1.out", scores)
    zero = "   1 CG   not    steps 0/1    0.0 min    0 calls $0.000\n"
    _write(runs / "scicode-runs" / "s109-sci-codex.out", zero)
    repo = tmp_path / "repo"
    _write(repo / "LICENSE", "Apache\n")
    monkeypatch.setattr(rs, "RUNS", runs)
    monkeypatch.setattr(rs, "REPO", repo)
    monkeypatch.setattr(rs, "GOOSE_DB", tmp_path / "none.db")
    fake = type("G", (), {"questions": staticmethod(lambda: {"deadbeef-0000": {}})})
    monkeypatch.setattr(rs, "_gaia", lambda: fake)
    out = tmp_path / "out"
    rs.build("scicode", out, rs.Identity(home=tmp_path / "Users" / IDENT_HOME))
    rows = _index(out)
    h, c = rows["p1-sci-r1", "1"], rows["s109-sci-codex", "1"]
    assert (h["resolved"], h["steps_passed"], h["steps_total"]) == (True, 1, 1)
    assert (h["wall_s"], h["calls"], h["cost_usd"]) == (150.0, 8, 0.003)
    assert h["patch_file"] == "traces/p1-sci-r1/1/solution.py"
    assert (c["resolved"], c["steps_passed"], c["agent"]) == (False, 0, "codex")
    assert (c["calls"], c["cost_usd"], c["wall_s"], c["trace_file"]) == (None, None, None, None)
    assert c["patch_file"] == "traces/s109-sci-codex/1/solution.py"
    assert not (out / "traces/s109-sci-codex/agent-out/1/stdout.log").exists()
