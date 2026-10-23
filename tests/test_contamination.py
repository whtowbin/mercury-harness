"""scripts/analysis/contamination.py against a real git repository: which gold lines count, and
when a sighting is evidence."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
_spec = importlib.util.spec_from_file_location(
    "contamination", REPO / "scripts" / "analysis" / "contamination.py"
)
assert _spec is not None and _spec.loader is not None
ct = importlib.util.module_from_spec(_spec)
sys.modules["contamination"] = ct
_spec.loader.exec_module(ct)

PATCH = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,2 +1,3 @@
 def f():
+    return compute_the_answer(carefully=True)
+    @decorator_used_elsewhere('already_in_repo_x')
+    x = 1
"""


def make_repo(root: Path) -> str:
    bare = root / "django.git"
    work = root / "work"
    (work / "pkg").mkdir(parents=True)
    (work / "pkg" / "core.py").write_text("def f():\n    pass\n")
    (work / "pkg" / "other.py").write_text("@decorator_used_elsewhere('already_in_repo_x')\n")

    def git(*a, cwd=work):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True)

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    base = git("rev-parse", "HEAD").stdout.strip()
    git("clone", "-q", "--bare", str(work), str(bare), cwd=root)
    return base


def test_gold_lines_drop_short_lines_and_lines_anywhere_in_the_repo(tmp_path, monkeypatch):
    base = make_repo(tmp_path)
    monkeypatch.setattr(ct, "DATA", tmp_path)
    assert ct.gold_lines(PATCH, base) == ["return compute_the_answer(carefully=True)"]


def test_a_sighting_counts_before_the_first_edit_and_not_after_writing_it():
    line = "return compute_the_answer(carefully=True)"
    before = ct.Run("r", "mini", "1", "mini", [{"text": "cat core.py", "output": f"    {line}\n"}])
    assert [x[3] for x in ct.leaks(before, [line])] == [True]

    after_edit = ct.Run(
        "r",
        "mini",
        "1",
        "mini",
        [
            {"text": "sed -i 's/pass/x/' core.py", "output": ""},
            {"text": "cat core.py", "output": line},
        ],
    )
    assert [x[3] for x in ct.leaks(after_edit, [line])] == [False]

    written = ct.Run(
        "r", "mini", "1", "mini", [{"text": f"echo '{line}' > core.py", "output": line}]
    )
    assert ct.leaks(written, [line]) == []


def test_every_iso_arm_is_recognized_with_the_longest_name_winning():
    """An arm missing from ARMS is silently skipped: §126's harness arms were never read."""
    import re

    iso_s126 = (REPO / "scripts" / "swebench" / "iso_s126.py").read_text()
    arms = re.findall(r'iso\.ARMS\["([^"]+)"\]', iso_s126)
    assert arms, "found no arms in iso_s126.py"
    for arm in arms:
        assert ct.split_dir(f"s131-dj-r1-{arm}") == ("s131-dj-r1", arm)


def _mini_traj(path: Path, command: str, output: str) -> None:
    path.parent.mkdir(parents=True)
    call = {"id": "c1", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"output": output})},
    ]
    path.write_text(json.dumps({"messages": msgs}))


def test_iso_mini_trajectories_are_read_with_round_and_arm_parsed(tmp_path, monkeypatch):
    """iso_agents.py writes <round>-<arm>/out/<task>/traj.json, not <round>/mini/*.traj.json."""
    _mini_traj(tmp_path / "s131-dj-r2-mini-m3" / "out" / "11790" / "traj.json", "ls", "x")
    _mini_traj(tmp_path / "s126-dj-r1-mini" / "out" / "11815" / "traj.json", "ls", "x")
    _mini_traj(tmp_path / "s126-dj-r1-mini" / "out" / "11815.abandoned-1" / "traj.json", "ls", "x")
    monkeypatch.setattr(ct, "RUNS", tmp_path)
    got = {(r.round, r.arm, r.task, r.agent) for r in ct.external_runs() if r.agent == "mini"}
    assert got == {
        ("s131-dj-r2", "mini-m3", "11790", "mini"),
        ("s126-dj-r1", "mini", "11815", "mini"),
    }


def test_a_curl_to_the_github_api_is_a_web_route():
    cmd = "curl -s https://api.github.com/repos/django/django/pulls/123/files"
    run = ct.Run("r", "mini-m3", "1", "mini", [{"text": cmd, "output": ""}])
    assert "web" in ct.routes(run)
    quiet = ct.Run("r", "mini-m3", "1", "mini", [{"text": "pytest tests/x.py", "output": ""}])
    assert "web" not in ct.routes(quiet)
