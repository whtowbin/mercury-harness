"""scripts/analysis/contamination.py against a real git repository: which gold lines count, and
when a sighting is evidence."""

import importlib.util
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
