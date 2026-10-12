"""Reads do not block; writes do — and the declaration is binding.

The scheduler used to conflict on every file a step declared, so four audit
steps in a traced plan that all read game.py and all wrote tests/test_game.py
ran strictly sequentially — and would have even if they had written separate
test files, because the shared *read* alone was enough to block them.

Separating reads from writes is only safe if the declaration is enforced. A
step that declares game.py and then writes test_game.py, while its sibling is
writing test_game.py too, is the race the scheduler was avoiding by being
pessimistic. So while a step runs beside another, writing outside its declared
set is refused.
"""

from __future__ import annotations

import pytest

from agent.approval import Approver
from agent.phases.plan import Step
from agent.phases.schedule import partition_disjoint
from agent.tools import ToolContext, WriteOutsideStep
from agent.tools.fs import edit_file, write_file
from agent.trace import Trace
from agent.workspace import Workspace


def step(sid, files, writes=()):
    return Step(id=sid, title="", intent="", acceptance="", files=list(files), writes=list(writes))


# -- scheduling -----------------------------------------------------------


def test_steps_that_only_share_a_read_run_together():
    """The traced case: four audits reading game.py, each writing its own test."""
    wave = [
        step(f"audit-{n}", ["game.py", f"tests/test_{n}.py"], [f"tests/test_{n}.py"])
        for n in ("scoring", "prefs", "collision", "edges")
    ]
    assert partition_disjoint(wave, 4) == [wave], "a shared read must not block"


def test_steps_that_write_the_same_file_never_run_together():
    wave = [step("a", ["game.py"], ["game.py"]), step("b", ["game.py"], ["game.py"])]
    assert [[s.id for s in b] for b in partition_disjoint(wave, 4)] == [["a"], ["b"]]


def test_a_write_conflicts_with_a_siblings_read():
    """A reader is entitled to see the file whole, not half-rewritten."""
    wave = [step("writer", ["game.py"], ["game.py"]), step("reader", ["game.py", "t.py"], ["t.py"])]
    assert len(partition_disjoint(wave, 4)) == 2


def test_a_step_declaring_no_writes_is_safe_beside_anything():
    wave = [step("readonly", ["game.py"], []), step("writer", ["other.py"], ["other.py"])]
    assert partition_disjoint(wave, 4) == [wave]


def test_a_step_declaring_no_files_at_all_still_runs_alone():
    """Undeclared is not the same as "writes nothing": it means unknown."""
    wave = [step("mystery", [], []), step("b", ["x.py"], ["x.py"])]
    assert [[s.id for s in b] for b in partition_disjoint(wave, 4)] == [["mystery"], ["b"]]


def test_a_plan_written_before_writes_existed_behaves_as_it_did():
    """No `writes` means every declared file may be modified — the old rule."""
    wave = [step("a", ["game.py"]), step("b", ["game.py"])]
    assert len(partition_disjoint(wave, 4)) == 2
    wave = [step("a", ["a.py"]), step("b", ["b.py"])]
    assert partition_disjoint(wave, 4) == [wave]


def test_the_concurrency_limit_is_still_a_ceiling():
    wave = [step(str(n), [f"{n}.py"], [f"{n}.py"]) for n in range(5)]
    assert all(len(b) <= 2 for b in partition_disjoint(wave, 2))


# -- enforcement ----------------------------------------------------------


def _ctx(tmp_path, writable):
    root = tmp_path / "ws"
    root.mkdir(exist_ok=True)
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
        writable=writable,
    )


def test_a_parallel_step_may_write_what_it_claimed(tmp_path):
    ctx = _ctx(tmp_path, frozenset({"mine.py"}))
    assert not write_file(ctx, "mine.py", "x = 1\n").startswith("error:")
    assert (ctx.workspace.root / "mine.py").exists()


def test_a_parallel_step_may_not_write_a_siblings_file(tmp_path):
    ctx = _ctx(tmp_path, frozenset({"mine.py"}))
    with pytest.raises(WriteOutsideStep) as caught:
        write_file(ctx, "theirs.py", "x = 1\n")
    assert "may not modify theirs.py" in str(caught.value)
    assert not (ctx.workspace.root / "theirs.py").exists(), "the write must not land"


def test_editing_a_siblings_file_is_refused_too(tmp_path):
    ctx = _ctx(tmp_path, frozenset({"mine.py"}))
    (ctx.workspace.root / "theirs.py").write_text("a = 1\n")
    with pytest.raises(WriteOutsideStep):
        edit_file(ctx, "theirs.py", "a = 1", "a = 2")
    assert (ctx.workspace.root / "theirs.py").read_text() == "a = 1\n"


def test_a_sequential_step_is_not_jailed_at_all(tmp_path):
    """writable=None is the default, so a run with concurrency off is exactly
    what it was before any of this existed."""
    ctx = _ctx(tmp_path, None)
    assert not write_file(ctx, "anything.py", "x = 1\n").startswith("error:")


def test_for_step_defaults_to_no_jail(tmp_path):
    ctx = _ctx(tmp_path, None)
    assert ctx.for_step().writable is None
    assert ctx.for_step(writable=frozenset({"a.py"})).writable == frozenset({"a.py"})
