"""A step's declared `writes` are enforced, independently of its `verify`.

The trace this exists for: a step named "Verify README summary" called
finish_step on its very first tool call, claiming the README's summary
sections were already correct, without ever opening the file. Its `writes`
was actually empty in that trace (a separate, still-open gap — see
DESIGN-LOG.md) but the mechanism these tests cover is what closes the case
where a step *does* declare `writes` and skips writing to it anyway: a plan
step's verify command tells you the work compiles or lints clean, never that
the work described by `writes` was done at all when there is no verify
command, or the verify command checks something else entirely (see
test_plan_gate.py's `test_writes_take_priority_over_files_for_the_verify_target`
for that half of the same bug).
"""

from __future__ import annotations

from agent.phases.execute import (
    _undeclared_shadow_files,
    _unwritten_declared_files,
    _verify_step,
)
from agent.phases.plan import Step


class _Baseline:
    def __init__(self, snapshots):
        self.snapshots = snapshots


class _Ctx:
    def __init__(self, root, touched, baseline_snapshots=None):
        self.touched = touched
        self.workspace = type("W", (), {"root": root})()
        # Default: every touched path counts as pre-existing (an edit, not a
        # new file), so _undeclared_shadow_files stays inert for every test
        # in this file that doesn't explicitly ask otherwise.
        self.baseline = _Baseline(
            baseline_snapshots if baseline_snapshots is not None else {p: "x" for p in touched}
        )


class _Trace:
    def __init__(self):
        self.events = []
        self.said = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))

    def say(self, message):
        self.said.append(message)


class _Settings:
    verify_steps = True


def _step(writes=(), verify="") -> Step:
    return Step(id="s", title="t", intent="i", acceptance="a", writes=list(writes), verify=verify)


# -- _unwritten_declared_files ---------------------------------------------


def test_a_declared_write_never_touched_is_reported(tmp_path):
    ctx = _Ctx(tmp_path, touched=[])
    assert _unwritten_declared_files(ctx, _step(writes=["README.md"])) == ["README.md"]


def test_a_declared_write_that_was_touched_is_not_reported(tmp_path):
    ctx = _Ctx(tmp_path, touched=["README.md"])
    assert _unwritten_declared_files(ctx, _step(writes=["README.md"])) == []


def test_a_step_with_no_declared_writes_is_never_flagged(tmp_path):
    """Scoped to `writes`, not `files`: a review-only step reads files
    without editing them by design, and must not be required to."""
    ctx = _Ctx(tmp_path, touched=[])
    assert _unwritten_declared_files(ctx, _step(writes=[])) == []


def test_only_the_untouched_subset_is_reported(tmp_path):
    ctx = _Ctx(tmp_path, touched=["a.py"])
    missing = _unwritten_declared_files(ctx, _step(writes=["a.py", "b.py"]))
    assert missing == ["b.py"]


# -- _verify_step: the feedback that drives a retry -------------------------


def test_an_empty_verify_no_longer_excuses_an_unwritten_declared_file(tmp_path):
    """Before this existed, verify="" meant the step was trusted
    unconditionally -- this is the loophole the traced README step used."""
    ctx = _Ctx(tmp_path, touched=[])
    step = _step(writes=["README.md"], verify="")
    problem = _verify_step(ctx, _Trace(), step, _Settings())
    assert "README.md" in problem
    assert "declared it would write" in problem


def test_a_step_that_wrote_its_declared_file_and_has_no_verify_passes(tmp_path):
    ctx = _Ctx(tmp_path, touched=["README.md"])
    step = _step(writes=["README.md"], verify="")
    assert _verify_step(ctx, _Trace(), step, _Settings()) == ""


def test_a_step_with_no_writes_declared_is_unaffected_by_this_check(tmp_path):
    """The pre-existing behaviour for a step that never declared writes: an
    empty verify is still nothing to run, still not a failure."""
    ctx = _Ctx(tmp_path, touched=[])
    step = _step(writes=[], verify="")
    assert _verify_step(ctx, _Trace(), step, _Settings()) == ""


def test_the_writes_problem_is_reported_even_when_verify_passes(tmp_path, monkeypatch):
    """A step can lint clean while still never having touched the file it
    was supposed to write -- verify passing is not evidence of that."""
    from agent.phases import execute

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (True, "All checks passed!"))
    ctx = _Ctx(tmp_path, touched=[])
    step = _step(writes=["README.md"], verify="uvx --quiet ruff check other.py")
    problem = _verify_step(ctx, _Trace(), step, _Settings())
    assert "README.md" in problem


def test_the_writes_problem_survives_an_environmental_verify_failure(tmp_path, monkeypatch):
    """An unrunnable verify command still shouldn't hide a genuinely
    unwritten declared file -- they are independent findings."""
    from agent.phases import execute

    monkeypatch.setattr(
        execute, "run_verify", lambda ctx, cmd: (False, "dyld[1]: Library not loaded: @rpath/x")
    )
    monkeypatch.setattr(execute.envfail, "repair", lambda root: (False, "nothing to sync"))
    ctx = _Ctx(tmp_path, touched=[])
    step = _step(writes=["README.md"], verify="uv run pytest -q")
    problem = _verify_step(ctx, _Trace(), step, _Settings())
    assert "README.md" in problem
    assert "declared it would write" in problem


def test_both_problems_are_reported_together_when_both_apply(tmp_path, monkeypatch):
    from agent.phases import execute

    monkeypatch.setattr(execute, "run_verify", lambda ctx, cmd: (False, "F821 undefined name"))
    ctx = _Ctx(tmp_path, touched=[])
    step = _step(writes=["README.md"], verify="uvx --quiet ruff check other.py")
    problem = _verify_step(ctx, _Trace(), step, _Settings())
    assert "README.md" in problem
    assert "verify command failed" in problem


# -- _undeclared_shadow_files -----------------------------------------------
# The trace this exists for: a step declared writes=["coin.py", "sprites.py"],
# then while chasing an import mid-step, created six files including a second
# sprites.py inside a tests/ directory it invented. Every run afterward
# silently ran two divergent copies of the test suite, and fixes to the real
# sprites.py never touched the shadow. Nothing flagged the fork.


def test_a_new_file_shadowing_an_existing_one_is_reported(tmp_path):
    (tmp_path / "sprites.py").write_text("def create_coin_sprite(): ...\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "sprites.py").write_text("def create_coin_sprite(): ...\n")
    ctx = _Ctx(
        tmp_path, touched=["tests/sprites.py"], baseline_snapshots={"tests/sprites.py": None}
    )  # None = did not exist before
    flagged = _undeclared_shadow_files(ctx, _step(writes=["coin.py"]), touched_before=frozenset())
    assert len(flagged) == 1
    assert "tests/sprites.py" in flagged[0]
    assert "sprites.py" in flagged[0]


def test_a_new_file_declared_in_writes_is_not_flagged(tmp_path):
    """A step that plans to add a genuinely new file, and says so, is fine --
    only an *undeclared* new file that shadows an existing one is suspicious."""
    (tmp_path / "sprites.py").write_text("...\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "sprites.py").write_text("...\n")
    ctx = _Ctx(
        tmp_path, touched=["tests/sprites.py"], baseline_snapshots={"tests/sprites.py": None}
    )
    flagged = _undeclared_shadow_files(
        ctx, _step(writes=["tests/sprites.py"]), touched_before=frozenset()
    )
    assert flagged == []


def test_a_new_file_with_a_unique_name_is_not_flagged(tmp_path):
    """Creating one small new helper file nobody thought to declare is
    normal and not the shape this check exists for."""
    (tmp_path / "tests").mkdir()
    ctx = _Ctx(
        tmp_path,
        touched=["tests/coin_helpers.py"],
        baseline_snapshots={"tests/coin_helpers.py": None},
    )
    flagged = _undeclared_shadow_files(ctx, _step(writes=[]), touched_before=frozenset())
    assert flagged == []


def test_editing_an_existing_file_is_never_flagged(tmp_path):
    """The baseline records real prior content -- this is an edit, not a new
    file, even though its basename would otherwise collide with itself."""
    (tmp_path / "sprites.py").write_text("old content\n")
    ctx = _Ctx(tmp_path, touched=["sprites.py"])  # default baseline: pre-existing
    flagged = _undeclared_shadow_files(ctx, _step(writes=[]), touched_before=frozenset())
    assert flagged == []


def test_multiple_init_files_are_never_flagged(tmp_path):
    """One __init__.py per package is completely ordinary."""
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    ctx = _Ctx(
        tmp_path, touched=["tests/__init__.py"], baseline_snapshots={"tests/__init__.py": None}
    )
    flagged = _undeclared_shadow_files(ctx, _step(writes=[]), touched_before=frozenset())
    assert flagged == []


def test_touched_before_the_step_is_not_considered_new(tmp_path):
    """A file another (earlier) step already created is not this step's
    shadow file, even if its name collides with something."""
    (tmp_path / "sprites.py").write_text("...\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "sprites.py").write_text("...\n")
    ctx = _Ctx(
        tmp_path, touched=["tests/sprites.py"], baseline_snapshots={"tests/sprites.py": None}
    )
    flagged = _undeclared_shadow_files(
        ctx, _step(writes=[]), touched_before=frozenset({"tests/sprites.py"})
    )
    assert flagged == []


def test_verify_step_reports_a_shadow_file_as_a_problem(tmp_path):
    (tmp_path / "sprites.py").write_text("...\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "sprites.py").write_text("...\n")
    ctx = _Ctx(
        tmp_path,
        touched=["coin.py", "tests/sprites.py"],
        baseline_snapshots={"coin.py": None, "tests/sprites.py": None},
    )
    step = _step(writes=["coin.py"], verify="")
    problem = _verify_step(ctx, _Trace(), step, _Settings(), touched_before=frozenset())
    assert "tests/sprites.py" in problem
    assert "duplicate" in problem
