"""Making the whole-suite test run itself faster, and safer to parallelise.

Three independent changes, each measured before being added:

1. `PYTHONDONTWRITEBYTECODE=1` on every subprocess the agent runs.
2. `-p no:randomly` alongside the existing `-p no:cacheprovider`.
3. `COVERAGE_CORE=sysmon` on the coverage-wrapped suite run (coverage 7.4+ on
   Python 3.12+): measured 52.5s -> 47.9s on a real 76-test suite.
4. An opt-in `test_parallel` setting that runs the whole-suite baseline and
   gate checks under `uv run --with pytest-xdist pytest -n auto`, mutually
   exclusive with coverage measurement in the same invocation (see
   `_run_suite`'s own docstring for why that pairing corrupts rather than
   speeds up), because turning it on unconditionally is not free: it broke
   two of this project's own tests that shared a fixture directory on disk,
   with no change to the code under test.
"""

from __future__ import annotations

from agent import toolchain
from agent.config import Settings
from agent.phases import finalcheck

# -- the environment every subprocess gets ---------------------------------


def test_bytecode_writes_are_disabled_by_default():
    """Every subprocess the agent runs is one-shot: a verify command, a test
    run, a step's shell call. None of them benefit from a .pyc cache that
    outlives the call, and it litters __pycache__ through a workspace that
    is not the agent's to leave artifacts in."""
    assert toolchain.clean_env()["PYTHONDONTWRITEBYTECODE"] == "1"


def test_an_explicit_bytecode_setting_is_not_overridden(monkeypatch):
    """setdefault, not an unconditional set -- a workspace or CI environment
    that deliberately wants bytecode written keeps getting it."""
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    assert toolchain.clean_env()["PYTHONDONTWRITEBYTECODE"] == "0"


def test_run_can_layer_extra_env_without_losing_the_workspace_s_own(monkeypatch, tmp_path):
    """extra_env adds to the call, it does not replace what workspace_env
    already set up -- the coverage-wrapped suite run still needs the
    workspace's own virtualenv on PATH as well as COVERAGE_CORE."""
    seen = {}

    def fake_run_capturing(argv, cwd, timeout, env):
        seen["env"] = env
        from types import SimpleNamespace

        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(toolchain, "resolve", lambda tool, root: [tool])
    monkeypatch.setattr(toolchain, "run_capturing", fake_run_capturing)
    toolchain.run("pytest", ["-q"], tmp_path, extra_env={"COVERAGE_CORE": "sysmon"})
    assert seen["env"]["COVERAGE_CORE"] == "sysmon"
    assert seen["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


# -- pytest flags: deterministic ordering, no cache I/O --------------------


def test_pytest_flags_disable_cacheprovider_and_randomly():
    assert "-p" in finalcheck._PYTEST_ARGS
    joined = " ".join(finalcheck._PYTEST_ARGS)
    assert "no:cacheprovider" in joined
    assert "no:randomly" in joined


# -- the coverage-wrapped run asks for the cheaper tracer -------------------


def test_the_coverage_run_requests_the_sysmon_core(monkeypatch, tmp_path):
    """coverage.py's default tracer is sys.settrace, instrumenting every
    line; sys.monitoring (Python 3.12+, coverage 7.4+) is built for this and
    costs far less per line. An unrecognised value degrades to a stderr
    warning and the default tracer rather than failing outright, so an older
    coverage.py is unaffected rather than broken."""
    calls = []
    monkeypatch.setattr(
        finalcheck.toolchain, "run", lambda *a, **k: (calls.append(k), (0, "1 passed", ""))[1]
    )

    class Ctx:
        class workspace:
            root = tmp_path

    finalcheck._run_suite(Ctx(), with_coverage=True)
    assert calls[0].get("extra_env", {}).get("COVERAGE_CORE") == "sysmon"


# -- parallel is opt-in, and never combined with coverage -------------------


def test_parallel_runs_pytest_under_xdist_when_requested(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        finalcheck.toolchain,
        "run",
        lambda tool, args, *a, **k: (calls.append((tool, args)), (0, "1 passed", ""))[1],
    )

    class Ctx:
        class workspace:
            root = tmp_path

    finalcheck._run_suite(Ctx(), parallel=True)
    tool, args = calls[0]
    assert tool == "uv"
    assert "pytest-xdist" in args
    assert "-n" in args and "auto" in args


def test_coverage_takes_priority_over_parallel_in_the_same_call(monkeypatch, tmp_path):
    """Plain `coverage run -m pytest -n auto` has every xdist worker write
    the same .coverage file at once and corrupt it; pytest-cov exists to
    coordinate that and this path does not use it. Coverage gates `ok`, so a
    wrong answer there is worse than a slow one -- it wins the conflict."""
    calls = []
    monkeypatch.setattr(
        finalcheck.toolchain,
        "run",
        lambda tool, args, *a, **k: (calls.append((tool, args)), (0, "1 passed", ""))[1],
    )

    class Ctx:
        class workspace:
            root = tmp_path

    finalcheck._run_suite(Ctx(), with_coverage=True, parallel=True)
    _tool, args = calls[0]
    assert "coverage" in args
    assert "pytest-xdist" not in args


def test_a_failed_xdist_bootstrap_falls_back_to_the_plain_suite(monkeypatch, tmp_path):
    """No network to fetch pytest-xdist must not cost the run its actual
    pass/fail signal, the same contract the coverage wrapper already has."""
    calls = []

    def fake_run(tool, args, *a, **k):
        calls.append(tool)
        if tool == "uv":
            return 1, "", "error: failed to resolve pytest-xdist"
        return 0, "1 passed", ""

    monkeypatch.setattr(finalcheck.toolchain, "run", fake_run)

    class Ctx:
        class workspace:
            root = tmp_path

    code, _out, _err, _recorded = finalcheck._run_suite(Ctx(), parallel=True)
    assert calls == ["uv", "pytest"], "must fall back to the plain suite"
    assert code == 0


def test_test_parallel_defaults_off():
    """Off by default: xdist needs test isolation a suite may not have --
    turning it on against this project's own suite broke two tests sharing a
    fixture directory, with no change to the code being tested."""
    assert Settings(workspace=".", goal="g").test_parallel is False
