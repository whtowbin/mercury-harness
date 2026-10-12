"""Telling the planner *why* baseline tests are red, and saying honestly what
the run did about them.

The baseline suite run held the whole pytest traceback and kept only node ids,
so a plan for "fix the failing tests" was written without seeing an assertion.
The end-of-run account also said "left alone" about tests the run had been
counted against, and the pre-run note said "will not be counted" while they
were.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agent import baseline_cache, failsig
from agent.loop import _failing_tests_context
from agent.phases.finalcheck import BaselineTests, _say_baseline
from agent.summary import render


@pytest.fixture(scope="module")
def red_project(tmp_path_factory):
    """A real pytest run, not a hand-written sample: two failing tests, one of
    them raising two frames below the assertion."""
    root = tmp_path_factory.mktemp("red")
    (root / "mod.py").write_text(
        "def f(x):\n    return g(x)\ndef g(x):\n    raise ValueError('bad %s' % x)\n"
    )
    (root / "test_mod.py").write_text(
        "from mod import f\ndef test_a():\n    assert f(1)\ndef test_b():\n    assert 1 == 2\n"
    )
    return root


def _pytest(root) -> str:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            ".",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout


def test_digest_names_the_deepest_project_frame_and_error(red_project):
    digest = dict(failsig.failure_digest(_pytest(red_project)))
    assert digest["test_a"] == "mod.py:4 in g -- ValueError: bad 1"


def test_digest_of_an_assertion_points_at_the_test_line(red_project):
    digest = dict(failsig.failure_digest(_pytest(red_project)))
    assert digest["test_b"] == "test_mod.py:5 in test_b -- assert 1 == 2"


def test_a_collection_error_reports_the_import_not_importlib(tmp_path):
    (tmp_path / "test_imp.py").write_text("import nosuchmod\ndef test_c():\n    pass\n")
    digest = failsig.failure_digest(_pytest(tmp_path))
    assert digest == [
        (
            "ERROR collecting test_imp.py",
            "test_imp.py:1 in <module> -- ModuleNotFoundError: No module named 'nosuchmod'",
        )
    ]


def test_a_passing_suite_has_an_empty_digest():
    assert failsig.failure_digest("3 passed in 0.01s") == []
    assert failsig.failure_digest("") == []


def test_digest_is_capped_in_count_and_size():
    sections = "".join(f"____ test_{i} ____\nm.py:{i}: in f\nE   Boom {i}\n" for i in range(20))
    digest = failsig.failure_digest(sections)
    assert len(digest) == failsig.MAX_DIGEST_FAILURES
    text = failsig.format_digest(digest, total=20)
    assert "(+15 more)" in text
    assert len(text) < failsig.MAX_DIGEST_CHARS + 40


def test_the_planner_is_shown_what_each_failure_says():
    baseline = BaselineTests(
        failures=["test_mod.py::test_a"], digest="  - test_a: mod.py:4 in g -- ValueError: bad 1"
    )

    class Settings:
        fix_existing_failures = True

    text = _failing_tests_context(baseline, Settings())
    assert "mod.py:4 in g -- ValueError: bad 1" in text
    assert "part of the work" in text


def test_a_baseline_without_a_digest_still_lists_the_names():
    baseline = BaselineTests(failures=["test_mod.py::test_a"])

    class Settings:
        fix_existing_failures = True

    text = _failing_tests_context(baseline, Settings())
    assert "test_mod.py::test_a" in text
    assert "What each reports" not in text


class _Trace:
    def __init__(self):
        self.said = []

    def say(self, text, **_):
        self.said.append(text)


class _Settings:
    def __init__(self, fix):
        self.fix_existing_failures = fix


def test_the_pre_run_note_says_it_will_try_when_failures_are_the_work():
    trace = _Trace()
    _say_baseline(BaselineTests(failures=["a::b"]), _Settings(True), trace)
    assert "will try to fix them" in trace.said[0]
    assert "not be counted" not in trace.said[0]


def test_the_pre_run_note_says_not_counted_only_when_they_are_out_of_scope():
    trace = _Trace()
    _say_baseline(BaselineTests(failures=["a::b"]), _Settings(False), trace)
    assert "will not be counted against it" in trace.said[0]


def test_an_old_cache_with_no_digest_is_reused_only_when_nothing_failed(tmp_path):
    baseline_cache.save(tmp_path, "s", ["a::b"], False, "  - a: m.py:1 in f -- E")
    loaded = baseline_cache.load(tmp_path, "s")
    assert loaded is not None
    assert loaded["digest"] == "  - a: m.py:1 in f -- E"
    path = tmp_path / ".harness" / baseline_cache.CACHE_FILE
    path.write_text(
        '{"workspace_state": "s", "failures": ["a::b"], "collection_broken": false}',
        encoding="utf-8",
    )
    reloaded = baseline_cache.load(tmp_path, "s")
    assert reloaded is not None
    assert "digest" not in reloaded


def _summary(**check):
    return {
        "ok": False,
        "changed_files": [],
        "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
        "final_check": {
            "ok": False,
            "tests_ran": True,
            "tests_passed": False,
            "new_failures": [],
            **check,
        },
        "plan": {"steps": []},
        "suggested_next_step": "",
        "quick_answer": "",
        "error": "",
    }


def test_summary_does_not_claim_it_left_alone_tests_it_was_asked_to_fix():
    text = render(
        "g",
        _summary(
            pre_existing_failures=["t.py::test_a"],
            fix_pre_existing=True,
            pre_existing_why="  - test_a: mod.py:4 in g -- ValueError: bad 1",
        ),
    )
    assert "left alone" not in text
    assert "not fixed" in text
    assert "mod.py:4 in g -- ValueError: bad 1" in text


def test_summary_still_says_left_alone_when_they_were_out_of_scope():
    text = render("g", _summary(pre_existing_failures=["t.py::test_a"], fix_pre_existing=False))
    assert "left alone" in text


def test_run_tests_traces_which_test_failed_and_where(tmp_path):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.tools.checks import run_tests
    from agent.trace import Trace
    from agent.workspace import Workspace

    trace = Trace(str(tmp_path / "t"), echo=False)
    ctx = ToolContext(
        workspace=Workspace(tmp_path), approver=Approver(auto_approve=True), trace=trace
    )
    (tmp_path / "mod.py").write_text("def f(x):\n    raise ValueError('bad')\n")
    (tmp_path / "test_mod.py").write_text("from mod import f\ndef test_a():\n    f(1)\n")
    run_tests(ctx, path="test_mod.py")
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    event = next(e for e in events if e["kind"] == "run_tests")
    assert event["failing"] == ["test_mod.py::test_a"]
    assert event["where"].startswith("mod.py:2 in f")


# -- failed_count: errors are failures too ---------------------------------


def test_failed_count_treats_errors_as_things_that_went_wrong():
    """A collection error reports `1 error` and no failures. Reading that as
    zero says the suite improved at the moment an edit stopped it running."""
    assert failsig.failed_count("1 error in 0.03s") == 1
    assert failsig.failed_count("3 errors in 0.04s") == 3
    assert failsig.failed_count("2 failed, 1 passed in 0.02s") == 2


def test_failed_count_is_unknown_rather_than_zero_when_it_cannot_tell():
    assert failsig.failed_count("pytest exited 2") is None
    assert failsig.failed_count("") is None
    assert failsig.failed_count("7 passed in 0.01s") == 0


def test_the_summary_is_the_banner_not_a_traceback_line_that_mentions_counts():
    """The final gate used to scan for any line with a digit and one of four
    keywords, taking the LAST such line, so a traceback line printed after the
    summary could stand in for it."""
    output = "1 failed, 2 passed in 0.03s\nE   AssertionError: 3 failed to connect\n"
    assert failsig.pytest_summary(output, 1) == "1 failed, 2 passed in 0.03s"


def test_the_summary_handles_banners_quiet_output_and_no_tests():
    assert (
        failsig.pytest_summary("========== 9 passed in 0.03s ==========\n", 0)
        == "9 passed in 0.03s"
    )
    assert failsig.pytest_summary("2 passed in 0.01s", 0) == "2 passed in 0.01s"
    assert failsig.pytest_summary("no tests ran in 0.01s", 5) == "no tests ran in 0.01s"
    assert (
        failsig.pytest_summary("1 failed, 2 passed, 1 warning in 1.20s (0:00:01)", 1)
        == "1 failed, 2 passed, 1 warning in 1.20s (0:00:01)"
    )


def test_the_summary_falls_back_to_the_exit_code():
    assert failsig.pytest_summary("total nonsense", 3) == "pytest exited 3"
    assert failsig.pytest_summary("", 0) == "pytest exited 0"
