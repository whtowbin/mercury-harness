"""testcmd.py against real django unittest-verbose fixtures and a real
`python -m unittest` process — see `agent/testcmd.py`'s module docstring for
why this adapter exists."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent import failsig, testcmd
from agent.phases import finalcheck

FIXTURES = Path(__file__).parent / "fixtures" / "testcmd"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# build_argv
# ---------------------------------------------------------------------------


def test_build_argv_substitutes_a_targets_placeholder_in_the_middle():
    argv = testcmd.build_argv("python tests/runtests.py {targets} --verbosity 2", ["pagination"])
    assert argv == ["python", "tests/runtests.py", "pagination", "--verbosity", "2"]


def test_build_argv_appends_targets_when_there_is_no_placeholder():
    argv = testcmd.build_argv("python -m pytest -x", ["tests/test_x.py"])
    assert argv == ["python", "-m", "pytest", "-x", "tests/test_x.py"]


def test_build_argv_drops_the_placeholder_entirely_for_empty_targets():
    argv = testcmd.build_argv("python tests/runtests.py {targets} --verbosity 2", [])
    assert argv == ["python", "tests/runtests.py", "--verbosity", "2"]


# ---------------------------------------------------------------------------
# parse_unittest against the real django fixtures
# ---------------------------------------------------------------------------


def test_parse_unittest_reads_a_clean_pass_as_all_ok():
    statuses = testcmd.parse_unittest(_read("django_pass.txt"))
    assert statuses["pagination.tests.PaginationTests.test_paginator_iteration"] == "ok"
    assert len(statuses) == 21
    assert all(status == "ok" for status in statuses.values())


def test_parse_unittest_keys_a_docstring_test_by_its_method_name_not_its_docstring():
    statuses = testcmd.parse_unittest(_read("django_pass.txt"))
    # test_page_getitem prints its docstring alone on the next line; the id
    # must still be the method's own name, not the docstring text.
    assert statuses["pagination.tests.ModelPaginationTests.test_page_getitem"] == "ok"
    assert statuses["pagination.tests.PaginationTests.test_get_page"] == "ok"


def test_parse_unittest_reads_a_real_error():
    statuses = testcmd.parse_unittest(_read("django_error.txt"))
    assert statuses["pagination.tests.PaginationTests.test_paginator_iteration"] == "ERROR"


def test_parse_unittest_credits_a_subtest_failure_to_the_closing_section_not_the_line():
    statuses = testcmd.parse_unittest(_read("django_subtest_fail.txt"))
    # The line reads "...rfc850 (...) ... test_parsing_year_less_than_70 (...)
    # ... ok" — the trailing "ok" belongs to the second test named on the
    # line, and the first only gets its real status from the closing
    # "FAIL:" section.
    assert statuses["utils_tests.test_http.HttpDateProcessingTests.test_parsing_rfc850"] == "FAIL"
    assert (
        statuses["utils_tests.test_http.HttpDateProcessingTests.test_parsing_year_less_than_70"]
        == "ok"
    )


def test_parse_unittest_keys_an_import_failure_by_the_loader_pseudo_class():
    statuses = testcmd.parse_unittest(_read("django_bad_label.txt"))
    assert statuses["unittest.loader._FailedTest.no_such_module"] == "ERROR"
    assert len(statuses) == 1


# ---------------------------------------------------------------------------
# pytest_shaped: the existing pytest-side parsers read its appended block
# correctly, for each fixture.
# ---------------------------------------------------------------------------


def test_pytest_shaped_pass_fixture_reads_as_a_clean_pytest_run():
    exit_code, out = testcmd.pytest_shaped(_read("django_pass.txt"), 0, 1.23, "/WORKSPACE")
    assert exit_code == 0
    summary = failsig.pytest_summary(out, exit_code)
    assert failsig.failed_count(summary) == 0
    ids, count, where = failsig.failure_details(out)
    assert ids == frozenset()
    assert count == 0
    assert where == ""
    assert failsig.failure_digest(out) == []
    assert finalcheck.failing_tests(out) == []


def test_pytest_shaped_error_fixture_reads_as_one_failing_id_with_a_workspace_frame():
    exit_code, out = testcmd.pytest_shaped(_read("django_error.txt"), 1, 1.23, "/WORKSPACE")
    assert exit_code == 1
    summary = failsig.pytest_summary(out, exit_code)
    assert failsig.failed_count(summary) == 1
    ids, count, where = failsig.failure_details(out)
    assert ids == frozenset({"pagination.tests.PaginationTests.test_paginator_iteration"})
    assert count == 1
    assert where.startswith("tests/pagination/tests.py:302 in test_paginator_iteration")
    assert "/PYTHON" not in where
    digest = failsig.failure_digest(out)
    assert digest == [
        (
            "pagination.tests.PaginationTests.test_paginator_iteration",
            "tests/pagination/tests.py:302 in test_paginator_iteration -- "
            "TypeError: 'Paginator' object is not iterable",
        )
    ]
    assert finalcheck.failing_tests(out) == [
        "pagination.tests.PaginationTests.test_paginator_iteration"
    ]


def test_pytest_shaped_subtest_fixture_fails_only_the_test_whose_subtest_failed():
    exit_code, out = testcmd.pytest_shaped(_read("django_subtest_fail.txt"), 1, 1.23, "/WORKSPACE")
    assert exit_code == 1
    ids, count, where = failsig.failure_details(out)
    assert ids == frozenset({"utils_tests.test_http.HttpDateProcessingTests.test_parsing_rfc850"})
    assert count == 1
    assert where.startswith("tests/utils_tests/test_http.py:340 in test_parsing_rfc850")
    assert finalcheck.failing_tests(out) == [
        "utils_tests.test_http.HttpDateProcessingTests.test_parsing_rfc850"
    ]


def test_pytest_shaped_bad_label_fixture_is_exit_2_and_keeps_the_library_frame_absolute():
    exit_code, out = testcmd.pytest_shaped(_read("django_bad_label.txt"), 1, 0.05, "/WORKSPACE")
    assert exit_code == 2
    ids, count, where = failsig.failure_details(out)
    assert ids == frozenset({"unittest.loader._FailedTest.no_such_module"})
    assert count == 1
    # The frame is stdlib, outside /WORKSPACE, so it stays absolute — that's
    # what tells failure_details it isn't project code.
    assert where.startswith("/PYTHON/lib/python3.8/unittest/loader.py:154 in loadTestsFromName")
    assert finalcheck.failing_tests(out) == ["unittest.loader._FailedTest.no_such_module"]


def test_pytest_shaped_does_not_let_unittests_own_failed_banner_pollute_failed_ids():
    # "FAILED (errors=1)" is unittest's own closing line and would otherwise
    # satisfy FAILED_ID_RE itself, adding a bogus "(errors=1)" failing id.
    _exit_code, out = testcmd.pytest_shaped(_read("django_error.txt"), 1, 1.23, "/WORKSPACE")
    assert "(errors=1)" not in failsig.FAILED_ID_RE.findall(out)


def test_pytest_shaped_keeps_the_original_output_ahead_of_the_appended_block():
    original = _read("django_pass.txt")
    _exit_code, out = testcmd.pytest_shaped(original, 0, 1.23, "/WORKSPACE")
    assert out.startswith(original.rstrip("\n"))


# ---------------------------------------------------------------------------
# A real process: python -m unittest against a small test file, not a fixture.
# ---------------------------------------------------------------------------

_SAMPLE_MODULE = '''\
import unittest


class SampleTests(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(1, 1)

    def test_fail(self):
        self.assertEqual(1, 2)

    def test_error(self):
        raise ValueError("boom")

    def test_docstring(self):
        """A docstring so unittest prints it on its own line."""
        self.assertEqual(1, 1)

    def test_subtest(self):
        for value in (1, 2):
            with self.subTest(value=value):
                self.assertEqual(value, 1)

    @unittest.skip("not needed")
    def test_skip(self):
        pass
'''


@pytest.fixture
def sample_run(tmp_path: Path) -> tuple[str, int, float]:
    (tmp_path / "test_sample.py").write_text(_SAMPLE_MODULE)
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_sample"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    combined = proc.stdout + proc.stderr
    return combined, proc.returncode, elapsed


def test_real_unittest_process_parses_cleanly(sample_run: tuple[str, int, float], tmp_path: Path):
    combined, _returncode, _elapsed = sample_run
    statuses = testcmd.parse_unittest(combined)
    by_name = {tid.rsplit(".", 1)[-1]: status for tid, status in statuses.items()}
    assert by_name["test_pass"] == "ok"
    assert by_name["test_fail"] == "FAIL"
    assert by_name["test_error"] == "ERROR"
    assert by_name["test_docstring"] == "ok"
    assert by_name["test_subtest"] == "FAIL"
    assert by_name["test_skip"] == "skipped"


def test_real_unittest_process_agrees_with_pytest_side_parsers(
    sample_run: tuple[str, int, float], tmp_path: Path
):
    combined, returncode, elapsed = sample_run
    exit_code, out = testcmd.pytest_shaped(combined, returncode, elapsed, tmp_path)
    assert exit_code == 1  # two failures and an error
    ids, count, where = failsig.failure_details(out)
    assert ids == {
        "test_sample.SampleTests.test_fail",
        "test_sample.SampleTests.test_error",
        "test_sample.SampleTests.test_subtest",
    }
    # 2 failed + 1 error: failed_count sums the banner's counts (it read only
    # the first, 2, until this adapter's real-process test showed it).
    assert count == 3
    # A relative frame pointing back at the sample file, not an absolute one.
    assert where.startswith("test_sample.py:")
    assert not where.startswith("/")
    summary = failsig.pytest_summary(out, exit_code)
    assert failsig.failed_count(summary) == 3
    assert set(finalcheck.failing_tests(out)) == ids


def test_failed_count_sums_failures_and_errors_from_the_summary_banner():
    assert failsig.failed_count("2 failed, 1 error, 3 passed in 0.10s") == 3
    assert failsig.failed_count("1 failed") == 1
    assert failsig.failed_count("5 passed in 0.01s") == 0
    assert failsig.failed_count("nothing useful") is None


def test_failed_count_reads_the_banner_not_a_traceback_line_that_mentions_failures():
    output = "E   AssertionError: 3 failed to connect\n==== 1 failed, 2 passed in 0.04s ===="
    assert failsig.failed_count(output) == 1
