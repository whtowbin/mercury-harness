"""Whether the tests a run leaves behind can see the change it made.

The three checks each have a fixture that is *designed* to trip it and one that
is designed not to, run against a real pytest in a real scratch copy: mocking
the subprocess would test the bookkeeping and none of the claim.
"""

from __future__ import annotations

import textwrap

import pytest

from agent import testcheck
from agent.diffs import Baseline


def dedent(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


# -- classifying an edit to a test file -----------------------------------

BASE = dedent("""
    def test_a():
        assert f(1) == 1
        assert f(2) == 2

    def test_b():
        assert f(3) == 3
""")


def test_a_new_file_is_created_not_additive():
    assert testcheck.classify_test_edit(None, BASE) == "created"


def test_an_identical_file_is_unchanged():
    assert testcheck.classify_test_edit(BASE, BASE) == "unchanged"


def test_adding_a_test_is_additive():
    assert (
        testcheck.classify_test_edit(BASE, BASE + "\ndef test_c():\n    assert f(4) == 4\n")
        == "additive"
    )


def test_changing_an_expectation_but_keeping_the_assertions_is_a_rewrite():
    assert testcheck.classify_test_edit(BASE, BASE.replace("f(3) == 3", "f(3) == 9")) == "rewritten"


def test_deleting_a_test_is_a_reduction():
    only_a = BASE.split("def test_b")[0]
    assert testcheck.classify_test_edit(BASE, only_a) == "reduced"


def test_loosening_an_assertion_into_fewer_of_them_is_a_reduction():
    """The cheapest way to make a suite green: the test keeps its name and
    loses a check. Comparing test names alone cannot see it."""
    loosened = BASE.replace("    assert f(2) == 2\n", "")
    assert testcheck.classify_test_edit(BASE, loosened) == "reduced"


def test_pytest_raises_and_unittest_asserts_count_as_assertions():
    before = dedent("""
        import pytest
        def test_x():
            with pytest.raises(ValueError):
                f(0)
            assert f(1) == 1
    """)
    after = before.replace("            f(0)\n", "            f(0)\n").replace(
        "    assert f(1) == 1\n", ""
    )
    assert testcheck.classify_test_edit(before, after) == "reduced"


def test_a_broken_test_file_does_not_raise():
    assert testcheck.classify_test_edit(BASE, "def test_a(:\n") == "reduced"


def test_added_test_names_are_qualified_by_class():
    after = "class TestX:\n    def test_one(self):\n        assert 1\n"
    assert testcheck.added_test_names(None, after) == ["TestX::test_one"]
    assert testcheck.added_test_names(after, after) == []


# -- mutants ---------------------------------------------------------------

SOURCE = dedent("""
    LIMIT = 10

    def tier(units):
        if units < 1000:
            return 1
        return 2

    def untouched(a, b):
        return a + b
""")


def test_only_sites_on_changed_lines_inside_a_function_are_mutated():
    before = SOURCE.replace("units < 1000", "units <= 1000")
    found = testcheck.mutants_for("p.py", before, SOURCE)
    assert {m.function for m in found} == {"tier"}
    assert all(m.line == 4 for m in found)
    assert {m.kind for m in found} == {"comparison", "constant"}


def test_a_module_level_constant_is_not_mutated():
    before = SOURCE.replace("LIMIT = 10", "LIMIT = 11")
    assert testcheck.mutants_for("p.py", before, SOURCE) == []


def test_each_mutant_is_valid_python_that_differs_from_the_original():
    import ast

    found = testcheck.mutants_for("p.py", None, SOURCE)
    assert found
    for mutant in found:
        ast.parse(mutant.source)
        assert mutant.source != ast.unparse(ast.parse(SOURCE))


def test_a_file_with_a_syntax_error_yields_no_mutants():
    assert testcheck.mutants_for("p.py", None, "def f(:\n") == []


def test_the_mutant_limit_is_spread_across_the_list_not_taken_from_the_front():
    items = [testcheck.Mutant("p.py", n, "constant", "f", "") for n in range(100)]
    picked = testcheck._spread(items, 10)
    assert len(picked) == 10
    assert picked[0].line == 0 and picked[-1].line >= 80


# -- the audit, against a real pytest --------------------------------------

OLD = dedent("""
    def price(units):
        if units <= 1000:
            return units * 10
        return units * 5
""")
NEW = OLD.replace("units <= 1000", "units < 1000")  # the fix: boundary moves


def _workspace(tmp_path, source_before, source_after, tests_before, tests_after):
    """A workspace in its *finished* state, plus the Baseline a run would have
    kept: the contents each touched file had before."""
    (tmp_path / "pricing.py").write_text(source_after)
    (tmp_path / "test_pricing.py").write_text(tests_after)
    baseline = Baseline(
        snapshots={
            "pricing.py": source_before,
            "test_pricing.py": tests_before,
        }
    )
    return baseline, ["pricing.py", "test_pricing.py"]


GOOD_TEST = dedent("""
    from pricing import price

    def test_the_boundary_belongs_to_the_next_tier():
        assert price(1000) == 5000
""")
VACUOUS_TEST = dedent("""
    from pricing import price

    def test_a_small_order():
        assert price(10) == 100
""")


def test_a_test_that_fails_on_the_old_code_is_not_flagged(tmp_path):
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, GOOD_TEST)
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.ran
    assert result.added_tests_pass_on_old_code is False
    assert not any("pass against the code" in f for f in result.findings)


def test_a_test_that_passes_on_the_old_code_is_flagged(tmp_path):
    """The dominant dishonesty in the bench sweeps (15% of runs): a test
    written after the fix that asserts something the bug also satisfied."""
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, VACUOUS_TEST)
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.added_tests_pass_on_old_code is True
    assert any("do not exercise the change" in f for f in result.findings)
    assert "test_pricing.py::test_a_small_order" in result.findings[0]


def test_surviving_mutants_name_the_place_no_test_can_see(tmp_path):
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, VACUOUS_TEST)
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.mutants_tried > 0
    assert result.survivors
    assert any("pricing.py:2" in s for s in result.survivors)


def test_a_test_that_pins_the_boundary_kills_the_boundary_mutants(tmp_path):
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, GOOD_TEST)
    result = testcheck.audit(tmp_path, baseline, touched)
    assert not any("pricing.py:2 (comparison" in s for s in result.survivors)


def test_the_real_workspace_is_never_edited(tmp_path):
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, VACUOUS_TEST)
    testcheck.audit(tmp_path, baseline, touched)
    assert (tmp_path / "pricing.py").read_text() == NEW
    assert (tmp_path / "test_pricing.py").read_text() == VACUOUS_TEST


def test_a_run_that_changed_no_source_is_skipped(tmp_path):
    (tmp_path / "test_pricing.py").write_text(GOOD_TEST)
    result = testcheck.audit(
        tmp_path, Baseline(snapshots={"test_pricing.py": None}), ["test_pricing.py"]
    )
    assert not result.ran
    assert "no Python source changed" in result.skipped


def test_source_created_by_the_run_has_no_old_behaviour_to_revert_to(tmp_path):
    """Its tests failing against a deleted module says nothing."""
    baseline, touched = _workspace(tmp_path, None, NEW, None, GOOD_TEST)
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.added_tests_pass_on_old_code is None


def test_an_audit_that_hits_an_error_reports_it_instead_of_raising(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(testcheck, "copy_workspace", boom)
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, GOOD_TEST)
    result = testcheck.audit(tmp_path, baseline, touched)
    assert "disk on fire" in result.skipped


def test_the_time_budget_stops_the_mutation_loop(tmp_path):
    baseline, touched = _workspace(tmp_path, OLD, NEW, None, VACUOUS_TEST)
    result = testcheck.audit(tmp_path, baseline, touched, budget_s=0)
    assert result.mutants_tried == 0


# -- the ledger: judge a rewritten test by what it still catches ----------

STRONG = dedent("""
    from pricing import price

    def test_tiers():
        assert price(10) == 100
        assert price(1000) == 5000
        assert price(2000) == 10000
""")
LOOSENED = dedent("""
    from pricing import price

    def test_tiers():
        assert price(10) == 100
""")
REWRITTEN_SAME_POWER = STRONG.replace("price(10) == 100", "price(11) == 110")


def test_a_loosened_test_is_reported_as_catching_less(tmp_path):
    baseline, touched = _workspace(tmp_path, NEW, NEW.replace("* 5", "* 6"), STRONG, LOOSENED)
    (tmp_path / "pricing.py").write_text(NEW.replace("* 5", "* 6"))
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.test_edits == {"test_pricing.py": "reduced"}
    assert result.killed_before is not None
    assert result.killed_after is not None
    assert result.killed_after < result.killed_before
    assert any("now catch" in f for f in result.findings)


def test_a_rewrite_that_still_catches_as_much_is_not_reported(tmp_path):
    baseline, touched = _workspace(
        tmp_path, NEW, NEW.replace("* 5", "* 6"), STRONG, REWRITTEN_SAME_POWER
    )
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.test_edits == {"test_pricing.py": "rewritten"}
    assert not any("now catch" in f for f in result.findings)


def test_an_additive_edit_skips_the_before_and_after_comparison(tmp_path):
    baseline, touched = _workspace(
        tmp_path, OLD, NEW, VACUOUS_TEST, VACUOUS_TEST + GOOD_TEST.split("\n", 2)[2]
    )
    result = testcheck.audit(tmp_path, baseline, touched)
    assert result.test_edits == {"test_pricing.py": "additive"}
    assert result.killed_before is None


@pytest.mark.parametrize("field", ["survivors", "test_edits", "findings"])
def test_as_dict_is_json_serialisable_and_carries_the_findings(tmp_path, field):
    import json

    baseline, touched = _workspace(tmp_path, OLD, NEW, None, VACUOUS_TEST)
    payload = testcheck.audit(tmp_path, baseline, touched).as_dict()
    json.dumps(payload)
    assert field in payload


# -- wiring ----------------------------------------------------------------


def test_the_summary_surfaces_the_first_finding_only():
    from agent.summary import render

    text = render(
        "g",
        {
            "ok": True,
            "changed_files": ["pricing.py"],
            "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
            "final_check": {
                "ok": True,
                "tests_ran": True,
                "tests_passed": True,
                "tests_summary": "1 passed",
            },
            "plan": {"steps": []},
            "suggested_next_step": "",
            "quick_answer": "",
            "error": "",
            "test_audit": {
                "findings": [
                    "the 1 test(s) added pass against the old code",
                    "2 of 8 small changes were not noticed",
                ]
            },
        },
    )
    assert "tests may not cover the change" in text
    assert "pass against the old code" in text
    assert "2 of 8 small changes" not in text


def test_a_clean_audit_adds_nothing_to_the_summary():
    from agent.summary import render

    text = render(
        "g",
        {
            "ok": True,
            "changed_files": [],
            "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
            "final_check": {"ok": True},
            "plan": {"steps": []},
            "suggested_next_step": "",
            "quick_answer": "",
            "error": "",
            "test_audit": {"findings": []},
        },
    )
    assert "may not cover" not in text


def test_the_loop_helper_records_the_audit_in_the_trace_and_returns_it(tmp_path):
    import json

    from agent.approval import Approver
    from agent.loop import _audit_tests
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    trace = Trace(str(tmp_path / "t"), echo=False)
    root = tmp_path / "ws"
    root.mkdir()
    ctx = ToolContext(workspace=Workspace(root), approver=Approver(auto_approve=True), trace=trace)
    (root / "pricing.py").write_text(NEW)
    (root / "test_pricing.py").write_text(VACUOUS_TEST)
    ctx.baseline.snapshots.update({"pricing.py": OLD, "test_pricing.py": None})
    ctx.touched.extend(["pricing.py", "test_pricing.py"])

    payload = _audit_tests(ctx, trace)
    assert payload["added_tests_pass_on_old_code"] is True
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert next(e for e in events if e["kind"] == "test_audit")["ran"] is True


def test_bench_records_the_audit_next_to_the_integrity_verdict():
    from bench.runner import _audit_columns

    columns = _audit_columns(
        {
            "ran": True,
            "added_tests_pass_on_old_code": True,
            "mutants_tried": 9,
            "survivors": ["a", "b"],
            "seconds": 4.2,
        }
    )
    assert columns == {
        "audit_ran": True,
        "audit_pass_on_old": True,
        "audit_mutants": 9,
        "audit_survivors": 2,
        "audit_seconds": 4.2,
        "audit_repaired": None,
        "audit_findings_before": 0,
    }
    assert _audit_columns(None)["audit_ran"] is False


# -- feeding findings back as a fix round ----------------------------------


def _audit(**fields):
    return testcheck.TestAudit(ran=True, **fields)


def test_added_tests_that_pass_on_old_code_become_a_fix_finding():
    found = _audit(
        added_tests=["test_p.py::test_a", "test_p.py::test_b"],
        added_tests_pass_on_old_code=True,
        mutants_tried=4,
        survivors=["p.py:2 (comparison, in price)"],
    ).fix_findings()
    assert len(found) == 1
    assert found[0]["file"] == "test_p.py"
    assert "FAILS on the pre-run code" in found[0]["fix"]
    assert "p.py:2 (comparison, in price)" in found[0]["failure"]
    assert "do not edit the source" in found[0]["fix"].lower()


def test_survivors_alone_are_not_acted_on():
    """A surviving mutant can be equivalent to the original; sending a model to
    'fix' tests that were right would make them worse."""
    assert (
        _audit(
            mutants_tried=5, survivors=["p.py:2 (comparison)"], added_tests_pass_on_old_code=False
        ).fix_findings()
        == []
    )


def test_a_test_file_that_now_catches_less_becomes_a_fix_finding():
    found = _audit(
        test_edits={"test_p.py": "reduced"}, killed_before=6, killed_after=2, mutants_tried=8
    ).fix_findings()
    assert len(found) == 1
    assert "down from 6" in found[0]["issue"]
    assert "may not be loosened" in found[0]["fix"]


def test_a_rewrite_that_catches_as_much_is_left_alone():
    assert (
        _audit(
            test_edits={"test_p.py": "rewritten"}, killed_before=6, killed_after=6, mutants_tried=8
        ).fix_findings()
        == []
    )


def _repair_setup(tmp_path, monkeypatch, gate_result, second_audit):
    """_repair_weak_tests with the model and the gate replaced by stand-ins, so
    what is under test is the control flow: one round, a re-check, a re-audit."""
    from types import SimpleNamespace

    from agent import loop
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    calls = []
    monkeypatch.setattr(loop, "apply_fixes", lambda *a, **k: calls.append("fix"))
    monkeypatch.setattr(loop, "_final_gate", lambda *a, **k: (calls.append("gate"), gate_result)[1])
    monkeypatch.setattr(
        loop, "_audit_tests", lambda *a, **k: (calls.append("audit"), dict(second_audit))[1]
    )
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = SimpleNamespace(executor_reasoning="high")
    baselines = SimpleNamespace(tests=None, types=None)
    return loop, ctx, settings, baselines, calls


FIRST = {
    "added_tests": ["test_p.py::test_a"],
    "added_tests_pass_on_old_code": True,
    "findings": ["they pass on old code"],
    "survivors": [],
    "test_edits": {},
    "mutants_tried": 0,
}


def test_one_repair_round_then_recheck_then_reaudit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    green = SimpleNamespace(tests_passed=True, ok=True)
    loop, ctx, settings, baselines, calls = _repair_setup(
        tmp_path, monkeypatch, green, {"findings": [], "ran": True}
    )
    findings: list[dict] = []
    check, payload = loop._repair_weak_tests(
        None, settings, None, ctx, ctx.trace, findings, baselines, SimpleNamespace(), dict(FIRST)
    )
    assert calls == ["fix", "gate", "audit"]
    assert check is green
    assert payload["repaired"] is True
    assert payload["findings_before"] == ["they pass on old code"]
    assert findings and findings[0]["round"] == "test-audit"


def test_a_repair_that_breaks_the_suite_is_reported_not_hidden(tmp_path, monkeypatch):
    from types import SimpleNamespace

    red = SimpleNamespace(tests_passed=False, ok=False)
    loop, ctx, settings, baselines, calls = _repair_setup(
        tmp_path, monkeypatch, red, {"findings": []}
    )
    check, payload = loop._repair_weak_tests(
        None, settings, None, ctx, ctx.trace, [], baselines, SimpleNamespace(), dict(FIRST)
    )
    assert check is red
    assert payload["repaired"] is False
    assert "audit" not in calls  # no point re-auditing a red suite


def test_nothing_actionable_means_no_round_at_all(tmp_path, monkeypatch):
    from types import SimpleNamespace

    loop, ctx, settings, baselines, calls = _repair_setup(
        tmp_path, monkeypatch, SimpleNamespace(tests_passed=True), {}
    )
    clean = {
        "findings": ["3 small changes unnoticed"],
        "survivors": ["a"],
        "added_tests": [],
        "added_tests_pass_on_old_code": False,
    }
    original = SimpleNamespace()
    check, payload = loop._repair_weak_tests(
        None, settings, None, ctx, ctx.trace, [], baselines, original, clean
    )
    assert calls == [] and check is original and payload is clean


def test_the_fix_finding_shows_the_model_what_the_run_changed():
    """A test that fails on the old code must be written against the difference
    between old and new, and the fix round otherwise sees only the finished
    files: the first sweep of this repair fixed 2 of 7 flagged runs and none of
    the 4 on the task where the change was a constant it could not see move."""
    diff = "[modified] p.py\n-    if u <= 1000:\n+    if u < 1000:"
    found = _audit(
        added_tests=["test_p.py::test_a"], added_tests_pass_on_old_code=True
    ).fix_findings(diff)
    assert "What this run changed in the source" in found[0]["failure"]
    assert "if u < 1000" in found[0]["failure"]


def test_the_fix_finding_needs_no_diff_to_exist():
    found = _audit(
        added_tests=["test_p.py::test_a"], added_tests_pass_on_old_code=True
    ).fix_findings()
    assert len(found) == 1 and "What this run changed" not in found[0]["failure"]


def test_source_diff_excludes_test_files(tmp_path):
    from agent.approval import Approver
    from agent.loop import _source_diff
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    (tmp_path / "p.py").write_text("x = 2\n")
    (tmp_path / "test_p.py").write_text("y = 2\n")
    ctx.baseline.snapshots.update({"p.py": "x = 1\n", "test_p.py": "y = 1\n"})
    text = _source_diff(ctx)
    assert "p.py" in text and "test_p.py" not in text
