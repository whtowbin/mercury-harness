"""Process signals read off a run's trace, and the two views built on them.

The counters replace scripts that were hand-written twice to answer the same
questions off the same traces, so these tests pin the shapes those scripts got
wrong: a test run with no edit before it says nothing about an edit, and a
failure count that falls is progress even while the suite is still red.
"""

from __future__ import annotations

import json

from bench import diagnostics, report


def ev(kind: str, **fields) -> dict:
    return {"kind": kind, **fields}


def call(tool: str, phase: str = "execute:s1") -> dict:
    return ev("tool_call", tool=tool, phase=phase)


def suite(summary: str, exit_code: int = 1, **extra) -> dict:
    return ev("run_tests", summary=summary, exit_code=exit_code, **extra)


# -- the failure-count reading --------------------------------------------


def test_failed_count_reads_failures_errors_and_passes():
    assert diagnostics._failed_count("1 failed, 6 passed in 0.02s") == 1
    assert diagnostics._failed_count("7 passed in 0.01s") == 0
    assert diagnostics._failed_count("3 errors in 0.04s") == 3
    assert diagnostics._failed_count("1 error in 0.03s") == 1


def test_an_unreadable_summary_is_unknown_not_zero():
    """Zero would read as "nothing failed" and score a wasted edit as progress."""
    assert diagnostics._failed_count("pytest exited 2") is None
    assert diagnostics._failed_count(None) is None


# -- unproductive edits ----------------------------------------------------


def test_an_edit_that_does_not_reduce_the_failure_count_is_wasted():
    result = diagnostics.from_events(
        [suite("2 failed, 1 passed"), call("edit_file"), suite("2 failed, 1 passed")]
    )
    assert result.unproductive_edits == 1
    assert result.longest_unproductive_streak == 1


def test_an_edit_that_reduces_the_count_is_not_wasted():
    result = diagnostics.from_events([suite("3 failed"), call("edit_file"), suite("1 failed")])
    assert result.unproductive_edits == 0


def test_a_rising_failure_count_is_wasted_too():
    result = diagnostics.from_events([suite("1 failed"), call("edit_file"), suite("6 failed")])
    assert result.unproductive_edits == 1


def test_two_test_runs_with_no_edit_between_them_judge_nothing():
    """Re-running the suite unchanged is not an edit that failed to work."""
    result = diagnostics.from_events([suite("2 failed"), call("read_file"), suite("2 failed")])
    assert result.unproductive_edits == 0


def test_the_streak_is_the_longest_run_not_the_total():
    result = diagnostics.from_events(
        [
            suite("5 failed"),
            call("edit_file"),
            suite("5 failed"),
            call("edit_file"),
            suite("5 failed"),
            call("edit_file"),
            suite("2 failed"),  # progress resets it
            call("edit_file"),
            suite("2 failed"),
        ]
    )
    assert result.unproductive_edits == 3
    assert result.longest_unproductive_streak == 2


def test_a_passing_run_ends_the_cycle():
    result = diagnostics.from_events(
        [suite("1 failed"), call("edit_file"), suite("2 passed", exit_code=0), call("edit_file")]
    )
    assert result.unproductive_edits == 0
    assert result.longest_unproductive_streak == 0


# -- the other counters ----------------------------------------------------


def test_reads_before_the_first_edit_stop_counting_once_it_edits():
    result = diagnostics.from_events(
        [call("read_files"), call("grep"), call("edit_file"), call("read_file")]
    )
    assert result.reads_before_first_edit == 2


def test_run_tests_is_not_counted_as_looking_before_leaping():
    """Running the suite produces the failure; it is not a read of the code."""
    result = diagnostics.from_events([call("run_tests"), call("edit_file")])
    assert result.reads_before_first_edit == 0


def test_plan_phase_tool_calls_are_counted_separately():
    result = diagnostics.from_events(
        [
            call("read_files", phase="plan"),
            call("submit_plan", phase="plan"),
            call("read_file", phase="execute:s1"),
        ]
    )
    assert result.plan_tool_calls == 2


def test_nudges_loops_and_retries_are_counted():
    result = diagnostics.from_events(
        [
            ev("no_progress_nudge"),
            ev("same_failure_nudge"),
            ev("loop_exhausted"),
            ev("step_retry"),
            ev("repeated_call"),
            ev("tool_error"),
            suite("1 error", collection_error=True),
        ]
    )
    assert result.nudges == 2
    assert result.loops_exhausted == 1
    assert result.step_retries == 1
    assert result.repeated_calls == 1
    assert result.tool_errors == 1
    assert result.collection_errors == 1
    assert result.failed_test_runs == 1


def test_an_empty_trace_yields_zeroes_not_an_error():
    assert diagnostics.from_events([]).as_dict() == diagnostics.Diagnostics().as_dict()


def test_events_from_an_older_harness_contribute_what_they_have():
    """No `phase`, no `summary`, no `exit_code` — must not raise."""
    result = diagnostics.from_events(
        [{"kind": "tool_call", "tool": "edit_file"}, {"kind": "run_tests"}, {"kind": "log"}]
    )
    assert result.test_runs == 1


# -- reading a file --------------------------------------------------------


def test_from_trace_reads_a_real_jsonl_file(tmp_path):
    path = tmp_path / "run-x.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in [suite("2 failed"), call("edit_file"), suite("2 failed")]),
        encoding="utf-8",
    )
    assert diagnostics.from_trace(path).unproductive_edits == 1


def test_a_missing_or_corrupt_trace_is_empty_not_fatal(tmp_path):
    """Telemetry must never fail a sweep that otherwise succeeded."""
    assert diagnostics.from_trace(tmp_path / "nope.jsonl").test_runs == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json", encoding="utf-8")
    assert diagnostics.from_trace(bad).test_runs == 0


# -- the views -------------------------------------------------------------


def _record(config: str, task: str, **fields) -> dict:
    return {"config": config, "task": task, "accuracy": 1.0, "calls": 10, "honest": True, **fields}


def test_the_diagnostics_view_shows_a_mean_per_config():
    text = report.render_diagnostics(
        [
            _record("a", "t", diag_unproductive_edits=4),
            _record("a", "t", diag_unproductive_edits=2),
            _record("b", "t", diag_unproductive_edits=0),
        ]
    )
    rows = {line.split()[0]: line for line in text.splitlines() if line.split() and line[0] in "ab"}
    assert "3.0" in rows["a"]
    assert "0.0" in rows["b"]


def test_a_task_every_config_scores_alike_on_is_reported_as_not_separating():
    text = report.render_discrimination(
        [
            _record("a", "flat"),
            _record("b", "flat"),
            _record("a", "sharp", accuracy=1.0),
            _record("b", "sharp", accuracy=0.5),
        ]
    )
    lines = {line.split()[0]: line for line in text.splitlines() if line.split()}
    assert lines["flat"].endswith("no")
    assert lines["sharp"].endswith("yes")


def test_a_task_at_the_accuracy_ceiling_still_separates_on_calls():
    """Accuracy has been at a ceiling for most of this project's sweeps; a
    task that costs one config twice the calls is still earning its budget."""
    text = report.render_discrimination([_record("a", "t", calls=8), _record("b", "t", calls=20)])
    row = next(line for line in text.splitlines() if line.startswith("t "))
    assert row.endswith("yes")


def test_a_task_at_the_ceiling_still_separates_on_honesty():
    """A real seeded task name, since honesty is only compared on tasks that
    have a seed directory to weaken tests in."""
    text = report.render_discrimination(
        [_record("a", "twin-paths", honest=True), _record("b", "twin-paths", honest=False)]
    )
    row = next(line for line in text.splitlines() if line.startswith("twin-paths"))
    assert row.endswith("yes")


def test_discrimination_needs_two_configs_to_say_anything():
    assert "at least two configs" in report.render_discrimination([_record("a", "t")])


def test_a_task_only_one_config_ran_is_skipped_rather_than_compared():
    text = report.render_discrimination(
        [_record("a", "both"), _record("b", "both"), _record("a", "lonely")]
    )
    assert "lonely" not in text


def test_honesty_is_not_compared_on_a_greenfield_task():
    """`_honest_rate` had this denominator bug once already: a task with no
    seed directory has no seeded test to weaken, so every run scores honest
    and a zero gap would read as "nothing separated" rather than "does not
    apply". `expr` is greenfield; `twin-paths` is seeded."""
    text = report.render_discrimination(
        [_record("a", "expr", honest=True), _record("b", "expr", honest=True)]
    )
    row = next(line for line in text.splitlines() if line.startswith("expr"))
    assert "—" in row


def test_honesty_is_compared_on_a_seeded_task():
    text = report.render_discrimination(
        [_record("a", "twin-paths", honest=True), _record("b", "twin-paths", honest=False)]
    )
    row = next(line for line in text.splitlines() if line.startswith("twin-paths"))
    assert "100%" in row
    assert row.endswith("yes")


# -- reliability, failure classes, compare ---------------------------------


def _run(config, task, accuracy=1.0, **fields):
    return {
        "config": config,
        "task": task,
        "accuracy": accuracy,
        "honest": True,
        "calls": 10,
        "cost_usd": 0.01,
        "model_latency_s": 30.0,
        "started_at": 1000.0,
        **fields,
    }


def test_pass_hat_k_needs_every_repeat_to_pass():
    text = report.render_reliability(
        [_run("a", "t1", 1.0), _run("a", "t1", 1.0), _run("a", "t2", 1.0), _run("a", "t2", 0.5)]
    )
    row = next(line for line in text.splitlines() if line.startswith("a "))
    assert "100%" in row and "50%" in row  # pass@k 100%, pass^k 50%


def test_a_task_run_once_is_left_out_of_reliability():
    text = report.render_reliability([_run("a", "t1"), _run("a", "t2"), _run("a", "t2")])
    row = next(line for line in text.splitlines() if line.startswith("a "))
    assert row.split()[1] == "1"


def test_classify_puts_the_first_thing_that_went_wrong_first():
    assert report.classify_failure(_run("a", "t", timed_out=True)) == "crashed"
    assert report.classify_failure(_run("a", "t", agent_error="boom")) == "crashed"
    assert report.classify_failure(_run("a", "t")) == "ok"
    assert (
        report.classify_failure(_run("a", "t", own_tests_miss_the_bug=True)) == "tests_miss_change"
    )
    assert report.classify_failure(_run("a", "t", honest=False)) == "dishonest"
    assert report.classify_failure(_run("a", "t", 0.5, diag_loops_exhausted=1)) == "loop"
    assert (
        report.classify_failure(_run("a", "t", 0.5, diag_longest_unproductive_streak=3)) == "loop"
    )
    assert (
        report.classify_failure(_run("a", "t", 0.5, diag_final_green=1))
        == "wrong_fix_believed_done"
    )
    assert report.classify_failure(_run("a", "t", 0.5, diag_final_green=0)) == "gave_up_red"


def test_a_run_that_looped_is_a_loop_problem_before_a_wrong_fix_problem():
    assert (
        report.classify_failure(_run("a", "t", 0.5, diag_loops_exhausted=2, diag_final_green=1))
        == "loop"
    )


def test_the_failure_view_counts_each_class_per_config():
    text = report.render_failures(
        [_run("a", "t"), _run("a", "t", 0.5, diag_final_green=1), _run("b", "t", timed_out=True)]
    )
    rows = {
        line.split()[0]: line.split()
        for line in text.splitlines()
        if line[:1] in "ab" and len(line.split()) > 5
    }
    assert rows["a"][1] == "2" and rows["b"][1] == "1"


def test_compare_calls_a_large_real_gap_different():
    from bench import stats

    a = [0.0] * 20 + [1.0] * 5
    b = [1.0] * 22 + [0.0] * 3
    result = stats.compare("x", a, b)
    assert result.distinguishable and result.verdict == "differs"


def test_compare_refuses_a_gap_that_is_only_noise():
    """The exact shape of the two false leads: ~20 points of honesty at 24 runs
    a side, which the bootstrap interval says nothing about."""
    from bench import stats

    a = [1.0] * 16 + [0.0] * 8  # 67% of 24
    b = [1.0] * 21 + [0.0] * 3  # 88% of 24
    result = stats.compare("honest", a, b)
    assert not result.distinguishable
    assert "not distinguishable" in result.verdict
    assert result.runs_needed and result.runs_needed > 24


def test_compare_gives_no_verdict_on_too_few_runs():
    from bench import stats

    assert "too few" in stats.compare("x", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).verdict


def test_compare_is_deterministic():
    from bench import stats

    a, b = [0.0, 1.0] * 15, [1.0, 1.0, 0.0] * 10
    assert stats.compare("x", a, b) == stats.compare("x", a, b)


def test_arms_that_ran_at_different_times_are_flagged_as_not_interleaved():
    from bench import stats

    assert not stats.interleaved([1.0, 2.0, 3.0], [100.0, 101.0])
    assert stats.interleaved([1.0, 50.0], [2.0, 51.0])


def test_the_compare_view_warns_when_the_arms_did_not_share_time():
    rows = [_run("a", "t", started_at=float(i)) for i in range(8)] + [
        _run("b", "t", started_at=1000.0 + i) for i in range(8)
    ]
    text = report.render_compare(rows, "a", "b")
    assert "did not run during the same stretch of time" in text


def test_the_compare_view_is_quiet_when_the_arms_interleave():
    rows = []
    for i in range(8):
        rows += [_run("a", "t", started_at=float(i)), _run("b", "t", started_at=float(i) + 0.5)]
    assert "WARNING" not in report.render_compare(rows, "a", "b")


def test_compare_names_the_configs_it_found_when_one_is_missing():
    assert "need runs for both" in report.render_compare([_run("a", "t")], "a", "b")


def test_unparseable_tool_arguments_count_as_tool_errors():
    result = diagnostics.from_events(
        [ev("tool_arg_parse_error"), ev("tool_call_failed"), call("read_file")]
    )
    assert result.tool_errors == 2
    assert result.tool_calls == 1


def test_the_final_suite_state_is_recorded_when_there_was_one():
    assert diagnostics.from_events([ev("final_tests", exit_code=0)]).final_green == 1
    assert diagnostics.from_events([ev("final_tests", exit_code=1)]).final_green == 0
    assert diagnostics.from_events([]).final_green == -1


# -- red-first: did the step watch a test fail before editing source -------


def edit(path: str, phase: str = "execute:s1") -> dict:
    return ev("tool_call", tool="edit_file", phase=phase, args={"path": path})


def test_a_step_that_ran_a_failing_test_before_editing_source_is_red_first():
    result = diagnostics.from_events([call("run_tests"), suite("1 failed"), edit("pricing.py")])
    assert (result.source_edit_steps, result.red_first_steps) == (1, 1)


def test_a_step_that_edited_source_with_no_test_run_is_not_red_first():
    result = diagnostics.from_events([edit("pricing.py")])
    assert (result.source_edit_steps, result.red_first_steps) == (1, 0)


def test_editing_a_test_file_first_does_not_count_as_editing_source():
    result = diagnostics.from_events(
        [edit("test_pricing.py"), call("run_tests"), suite("1 failed"), edit("pricing.py")]
    )
    assert (result.source_edit_steps, result.red_first_steps) == (1, 1)


def test_a_green_first_run_is_not_red_first():
    result = diagnostics.from_events(
        [call("run_tests"), suite("3 passed", exit_code=0), edit("pricing.py")]
    )
    assert (result.source_edit_steps, result.red_first_steps) == (1, 0)


def test_only_the_first_test_run_before_the_edit_decides():
    result = diagnostics.from_events(
        [
            call("run_tests"),
            suite("3 passed", exit_code=0),
            call("run_tests"),
            suite("1 failed"),
            edit("pricing.py"),
        ]
    )
    assert result.red_first_steps == 0


def test_steps_are_counted_separately():
    result = diagnostics.from_events(
        [
            call("run_tests", "execute:a"),
            suite("1 failed"),
            edit("p.py", "execute:a"),
            edit("q.py", "execute:b"),
        ]
    )
    assert (result.source_edit_steps, result.red_first_steps) == (2, 1)


def test_edit_paths_are_read_from_stringified_args_too():
    """Older traces stored the args as text."""
    result = diagnostics.from_events(
        [
            {
                "kind": "tool_call",
                "tool": "edit_file",
                "phase": "execute:s1",
                "args": "{'path': 'test_x.py', 'old_string': 'a'}",
            }
        ]
    )
    assert result.source_edit_steps == 0


# -- which model actually answered -----------------------------------------


def _call(model, served="", provider=""):
    return ev("llm_call", model=model, served_model=served, provider=provider)


def test_served_models_pair_the_requested_and_the_served_id():
    found = diagnostics.served_models(
        [
            _call("a/m", "a/m-2026", "Acme"),
            _call("a/m", "a/m-2026", "Acme"),
            _call("a/m", "a/m-2027", "Acme"),
        ]
    )
    assert found == ["a/m -> a/m-2026 @ Acme", "a/m -> a/m-2027 @ Acme"]


def test_a_call_with_no_served_id_contributes_nothing():
    assert diagnostics.served_models([_call("a/m")]) == []


def test_a_missing_provider_is_shown_as_a_question_mark():
    assert diagnostics.served_models([_call("a/m", "a/m-1")]) == ["a/m -> a/m-1 @ ?"]


def test_the_models_view_flags_two_pairings_for_one_requested_id():
    rows = [
        {"served_models": ["a/m -> a/m-1 @ Acme"], "started_at": 1000.0},
        {"served_models": ["a/m -> a/m-2 @ Acme"], "started_at": 5000.0},
    ]
    text = report.render_models(rows)
    assert "a/m was served under 2 different" in text


def test_the_models_view_is_quiet_when_one_pairing_served_every_run():
    rows = [{"served_models": ["a/m -> a/m-1 @ Acme"], "started_at": 1000.0}] * 3
    text = report.render_models(rows)
    assert "NOTE" not in text and "3" in text


def test_the_models_view_says_so_when_no_run_recorded_it():
    assert "no served-model data" in report.render_models([{"started_at": 1.0}])


def test_a_pass_verdict_that_lists_findings_is_counted():
    result = diagnostics.from_events(
        [
            ev("review_result", verdict="pass", findings=[{"issue": "x"}]),
            ev("review_result", verdict="pass", findings=[]),
            ev("review_result", verdict="needs_changes", findings=[{"issue": "y"}]),
        ]
    )
    assert result.verdict_contradicts_findings == 1


# -- plan-quality proxies --------------------------------------------------


def test_the_plan_event_yields_specificity_counters():
    result = diagnostics.from_events(
        [
            ev(
                "plan",
                draft=["understood: fix the boundary", "check tier_for and price both"],
                steps=[
                    {
                        "tests": ["test_a", "test_b"],
                        "signatures": ["f(x)"],
                        "files": ["a.py", "b.py"],
                    },
                    {"tests": ["test_c"], "signatures": [], "files": ["a.py"]},
                ],
            )
        ]
    )
    assert result.plan_steps == 2
    assert result.plan_draft_words == 9
    assert result.plan_named_tests == 3
    assert result.plan_signatures == 1
    assert result.plan_declared_files == 3


def test_the_last_plan_event_is_the_one_counted():
    result = diagnostics.from_events(
        [
            ev("plan", draft=["one"], steps=[{"tests": ["t"]}]),
            ev("plan", draft=["one two three"], steps=[{"tests": ["a", "b"]}, {"tests": []}]),
        ]
    )
    assert (result.plan_steps, result.plan_draft_words, result.plan_named_tests) == (2, 3, 2)


def test_a_plan_with_no_steps_or_draft_counts_zero():
    result = diagnostics.from_events([ev("plan")])
    assert result.plan_steps == result.plan_draft_words == result.plan_named_tests == 0


def test_a_plan_event_followed_by_tool_calls_does_not_crash():
    """The plan branch once named its local `steps`, shadowing the per-step
    state dict, so every real run (a plan event, then tool calls) raised
    AttributeError inside the bench worker. The unit tests above each fed one
    kind of event; a real trace interleaves them."""
    result = diagnostics.from_events(
        [
            ev("plan", draft=["understood: x"], steps=[{"tests": ["t"]}]),
            call("run_tests", "execute:s1"),
            suite("1 failed"),
            edit("pricing.py", "execute:s1"),
            ev("review_result", verdict="pass", findings=[]),
            ev("final_tests", exit_code=0),
        ]
    )
    assert result.plan_named_tests == 1
    assert (result.source_edit_steps, result.red_first_steps) == (1, 1)
    assert result.final_green == 1


def test_a_counter_bug_costs_a_column_not_the_run(monkeypatch, tmp_path):
    from bench import runner

    def boom(_path):
        raise RuntimeError("counter bug")

    monkeypatch.setattr(runner.diagnostics, "from_trace", boom)
    record = runner.flatten({"trace_path": str(tmp_path / "t.jsonl"), "scores": {}})
    assert "counter bug" in record["diag_error"]


def test_the_backfill_fills_counters_added_after_a_record_was_written(tmp_path):
    """Records from before the plan-quality counters carried the older diag_*
    columns, so a backfill keyed on one of those skipped them and the view
    showed the newer columns as zero."""
    import json

    from bench.cli import _backfill_diagnostics

    trace = tmp_path / "run.jsonl"
    trace.write_text(
        json.dumps(ev("plan", draft=["one two"], steps=[{"tests": ["a", "b"]}])), encoding="utf-8"
    )
    record = {"trace_path": str(trace), "diag_test_runs": 3}  # an older record
    _backfill_diagnostics([record])
    assert record["diag_plan_named_tests"] == 2
    assert record["diag_plan_draft_words"] == 2


def test_gate_holds_are_counted():
    result = diagnostics.from_events([ev("tdd_gate_held"), ev("tdd_gate_held")])
    assert result.tdd_holds == 2


def test_call_site_holds_are_counted():
    events = [{"kind": "callsite_gate_held", "phase": "plan", "tool": "submit_plan"}]
    assert diagnostics.from_events(events).callsite_holds == 1
