"""Deterministic trace analysis, validated against the exact shapes a real
trace-mining session found by hand on the platformer app's traces: the
coin-run's undeclared tests/ directory, the pygame-font bug recurring across
runs, and the run-outcome table rebuilt with a jq one-liner each time."""

from __future__ import annotations

import json

from agent import tracetools as tt


def _write(tmp_path, name, events):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


# -- loading -------------------------------------------------------------


def test_load_trace_reads_events_in_order(tmp_path):
    path = _write(
        tmp_path,
        "run-a.jsonl",
        [
            {"kind": "run_start", "goal": "g"},
            {"kind": "log", "message": "hi"},
        ],
    )
    events = tt.load_trace(path)
    assert [e["kind"] for e in events] == ["run_start", "log"]


def test_load_trace_skips_a_malformed_line_instead_of_failing(tmp_path):
    path = tmp_path / "run-a.jsonl"
    path.write_text('{"kind": "run_start"}\nnot json\n{"kind": "log"}\n')
    events = tt.load_trace(path)
    assert [e["kind"] for e in events] == ["run_start", "log"]


def test_load_traces_keys_by_filename(tmp_path):
    _write(tmp_path, "run-a.jsonl", [{"kind": "run_start"}])
    _write(tmp_path, "run-b.jsonl", [{"kind": "run_start"}])
    assert set(tt.load_traces(tmp_path)) == {"run-a.jsonl", "run-b.jsonl"}


# -- find_calls / next_event / pair_calls_with_outcomes -------------------


def test_find_calls_filters_by_tool_and_phase():
    events = [
        {"kind": "tool_call", "tool": "edit_file", "phase": "execute:s1", "iteration": 0},
        {"kind": "tool_call", "tool": "read_file", "phase": "execute:s1", "iteration": 1},
        {"kind": "tool_call", "tool": "edit_file", "phase": "execute:s2", "iteration": 0},
    ]
    found = tt.find_calls(events, tool="edit_file", phase="execute:s1")
    assert len(found) == 1
    assert found[0][0] == 0


def test_next_event_finds_the_first_matching_kind_within_the_window():
    events = [
        {"kind": "tool_call"},
        {"kind": "log", "message": "…"},
        {"kind": "tool_call_failed", "result": "error: x"},
    ]
    ev = tt.next_event(events, 0, kinds=("tool_call_failed",))
    assert ev is not None and ev["result"] == "error: x"


def test_next_event_returns_none_outside_the_window():
    events = [{"kind": "tool_call"}] + [{"kind": "log"}] * 5 + [{"kind": "tool_error"}]
    assert tt.next_event(events, 0, kinds=("tool_error",), within=3) is None


def test_pair_calls_with_outcomes_marks_a_failure(tmp_path):
    events = [
        {"kind": "tool_call", "tool": "edit_files", "phase": "execute:s1", "iteration": 1},
        {"kind": "log", "message": "editing 0 file(s)"},
        {
            "kind": "tool_call_failed",
            "tool": "edit_files",
            "result": "error: edits must be a JSON array",
        },
        {"kind": "tool_call", "tool": "edit_files", "phase": "execute:s1", "iteration": 2},
        {"kind": "edit_file", "path": "m.py"},
    ]
    paired = tt.pair_calls_with_outcomes(events, "edit_files")
    assert len(paired) == 2
    assert paired[0]["outcome"] == "failed"
    assert paired[1]["outcome"] == "ok"


# -- declared_vs_actual / new_files_not_in_plan ----------------------------


def test_declared_vs_actual_reports_an_unwritten_declared_file():
    events = [
        {"kind": "plan", "steps": [{"id": "s1", "writes": ["README.md", "a.py"]}]},
        {"kind": "edit_file", "path": "a.py"},
    ]
    rows = tt.declared_vs_actual(events)
    assert rows == [
        {"step": "s1", "declared_writes": ["README.md", "a.py"], "unwritten": ["README.md"]}
    ]


def test_new_files_not_in_plan_matches_the_real_coin_run_shape(tmp_path):
    """The exact shape found in a real trace: a step declared
    writes=[coin.py, sprites.py], then created six files, four of them
    undeclared and three of those shadowing an existing root-level file."""
    (tmp_path / "sprites.py").write_text("...")
    (tmp_path / "coin.py").write_text("...")
    (tmp_path / "test_game.py").write_text("...")
    (tmp_path / "tests").mkdir()

    events = [
        {
            "kind": "plan",
            "steps": [
                {"id": "add-coin-class", "writes": ["coin.py", "sprites.py"]},
            ],
        },
        {"kind": "write_file", "path": "tests/test_coin.py", "overwrote": False},
        {"kind": "write_file", "path": "tests/__init__.py", "overwrote": False},
        {"kind": "write_file", "path": "tests/coin.py", "overwrote": False},
        {"kind": "write_file", "path": "tests/sprites.py", "overwrote": False},
        {"kind": "write_file", "path": "coin.py", "overwrote": True},
    ]
    rows = tt.new_files_not_in_plan(events, workspace_root=tmp_path)
    by_path = {r["path"]: r["shadows"] for r in rows}
    assert by_path["tests/test_coin.py"] == []  # genuinely new, no collision
    assert by_path["tests/__init__.py"] == []  # excluded on purpose
    assert by_path["tests/coin.py"] == ["coin.py"]
    assert by_path["tests/sprites.py"] == ["sprites.py"]
    assert "coin.py" not in by_path  # declared, not flagged


def test_new_files_not_in_plan_without_workspace_root_skips_the_collision_check():
    events = [
        {"kind": "plan", "steps": [{"id": "s1", "writes": []}]},
        {"kind": "write_file", "path": "tests/sprites.py", "overwrote": False},
    ]
    rows = tt.new_files_not_in_plan(events)
    assert rows == [{"path": "tests/sprites.py", "shadows": []}]


def test_undeclared_writes_flags_an_edit_to_an_already_existing_file():
    """The repeated shape found auditing a real sweep: a step declares
    writes=[convert.py], edits convert.py *and* test_convert.py (which
    already exists — not a new file, so new_files_not_in_plan never sees
    it), and declared_vs_actual reports a clean run because convert.py, the
    only declared path, really was written."""
    events = [
        {"kind": "plan", "steps": [{"id": "fix-conversion", "writes": ["convert.py"]}]},
        {"kind": "edit_file", "path": "convert.py"},
        {"kind": "edit_file", "path": "test_convert.py"},
    ]
    assert tt.declared_vs_actual(events) == [
        {"step": "fix-conversion", "declared_writes": ["convert.py"], "unwritten": []}
    ]
    assert tt.undeclared_writes(events) == ["test_convert.py"]


def test_undeclared_writes_empty_when_everything_touched_was_declared():
    events = [
        {"kind": "plan", "steps": [{"id": "s1", "writes": ["a.py", "test_a.py"]}]},
        {"kind": "edit_file", "path": "a.py"},
        {"kind": "write_file", "path": "test_a.py", "overwrote": False},
    ]
    assert tt.undeclared_writes(events) == []


def test_undeclared_writes_is_plan_wide_not_per_step():
    """A file written during one step but declared by a different step is
    not undeclared — ownership across the whole plan is what's checked,
    same scope as new_files_not_in_plan."""
    events = [
        {
            "kind": "plan",
            "steps": [
                {"id": "s1", "writes": ["a.py"]},
                {"id": "s2", "writes": ["b.py"]},
            ],
        },
        {"kind": "edit_file", "path": "b.py"},
    ]
    assert tt.undeclared_writes(events) == []


# -- verify_coverage_gaps ----------------------------------------------------


def test_verify_coverage_gaps_flags_a_lint_only_step():
    events = [
        {
            "kind": "step_verify",
            "step": "add-coin-class",
            "command": "uvx --quiet ruff check coin.py sprites.py",
            "passed": True,
        },
    ]
    assert tt.verify_coverage_gaps(events) == [
        {"step": "add-coin-class", "commands": ["uvx --quiet ruff check coin.py sprites.py"]}
    ]


def test_verify_coverage_gaps_does_not_flag_a_step_that_ran_pytest():
    events = [
        {
            "kind": "step_verify",
            "step": "add-lives-system",
            "command": "uv run pytest -q tests/test_game.py::TestGame",
            "passed": True,
        },
    ]
    assert tt.verify_coverage_gaps(events) == []


def test_verify_coverage_gaps_does_not_flag_a_step_with_no_verify_event():
    assert tt.verify_coverage_gaps([{"kind": "log", "message": "hi"}]) == []


def test_verify_coverage_gaps_is_flagged_only_if_every_command_for_the_step_skips_tests():
    events = [
        {"kind": "step_verify", "step": "s1", "command": "ruff check a.py", "passed": True},
        {"kind": "step_verify", "step": "s1", "command": "pytest tests/test_a.py", "passed": True},
    ]
    assert tt.verify_coverage_gaps(events) == []


# -- plan_conformance ---------------------------------------------------------


def test_plan_conformance_flags_a_signature_never_written(tmp_path):
    (tmp_path / "main.py").write_text("class Game:\n    def update(self): pass\n")
    events = [
        {
            "kind": "plan",
            "steps": [
                {
                    "id": "add-lives-system",
                    "signatures": ["Game.update(self)", "Game.draw(self)"],
                    "tests": [],
                }
            ],
        }
    ]
    result = tt.plan_conformance(events, tmp_path)
    assert result["missing_signatures"] == [
        {"step": "add-lives-system", "signature": "Game.draw(self)", "name": "draw"}
    ]


def test_plan_conformance_flags_a_test_never_written(tmp_path):
    (tmp_path / "test_game.py").write_text("def test_a(): pass\n")
    events = [
        {
            "kind": "plan",
            "steps": [
                {
                    "id": "add-lives-tests",
                    "signatures": [],
                    "tests": ["test_game_starts_with_3_lives: has 3 lives"],
                }
            ],
        }
    ]
    result = tt.plan_conformance(events, tmp_path)
    assert result["missing_tests"] == [
        {
            "step": "add-lives-tests",
            "test": "test_game_starts_with_3_lives: has 3 lives",
            "name": "test_game_starts_with_3_lives",
        }
    ]


def test_plan_conformance_reports_nothing_missing_when_everything_exists(tmp_path):
    (tmp_path / "main.py").write_text("def foo(): pass\n")
    events = [
        {
            "kind": "plan",
            "steps": [
                {
                    "id": "s1",
                    "signatures": ["foo()"],
                    "tests": [],
                }
            ],
        }
    ]
    result = tt.plan_conformance(events, tmp_path)
    assert result == {"missing_signatures": [], "missing_tests": []}


# -- goal_requirement_gaps ----------------------------------------------------


def test_goal_requirement_gaps_flags_a_word_absent_from_the_plan():
    events = [
        {"kind": "run_start", "goal": "coins disappear when collected"},
        {
            "kind": "plan",
            "steps": [
                {
                    "title": "Add health bar",
                    "intent": "collect coins",
                    "acceptance": "coin removed on pickup",
                }
            ],
        },
    ]
    assert "disappear" in tt.goal_requirement_gaps(events)


def test_goal_requirement_gaps_excludes_a_word_present_in_the_plan():
    events = [
        {"kind": "run_start", "goal": "player loses a life"},
        {
            "kind": "plan",
            "steps": [{"title": "Lives system", "intent": "player loses a life", "acceptance": ""}],
        },
    ]
    assert "loses" not in tt.goal_requirement_gaps(events)


def test_goal_requirement_gaps_skips_stopwords_and_short_words():
    events = [
        {"kind": "run_start", "goal": "add a way for it to do that"},
        {"kind": "plan", "steps": []},
    ]
    assert tt.goal_requirement_gaps(events) == []


# -- cross-trace pattern mining ---------------------------------------------


def test_grep_traces_finds_a_recurring_pattern_across_files(tmp_path):
    _write(tmp_path, "run-a.jsonl", [{"kind": "log", "message": "font not initialized"}])
    _write(tmp_path, "run-b.jsonl", [{"kind": "log", "message": "all clear"}])
    _write(tmp_path, "run-c.jsonl", [{"kind": "log", "message": "Font Not Initialized here too"}])
    hits = tt.grep_traces(tmp_path, "font not initialized")
    assert set(hits) == {"run-a.jsonl", "run-c.jsonl"}  # case-insensitive


def test_grep_traces_treats_the_pattern_as_a_literal_substring(tmp_path):
    """Not a regex -- '.' in a search term must not act as a wildcard."""
    _write(tmp_path, "run-a.jsonl", [{"kind": "log", "message": "pygame.font.SysFont broke"}])
    _write(tmp_path, "run-b.jsonl", [{"kind": "log", "message": "pygameXfontXSysFont broke"}])
    hits = tt.grep_traces(tmp_path, "pygame.font.SysFont")
    assert set(hits) == {"run-a.jsonl"}


def test_run_outcomes_reads_goal_ok_and_cost(tmp_path):
    _write(
        tmp_path,
        "run-a.jsonl",
        [
            {"kind": "run_start", "goal": "fix the bug"},
            {"kind": "final_check", "ok": False},
            {"kind": "totals", "cost_usd": 0.0466, "calls": 103},
        ],
    )
    rows = tt.run_outcomes(tmp_path)
    assert rows == [
        {
            "trace": "run-a.jsonl",
            "goal": "fix the bug",
            "ok": False,
            "cost_usd": 0.0466,
            "calls": 103,
        }
    ]


def test_run_outcomes_uses_the_last_final_check_when_there_were_retries(tmp_path):
    _write(
        tmp_path,
        "run-a.jsonl",
        [
            {"kind": "run_start", "goal": "g"},
            {"kind": "final_check", "ok": False},
            {"kind": "final_check", "ok": True},
            {"kind": "totals", "cost_usd": 0.01, "calls": 5},
        ],
    )
    rows = tt.run_outcomes(tmp_path)
    assert rows[0]["ok"] is True


def test_run_outcomes_handles_a_trace_with_no_final_check_or_totals(tmp_path):
    _write(tmp_path, "run-a.jsonl", [{"kind": "run_start", "goal": "just planning"}])
    rows = tt.run_outcomes(tmp_path)
    assert rows == [
        {
            "trace": "run-a.jsonl",
            "goal": "just planning",
            "ok": None,
            "cost_usd": None,
            "calls": None,
        }
    ]


# -- per-run statistics -------------------------------------------------------


def test_tool_usage_counts_and_sorts_by_frequency():
    events = [{"kind": "tool_call", "tool": "read_file"}] * 3 + [
        {"kind": "tool_call", "tool": "edit_file"}
    ] * 5
    assert list(tt.tool_usage(events)) == ["edit_file", "read_file"]


def test_file_statistics_counts_reads_and_edits_separately():
    events = [
        {"kind": "tool_call", "tool": "read_file", "args": {"path": "a.py"}},
        {"kind": "tool_call", "tool": "read_file", "args": {"path": "a.py"}},
        {"kind": "edit_file", "path": "a.py"},
        {"kind": "write_file", "path": "b.py"},
    ]
    stats = tt.file_statistics(events)
    assert stats["a.py"] == {"reads": 2, "edits": 1}
    assert stats["b.py"] == {"reads": 0, "edits": 1}


def test_cost_breakdown_groups_by_phase_prefix():
    events = [
        {"kind": "llm_call", "phase": "plan", "cost_usd": 0.001},
        {"kind": "llm_call", "phase": "execute:step-1", "cost_usd": 0.002},
        {"kind": "llm_call", "phase": "execute:step-2", "cost_usd": 0.003},
    ]
    breakdown = tt.cost_breakdown(events)
    assert breakdown["plan"] == {"cost_usd": 0.001, "calls": 1}
    assert breakdown["execute"] == {"cost_usd": 0.005, "calls": 2}


def test_summarize_trace_pulls_the_headline_facts():
    events = [
        {"kind": "run_start", "goal": "fix the thing", "workspace": "/w"},
        {"kind": "plan", "steps": [{"id": "s1", "writes": ["a.py"]}]},
        {"kind": "final_check", "ok": True},
        {"kind": "totals", "cost_usd": 0.02, "calls": 10},
        {"kind": "edit_file", "path": "a.py"},
    ]
    summary = tt.summarize_trace(events)
    assert summary["goal"] == "fix the thing"
    assert summary["steps_planned"] == 1
    assert summary["final_ok"] is True
    assert summary["cost_usd"] == 0.02
    assert summary["declared_vs_actual"] == [
        {"step": "s1", "declared_writes": ["a.py"], "unwritten": []}
    ]
