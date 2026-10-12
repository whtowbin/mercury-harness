"""Three fixes from the platformer app's 2026-09-30 run (DESIGN-LOG §98): the planner keeps a
question, the summary says what the run found, and a fix whose finding already passes is
skipped."""

from __future__ import annotations

import json

from agent.approval import Approver
from agent.config import Settings
from agent.interact import Interactor
from agent.phases import review as review_module
from agent.phases.clarify import MAX_QUESTIONS
from agent.summary import render
from agent.tools import ToolContext
from agent.trace import Trace
from agent.workspace import Workspace

# -- 1. clarify leaves one question for the planner -------------------------------


def test_clarify_has_its_own_five_and_leaves_the_runs_three_for_later():
    """Clarify's batch never spends the planner's questions (DESIGN-LOG §98, §101)."""
    assert MAX_QUESTIONS == 5
    asked: list[str] = []
    interactor = Interactor(ask=lambda q, o=None, r=None: asked.append(q) or "an answer")
    answers = interactor.ask_many([(f"clarify {i}?", None) for i in range(7)])
    assert answers[:5] == ["an answer"] * 5
    assert all("5-question limit" in a for a in answers[5:])
    later = [interactor.ask(f"planner {i}?") for i in range(4)]
    assert later[:3] == ["an answer"] * 3 and "3-question limit" in later[3]
    assert asked == [*(f"clarify {i}?" for i in range(5)), "planner 0?", "planner 1?", "planner 2?"]


def test_a_form_gets_the_whole_batch_at_once():
    seen: list[list] = []

    def form(items):
        seen.append(items)
        return ["a", "b"]

    interactor = Interactor(ask=lambda q, o=None: "one at a time", ask_many=form)
    assert interactor.ask_many([("first?", ["x", "y"]), ("second?", None)]) == ["a", "b"]
    assert seen == [[("first?", ["x", "y"], None), ("second?", None, None)]]


def test_with_questions_off_the_batch_gets_the_conservative_refusal():
    answers = Interactor(enabled=False).ask_many([("a?", None), ("b?", None)])
    assert all("turned off" in a and "conservative" in a for a in answers)


def test_over_the_limit_the_refusal_asks_for_what_the_goal_wants():
    interactor = Interactor(ask=lambda q, o=None, r=None: "an answer", max_questions=3)
    answers = [interactor.ask(f"q{i}") for i in range(4)]
    assert answers[:3] == ["an answer"] * 3
    assert "3-question limit" in answers[3]
    assert "delivers what the goal asks for" in answers[3]
    assert "conservative" not in answers[3]


def test_with_nobody_to_ask_the_refusal_still_prefers_the_conservative_option():
    """Benchmarks run with questions off; their honesty tasks rely on this wording."""
    answer = Interactor(enabled=False).ask("should I change it?")
    assert "turned off" in answer and "conservative" in answer


# -- 2. the summary says what the run found ----------------------------------------

PLATFORMER_APP_DRAFT = [
    "understood: User wants to know why the flag isn't appearing. Flagpole exists at x=1100 "
    "but SCREEN_WIDTH=800 making it off-screen.",
    "Issue 1: Flagpole at x=1100 is off-screen. Assumption: keep flagpole off-screen.",
    "Test approach: Add tests for jump mechanics.",
    "Plan: 1) Fix ground collision. 2) Add flagpole tests.",
]


def _summary(draft, findings=()):
    return {
        "ok": True,
        "plan": {"goal": "g", "draft": draft, "findings": list(findings), "steps": []},
        "steps": {},
        "changed_files": ["main.py"],
    }


def test_the_summary_shows_the_diagnosis_and_the_assumption():
    text = render("g", _summary(PLATFORMER_APP_DRAFT, ["decision: keep x=1100 because level end"]))
    found = text.split("found:")[1].split("built:")[0]
    assert "SCREEN_WIDTH=800 making it off-screen" in found
    assert "Assumption: keep flagpole off-screen" in found
    assert "decision: keep x=1100" in found
    assert "Test approach" not in found and "Plan: 1)" not in found


def test_no_found_section_without_a_diagnosis():
    assert "found:" not in render("g", _summary(["Plan: add a flag", "Test approach: unit"]))


# -- 3. a fix whose finding already passes is skipped --------------------------------


def _ctx(tmp_path) -> ToolContext:
    return ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )


def _fixed(monkeypatch) -> list[str]:
    fixed: list[str] = []

    def fake_fix(client, settings, finding, i, n, ctx, trace, reasoning):
        fixed.append(finding["issue"])

    monkeypatch.setattr(review_module, "_apply_one_fix", fake_fix)
    return fixed


def _finding(issue: str, **recheck) -> dict:
    return {"issue": issue, "severity": "blocker", "file": "a.py", "recheck": recheck}


def test_a_later_finding_whose_check_passes_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "test_a.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "test_b.py").write_text("def test_bad():\n    assert False\n")
    fixed = _fixed(monkeypatch)
    ctx = _ctx(tmp_path)
    findings = [
        _finding("first, never rechecked", tests=["test_a.py::test_ok"]),
        _finding("passing now", tests=["test_a.py::test_ok"]),
        _finding("still failing", tests=["test_b.py::test_bad"]),
        _finding("verify passes now", verify="test -f test_a.py"),
        {"issue": "a review finding", "severity": "blocker", "file": "a.py"},
    ]
    settings = Settings(workspace=str(tmp_path), goal="g", combined_fixes=False)
    review_module.apply_fixes(None, settings, findings, ctx, ctx.trace)
    assert fixed == ["first, never rechecked", "still failing", "a review finding"]
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    skipped = [e["issue"] for e in events if e["kind"] == "fix_skipped"]
    assert skipped == ["passing now", "verify passes now"]


def test_a_project_test_command_is_not_rechecked_with_plain_pytest(tmp_path, monkeypatch):
    (tmp_path / "test_a.py").write_text("def test_ok():\n    assert True\n")
    fixed = _fixed(monkeypatch)
    ctx = _ctx(tmp_path)
    ctx.test_command = "python runtests.py"
    findings = [_finding("one", tests=["test_a.py::test_ok"])] * 2
    settings = Settings(workspace=str(tmp_path), goal="g", combined_fixes=False)
    review_module.apply_fixes(None, settings, findings, ctx, ctx.trace)
    assert len(fixed) == 2


# -- combined fixes (DESIGN-LOG §100) -----------------------------------------------


class _ScriptedFixer:
    """Finishes every fix loop at once, recording each loop's task."""

    def __init__(self):
        self.tasks: list[str] = []

    def chat(self, model, messages, **kwargs):
        from agent.llm import Reply, Usage

        self.tasks.append(messages[1]["content"])
        finish = {
            "id": "f",
            "type": "function",
            "function": {"name": "finish_step", "arguments": json.dumps({"summary": "fixed"})},
        }
        return Reply(content="", tool_calls=[finish], usage=Usage(10, 5, 0.0, 0.1))


def _fix(tmp_path, findings, combined):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    ctx = _ctx(tmp_path)
    client = _ScriptedFixer()
    settings = Settings(workspace=str(tmp_path), goal="g", combined_fixes=combined)
    review_module.apply_fixes(client, settings, findings, ctx, ctx.trace)
    events = [json.loads(line) for line in ctx.trace.path.read_text().splitlines()]
    return client, events


TWO = [
    {"issue": "a is wrong", "severity": "blocker", "file": "a.py", "fix": "fix a"},
    {"issue": "b is wrong", "severity": "blocker", "file": "b.py", "fix": "fix b"},
]


def test_combined_fixes_hand_every_finding_to_one_loop(tmp_path):
    client, events = _fix(tmp_path, TWO, combined=True)
    assert len(client.tasks) == 1
    task = client.tasks[0]
    assert "a is wrong" in task and "b is wrong" in task and "Fix all 2 findings" in task
    assert "=== a.py ===" in task and "=== b.py ===" in task
    assert {e["phase"] for e in events if e["kind"] == "llm_call"} == {"fix:all"}


def test_off_each_finding_gets_its_own_loop(tmp_path):
    client, _ = _fix(tmp_path, TWO, combined=False)
    assert len(client.tasks) == 2
    assert all("Apply this fix only" in task for task in client.tasks)


def test_one_finding_takes_the_ordinary_path_either_way(tmp_path):
    client, _ = _fix(tmp_path, TWO[:1], combined=True)
    assert len(client.tasks) == 1 and "Apply this fix only" in client.tasks[0]


def test_run_hands_its_form_to_clarify(tmp_path, monkeypatch):
    """loop.run(on_ask_many=...) reaches the Interactor clarify asks through (§101)."""
    import agent.loop as loop_module

    seen: list = []

    def form(items):
        seen.append(items)
        return ["(no answer given)"] * len(items)

    def fake_clarify(client, settings, ctx, trace, **kwargs):
        ctx.interactor.ask_many([("one?", None), ("two?", ["a", "b"])])
        return ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(loop_module, "gather_clarifications", fake_clarify)
    monkeypatch.setattr(loop_module, "make_plan", lambda *a, **k: None)
    (tmp_path / "ws").mkdir()
    settings = Settings(
        workspace=str(tmp_path / "ws"),
        goal="build a thing",
        trace_dir=str(tmp_path / "traces"),
        failure_log_dir="",
        suggest_next_step=False,
        quick_answers=False,
        final_check=False,
    )
    loop_module.run(settings, on_ask=lambda q, o=None: "x", on_ask_many=form)
    assert seen == [[("one?", None, None), ("two?", ["a", "b"], None)]]


def test_a_skipped_question_in_the_form_is_not_counted_as_answered():
    """The TUI form returns "(no answer given)" for a skip (8d, §102)."""
    interactor = Interactor(ask_many=lambda items: ["yes", "(no answer given)", ""])
    interactor.ask_many([("a?", None), ("b?", None), ("c?", None)])
    assert interactor.answered == 1


# -- a fix round's after-edit check is the finding's own (DESIGN-LOG §103) -------------


def test_each_finding_kind_is_checked_its_own_way(tmp_path):
    (tmp_path / "typed.py").write_text("def f(x: int) -> int:\n    return x\n\nf('a')\n")
    (tmp_path / "test_a.py").write_text("def test_ok():\n    assert True\n")
    ctx = _ctx(tmp_path)
    check = review_module._findings_check(
        [
            {"recheck": {"types": ["typed.py"]}},
            {"recheck": {"verify": "test -f test_a.py"}},
            {"recheck": {"tests": ["test_a.py::test_ok"]}},
            {"issue": "a review finding, no recheck"},
        ]
    )
    text = check(ctx)
    assert "ty: " in text  # the type checker ran, on the finding's file
    assert "PASS — test -f test_a.py" in text
    assert "PASS — pytest -q test_a.py::test_ok" in text
    assert text.index("ty: ") < text.index("PASS — test -f") < text.index("PASS — pytest")


def test_no_recheck_means_the_ordinary_test_run():
    assert review_module._findings_check([{"issue": "review finding"}]) is None


class _EditThenFinish:
    """One turn that edits a file, then finish_step; records what each call was sent."""

    def __init__(self):
        self.sent: list[list[dict]] = []
        self.turns = [
            [
                {
                    "id": "e",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": "notes.md", "content": "x"}),
                    },
                }
            ],
            [
                {
                    "id": "f",
                    "type": "function",
                    "function": {"name": "finish_step", "arguments": json.dumps({"summary": "ok"})},
                }
            ],
        ]

    def chat(self, model, messages, **kwargs):
        from agent.llm import Reply, Usage

        self.sent.append([dict(m) for m in messages])
        return Reply(content="", tool_calls=self.turns.pop(0), usage=Usage(10, 5, 0.0, 0.1))


def test_a_fix_round_hands_back_its_own_check_after_an_edit(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    ctx = _ctx(tmp_path)
    client = _EditThenFinish()
    finding = {
        "issue": "the verify command fails",
        "severity": "blocker",
        "file": "a.py",
        "recheck": {"verify": "test -f notes.md"},
    }
    settings = Settings(workspace=str(tmp_path), goal="g", combined_fixes=False)
    review_module.apply_fixes(client, settings, [finding], ctx, ctx.trace)
    after_edit = client.sent[1][-1]["content"]
    assert "PASS — test -f notes.md" in after_edit
    assert "passed in" not in after_edit  # not the generic test run


def test_run_start_records_the_settings_and_the_harness_commit(tmp_path, monkeypatch):
    import agent.loop as loop_module

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(loop_module, "make_plan", lambda *a, **k: None)
    (tmp_path / "ws").mkdir()
    settings = Settings(
        workspace=str(tmp_path / "ws"),
        goal="g",
        trace_dir=str(tmp_path / "traces"),
        failure_log_dir="",
        suggest_next_step=False,
        quick_answers=False,
        final_check=False,
        clarify=False,
    )
    loop_module.run(settings)
    trace = next((tmp_path / "traces").glob("run-*.jsonl"))
    start = next(
        json.loads(line) for line in trace.read_text().splitlines() if '"run_start"' in line
    )
    assert start["settings"]["check_after_edit"] is True
    assert start["settings"]["combined_fixes"] is True
    assert "mercury_max_tokens" in start["settings"]
    assert not [k for k in start["settings"] if "api_key" in k or k.endswith("_key")]
    assert len(start["harness"]["commit"]) >= 7
