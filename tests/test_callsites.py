"""CallSiteGate: a rename plan must write every file that uses the name.

Built after a rename-callers run whose plan assigned three of five caller
modules and never mentioned the others (DESIGN-LOG §62.1). Tested against the
real rename-callers seed, not a toy.
"""

from __future__ import annotations

import json
import shutil

import pytest

from agent.callsites import CallSiteGate, files_using, names_to_follow, uncovered
from bench import tasks


class _Trace:
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))


@pytest.fixture
def seed(tmp_path):
    task = tasks.get("rename-callers")
    shutil.copytree(task.seed_dir, tmp_path, dirs_exist_ok=True)
    return tmp_path, task.goal


def _submit(steps):
    return {
        "function": {"name": "submit_plan", "arguments": json.dumps({"draft": [], "steps": steps})}
    }


def test_the_renamed_name_and_every_file_using_it_are_found(seed):
    root, goal = seed
    names = names_to_follow(goal, root)
    assert names == {"make_line"}
    uses = files_using(root, names)["make_line"]
    assert {"pricing.py", "cart.py", "invoice.py", "promo.py", "report.py", "export.py"} <= uses


def test_a_goal_that_changes_no_name_is_left_alone(seed):
    root, _ = seed
    assert names_to_follow("Fix the bug in make_line() when qty is zero.", root) == set()


@pytest.mark.parametrize("task", ["ledger-multi", "slowsuite", "throttle"])
def test_bug_fix_goals_that_only_mention_signatures_do_not_trigger(task):
    """Each triggered the first version: `remove` inside remove_and_report,
    "without changing any public signature", "do not change the class's
    public signature", "should not delete or skip any of them"."""
    t = tasks.get(task)
    assert names_to_follow(t.goal, t.seed_dir) == set()


def test_no_other_bench_goal_triggers():
    triggered = {
        name
        for name, t in tasks.TASKS.items()
        if t.seed_dir is not None and names_to_follow(t.goal, t.seed_dir)
    }
    assert triggered == {"rename-callers"}


def test_a_name_the_workspace_does_not_define_is_ignored(seed):
    root, _ = seed
    assert names_to_follow("Rename `fetch_all()` to `fetch()` everywhere.", root) == set()


def test_the_real_missed_callers_plan_is_held_once(seed):
    """The §62.1 plan: pricing, cart, invoice and the test, nothing else."""
    root, goal = seed
    trace = _Trace()
    gate = CallSiteGate(goal, root, trace)
    plan = _submit(
        [
            {"id": "rename-make-line", "writes": ["pricing.py"]},
            {"id": "update-cart", "writes": ["cart.py"]},
            {"id": "update-invoice", "writes": ["invoice.py"]},
            {"id": "update-tests", "writes": ["test_pricing.py"]},
        ]
    )
    held = gate.hold(plan)
    assert held is not None
    for missing in ("promo.py", "report.py", "export.py"):
        assert missing in held
    assert gate.hold(plan) is None, "held once; a resubmission goes through"
    assert trace.events[0][0] == "callsite_check"


def test_a_plan_that_covers_every_file_goes_straight_through(seed):
    root, goal = seed
    gate = CallSiteGate(goal, root, _Trace())
    files = sorted(files_using(root, {"make_line"})["make_line"])
    assert gate.hold(_submit([{"id": "all", "writes": files}])) is None


def test_steps_without_writes_count_their_files():
    missing = uncovered({"f": {"a.py", "b.py"}}, [{"id": "s", "files": ["a.py", "./b.py"]}])
    assert missing == {}


def test_the_gate_chain_runs_both_gates_and_names_the_one_that_held(seed):
    from agent.runner import GateChain

    root, goal = seed

    class Never:
        event = "never_held"
        observed = 0

        def observe(self, call):
            Never.observed += 1

        def hold(self, call):
            return None

    chain = GateChain(Never(), None, CallSiteGate(goal, root, _Trace()))
    plan = _submit([{"id": "s", "writes": ["pricing.py"]}])
    assert chain.hold(plan) is not None
    assert chain.event == "callsite_gate_held"
    chain.observe(plan)
    assert Never.observed == 1


def test_the_check_is_on_by_default():
    """On after DESIGN-LOG §66.1; mercury-no-callsite-check keeps the old
    behaviour runnable so a later sweep can price it again."""
    from agent.config import Settings
    from bench import configs

    assert Settings(workspace=".", goal="g").callsite_check is True
    assert configs.get("mercury-only").callsite_check is True
    assert configs.get("mercury-no-callsite-check").callsite_check is False


def test_a_declared_path_loses_only_a_leading_dot_slash():
    """`lstrip("./")` stripped characters, so `.config.py` became `config.py` and
    `../x.py` became `x.py`."""
    uses = {"f": {".config.py", "x.py"}}
    steps = [{"id": "s", "writes": ["./.config.py", "../x.py"]}]
    assert uncovered(uses, steps) == {"f": ["x.py"]}
