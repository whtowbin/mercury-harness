#!/usr/bin/env python3
"""Differential replay: prove a behaviour-preserving change changed nothing.

A live benchmark sweep of ~30 runs cannot see a small behavioural change (its
own power estimate says ~100 runs a side for a modest gap) and its noise is the
model's sampling. This removes the model: the same seeded, scripted inputs go
through the code in two working trees, and every message, outcome and trace
event is compared exactly.

    # in the tree under test, and again in a checkout of the baseline commit
    git worktree add /tmp/baseline <commit> && (cd /tmp/baseline && uv sync)
    uv run python scripts/replay_diff.py dump new.json
    (cd /tmp/baseline && env -u VIRTUAL_ENV uv run python "$OLDPWD/scripts/replay_diff.py" \\
        dump base.json)
    # a control: the baseline against itself, so nondeterminism is not mistaken
    # for a difference
    (cd /tmp/baseline && env -u VIRTUAL_ENV uv run python "$OLDPWD/scripts/replay_diff.py" \\
        dump base2.json)
    uv run python scripts/replay_diff.py compare base.json new.json --control base2.json

Targets (`--targets`, default all):
    loop    agent.runner.run_tool_loop with a scripted model: nudges, stalls,
            compaction, the red-first gate, budget and guard exceptions
    plan    agent.phases.plan.finalize_plan on random plans
    audit   agent.testcheck.audit (mutation and old-tests ledger)

What is deliberately kept out because it is nondeterministic: `grep` (ripgrep
returns files in a varying order), wall-clock fields, and temp paths. Coverage is
printed so a target that exercised nothing cannot pass silently.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import Any

# Fields that differ run to run for reasons that are not behaviour.
VOLATILE = {"t", "ts", "time", "elapsed", "elapsed_s", "latency_s", "wall_s", "path", "at"}
VOLATILE |= {"seconds", "duration_s", "total_s", "ttft_s"}
NUDGE_KINDS = ("nudge", "stalled", "loop_exhausted", "tdd_gate_held", "repeated_call", "compacted")


def _events(trace, tmp: Path) -> str:
    trace._fh.flush()
    events = []
    for line in trace.path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(
            {k: v for k, v in event.items() if k not in VOLATILE and not isinstance(v, float)}
        )
    return json.dumps(events, sort_keys=True, default=str).replace(str(tmp), "<T>")


def _new_context(tmp: Path):
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    trace = Trace(tempfile.mkdtemp(), echo=False)
    return ToolContext(workspace=Workspace(tmp), approver=Approver(auto_approve=True), trace=trace)


# --- loop ---------------------------------------------------------------------

_TOOLS = [
    "read_file", "read_file", "read_files", "edit_file", "edit_file", "glob_files", "list_dir",
    "run_python", "write_file", "bad_tool", "probe", "probe", "probe", "tests",
]  # fmt: skip
_PROBES = ["from a import x\nprint(x)", "import a\nprint(a.x)", "from b import x"]


def _tool_call(rng: random.Random, counter: list[int]) -> dict:
    kind = rng.choice(_TOOLS)
    path = rng.choice(["a.py", "b.py", "c.py"])
    value = rng.randint(1, 4)
    args: dict = {
        "read_file": {"path": path, "offset": rng.randint(1, 3)},
        "read_files": {"paths": [path, "b.py"]},
        "edit_file": {
            "path": path,
            "old_string": f"x = {value}",
            "new_string": f"x = {rng.randint(1, 4)}",
        },
        "glob_files": {"pattern": "*.py"},
        "list_dir": {"path": "."},
        "run_python": {"code": f"print({value})"},
        "write_file": {"path": path, "content": f"x = {value}\n"},
        "bad_tool": {"nothing": 1},
        "probe": {"code": rng.choice(_PROBES)},
        "tests": {},
    }[kind]
    name = {"probe": "run_python", "tests": "run_tests"}.get(kind, kind)
    counter[0] += 1
    return {"id": f"c{counter[0]}", "function": {"name": name, "arguments": json.dumps(args)}}


def _script(rng: random.Random):
    from agent.llm import Reply, Usage

    usage = Usage(10, 5, 0.001, 0.1)
    counter = [0]
    replies = []
    for i in range(rng.randint(3, 14)):
        if rng.random() < 0.12:
            replies.append(Reply(content=rng.choice(["", "thinking"]), usage=usage))
            continue
        calls = [_tool_call(rng, counter) for _ in range(rng.choice([1, 1, 1, 2, 3]))]
        replies.append(Reply(content=f"step {i}", tool_calls=calls, usage=usage))
    if rng.random() < 0.7:
        counter[0] += 1
        finish = {"summary": "s", "verified": "false"}
        call = {
            "id": f"c{counter[0]}",
            "function": {"name": "finish_step", "arguments": json.dumps(finish)},
        }
        replies.append(Reply(content="fin", tool_calls=[call], usage=usage))
    return replies


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, model, messages, tools=None, tool_choice=None, temperature=None,
             max_tokens=None, reasoning_effort=None):  # fmt: skip
        from agent.llm import Reply, Usage

        if not self.replies:
            return Reply(content="done", usage=Usage(10, 5, 0.001, 0.1))
        return self.replies.pop(0)


def dump_loop(seed: int) -> dict:
    from agent.runner import run_tool_loop
    from agent.tools import execution_registry

    rng = random.Random(seed)
    tmp = Path(tempfile.mkdtemp())
    for name in ("a.py", "b.py", "c.py"):
        (tmp / name).write_text("x = 1\nprint(x)\n")
    (tmp / "test_a.py").write_text(
        "from a import x\n\n"
        "def test_one():\n    assert x == 2\n\n"
        "def test_two():\n    assert x == 3\n"
    )
    ctx = _new_context(tmp)
    options: dict[str, Any] = {
        "phase": rng.choice(["execute:s", "fix:1", "plan", "review"]),
        "max_iters": rng.choice([4, 8, 12]),
        "prompt_token_budget": rng.choice([None, None, 700]),
        "salvage": rng.random() < 0.3,
        "tdd_tests": rng.choice([None, None, ["test_a"]]),
        "max_usd": rng.choice([None, None, 0.004]),
    }
    try:
        result, messages = run_tool_loop(
            client=_Client(_script(rng)), model="fake", system="sys", user="usr",
            registry=execution_registry(), ctx=ctx, trace=ctx.trace, **options,
        )  # fmt: skip
        outcome: list = ["ok", repr(result)[:300]]
    except Exception as exc:
        outcome, messages = ["exc", type(exc).__name__, str(exc)[:200]], None
    text = lambda obj: json.dumps(obj, sort_keys=True, default=str).replace(str(tmp), "<T>")  # noqa: E731
    return {"outcome": text(outcome), "messages": text(messages), "events": _events(ctx.trace, tmp)}


# --- plan ---------------------------------------------------------------------

_VERIFY = [
    "", "pytest", "python -m pytest tests/", "uv run pytest -q", "ruff --version",
    "python --version",
    "pytest test_a.py::test_one", "uv run ruff check .", "pytest tests -x", "   ", "echo ok",
    "python -c 'print(1)'", "pytest -k foo",
]  # fmt: skip


def dump_plan(seed: int) -> dict:
    from agent.config import Settings
    from agent.phases.plan import finalize_plan

    rng = random.Random(seed)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.py").write_text("x = 1\n")
    (tmp / "test_a.py").write_text("from a import x\n\ndef test_one():\n    assert x == 1\n")
    if rng.random() < 0.5:
        (tmp / "pyproject.toml").write_text("[project]\nname='p'\nversion='0'\n")
    ctx = _new_context(tmp)
    settings = Settings(workspace=str(tmp), goal="do the thing")
    settings.scope_step_verify = rng.random() < 0.7
    steps: list = []
    for i in range(rng.randint(0, 4)):
        if rng.random() < 0.2:
            steps.append("prose step")
        steps.append({
            "id": f"s{i}", "title": f"t{i}", "intent": "i", "acceptance": "a",
            "verify": rng.choice(_VERIFY),
            "files": rng.sample(["a.py", "test_a.py", "b.py"], rng.randint(0, 2)),
            "writes": rng.sample(["a.py", "b.py"], rng.randint(0, 2)),
        })  # fmt: skip
    try:
        plan = finalize_plan({"steps": steps, "draft": ["d1", "d2"]}, settings, ctx, ctx.trace)
        outcome = json.dumps(dataclasses.asdict(plan), sort_keys=True, default=str)
    except Exception as exc:
        outcome = f"EXC {type(exc).__name__}: {str(exc)[:100]}"
    return {"outcome": outcome, "messages": "", "events": _events(ctx.trace, tmp)}


# --- audit --------------------------------------------------------------------

_OLD = dedent("""
    def price(units):
        if units <= 1000:
            return units * 10
        return units * 5
""")
_NEW = _OLD.replace("units <= 1000", "units < 1000")
_NEW2 = _NEW.replace("units * 5", "units * 6")
_STRONG = dedent("""
    from pricing import price

    def test_tiers():
        assert price(10) == 100
        assert price(1000) == 5000
        assert price(2000) == 10000
""")
_LOOSENED = _STRONG.split("assert price(1000)")[0]
_VACUOUS = "from pricing import price\n\ndef test_x():\n    assert price(1) > 0\n"
_SAME_POWER = _STRONG.replace("price(10) == 100", "price(11) == 110")
_ADDED = _STRONG + "\ndef test_added():\n    assert price(1000) == 5000\n"
_SOURCES = [(_OLD, _NEW), (_OLD, _NEW2), (_NEW, _NEW)]
_TESTS = [
    (_STRONG, _STRONG), (_STRONG, _LOOSENED), (_STRONG, _SAME_POWER),
    (_STRONG, _VACUOUS), (_STRONG, _ADDED), (_LOOSENED, _STRONG),
]  # fmt: skip


def dump_audit(case: int) -> dict:
    from agent import testcheck
    from agent.diffs import Baseline

    (source_before, source_after), (tests_before, tests_after) = list(
        itertools.product(_SOURCES, _TESTS)
    )[case]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "pricing.py").write_text(source_after)
    (tmp / "test_pricing.py").write_text(tests_after)
    baseline = Baseline(snapshots={"pricing.py": source_before, "test_pricing.py": tests_before})
    try:
        result = testcheck.audit(tmp, baseline, ["pricing.py", "test_pricing.py"])
        fields = dataclasses.asdict(result)
        fields.pop("seconds", None)
        outcome = json.dumps(fields, sort_keys=True, default=str).replace(str(tmp), "<T>")
    except Exception as exc:
        outcome = f"EXC {type(exc).__name__}: {str(exc)[:200]}"
    return {"outcome": outcome, "messages": "", "events": ""}


TARGETS = {
    "loop": (dump_loop, lambda n: range(n)),
    "plan": (dump_plan, lambda n: range(n)),
    "audit": (dump_audit, lambda n: range(len(_SOURCES) * len(_TESTS))),
}


# --- commands -----------------------------------------------------------------


def cmd_dump(args: argparse.Namespace) -> int:
    out: dict[str, dict] = {}
    for name in args.targets:
        function, cases = TARGETS[name]
        for case in cases(args.seeds):
            out[f"{name}:{case}"] = function(case)
    Path(args.out).write_text(json.dumps(out, sort_keys=True))
    print(f"wrote {len(out)} cases to {args.out}")
    return 0


def _differing(a: dict, b: dict) -> list[str]:
    return [k for k in a if a[k] != b.get(k)]


def _first_difference(a: dict, b: dict) -> str:
    for field in ("outcome", "messages", "events"):
        x, y = a[field], b[field]
        if x != y:
            i = next(
                (i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q),
                min(len(x), len(y)),
            )
            around = slice(max(0, i - 80), i + 160)
            return f"{field} @ {i}\n  A: {x[around]}\n  B: {y[around]}"
    return ""


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    differing = _differing(a, b)
    print(f"{len(a)} cases; A vs B differing: {len(differing)}")
    if args.control:
        control = json.loads(Path(args.control).read_text())
        print(f"control (A vs {args.control}) differing: {len(_differing(a, control))}")
    events = Counter(
        e.get("kind") for v in a.values() if v["events"] for e in json.loads(v["events"])
    )
    print(
        "coverage:",
        {k: n for k, n in sorted(events.items()) if any(w in str(k) for w in NUDGE_KINDS)},
    )
    outcomes = Counter(json.loads(v["outcome"])[0] for k, v in a.items() if k.startswith("loop:"))
    print("loop outcomes:", dict(outcomes))
    for key in differing[:3]:
        print(f"\n{key}: {_first_difference(a[key], b[key])}")
    return 1 if differing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    dump = sub.add_parser("dump", help="run the scripted inputs through this tree's code")
    dump.add_argument("out")
    dump.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    dump.add_argument("--seeds", type=int, default=300, help="cases for loop and plan")
    compare = sub.add_parser("compare", help="compare two dumps; exit 1 if any case differs")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--control", help="a second dump of A's tree, to expose nondeterminism")
    args = parser.parse_args(argv)
    return cmd_dump(args) if args.cmd == "dump" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
