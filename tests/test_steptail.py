"""The dynamic tail's state sections, the planner tail, and Mercury's output cap
(DESIGN-LOG §83). The sections are read from a loop's own tool calls and results,
in the shapes run_tool_loop records them."""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent.phases.plan import Step
from agent.steptail import TailItems, planner_tail, step_tail
from bench import configs

STEP = Step(id="evict", title="Evict the oldest", intent="i", acceptance="test_evict passes")

FAIL = """____ test_evict ____
cache.py:12: in evict
    return self.items.pop()
E   IndexError: pop from empty list
FAILED test_cache.py::test_evict - IndexError: pop from empty list
==== 1 failed, 3 passed in 0.02s ===="""
PASS = "==== 4 passed in 0.02s ===="


def turn(n: int, name: str, args: dict, result: str) -> list[dict]:
    """One assistant tool call and its result, as the loop's history holds them."""
    cid = f"c{n}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": cid, "content": result},
    ]


def history(*turns: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for t in turns:
        out += t
    return out


BOTH = TailItems(errors=True, ledger=True)


def test_without_items_the_tail_is_the_round_one_tail():
    msgs = history(turn(1, "run_tests", {}, FAIL))
    assert step_tail(STEP, 2, 14, msgs) == step_tail(STEP, 2, 14)


def test_a_failing_run_is_named_with_where_it_failed():
    text = step_tail(STEP, 2, 14, history(turn(1, "run_tests", {}, FAIL)), TailItems(errors=True))
    assert "failing: test_cache.py::test_evict" in text
    assert "where: cache.py:12 in evict -- IndexError: pop from empty list" in text
    assert "in a row" not in text  # one run is not a streak


def test_the_same_failure_twice_is_called_out():
    msgs = history(
        turn(1, "run_tests", {}, FAIL),
        turn(2, "edit_file", {"path": "cache.py", "old_string": "a", "new_string": "b"}, "ok"),
        turn(3, "run_tests", {}, FAIL),
    )
    assert "failed 2 runs in a row" in step_tail(STEP, 4, 14, msgs, TailItems(errors=True))


def test_no_failure_section_once_the_tests_pass():
    msgs = history(turn(1, "run_tests", {}, FAIL), turn(2, "run_tests", {}, PASS))
    assert "<last_test_failure>" not in step_tail(STEP, 3, 14, msgs, TailItems(errors=True))


def test_the_ledger_counts_edits_misses_and_test_runs():
    msgs = history(
        turn(1, "edit_file", {"path": "cache.py"}, "error: old_string not found in cache.py."),
        turn(2, "edit_files", {"edits": [{"path": "cache.py"}, {"path": "test_cache.py"}]}, "ok"),
        turn(3, "run_tests", {}, PASS),
    )
    text = step_tail(STEP, 4, 14, msgs, TailItems(ledger=True))
    assert "edited: cache.py x2, test_cache.py x1" in text
    assert "edits that did not apply: 1" in text
    assert "test runs: 1; last: ==== 4 passed in 0.02s ====" in text


def test_nothing_done_yet_adds_no_ledger():
    msgs = history(turn(1, "read_file", {"path": "cache.py"}, "x = 1"))
    assert "<work_so_far>" not in step_tail(STEP, 2, 14, msgs, TailItems(ledger=True))


def test_the_rules_stay_last_after_the_state():
    msgs = history(turn(1, "run_tests", {}, FAIL))
    text = step_tail(STEP, 2, 14, msgs, BOTH)
    assert (
        text.index("<current_step>")
        < text.index("<last_test_failure>")
        < text.index("<work_so_far>")
    )
    assert text.rstrip().endswith("</rules>")
    assert text.index("<work_so_far>") < text.index("<rules>")


def test_the_planner_tail_repeats_the_goal_and_counts_down():
    text = planner_tail("Add eviction to the cache", 3, 12)
    assert "<goal>Add eviction to the cache</goal>" in text
    assert "Turns left: 9 of 12." in text
    assert "submit the plan now" in planner_tail("g", 11, 12)
    assert planner_tail("x" * 5000, 1, 12).count("x") < 700  # a reminder, not a copy


# -- Mercury's output cap -------------------------------------------------------


def test_the_cap_is_sent_for_mercury_only_and_never_overrides_an_explicit_value():
    import http.server
    import threading

    from agent.llm import OpenRouterClient

    bodies: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenRouterClient(
            "k",
            base_url=f"http://127.0.0.1:{server.server_port}",
            stream=False,
            mercury_max_tokens=16_000,
        )
        msgs = [{"role": "user", "content": "hi"}]
        client.chat("inception/mercury-2.5", msgs)
        client.chat("inception/mercury-2.5", msgs, max_tokens=500)
        client.chat("z-ai/glm-5.3-flash", msgs)
        client.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert [b.get("max_tokens") for b in bodies] == [16_000, 500, None]


# -- the arms --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "control", "fields"),
    [
        ("mercury-inception-tail-errors", "mercury-inception-tail", {"executor_tail_items"}),
        ("mercury-inception-tail-ledger", "mercury-inception-tail", {"executor_tail_items"}),
        ("mercury-inception-tail-plan", "mercury-inception-tail", {"planner_tail"}),
        # mercury-inception-cap16k is retired: the cap became the default (§95.1). The
        # controls that price the two §95.1 defaults take its place.
        ("mercury-no-output-cap", "mercury-only", {"mercury_max_tokens"}),
        ("mercury-no-stall-bound", "mercury-only", {"mercury_content_stall_s"}),
    ],
)
def test_each_arm_is_its_control_plus_one_thing(name, control, fields):
    base = dataclasses.asdict(configs.CONFIGS[control])
    arm = dataclasses.asdict(configs.CONFIGS[name])
    assert {k for k in base if base[k] != arm[k]} - {"name", "description"} == fields


def test_the_failure_section_reads_a_real_run_tests_result(tmp_path):
    """The fixture above is hand-written pytest output; run_tests reformats it.
    This runs the real tool on a real failing test and parses what it returns."""
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.tools.checks import run_tests
    from agent.trace import Trace
    from agent.workspace import Workspace

    (tmp_path / "cache.py").write_text("def evict(items):\n    return items.pop()\n")
    (tmp_path / "test_cache.py").write_text(
        "from cache import evict\n\ndef test_evict():\n    assert evict([]) is None\n"
    )
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    text = step_tail(STEP, 2, 14, history(turn(1, "run_tests", {}, run_tests(ctx))), BOTH)
    assert "failing: test_cache.py::test_evict" in text
    assert "where: cache.py:2 in evict -- IndexError: pop from empty list" in text
    assert "test runs: 1; last: 1 failed in" in text


# -- the metrics the §83 arms are judged on --------------------------------------


def test_diagnostics_count_calls_by_phase_output_and_stalls():
    from bench import diagnostics

    def llm(phase, out, finish="tool_calls"):
        return {
            "kind": "llm_call",
            "phase": phase,
            "completion_tokens": out,
            "finish_reason": finish,
        }

    events = [
        llm("plan", 100),
        llm("execute:s1", 60_000),
        llm("execute:s1", 16_000, "length"),
        llm("fix:f1", 50),
        llm("review", 10),
        {"kind": "stalled", "phase": "execute:s1"},
    ]
    d = diagnostics.from_events(events)
    assert (d.plan_calls, d.execute_calls, d.fix_calls, d.review_calls) == (1, 2, 1, 1)
    assert d.out_tokens == 76_160 and d.max_call_out_tokens == 60_000
    assert d.length_stops == 1 and d.stalls == 1


def test_compare_reports_the_per_phase_metrics():
    from bench.report import render_compare

    def rec(config, execute, fix, plan, acc, t):
        return {
            "config": config,
            "accuracy": acc,
            "calls": execute + fix + plan,
            "diag_execute_calls": execute,
            "diag_fix_calls": fix,
            "diag_plan_calls": plan,
            "started_at": t,
        }

    records = [rec("a", 9, 7, 2, 1.0, i) for i in range(6)] + [
        rec("b", 6, 5, 4, 1.0 if i else 0.5, i) for i in range(6)
    ]
    text = render_compare(records, "a", "b")
    assert "execute+fix calls" in text and "plan calls" in text and "perfect runs" in text


# -- round three: api facts, unchecked library calls, key facts (DESIGN-LOG §84) ---

from agent.steptail import api_section, library_calls, with_key_fact  # noqa: E402

ORTHO = Step(
    id="qr",
    title="Orthonormalize with gram_schmidt",
    intent="Implement gram_schmidt(A) returning Q",
    acceptance="Q.T @ Q is the identity",
    signatures=["gram_schmidt(A: np.ndarray) -> np.ndarray"],
)


def test_api_facts_keep_those_that_name_the_step():
    facts = (
        "np.linalg.qr(A) returns (Q, R); gram_schmidt must match its signs",
        "scipy.special.sph_harm takes (m, n, azimuth, polar)",
    )
    text = api_section(ORTHO, facts)
    assert "gram_schmidt must match" in text and "sph_harm" not in text


def test_api_facts_fall_back_to_all_when_none_match_and_stay_capped():
    text = api_section(ORTHO, ("unrelated fact one", "unrelated fact two"))
    assert "fact one" in text and "fact two" in text
    assert len(api_section(ORTHO, tuple("x" * 400 for _ in range(10)))) < 1_000


def test_library_calls_follow_aliases_and_skip_the_stdlib():
    code = (
        "import numpy as np\nfrom scipy import special, linalg as la\nimport math\n"
        "def f(x):\n    return np.linalg.norm(x) + special.jv(0, x) + la.qr(x) + math.sqrt(x)\n"
    )
    assert library_calls(code) == ["np.linalg.norm", "special.jv", "la.qr"]
    assert library_calls("import numpy as np\nx = np.array([1]) + np.dot(a, b)\n") == []


def test_uninspected_calls_in_new_code_are_named_until_looked_up():
    edit = turn(
        1,
        "edit_file",
        {"path": "solution.py", "old_string": "pass", "new_string": "    return special.jv(0, x)"},
        "ok",
    )
    file_text = "from scipy import special\n"
    items = TailItems(inspect=True)
    before = step_tail(ORTHO, 2, 14, history(edit), items, file_text=file_text)
    assert "special.jv" in before and "<unchecked_library_calls>" in before
    looked = turn(2, "inspect_module", {"module": "scipy.special", "member": "jv"}, "jv(v, z)")
    after = step_tail(ORTHO, 3, 14, history(edit, looked), items, file_text=file_text)
    assert "<unchecked_library_calls>" not in after


def test_the_key_fact_is_repeated_and_the_schema_asks_for_it():
    step = Step(id="s", title="t", intent="i", acceptance="a", key_fact="azimuth before polar")
    assert "<key_fact>azimuth before polar</key_fact>" in step_tail(
        step, 1, 14, None, TailItems(key=True)
    )
    assert Step.from_dict({"id": "s", "key_fact": "radians"}, 0).key_fact == "radians"

    from agent.tools import planning_registry

    submit = next(t for t in planning_registry().tools() if t.name == "submit_plan")
    props = with_key_fact(submit).parameters["properties"]["steps"]["items"]["properties"]
    assert "key_fact" in props


def test_calls_written_into_test_files_are_not_flagged():
    edit = turn(
        1,
        "write_file",
        {"path": "test_solution.py", "content": "from scipy import special\nspecial.jv(0, 1)\n"},
        "ok",
    )
    assert "<unchecked_library_calls>" not in step_tail(
        ORTHO, 2, 14, history(edit), TailItems(inspect=True)
    )
