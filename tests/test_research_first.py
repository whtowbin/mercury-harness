"""Research before planning (agent/research.py, DESIGN-LOG §77)."""

from __future__ import annotations

import json

from agent.config import Settings
from agent.phases.plan import _planner_system
from agent.research import PLAN_RESEARCH_FIRST, ResearchGate, goal_libraries


def _call(name: str, **args) -> dict:
    return {"id": "c", "function": {"name": name, "arguments": json.dumps(args)}}


def test_the_goal_s_own_imports_name_the_libraries_to_look_up():
    goal = "Use only:\nimport pandas as pd\nfrom sklearn.linear_model import Ridge\nimport math\n"
    assert goal_libraries(goal) == ["pandas", "sklearn.linear_model"]


def test_numpy_and_scipy_are_not_looked_up_on_the_gate_s_orders():
    """Their lookups were trivia (np.zeros, np.sqrt) and no failure misused them (§79.2)."""
    goal = "import numpy as np\nfrom scipy.integrate import solve_ivp\n"
    assert goal_libraries(goal) == []
    gate = ResearchGate(goal)
    assert gate.hold(_call("submit_plan", draft=[], steps=[])) is None


def test_the_standard_library_and_the_workspace_s_own_code_are_not_libraries(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "helpers.py").write_text("")
    goal = "import heapq\nfrom dataclasses import dataclass\nimport shop.cart\nimport helpers\n"
    assert goal_libraries(goal, tmp_path) == []
    assert goal_libraries("import requests", tmp_path) == ["requests"]


def test_a_goal_that_uses_no_library_is_never_held():
    """The first version held greenfield `expr.py` and `semver.py` plans (§79)."""
    gate = ResearchGate("Create expr.py with evaluate(expression: str) -> float.")
    assert gate.hold(_call("submit_plan", draft=[], steps=[])) is None


def test_reading_the_workspace_does_not_count_as_checking_a_library():
    gate = ResearchGate("import pandas as pd")
    gate.observe(_call("read_file", path="README.md"))
    assert gate.hold(_call("submit_plan", draft=[], steps=[])) is not None


def test_a_plan_submitted_without_any_lookup_is_held_once_with_concrete_lookups():
    gate = ResearchGate("import pandas as pd\nfrom sklearn.linear_model import Ridge")
    held = gate.hold(_call("submit_plan", draft=[], steps=[]))
    assert held and 'inspect_module("sklearn.linear_model")' in held
    assert gate.hold(_call("submit_plan", draft=[], steps=[])) is None  # once


def test_a_planner_that_looked_something_up_is_not_held():
    gate = ResearchGate("import pandas as pd")
    gate.observe(_call("inspect_module", module="pandas"))
    assert gate.hold(_call("submit_plan", draft=[], steps=[])) is None


def test_the_research_order_puts_library_docs_before_the_web():
    text = " ".join(PLAN_RESEARCH_FIRST.split())
    assert text.index("inspect_module") < text.index("read_skill") < text.index("web_search")


def test_the_block_is_opt_in_and_the_default_prompt_is_unchanged():
    off = _planner_system(Settings(workspace=".", goal="g"))
    on = _planner_system(Settings(workspace=".", goal="g", research_first=True))
    assert PLAN_RESEARCH_FIRST not in off
    assert on.endswith(PLAN_RESEARCH_FIRST) or PLAN_RESEARCH_FIRST in on


def test_a_provider_preference_reaches_the_request_body():
    """Against a real local server, not a mocked post (the user's rule for I/O)."""
    import http.server
    import threading

    from agent import llm

    bodies: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply = json.dumps(
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}
        forced = {"type": "function", "function": {"name": "t"}}
        for sort in ("latency", ""):
            client = llm.OpenRouterClient(
                api_key="x", base_url=base, stream=False, provider_sort=sort
            )
            client.chat(model="z-ai/glm-5.3-flash", messages=[{"role": "user", "content": "hi"}])
            client.chat(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                tools=[tool],
                tool_choice=forced,
            )
    finally:
        server.shutdown()
    assert bodies[0]["provider"] == {"sort": "latency"}
    assert "provider" not in bodies[2]
    # tool_choice travels with tools, whatever the provider preference: inserting the
    # provider line once moved it under `if self.provider_sort`, so a run without a sort
    # stopped forcing the last word's tool, and one with a sort sent tool_choice with no
    # tools (caught by a background security review).
    assert "tool_choice" not in bodies[0] and "tool_choice" not in bodies[2]
    assert bodies[1]["tool_choice"] == forced and bodies[3]["tool_choice"] == forced


# -- findings: what the planner looked up reaches every executor step whole --------


def _loop_messages(module: str, member: str, result: str) -> list[dict]:
    """A planning loop's history around one inspect_module call, as runner builds it."""
    args = json.dumps({"module": module, "member": member})
    return [
        {"role": "system", "content": "s"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "inspect_module", "arguments": args}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "inspect_module", "content": result},
    ]


def test_the_signature_comes_from_a_real_inspect_module_result(tmp_path):
    """Parsed from the tool's real output, not a hand-written imitation of it."""
    from agent.approval import Approver
    from agent.research import looked_up_signatures
    from agent.tools import ToolContext
    from agent.tools.introspect import inspect_module
    from agent.trace import Trace
    from agent.workspace import Workspace

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    real = inspect_module(ctx, "json", "dumps")

    [(label, signature)] = looked_up_signatures(_loop_messages("json", "dumps", real))

    assert label == "json.dumps"
    assert signature.startswith("json.dumps(obj") and "indent" in signature


def test_a_looked_up_signature_the_steps_use_is_added_when_the_planner_left_it_out():
    from agent.research import complete_findings

    lookups = [("scipy.integrate.solve_ivp", "solve_ivp(fun, t_span, y0, method='RK45')")]
    steps = json.dumps([{"intent": "integrate with solve_ivp over t_span"}])

    added = complete_findings([], lookups, steps)
    assert added == [
        "scipy.integrate.solve_ivp: solve_ivp(fun, t_span, y0, method='RK45') (inspect_module)"
    ]
    # Already stated by the planner, or never used by the steps: nothing added.
    assert complete_findings(["solve_ivp returns .y (inspect_module)"], lookups, steps) == [
        "solve_ivp returns .y (inspect_module)"
    ]
    assert complete_findings([], lookups, json.dumps([{"intent": "use odeint"}])) == []


def test_the_handoff_drops_whole_items_never_cuts_one():
    from agent.research import FINDINGS_HANDOFF_CHARS, findings_handoff

    items = [f"fact {i}: " + "x" * 600 for i in range(5)]
    block = findings_handoff(items)
    assert len(block) <= FINDINGS_HANDOFF_CHARS + 200
    kept = [line for line in block.splitlines() if line.startswith("- ")]
    assert kept and all(line.endswith("x" * 600) for line in kept)
    assert len(kept) < len(items)
    assert findings_handoff([]) == ""


def test_findings_are_offered_only_with_research_first_and_the_default_schema_is_untouched():
    from agent.research import with_findings
    from agent.tools.control import SUBMIT_PLAN, submit_plan

    assert "findings" not in SUBMIT_PLAN.parameters["properties"]
    assert "findings" in with_findings(SUBMIT_PLAN).parameters["properties"]
    assert "findings" not in SUBMIT_PLAN.parameters["properties"]

    from types import SimpleNamespace

    ctx = SimpleNamespace(result={}, done=False)
    submit_plan(ctx, draft=[], steps=[], findings=["a (web)"])
    assert ctx.result["findings"] == ["a (web)"]
    submit_plan(ctx, draft=[], steps=[])
    assert "findings" not in ctx.result


def test_the_plan_carries_findings_and_every_step_prompt_shows_them(tmp_path):
    from agent.approval import Approver
    from agent.config import Settings
    from agent.phases.execute import _step_prompt
    from agent.phases.plan import finalize_plan
    from agent.research import findings_handoff
    from agent.tools import ToolContext
    from agent.trace import Trace
    from agent.workspace import Workspace

    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False),
    )
    settings = Settings(workspace=str(tmp_path), goal="g", research_first=True)
    steps = [
        {"id": s, "title": s, "intent": "i", "acceptance": "a", "files": ["a.py"], "verify": ""}
        for s in ("one", "two")
    ]
    plan = finalize_plan(
        {"draft": [], "steps": steps, "findings": ["k = 1.38e-23 J/K (web)"]},
        settings,
        ctx,
        ctx.trace,
    )
    assert plan.findings == ["k = 1.38e-23 J/K (web)"]
    for i, step in enumerate(plan.steps):
        prompt = _step_prompt(ctx, plan, step, i, 4_000, draft=findings_handoff(plan.findings))
        assert "k = 1.38e-23 J/K (web)" in prompt


def test_the_prompt_asks_for_conventions_before_lookups():
    text = " ".join(PLAN_RESEARCH_FIRST.split())
    assert text.index("conventions") < text.index("inspect_module")
    assert "np.random.seed" in text and "decision:" in text
