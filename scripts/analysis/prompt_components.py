"""Where a call's prompt tokens go, and which parts the recorded runs ever use (DESIGN-LOG §129).

  uv run python scripts/analysis/prompt_components.py [TRACE_DIR ...]

No model call. Prints the tokens (approx_tokens, the harness's own estimate) of the executor's and
the planner's fixed parts: the system prompt and each tool schema. With trace directories it adds,
for each execution tool, how many recorded calls used it, so a schema that costs tokens on every
call and is called almost never shows up at the top.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from agent import prompts  # noqa: E402
from agent.llm import approx_tokens  # noqa: E402
from agent.tools import execution_registry, planning_registry  # noqa: E402
from agent.tracetools import sql_connection  # noqa: E402

CALLS = "select data->>'tool' tool, count(*) n from events where kind = 'tool_call' group by 1"


def schema_tokens(registry) -> dict[str, int]:
    return {
        s["function"]["name"]: approx_tokens(json.dumps(s["function"])) for s in registry.schemas()
    }


def calls_by_tool(dirs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dirs:
        for tool, n in sql_connection(Path(d)).sql(CALLS).fetchall():
            out[tool] = out.get(tool, 0) + n
    return out


def main(dirs: list[str]) -> None:
    used = calls_by_tool(dirs) if dirs else {}
    for label, system, registry in (
        ("executor", prompts.EXECUTOR_SYSTEM, execution_registry()),
        ("planner", prompts.PLANNER_SYSTEM, planning_registry()),
    ):
        schemas = schema_tokens(registry)
        print(f"\n{label}: system prompt {approx_tokens(system):,} tokens (template, unfilled)")
        print(f"  {len(schemas)} tool schemas, {sum(schemas.values()):,} tokens in all")
        for name, tokens in sorted(schemas.items(), key=lambda kv: -kv[1]):
            print(f"    {name:28s} {tokens:5d} tokens  {used.get(name, 0):6d} recorded calls")


if __name__ == "__main__":
    main(sys.argv[1:])
