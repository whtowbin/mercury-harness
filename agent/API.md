# Agent-to-Agent API

This document describes how to use this agent as a CLI for other models to call, including how to pass a pre-made plan to skip the planning phase.

## Overview

The agent consists of several phases:
1. **Clarify** (optional) — ask clarifying questions about the goal
2. **Plan** — generate a multi-step plan
3. **Execute** — run each step of the plan
4. **Review** — verify the work and fix failures
5. **Final Check** — re-verify everything at the end

Normally all phases run. You can skip the planning phase by providing a pre-made plan.

## Passing a Plan from Another Agent

### Usage

```bash
other-model-cli generate-plan | harness run --workspace /path/to/workspace --stdin-plan
```

Or save the plan to a file first:

```bash
other-model-cli generate-plan > plan.json
harness run goal --workspace /path/to/workspace --plan plan.json
```

### Plan JSON Schema

The plan is a JSON object with this structure:

```json
{
  "goal": "what the plan is trying to accomplish",
  "draft": ["optional", "reasoning steps", "the model", "took to create the plan"],
  "steps": [
    {
      "id": "step-1",
      "title": "A short title for this step",
      "intent": "What this step is for (the Why)",
      "acceptance": "How you'll know this step is done",
      "signatures": [
        "List of function/class signatures this step should write, e.g. 'def foo(x: int) -> str'",
        "Use these if the planner analyzed existing code"
      ],
      "tests": [
        "Test names or descriptions this step should make pass, e.g. 'test_foo_returns_string'",
        "Optional but helpful for verification"
      ],
      "files": [
        "List of files this step touches or creates",
        "src/main.py",
        "tests/test_main.py"
      ],
      "writes": [
        "Subset of 'files' that this step may modify.",
        "Empty or missing means 'all of them' (conservative)",
        "src/main.py"
      ],
      "verify": "A shell command to verify this step is correct, e.g. 'uv run pytest tests/test_main.py'",
      "depends_on": [
        "step-0"
      ]
    }
  ]
}
```

### Field Descriptions

**Top level:**
- `goal` (string, required) — What the plan is trying to accomplish. Will override the command-line goal if provided.
- `draft` (array of strings, optional) — The reasoning steps the planner took. Purely informational; goes in the trace but doesn't affect execution.
- `steps` (array, required) — The list of steps to execute.

**Per step:**
- `id` (string, required) — Unique identifier for this step. Used for dependencies and tracing.
- `title` (string, required) — A short human-readable title.
- `intent` (string, required) — Why this step exists—the goal it's accomplishing within the larger plan.
- `acceptance` (string, required) — How you'll know this step is done. What success looks like.
- `signatures` (array of strings, optional) — Function/class signatures the step should write, extracted from existing code or the requirements. Helps the executor understand the interface contract.
- `tests` (array of strings, optional) — Test names or descriptions that should pass after this step. Helps with verification.
- `files` (array of strings, optional) — Files this step touches or creates. If empty, treated as "all files".
- `writes` (array of strings, optional) — Subset of `files` this step may modify. If empty or missing, treated as "all files" (the conservative reading).
- `verify` (string, optional) — A shell command to verify the step is correct. Examples: `uv run pytest tests/test_main.py`, `ruff check src/`, `py.test -xvs`. Empty means no verification.
- `depends_on` (array of strings, optional) — IDs of steps that must finish before this one. Allows parallel execution of independent steps.

### Minimal Valid Plan

```json
{
  "goal": "add a hello function",
  "steps": [
    {
      "id": "step-1",
      "title": "Write hello function",
      "intent": "Provide a hello() function that returns a greeting",
      "acceptance": "hello() returns 'Hello, World!'",
      "files": ["hello.py"],
      "verify": "uv run python -c \"from hello import hello; assert hello() == 'Hello, World!'\""
    }
  ]
}
```

## Exit Codes

- `0` — Run completed successfully and passed all checks.
- `1` — Run failed, or completed but did not pass its own checks (failing tests, failed verify commands, etc.).
- `2` — Invalid arguments or configuration.
- `130` — Interrupted by user (Ctrl-C).

## Example: Agent-to-Agent Pipeline

Generate a plan with one model, execute with another:

```bash
#!/bin/bash
WORKSPACE=/tmp/agent-sandbox

# Model A generates the plan
PLAN=$(mercury-model-cli \
  --prompt "Add a --verbose flag to the CLI" \
  --format json-plan)

# Model B executes it
echo "$PLAN" | harness run \
  --workspace "$WORKSPACE" \
  --executor anthropic/claude-opus-5 \
  --stdin-plan
```

This lets you:
- Use a cheaper/faster model for planning
- Use a specialized model (e.g., a code-specialized variant) for execution
- Run the planner and executor on different hardware
- Implement custom planning logic in your own system

## Input/Output

### Input

When using `--stdin-plan`, the agent reads a single JSON document from stdin until EOF. This blocks the input stream, so provide the complete plan before the harness reads anything else.

### Output

The agent prints logs and progress to stdout/stderr during the run. At the end, if you pass `--out summary.json`, the summary is written as JSON to that file.

The summary includes:
- Whether the run passed its own checks (`ok`)
- Any error message (`error`)
- Cost and token usage (`totals`)
- What was built (`session_summary`)
- Detailed step results (`steps`)
- Review findings (`findings`)
- A suggested next step (`suggested_next_step`)

## Notes

### Plan Immutability

Once a plan is loaded (from file or stdin), it is not modified during execution. The executor may:
- Mark steps as passed/failed
- Record verify results
- Add error messages

But it cannot add new steps or change the plan structure.

### Step Dependencies

Steps with no `depends_on` or an empty `depends_on` run as soon as all their dependencies finish. The executor respects the dependency graph and may run independent steps in parallel (controlled by `--step-concurrency`).

### Verify Commands

Verify commands must be shell commands (bash/sh). They are run in the workspace root. Exit code 0 = pass, nonzero = fail.

Common verify patterns:
- `uv run pytest tests/` — Run the test suite
- `ruff check src/` — Lint the code
- `uv run python -c "..."` — Quick inline Python check
- `git diff --check` — Verify formatting

### Tool Probes Are Rejected

Verify commands that only check whether a tool exists (e.g., `--version`, `which`, `pip show`) are rejected by the verify gate. The executor needs to verify the *work is correct*, not that the tool is installed.

### Error Handling

If a step fails its verify command and you have `--escalate` enabled (the default), the executor retries with reasoning turned up. If that still fails, the step is marked failed and the final check phase may attempt to fix it.

If a step hits a tool error (e.g., the test runner crashes, a subprocess times out), those errors are counted. Once a step hits `--max-step-errors` (default 12), the step is abandoned and execution moves to the next one.

## See Also

- `harness run --help` — All CLI flags
- `agent/phases/plan.py` — Step and Plan dataclass definitions
- `agent/loop.py` — The run orchestrator
