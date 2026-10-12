---
name: uv
description: Using uv for Python environments and dependencies inside this workspace.
# `pip` is deliberately absent: triggers match at a word start, so it fired on
# every goal mentioning a "pipe" (a real, recurring one here — a flappy-bird app
# workspace), pulling 476 tokens of dependency guidance into steps about game
# geometry. Anything genuinely about pip is also about installing, a
# dependency, or a package, all of which are listed.
triggers: [pyproject.toml, uv.lock, uv, install, dependency, depend, package, requirement, venv, environment]
phases: [execute, fix, plan]
---

uv is a fast Python package and project manager. It replaces pip, pip-tools,
pipx, virtualenv and poetry. Use it for anything touching dependencies or
environments. Adapted from Astral's uv skill for this harness.

## In this workspace specifically

**pytest is already available. Do not install it.** `run_tests` resolves pytest
through `uvx` whether or not the workspace declares it. `uv add --dev pytest` is
a wasted round trip, and it is the single most common wasted call in this agent.

**`uv pip install` and `uv pip uninstall` are blocked.** They install into the
first virtualenv found by searching *upward* from the working directory, which
can be an environment outside the workspace. Use `uv add`, which targets this
project. The harness will refuse the command and tell you this.

**You are already in the workspace root.** Do not prefix commands with `cd`.

## Commands

```bash
uv add requests            # add a dependency (records it in pyproject.toml)
uv add --dev pytest-cov    # add a dev dependency
uv remove requests         # remove one
uv sync                    # install from the lockfile
uv run <command>           # run inside the project environment
uv run python script.py    # run a script in the project environment
uv init                    # create a new project here
uvx <tool>                 # run a tool without installing it
```

## Rules

- Never `pip install`. Use `uv add`, so the dependency lands in `pyproject.toml`.
- Never run `python` directly when the workspace is a uv project; use `uv run python`.
- Prefer the `run_python` tool over `uv run python -c "..."`: no shell quoting,
  no approval prompt, and it already runs with the workspace importable.
- Do not hand-manage `.venv` with `python -m venv` or `source activate`.
- Only add a dependency the task actually needs. Prefer the standard library.

## Documentation

https://docs.astral.sh/uv/llms.txt
