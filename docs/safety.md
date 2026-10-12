# Safety model

## The path jail does not constrain subprocesses

This is the thing to understand before pointing the harness anywhere real.
`Workspace` refuses any file path outside the workspace root, symlinks
included — but a shell command is a different matter, because Python build
tools search *upward*. `uv pip install` with no virtualenv in the workspace
resolves to the nearest one above it and installs there.

That is not hypothetical: an early run of this harness used `sandbox/` inside
the project directory, and the agent's `uv pip install -e .` installed its own
package into the harness's own `.venv` and evicted a dependency, breaking the
run in progress. Three defences now exist:

- **Nested workspaces are refused.** If any ancestor of the workspace holds a
  `pyproject.toml` or a `.venv`, the run stops before the first model call
  (`agent.workspace.NestedWorkspaceError`). `--allow-nested` overrides it.
- **`uv pip install/uninstall/sync` is blocked** unless the workspace owns a
  `.venv`, with an error steering to `uv add`. Pinning `VIRTUAL_ENV` is *not*
  sufficient on its own — uv ignores a pin to a directory that does not exist
  and resumes searching upward.
- **Subprocesses get a scrubbed environment.** `PYTHONPATH`, `PYTHONHOME` and
  any inherited `VIRTUAL_ENV` are dropped, and `UV_PROJECT_ENVIRONMENT` points
  into the workspace (`agent/toolchain.py::workspace_env`).

## Shell approval

`run_bash` runs allowlisted commands (`uv`, `ruff`, `ty`, `pytest`, `git`,
read-only shell tools) freely and prompts for anything else; a denylist blocks
destructive forms outright (`agent/config.py::BASH_ALLOWLIST`,
`BASH_DENYLIST_SUBSTRINGS`). `-y`/`--yes` skips the prompt — use it only on a
throwaway workspace.

A compound command (`;`, `&&`, `||`, `|`, backticks, `$(...)`) is never made
safe by an allowlisted first word: `git status && rm -rf ~/.ssh` would run the
whole thing the moment `git` is recognised if that were allowed, so a compound
command always needs an explicit approval, matching the exact
command/snippet — approving one `git` invocation does not blanket-approve every
future command that starts with `git`.

`run_python` (a Python snippet with the workspace importable) carries the same
risk through a different door: it is arbitrary Python, not shell syntax, so the
allowlist/denylist don't apply — but it goes through the same
approve-or-deny gate.

### `--auto-approve-python`: a narrower trust level than `-y`

`run_python` is what an executor step reaches for constantly to check its own
work — running the function it just wrote, importing a module to confirm it
parses — and on a task with several steps that means the same approve-or-deny
prompt many times over, for calls that are all routine verification rather
than anything a plan hadn't already scoped. `--auto-approve-python` skips
that gate for `run_python` specifically, while `run_bash` (installs, git,
anything else) keeps gating normally — a narrower trust level than `-y`,
which approves both. `run_python` is only ever offered during the execute and
fix phases (`agent/tools/__init__.py::execution_registry`), never planning, so
every call it gates already happens after a plan exists.

This is a trust decision, not a safety classifier: it does not inspect what
the snippet actually does before approving it, so treat it the same as `-y` —
appropriate for a throwaway workspace, not for pointing the agent at
something you care about.

## Everything else

- **Spend cap.** Checked before every model call; the run stops rather than overruns.
- **Resource guards** (`agent/guards.py`): a per-step subprocess time budget, a
  workspace size ceiling, a per-file write cap, and a per-step error budget
  that abandons a step going nowhere.
- **Loop detection.** Three identical tool calls in one step and the model is
  told so directly, rather than letting it spin.
- **A stream deadline on every model call.** A provider's keep-alive comments
  can hold a streaming HTTP connection open past the point it's making
  progress — the request-level read timeout resets on each one. A wall-clock
  deadline independent of that (`agent/llm.py::OpenRouterClient._stream_once`)
  catches it instead of letting one stuck call run indefinitely.
- **Tools never raise.** Every failure comes back to the model as a readable
  error string, because a recoverable error is a turn the model can use.
