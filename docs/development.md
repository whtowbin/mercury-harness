# Development

## Layout

```text
agent/
  cli.py         argument parsing, `harness run`/`config`/`skills`
  tui.py         the live Textual dashboard (`--tui`)
  loop.py        orchestrator: plan -> execute -> review -> fix
  runner.py      the generic tool-calling loop every phase is built from
  llm.py         OpenRouter client: retries, streaming, usage and cost accounting
  prompts.py     every prompt, in one file, so they are easy to A/B
  workspace.py   the path jail, and nested-workspace detection
  approval.py    the shell gate
  toolchain.py   locating ruff/ty/pytest, and the isolated subprocess environment
  validate.py    ast.parse -> Ruff -> ty, and the automatic post-write check
  guards.py      subprocess time, disk, write size and error budgets
  diffs.py       pre-run snapshots and the run diff
  trace.py       JSONL run record
  interact.py    ask_user's pluggable backend (stdin, TUI, or "no TTY" fallback)
  skills.py      loading, selecting and rendering skill files
  userconfig.py  persisted CLI preferences and the user skills directory
  phases/        plan.py (chain of draft) / execute.py (chunked) / review.py (large chunks) / schedule.py (parallel steps)
  tools/         fs.py, shell.py, checks.py, introspect.py, vcs.py, ask.py,
                 control.py (terminal tools), base.py (registry, context, cache)
  skills_data/   uv.md, ruff.md, ty.md, verify.md — bundled with the package,
                 override with --skills-dir
bench/
  cli.py         `bench list`/`run`/`report`, and the per-model concurrency gate
  tasks.py       task definitions and the interface contract each one pins
  fixtures/      hidden tests and timed workloads, one directory per task
  configs.py     named agent configurations to compare
  runner.py      one run: temp workspace, agent, scoring, one flat record
  metrics.py     accuracy (JUnit XML), execution timing, lint and type health
  report.py      aggregation and the comparison table
tests/           tool-layer, loop-wiring, streaming, isolation and benchmark tests
docs/            this site
```

## Running the checks

```bash
uv run pytest -q                      # no API key needed; the network is stubbed
uv run pytest -n auto -m "not slow"   # same suite, minus a few OS-timing tests, in parallel
uv run ruff check . && uv run ruff format .
uv run ty check .
```

The parallel form is for iteration (~13s vs ~85s sequential, measured); see
DESIGN-LOG §27-28 for the reasoning.
Run plain `pytest -q` — every test, one at a time — before a commit or in CI.

This project uses the Astral toolchain throughout — `uv` for the environment
and dependencies, `ruff` for linting and formatting, `ty` for type checking —
never plain `pip`/`venv`.

## Where the time goes

Wall-clock time is dominated by two things the harness cannot make faster on its own:
model round trips and subprocess time (pytest, ty, git). The harness's own Python was
profiled twice and is not a CPU bottleneck anywhere (DESIGN-LOG §20, §26): the one real
cost was unpruned filesystem walks into `.venv` and `node_modules`, fixed in §20 (69x on
the manifest walk), and the test suite's own slowness was subprocess isolation, fixed by
parallelising (§28) and moving the TUI tests in-process (§39). The original investigation
is recorded in DESIGN-LOG §20. Measure with `bench report`'s
`model` and `s/call` columns before assuming the harness is slow.

## Packaging

The project builds as `mercury-harness`, installing two commands (`harness`, `bench`).

```bash
uv build              # writes dist/*.whl and dist/*.tar.gz
uv publish            # needs a PyPI token: uv publish --token <pypi-token>,
                       # or PYPI_TOKEN / UV_PUBLISH_TOKEN in the environment
```

**The one thing worth knowing if you touch packaging.** `agent/skills.py` and
`bench/tasks.py` both resolve their data directories relative to their own
`__file__`, not the repo root — `agent/skills_data/*.md` and
`bench/fixtures/**` ship inside the wheel as package data
(`[tool.hatch.build.targets.wheel] artifacts` in `pyproject.toml`) for exactly
this reason. A repo-root-relative path resolves to a real directory in a
source checkout and to nothing at all once `pip install`ed — the tool
installs cleanly and then silently runs with no skills anywhere but the
original checkout. `tests/test_toolchain.py` asserts the package-relative path
directly.

## Adding a benchmark task

1. Write the goal in `bench/tasks.py`, pinning the module path and every
   function signature the hidden tests will import.
2. Put the hidden tests in `bench/fixtures/<name>/hidden_test.py`.
3. Optionally add `perf.py`, printing `{"seconds": ...}` for the
   execution-speed axis.
4. **Validate the hidden tests against a reference implementation before
   trusting them.** A wrong hidden test silently penalises every agent, and
   the benchmark will look authoritative while being wrong.

## Building these docs locally

```bash
uv run --with-requirements docs/requirements.txt sphinx-build -b html docs docs/_build/html
```
