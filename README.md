# mercury-harness

A plan/execute coding agent for Mercury 2.5, Inception's diffusion language model, and the
benchmark that measured it. The agent plans a task as small, fully specified steps, has Mercury
carry out each one, and ends with a final check that makes no model call. The benchmark is a
set of seeded tasks, an A/B method with pre-registered decision rules and trace tooling.

**Research code, not a product.** What it found: the model sets the ceiling, so the agent
changed what a run cost far more than what it solved (with Mercury, a 100-line loop matched
this 33,000-line harness on django issues); most mechanisms added on intuition did nothing
measurable; and a diffusion model's stalls, not its speed, decide wall time. A stronger model
was the only lever that separated anything. This is an independent project, not affiliated
with or endorsed by Inception Labs.

## Install

Needs git, [uv](https://docs.astral.sh/uv/), Python 3.11 or later, and macOS or Linux (Windows
is not supported). It is installed from GitHub, not PyPI:

```bash
uv tool install git+https://github.com/whtowbin/mercury-harness
```

Optional extras: `tui` (the dashboard), `analysis` (`harness trace sql`) and `keyring` (system
keyring on Linux), e.g. `uv tool install 'mercury-harness[tui] @ git+https://github.com/whtowbin/mercury-harness'`.

## Set up and run

```bash
harness setup        # choose OpenRouter or Inception, paste a key; it is checked and stored
mkdir -p ~/agent-sandbox
harness run -w ~/agent-sandbox "build a CLI todo list with JSON persistence and pytest tests"
```

`harness setup` stores the key in the macOS Keychain, the Linux system keyring (with the
`keyring` extra) or a file only you can read. The workspace must be outside any Python project.
By default every phase uses Mercury, review is off (`--review` turns it on) and the final check
runs. [Quickstart](https://mercury-harness.readthedocs.io/en/latest/quickstart.html) and
[usage](https://mercury-harness.readthedocs.io/en/latest/usage.html) cover keys, follow-ups,
the dashboard, preferences and reading traces.

## Read more

- **Docs site**: <https://mercury-harness.readthedocs.io> (the same pages are in [docs/](docs/)).
- **Blog**: [Mercury 2.5: a cheap diffusion LLM with a hard ceiling](https://www.henrytowbin.com/posts/2026-mercury-coding-agent/)
  and its [supplement](https://www.henrytowbin.com/notes/2026-mercury-coding-agent-supplement/) with the full tables.
- **[FINDINGS.md](FINDINGS.md)**: the results in one table, each with its run count, interval
  where there was a rule, section and results file.
- **[DESIGN-LOG.md](DESIGN-LOG.md)**: how it was built and what each step measured. Read its
  "Start here" list, then only the section you need.
- **[TUNABLES.md](TUNABLES.md)**: every load-bearing constant, with its evidence or a plain
  "not measured".
- **[docs/configuration-comparison.md](docs/configuration-comparison.md)**: every optional flag,
  where it was tested and the outcome.
- **[docs/reproduce.md](docs/reproduce.md)**: how to rerun each suite, what it costs, and how to
  check the numbers against the released result files.
- **[CHANGELOG.md](CHANGELOG.md)**: what v0.1.0 contains.

## Data

Per-run results and a per-call metrics table are assets of the
[v0.1.0 release](https://github.com/whtowbin/mercury-harness/releases/tag/v0.1.0). No traces are
published, and GAIA appears as aggregate scores only.

## How the agent works

Three phases and a check that makes no model call. A planner explores read-only and submits
a structured plan of small, fully specified steps; each step is executed in its own small
context with only its declared files; a reviewer sees whole files and hands each finding
back as a scoped fix; then a deterministic final check re-runs every step's verify command,
the project's tests and a type check, and sends failures back through the fix path.
Details: [docs/architecture.md](docs/architecture.md), [docs/safety.md](docs/safety.md).

By default Mercury does the planning and execution phases, and the final check runs; the
review phase is off unless you pass `--review` (it cost about a fifth of the spend and time and
no suite showed a loss without it, DESIGN-LOG §124.10). With all three phases on the harness's own fixtures that matched a
hybrid with a stronger planner and reviewer at a fraction of the cost and time; on real
django issues and SciCode problems a stronger planner or research-first planning helps, and
the single-run numbers behind that are in FINDINGS.md with their caveats.

## History

Built in three weeks of September 2026, largely by Claude Code sessions working under
Henry Towbin's direction; the design log is where the sessions argued with the data.
Several sessions ran at once, which is why the log is written in sections. This repository is
published as one commit: the development history (about 400 commits) stays private, so a commit
hash cited in the design log refers to that history.

## Reproduce the experiments

[docs/reproduce.md](docs/reproduce.md) has per-suite setup (django, SciCode and GAIA need data
you fetch yourself), costs and times. To check the numbers without running anything:

```bash
git clone https://github.com/whtowbin/mercury-harness && cd mercury-harness && uv sync
gh release download v0.1.0 -R whtowbin/mercury-harness -p 'mercury-harness-results-*'
tar xzf mercury-harness-results-v0.1.0.tar.gz   # results/: the per-run records the docs cite
uv run pytest -n auto -m "not slow"             # 2,500+ tests, no key needed, ~13s
uv run bench report "results/*.jsonl" --view history --a mercury-only
```

[docs/benchmarking.md](docs/benchmarking.md) has the report views and the recipe for long
sweeps. The toolchain is uv, Ruff, ty, complexipy and prek (`prek install` once per clone);
[docs/development.md](docs/development.md) has the layout.

## Documentation

[Quickstart](docs/quickstart.md), [usage](docs/usage.md), [CLI reference](docs/cli.md),
[skills](docs/skills.md), [architecture](docs/architecture.md), [safety model](docs/safety.md),
[benchmarking](docs/benchmarking.md), [agent-to-agent API](agent/API.md) (pass a pre-made plan
as JSON). Build the site locally with
`uv run --with-requirements docs/requirements.txt sphinx-build -b html docs docs/_build/html`.

## License

Apache 2.0, see [LICENSE](LICENSE).
