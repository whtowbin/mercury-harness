# Reproducing the experiments

What it takes to rerun the sweeps behind {doc}`configuration-comparison` and `FINDINGS.md`, or to
check them without running anything. Costs and times are per run, from the DESIGN-LOG
section cited; they are what Mercury 2.5 billed through OpenRouter in early October 2026 and will
drift with prices and provider load. The other half of the work is data you have to fetch yourself,
and the page says where each suite needs that.

| suite | install | needs besides the harness | cost and time per run | works on Linux |
|---|---|---|---|---|
| seeded fixtures | `uv sync` | nothing (the 16 active tasks ship in `bench/fixtures/`) | about $0.002-0.005, 30-60 s of model time | yes |
| django (SWE-bench Verified) | `uv sync` | task files and a django clone you build; Python 3.8 through uv | about $0.008-0.012, 2.4-3.8 min | the isolated driver needs macOS `sandbox-exec` |
| SciCode | `uv sync` | SciCode's problems and its 1 GB reference file; about 1 GB a round for venvs | about $0.07 a 15-problem round, about 17 min a round | the isolated driver needs macOS `sandbox-exec` |
| GAIA | `uv sync` | gated Hugging Face access; a search backend | about $0.011 and 2.3 min a question, plus search | search and driver yes; results not republishable |

## Before any sweep

```bash
git clone https://github.com/whtowbin/mercury-harness && cd mercury-harness
uv sync
uv run harness setup                          # an OpenRouter key (Mercury is served through it)
uv run pytest -n auto -m "not slow"           # no key needed, about 13 s
uv run bench list                             # the configurations and tasks
```

Every sweep spends real money. `bench run` prints an estimate and asks before it starts (`-y`
skips the question), `harness run --max-usd` caps one run, and the drivers below take their own
caps (`--max-spend` in `iso.py`, `SCI_MAX_USD`, `ISO_MAX_USD`). Read the repository's rule for a
sweep first ({doc}`benchmarking`): a history view to drop tasks at a ceiling or floor, then a
two-repeat `bench scout`, then the long run with its decision rule written first.

Which configuration is "the harness default"? The harness has review off. Among the bench
configurations that is `mercury-noreview`; `mercury-only` is the same with review on, which is what
the sweeps up to §126 called the default.

## Seeded fixtures

Sixteen small bug-fix and honesty tasks, each a seeded project with hidden tests and an
integrity check (`bench/fixtures/`, `bench/tasks.py`). They ship in the repository, so this is the
one suite that runs with nothing to download.

```bash
OUT=results/my-fixtures.jsonl
uv run bench scout --configs mercury-noreview,mercury-only            # 2 repeats, 300 s each
uv run bench run -y --configs mercury-noreview --tasks all --repeats 3 --parallel 4 --out $OUT
uv run bench report $OUT                                              # accuracy, honesty, cost, time
uv run bench report $OUT --view diagnostics
```

Cost: $0.0022 a run with review off (the default) and $0.0030 with it on, at about 32 and 46 s of model
time (§126.4). 144 runs of three configurations cost $0.61 and took under an hour. Run long
sweeps in batches of one or two repeats appended to one `--out` with `--repeat-offset`, so one bad
provider hour is spread over every arm ({doc}`benchmarking`). The fixtures sit near their accuracy
ceiling (0.90-1.00), so they are a regression guard, not a way to tell two configurations apart.

## django (SWE-bench Verified)

Ten django tasks (9296, 11790, 11815, 11848, 11880, 11885, 11951, 11964, 11999, 12708), scored by
the task's FAIL_TO_PASS and PASS_TO_PASS tests (SWE-bench is MIT licensed, django BSD-3).

**What you have to set up.** The repository does not ship the task data, because the files hold the
gold patches, and it does not ship a script that builds it. `scripts/swebench/swe.py` expects, under
`SWE_DATA` (default `traces/swebench-data/`), a bare clone `django.git` and one
`django__django-<id>.json` per task with the keys `id`, `base`, `patch`, `test_patch`, `f2p`, `p2p`,
`problem` and `eval` (the runtests command line). They come from the SWE-bench Verified rows
(`base_commit`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `problem_statement`) and a
bare `git clone --bare https://github.com/django/django django.git`; you assemble them yourself.
Each task also gets a Python 3.8 environment through `uv venv`, which uv downloads.

**Run it isolated.** An earlier setup let agents read the fix from django's git history and from
other runs' folders, and eleven wins were voided for it (§118). `scripts/swebench/iso.py` gives every
run a fresh one-commit repository made from the task's base tree, a run directory of its own, and
runs the harness under `sandbox-exec` with the home folder unreadable except the run's own paths:

```bash
cd scripts/swebench
export ISO_ARM_PUBLIC="v0.1.0"                  # an arm: a commit of this repo, then extra flags
export ISO_ARM_PUBLIC_REVIEW="v0.1.0 --review"
uv run python iso.py run mine --ids 11790,11951,12708 --arms public,public-review \
    --repeats 3 --parallel 4 --max-spend 1.50     # builds a frozen, slim harness tree per arm
uv run python iso.py table mine                   # resolved, calls and cost by arm and task
uv run python ../analysis/contamination.py --rounds mine-r1 ...   # did a run read the fix first?
```

The arms `rel` and `sep30` built into `iso.py` name commits of the private development history
(343bca8, 5f1b02a) that the public repository does not have, so define your own arms as above.
Every arm runs Mercury in every phase with the defaults of its commit plus the flags you give.

`sandbox-exec` exists on macOS only. On Linux the harness itself runs, but the isolated drivers do
not, and you would need your own jail; the unsandboxed `swe.py setup|run|score` reproduces the
older, contaminated setup and should not be used for a comparison.

Cost and time: $0.0123 and 3.8 min a run with review on, $0.0076 and 2.4 min with it off (§126.2). 150
runs of five arms cost $2.17. At 30 runs a side only a gap of six tasks or more is visible, and the same
configuration resolved 17, 19 and 13 of 30 over three sweeps (§129.6), so repeat before believing a
difference. Always pass `--web-deny` for the benchmark's own sources (the drivers do) and read
`harness trace leaks` over the traces afterwards (§80).

## SciCode

SciCode's development split (15 problems, 48 sub-steps) and test split (65 problems, 288 sub-steps),
scored with SciCode's own tests and reference data, which the agent never sees (Apache-2.0).

**What you have to set up.** Under `SCI_DATA` (default `traces/scicode-data/`), SciCode's own
files, which are not in this repository: problems_dev.jsonl and problems_test.jsonl,
drive/test_data.h5 (1 GB, from the SciCode release) and repo/ (the scicode package). Then once `uv run python sci.py env`. Every workspace gets its own `.venv`, about
1 GB a round, so keep `SCI_RUNS` out of the repository (`scripts/scicode/README.md`).

```bash
cd scripts/scicode
uv run python sci.py env
uv run python sci.py setup mine dev
SCI_MODELS="--all-mercury --no-review" uv run python sci.py run mine
uv run python sci.py score mine > mine.out
```

Cost and time: $0.070 a round of 15 problems with review on and $0.043 off, 15 rounds in 4 h 16 min,
about 17 min each (§126.3; a problem takes 2-3 min, §95.1). Single rounds vary by 2-3 problems
(5-9 of 15 over the history), so use three rounds an arm. `SCI_MAX_USD` caps a problem
(default $2.00); `sandboxed_harness.sh` is the isolated variant, again macOS only. The official
protocol (one call a step, scientist-annotated background) is `sample.py`; with it the bare model
scored 36.5% of the 288 test-split sub-steps against the 38% Artificial Analysis reports (§130.2).

## GAIA (aggregate scores only)

127 text-only validation questions, scored by GAIA's quasi-exact match. **GAIA's terms ask that the
validation set and its answers not be reshared outside a gated repository, so this repository
publishes level totals only** (59 of 127 on the release harness, 61 of 127 for the 30 Sep harness on
the same search, §96.4; per level in {doc}`configuration-comparison`). Run it for yourself, and keep
the per-question results to yourself.

**What you have to set up.** A Hugging Face account that has accepted GAIA's terms
(`gaia-benchmark/GAIA` is gated), `hf auth login`, then `uv run python scripts/gaia/gaia.py fetch`. And
web search, because GAIA needs it: Brave's API (`GAIA_SEARCH=brave`, `BRAVE_API_KEY`) or a local
SearXNG (`scripts/searxng/up.sh`, needs Docker). SearXNG is free but the engines behind it block
scripted queries within a day, and GAIA scores swung with it (22 to 15 of 40 for the same harness,
§112); use Brave for anything you want to compare across days, and run `gaia.py canary` first
(§115).

```bash
uv run python scripts/gaia/gaia.py setup mine
GAIA_SEARCH=brave uv run python scripts/gaia/gaia.py run mine
uv run python scripts/gaia/gaia.py score mine
```

Cost: about $0.011 and 2.3 min a question, $1.4-1.8 per full pass for Mercury (§93.1, §96.4), plus
about $5 of Brave searches (7-8 searches a question; §116.1 priced a pass at about $5, §115 priced
the final sweep's two passes at $10-11).

## Check the numbers without running anything

The traces are not published: they hold prompts, replies, tool arguments and fetched pages, and
GAIA's content is restricted. What is published are the per-run results and a per-call metrics
table, as assets of the `v0.1.0` release
(<https://github.com/whtowbin/mercury-harness/releases/tag/v0.1.0>):

| file | what |
|---|---|
| `mercury-harness-results-v0.1.0.tar.gz` | the per-run results files (`results/*.jsonl` and a few text summaries), as `bench` wrote them |
| `mercury-harness-calls-v0.1.0.parquet` | one row per model call or failed attempt in the django, SciCode and fixture runs: tokens, cost, latency, time to first token, provider, finish reason; numbers and labels only |
| DATA.md (a release asset) | the columns and row counts of both files, generated from them |

```bash
gh release download v0.1.0 -R whtowbin/mercury-harness -p 'mercury-harness-*' -p DATA.md
tar xzf mercury-harness-results-v0.1.0.tar.gz            # in a clone: creates results/
uv run bench report "results/*.jsonl" --view history --a mercury-only
uv run bench report results/s126-fixtures.jsonl --view compare --a mercury-only --b mercury-noreview
uv run --with duckdb python -c "
import duckdb
print(duckdb.sql('''select suite, model, sum(cost_usd) as usd, count(*) as calls
                    from 'mercury-harness-calls-v0.1.0.parquet' group by all order by usd desc'''))"
```

`DESIGN-LOG.md` and `FINDINGS.md` cite these files by name (`results/s126-dj.jsonl`, ...). The
design log's commit hashes refer to the private development history; the public repository is one
commit tagged `v0.1.0`. A file the design log names that is not in the tarball was not released. A sweep's
`results/*.jsonl` can be compared with `bench report --view compare`, which gives a bootstrap
interval and a verdict per metric.

## Sandboxing the other agents

The comparisons with mini-swe-agent, Aider, Goose, Codex CLI and Claude Code (§94, §109, §126, §127)
use one driver module per agent in `scripts/agents/` and `scripts/swebench/iso_agents.py`. An agent
with a shell can read what the harness's structured file tools cannot, so each runs inside what
`iso.py` gives a harness arm:

- a fresh one-commit repository per run, made from the task's base tree, with no other refs, so the
  fix is not in the history; HEAD is checked against the base tree and the status must be clean
  before the run starts;
- `sandbox-exec` with the home folder unreadable except the run's own directory, the agent's tools
  and state, and the uv, git and keychain files it needs (`agents.sandboxed()`), an allowlisted
  environment, and a temp folder of its own;
- the hidden test patch applied by the scorer from stdin, so no copy of it is ever in a workspace;
- no network confinement. Goose's shell has the web, so `leaks()` is read after the run, and
  `--web-deny`/`harness trace leaks` apply to the harness arms;
- Codex runs with its native web search off, Claude Code with six tools, no connectors and its own
  sandbox, and a guard on the weekly allowance it reads from the stream.

`scripts/analysis/contamination.py` reports a run that saw a gold line before writing it. Before the
isolation, eleven resolved runs had read the fix from git history, other worktrees or the answer key
and were voided (§118); after it, §126's check found none. The isolation is macOS-only and was built
to answer one contamination question, not as a security boundary: do not point it at code you do not
trust.

## What this does not cover

- No one-command reproduction. The suites above need data you assemble and, for two of them, a Mac.
- The release candidate's exact sweep was run from a private tree; the design log's commit hashes
  name it. Rerunning on the public tag gives the same configurations, not the same runs, and
  Mercury is sampled at temperature 1.0 whatever you ask (§85, §125.1).
- The provider changed under the runs: Mercury's cached share of input rose from 5-22% to 50-72% on
  2 Oct with no harness change (§122), and Brave replaced SearXNG for GAIA. Compare same-day arms only.
