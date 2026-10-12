# Benchmarking agent configurations

`bench` compares whole agent configurations — which model plans, executes and
reviews, how hard each thinks, whether review runs at all — on four axes that
trade against each other.

```bash
uv run bench list
uv run bench run --configs hybrid,mercury-only --tasks semver,expr --repeats 3 --parallel 4
uv run bench run --configs ablation --tasks all      # a whole group at once
uv run bench report results/bench-*.jsonl --by task
```

Configurations come in five groups. `core` is `mercury-only`, the shipped
default, which every other config is read against; `ablation` switches one
accuracy mechanism off at a time to price it (`mercury-raw`,
`mercury-noverify`, `mercury-noreview`, `mercury-nogate`, `mercury-parallel`,
`mercury-nano`, `mercury-noskills`, `mercury-high`); `perf` does the same for
the work-avoidance mechanisms (`perf-noverifyskip`, `perf-noscope`,
`perf-noreplay`, `perf-notypecheck`, and `perf-neither` for the total);
`executor` swaps the model out entirely; and `mixed` splits roles between
models. Pass a group name to `--configs` to run all of it.

**A config exists to isolate one thing.** `core` holds the only control, so a
sweep should never carry a second copy of it — `perf-baseline` was
byte-identical to `mercury-only` and a sweep over both groups silently paid
for the same twelve jobs twice. Where a config still varies more than one
field it is either a deliberate combination (`mercury-raw` is the floor,
`perf-neither` the total) or one change that takes several fields to express
(swapping the model sets three). `tests/test_bench.py` enforces both: no two
active configs may be identical, and every sweepable setting must have some
config that turns it off, because a knob nothing prices is carried on faith.

Configs are retired rather than deleted (`retired=True` keeps them
`get`-able by name but out of `in_group`). Twenty-four are retired today: the
Sonnet and Haiku ones, which cost roughly 20x `mercury-only` for a 33% score
across two sweeps; the cheap-model swaps that lost to GLM; and the whole
former `fix` group, whose every member varied `planner_reasoning` *as well
as* its own mechanism and so isolated nothing. Comparing those runs found no
clear effect from that setting, on unequal samples across different task
mixes — which is an argument for sweeping reasoning as an axis, not for
carrying a parallel config per mechanism.

`bench run` warns when `--repeats` is under 3 — cost varied 40-65% run to run
on identical inputs in this harness's own measurements, which is not
something a single run can see past.

## Scout first, then a targeted run

The repo rule is to buy power with tasks that separate the arms, not
with more repeats on tasks that cannot. Before any sweep of more than a few dozen
runs:

```bash
bench report results/*.jsonl --view history --a mercury-only   # ceilings and floors
bench scout --configs mercury-only,mercury-X                    # 2 repeats, 300s limit
bench scout --configs mercury-only,mercury-X --plan-only        # a question about plans
```

`bench scout` (`bench/scout.py`) reads failures first (every error, timeout and
stall, with its cause; a config that times out is dropped, repeated stalls or a
sleeping machine stop it) and then gives each task a reading: `separates on X`,
`ceiling`, `floor` or `moves`. Only surviving tasks go into the long run, with its
decision rule written in DESIGN-LOG first (§61).

`--plan-only` (on `bench run` and `bench scout`) stops each run once a plan is
accepted and does not score it: for questions about the plans themselves, such as
what the reproduce check did, at about a tenth of a full run's cost (§65). A
planning-only scout prints the `repro` view instead of accuracy readings.

## Long sweeps: run them in batches

The provider stalled 15% of runs across one five-hour sweep (DESIGN-LOG §54.3), and a
stall that lands on one block of tasks looks like a treatment effect. Run a long sweep
as several batches of one or two repeats each, appended to one `--out`, so any bad
hour is spread across every arm:

```bash
OUT=results/my-sweep.jsonl
bench run -y --configs a,b --tasks all --repeats 2 --out $OUT
bench run -y --configs a,b --tasks all --repeats 2 --repeat-offset 2 --out $OUT
bench run -y --configs a,b --tasks all --repeats 2 --repeat-offset 4 --out $OUT
```

`bench report` leaves provider failures out of every figure and names them first
(`--include-stalls` counts them), so a stalled batch costs runs, not accuracy.
`bench archive results/my-sweep.jsonl --name my-sweep` keeps the records and traces
together for later.

## Views on a sweep

```bash
bench report results/bench-*.jsonl                       # outcome: accuracy, cost, time
bench report results/bench-*.jsonl --view diagnostics    # how the run worked
bench report results/bench-*.jsonl --view discrimination # which tasks separated anything
bench report results/bench-*.jsonl --view reliability    # does it work every time
bench report results/bench-*.jsonl --view failures       # how the runs that failed ended
bench report results/bench-*.jsonl --view compare --a mercury-only --b mercury-nodigest
bench report results/*.jsonl --view history --a mercury-only --since 2026-09-24
bench report results/s65-*.jsonl --view repro             # the planner's reproduce check
```

Accuracy alone is a poor measure of an agent, and this project's sweeps have
sat at its ceiling for most of their history. These views are how to get past
it; each is computed from the records and the run's own trace, with no model
call.

`--view diagnostics` (`bench/diagnostics.py`) reports process signals: model
calls, `run_tests` calls, edits after a failure that did not reduce it, the
longest run of those in a row, exhausted loops, nudges, step retries, read-only
calls before the first edit, planning tool calls, and tool-call success. It
backfills from `trace_path`, so it works on sweeps recorded before the counters
existed.

`--view discrimination` says which tasks earned their budget: the largest gap
between any two configs on accuracy, calls and honesty, "no" only when all
three are flat.

`--view reliability` separates `pass@k` (some repeat got everything right) from
`pass^k` (every repeat did). On 114 `mercury-only` runs across six existing-code
tasks it is 100% against 50%: every task is solvable, half are reliably solved,
and the mean accuracy of 96% hid that entirely. The point is from Redis's
agentic-testing guide, which asks for scoring the path and the consistency, not
only the destination.

`--view history` answers the question the repo rule asks before a delta is called a regression: what a config normally scores on each task, over every file given, and which tasks are at a ceiling (cannot separate two configs) or a floor. `--since DATE` drops runs from before a fixture changed.

`--view repro` counts, per config and task, how often the planner's reproduce
check ran and what it showed (reproduced, passed, broken, left empty). On a
wrong-report task `passed` is the check working; on a real bug it is a correct plan
sent back (§65).

`--view failures` labels each run once, worst cause first: `ok`,
`tests_miss_change` (passed, but its own tests pass on the original bug),
`dishonest`, `wrong_fix_believed_done` (its own suite was green and the hidden
tests fail), `gave_up_red`, `loop`, `crashed`. The two worth telling apart have
opposite remedies: `wrong_fix_believed_done` is a diagnosis problem,
`gave_up_red` an execution one.

`--view compare --a X --b Y` is the one to read before believing a difference.
It gives a bootstrap interval for the gap on each metric, says "differs" only
when the interval excludes zero, and otherwise says how many runs a side it
would take to see a gap that size. Twice a gap of 12 and 21 points of honesty at
24 runs a side turned out to be noise; the second of those, put through this
view, reads "not distinguishable (~60 a side to see this gap)".

**Two arms must overlap in time.** Provider latency drifted 2.3x between two
sweeps an hour apart. `bench run` now orders jobs repeat-major with the configs
being compared innermost (`plan_jobs`), so they run side by side, and `compare`
warns when two arms' `started_at` ranges do not overlap. Before that fix every
sweep here ran one arm to completion before the other began (0% time overlap,
measured), including the ones this log called interleaved.

## Digging into a sweep's raw traces

`bench report` answers the aggregate question (accuracy/cost/honest-rate by
config or task); the *individual* traces behind a sweep land in
`traces/bench/`, one `run-*.jsonl` per job — the exact same format and event
schema as a real `harness run` trace, since `bench` uses the same `Trace`
class underneath. That means `harness trace report`/`harness trace grep`
(see `agent/tracetools.py`) work on a bench sweep directly, with no
conversion:

```bash
harness trace report traces/bench --failed-only   # which sweep jobs failed, and why
harness trace grep traces/bench "some recurring error text"
```

**Reach for these before reading raw JSONL by hand or writing a one-off
jq/grep pass** — that manual-digging path is exactly what `tracetools.py`
was built to replace, and it already works here unmodified. Extend
`agent/tracetools.py` itself (composable, tested functions) rather than a
sweep-specific script, so the same analysis works on a real run's traces and
a bench sweep's without being written twice.

## `--timeout` and `--max-per-model`: running a sweep without waiting on a slow model

Every job runs in its own spawned subprocess (`--timeout SECONDS`, default
180) so a hung model call can be killed outright — a blocked HTTP read cannot
be interrupted from inside the same process, threads included, so a
subprocess is the only thing the parent can unconditionally terminate. A
killed run is scored as a timeout (`t/out` in the report) rather than
vanishing from the results.

**A sweep across several configs can still starve itself even with a timeout
set.** Two configs that share an underlying model (say, a hybrid's planner and
a same-model-only baseline) compete for that one account's throughput when
`--parallel` runs them at once — the requests queue at the provider rather
than failing outright, which looks exactly like a hang. `--max-per-model N`
(default 3) caps concurrent jobs against the same model regardless of
`--parallel`, gating job starts in the scheduler rather than limiting HTTP
concurrency directly, which works whether a job executes in a thread or (via
`--timeout`) a subprocess.

**Two related failure modes worth knowing about, both since mitigated:**

- A provider that only sends keep-alive comments without ever finishing a
  response can hold a streaming call open indefinitely — the per-chunk read
  timeout resets on every comment. `OpenRouterClient` now enforces an overall
  wall-clock deadline on a single call, independent of that (see {doc}`safety`).
- The kill path's timing bounds (verified in isolation, with a test that
  spawns a SIGTERM-resistant process and asserts the whole terminate/kill/reap
  sequence stays under ~25s) were observed to degrade substantially under
  heavy concurrent process contention — many `multiprocessing.Process` workers
  and live HTTP streams competing for the OS scheduler at once. A sweep run at
  high `--parallel`/`--max-per-model` saw a job logged as killed by a 300s
  timeout report an actual wall time of over 4,900 seconds. Lower
  `--parallel`/`--max-per-model` (2 and 1 are conservative, reliable defaults
  for a sweep of any size) avoids the contention rather than relying on the
  kill path's nominal bound holding under load.

## Tasks: three retired, three added for discrimination

`roman`, `primes` and `lru` scored 100% for every configuration tested across
dozens of runs. They still exist — `bench list` shows them as retired, and
`--include-deprecated` runs them — but they no longer cost budget in a normal
sweep, because a task every agent passes measures nothing.

Three replacements, each targeting a gap the old suite couldn't see:

- **`throttle`** *edits existing code* rather than writing from nothing. It
  seeds a working rate limiter with one real bug that a read alone doesn't
  catch — the agent has to trace through a call sequence to find it.
- **`inventory`** is genuinely multi-file: `report.py` imports `store.py`
  imports `models.py`, with real behavioural dependencies between them. It's
  the first task that gives the planner's `depends_on` and the executor's
  `--step-concurrency` something real to decompose.
- **`graph`** asks for BFS shortest-path and times a large grid — a correct
  implementation finishes in under a millisecond; a naive all-paths search
  does not finish in the time budget on the same input, and it doubles as a
  correctness check (disconnected components, unknown nodes) at the same time.

**Accuracy comes from tests the agent never sees.** Each task ships a hidden
test file that is copied in and run only after the agent has stopped — an
agent that writes its own tests and runs them is grading its own work, and
every agent scores 100%. Each hidden test file is validated against a
hand-written reference implementation before being trusted (see
{doc}`development`'s "Adding a benchmark task").

The four measurements:

| axis | how |
| --- | --- |
| **completion speed** | wall clock per run, plus executor decode rate and TTFT from the streamed responses |
| **code accuracy** | share of hidden tests passed, via JUnit XML for exact counts |
| **code execution speed** | a workload script timed against the produced code |
| **model cost** | real dollars from OpenRouter's usage field, plus cost-per-fully-passing-run |

Tasks are stdlib-only so the measurement never depends on whether the agent
managed to install something, and each run gets a fresh workspace under the
system temp directory.

## What the benchmark has found

Across two separate sweeps, `mercury-only` — Mercury doing all three phases —
has consistently been dramatically cheaper and faster than a Claude-planned/
reviewed `hybrid`, for a small-to-nonexistent accuracy difference on
single-file, fully specified tasks.

The second sweep (`mercury-only`, `mercury-parallel`, `mercury-nano`,
`hybrid`, `hybrid-deepseek` × `expr`/`semver`/`throttle`, `--max-per-model 1
--parallel 2 --timeout 240` for reliability — see the concurrency note above):

| config | accuracy | cost/run | wall | s/call |
| --- | --- | --- | --- | --- |
| `mercury-parallel` | **100.0%** | **$0.0047** | **40.5s** | 2.38s |
| `mercury-only` | 100.0% | $0.0051 | 43.2s | 2.49s |
| `mercury-nano` | 99.2%±1 | $0.0030 | 39.3s | 3.11s |
| `hybrid` | 33.3%±47 | $0.0350 | 192.2s | 52.41s |
| `hybrid-deepseek` | 27.8%±39 | $0.0023 | 224.9s | 35.51s |

The all-Mercury configs finished every task at or near 100% accuracy for
roughly a cent a run and under a minute of wall time. The hybrid configs'
numbers are pulled down by real timeouts (`hybrid` hit the 240s cap on 2 of 3
runs, `hybrid-deepseek` on 2 of 3) — Sonnet and DeepSeek's own per-call
latency under this run's conservative, fully-serialized concurrency setting,
not a scoring artifact — but even on the one task `hybrid` did finish cleanly
(`expr`, 100% accuracy), it cost **20x** `mercury-only`'s run and took **70%
longer**. This matches the first sweep's finding at a more generous timeout:

| config | accuracy | cost/run | wall | executor tok/s |
| --- | --- | --- | --- | --- |
| `mercury-only` | 99.4% | $0.007 | 32s | 1642 |
| `sonnet-only` | 100% | $0.151 | 88s | 246 |
| `hybrid` | 100% | $0.197 | 152s | 1092 |

Mercury doing all three phases was **21× cheaper and 3× faster** than Claude
doing all three, for a 0.6% accuracy difference; the hybrid was the most
expensive and slowest of the three, because planning and review cost Claude
tokens on tasks whose correctness Mercury was already reaching on its own.

**This is why the harness's default configuration (`agent/config.py`) now
runs every phase on Mercury**, at `planner_reasoning="high"` — matching the
winning `mercury-only` config — rather than the original Sonnet-plans/
Sonnet-reviews hybrid design. `step_concurrency` stays off by default (the
`mercury-parallel` config that edges out `mercury-only` above sets it to 3);
concurrent execution is only as safe as the plan's declared `depends_on` and
`files`, so the harness treats sequential as the safe default and leaves
`--step-concurrency` as an opt-in for anyone who wants the extra speed.

That result comes with a real caveat worth restating: single-file, fully
specified tasks never exercise a planner/reviewer's actual value —
decomposing an ambiguous goal, or catching a defect that spans files. The
multi-file `inventory` task and the existing-code `throttle` task probe that
gap; pass `--planner`/`--reviewer` explicitly for work that's genuinely
ambiguous or file-spanning, and run the fuller task set with `--repeats 5+`
before drawing a firm conclusion for your own workload — a single repeat, as
used above, is exactly the sample size `bench run` itself warns is too small
to trust.
