# Architecture

## Before planning: is this even a coding task?

`--quick-answers` (on by default) tries one cheap, tool-free completion
before planning starts (`agent/loop.py::try_quick_answer`): given the goal
alone, the model either answers it directly — for a question, explanation,
or comparison that needs no file changes — or replies with the sentinel
`NEEDS_PLAN`, in which case `run()` falls through to the normal pipeline
below exactly as if quick-answers were off. A wrong guess costs one extra
small call before the plan step that would have run anyway; a right guess
skips planning, execution, and review entirely — no `=== phase 1: plan ===`
banner, no steps, just the answer. `--plan FILE` (re-running a saved plan)
skips this check too — that's already an explicit decision to do real work.

## What the agent remembers

Every run used to start knowing nothing about the runs before it. Ask the
harness *"why did that test fail?"* straight after the run that broke it and
the goal was unanswerable — the clarifier would ask *which* test, and there
was no answer available to anyone, because the previous run's findings lived
only in a trace file nothing read back.

`agent/memory.py` keeps two files under `<workspace>/.harness/`:

- **`memory.md`** persists across sessions. The harness appends one section
  per run — goal, files touched, outcome, what was left unresolved, what was
  suggested next — and reads the relevant parts back into the next run's
  clarify and planning prompts. It lives in the workspace rather than a
  central config directory so it travels with the project, is diffable, and
  can be committed or hand-edited. A throwaway workspace losing its memory
  when deleted is correct, not a limitation.
- **`scratch.md`** is working memory for one session, rotated at the start of
  every run. The `remember` tool writes here and nowhere else: a model's
  in-flight notes are frequently wrong, and a wrong note that persists across
  sessions is worse than no note. What survives is the harness's own account
  of what actually happened. `recall` searches both.

Retrieval is keyword and filename overlap over markdown sections — not a
database, not an embedding index. At this scale (tens of kilobytes, tens of
sections) scanning the file costs microseconds, and being able to `grep` the
agent's memory, read it in a diff, and fix it in an editor is worth more than
any accuracy a heavier index would buy. Filename matches outweigh word
matches, because two runs touching the same file are related in a way two runs
sharing the word "parser" may not be; the most recent section is always
included regardless of score, because a follow-up phrased as *"why did that
fail?"* shares almost no vocabulary with the run that caused it. Once the file
grows past several times the per-run recall budget, one cheap call condenses
everything older than the recent entries into a summary.

Bench forces this off: memory would make repeat 3 of a task a different
experiment from repeat 1, and two configs run in different orders would stop
being comparable.

## Is the goal even specific enough to build from?

`--clarify` (on by default, needs `--ask` and a terminal) is the same trick
applied to a different question: one cheap, tool-free call
(`agent/phases/clarify.py`) sees the goal and the workspace map and replies
either `CLEAR` or up to three questions, which are asked through the ordinary
`Interactor` and folded into the planner's prompt as plain text.

The planner has had `ask_user` all along and almost never used it, for two
compounding reasons: `PLANNER_SYSTEM` tells it — correctly — that most goals
do not need clarification, and a fast model driving a tool loop toward
`submit_plan` has every incentive to keep going rather than stop and ask. The
result was an agent that would confidently build the wrong thing from a
three-word goal. Giving the question its own call, with nothing competing
against it, is what makes it actually happen. Nothing here is binding on the
planner, so a skipped or garbled answer degrades to "the planner knows
slightly less" rather than to a broken run.

## The phases

1. **Plan** (Mercury 2.5 by default; `--planner` names any other model, and a stronger
   autoregressive planner is the usual swap). Read-only exploration, then
   Chain-of-Draft prompting: brief reasoning notes (or, with `--draft-style prose`, full sentences), followed by a
   structured plan of small, self-contained steps (`agent/phases/plan.py`).
   When the goal reports a problem, the plan carries a `reproduce` snippet
   that the harness runs on a throwaway copy before accepting it; if the
   problem does not show, the plan goes back once to decide whether the
   report is mistaken (`--plan-probe repro`, the default). `--plan-probe
   require` lets the planner run code itself and holds its first plan until
   it has, which catches more wrong reports at about two extra calls a run
   (DESIGN-LOG §54-54.2).
   With `--review-plan`, the finished plan is shown before anything executes:
   accept it, stop, or send it back with feedback that goes into the next
   planning call. A misunderstanding caught here costs one call; the same
   misunderstanding caught after execution costs the run.
2. **Execute** (Mercury 2.5, or whatever `--executor` names). Each step runs
   in its own small context — only its own declared files and the summaries
   of prior steps, around 12k prompt tokens by default (`agent/phases/execute.py`).
3. **Review** (off by default: `--review` turns it on; Mercury unless `--reviewer` says
   otherwise). Large chunks, whole files — the reviewer sees
   everything at once, which is structurally what the executor cannot do. Each
   finding is handed back to the executor as one scoped fix (`agent/phases/review.py`).
   Findings accumulate across rounds rather than being replaced by the last
   round's, so a run that found three blockers and repaired them reports that,
   instead of reporting that nothing was ever wrong.
4. **Final check** (no model at all). See below.

Review is off by default because it is 18-22% of spend and 20-25% of model time and no
later suite showed a loss without it: django 17 of 30 against 17, SciCode level, the fixtures
level, and on GAIA a replay of 173 reviewed answers found +1 (DESIGN-LOG §124.10, §126). An
earlier 20-question GAIA test had review off 4 answers worse (§93.1); the replay did not
reproduce it. Without
it the run goes from execute straight to the final check, whose fix rounds still act on
failures.

`--direct` (off by default, DESIGN-LOG §129) skips phase 1: the executor gets one step for the
whole goal and finds the files itself (`agent/phases/plan.py::direct_plan`), then the same
executor, gates and final check run. `--stop-after plan|execute|review` ends a run early, and
`--plan FILE` runs a saved plan.

Every model-driven phase drives the same generic tool-calling loop
(`agent/runner.py`): call the model, dispatch its tool calls, append the
results, repeat until a terminal tool is called, the model stalls, or the
budget runs out.

## The final check: the only phase that can contradict the model

Planning, execution and review are all a model's opinion about whether the work is right. The
step-level `verify` commands come closest to evidence, but each runs in
isolation immediately after its own step, so a later step that breaks an
earlier one's work passes unnoticed — and the review phase, the only thing
that looks at the whole surface and optional, is a judgement call.

`agent/phases/finalcheck.py` makes no model calls. It re-runs every step's
verify command against the *finished* state, runs the project's test suite,
and type-checks the changed files. A command either exited zero or it did not.
Failures become blocker findings and go through the ordinary fix path, and
then **the check runs again** — which also closes the review loop's last gap,
where the final round of fixes used to be applied and never looked at again,
leaving the edits most likely to be wrong as the only ones nothing verified.

**Pre-existing failures are named, and are usually the work.** Before
planning starts, the suite is run once and the failing test ids recorded
(`collect_baseline`). Those ids go into the planner's prompt — without that,
a run told to "fix the failing tests" was planned by a model that had never
been told which tests those were, and spent its budget rediscovering them or
asking the human to run pytest for it.

Whether they *count against* the run is `--fix-existing-failures`, on by
default. Excusing them unconditionally was a mistake: it is most of why
anyone points this agent at a repository at all, and it produced runs that
reported PASS over a suite that could not even be collected. `--no-fix-existing-failures`
restores the narrower behaviour for a workspace that merely happens to
contain unrelated red tests, where chasing them would spend money on code the
goal never mentioned. Either way the gate distinguishes new breakage from old,
and a pre-existing failure that has since been repaired stops being reported
as failing.

**It iterates.** Check, fix, re-check, up to `MAX_FIX_ROUNDS`, stopping the
moment a round leaves the failure set byte-for-byte unchanged: a fix that
fixed nothing will not fix anything on an identical retry, so reporting
honestly beats spending two more rounds proving it.

**Green is not the same as exercised.** A run once produced five passing
tests about the two features that were broken, by reaching into the objects
and setting state by hand rather than running the code it had just written.
`agent/phases/coverage.py` asks the narrower, checkable question: of the lines
this run added or modified, how many did the suite execute? The suite runs
once, under `coverage run`, and the data is read afterwards by a command that
executes no tests — so the measurement costs one fast subprocess rather than a
second run. A file whose changed lines are barely touched becomes a blocker
finding like any other.

The threshold is a fraction (`min_changed_coverage`, default 0.5) rather than
"no lines covered", which sounds stricter and is unreachable: `def f(...)`
executes on import, so a function nothing ever calls still reports one covered
line. Two limits are worth stating plainly — a bug in a line the run did not
touch is invisible to this, and a line being executed says only that it ran,
not that anything asserted about the result. It narrows the gap between green
and working; it does not close it.

Two consequences worth knowing:

- A step that has no `verify` command opts out of the per-step signal
  entirely, so `agent/phases/plan.py::_synthesize_verify` fills one in when
  the planner leaves it blank — `uv run pytest -q <the step's test files>`,
  or `uv run ruff check <its modules>` (whose default rules include pyflakes,
  so it catches undefined names and bad imports, not only formatting). It is
  scoped to the step's own files on purpose: a plan whose tests arrive in step
  3 would otherwise fail step 1 against tests that do not exist yet.
- The run's summary now carries `ok`, `steps` (failed/abandoned/unverified)
  and `final_check`. The CLI exits non-zero and says what failed; the TUI
  says "finished with failures" instead of a green "done." A run whose steps
  failed verification used to be indistinguishable from a clean one.

## Tools

The executor gets every tool; the planner and reviewer get a read-only subset.

| tool | what it's for |
| --- | --- |
| `read_file`, `write_file`, `edit_file` | exact-match search/replace edits, with near-miss hints on a failed anchor |
| `read_files` | several files in one call — a round trip costs about a second regardless of size |
| `edit_files` | several search/replace edits in one call, to one or many files |
| `move_file`, `delete_file` | rename/move or remove a file, without routing through `run_bash` |
| `run_python` | a Python snippet with the workspace importable, no shell quoting |
| `grep`, `glob_files`, `list_dir` | finding things |
| `check_code` | Ruff, plus ty for cross-file type errors; `fix=true` applies safe autofixes |
| `check_complexity` | cognitive complexity per function via complexipy — see below |
| `run_tests` | pytest with a parsed summary and the failing assertions |
| `inspect_module` | real signature, docstring and members of any installed module, class or function |
| `pypi_info` | confirm a package exists, its version and dependencies, before depending on it |
| `fetch_url` | fetch a known URL, HTML stripped to readable text |
| `web_search` | search the web (needs `SEARXNG_URL` or `BRAVE_API_KEY` — see below) |
| `show_changes`, `git_diff`, `git_log` | diff of everything this run changed, with or without git |
| `git_add`, `git_commit` | stage and commit, executor-only — see below |
| `run_bash` | everything else, metered against a per-step time budget |
| `ask_user` | a clarifying question when a wrong guess would be expensive to undo; `options` gives concrete choices and `recommended` names the one it would pick |
| `remember`, `recall` | leave a note for a later step; search what earlier runs did |

### `git_add`/`git_commit`, and why they're executor-only

`git` is already allowlisted for `run_bash`, so these don't change what the
agent can *do* — they exist because a hand-written
`run_bash("git commit -m \"...\"")` has to get shell-quoting right for the
message, multi-line messages included, and a mis-escaped quote is a common,
easy-to-hit failure. `git_commit`'s message is passed as a real argv element,
never through a shell, so none of that applies. `git_add`/`git_commit` are
only ever offered to the executor (`agent/tools/vcs.py::VCS_WRITE_TOOLS`,
kept separate from the read-only `VCS_TOOLS` every phase gets) — the same
enforced-not-asserted separation that keeps `run_bash`/`run_python` out of
`review_registry`: the reviewer critiques, the executor fixes and commits.

### Complexity, and why it gets its own tool

Generated code accretes branches one case at a time. Each addition is locally
reasonable; the total is a function nobody can safely change. `check_complexity`
runs [complexipy](https://complexipy.com), which measures **cognitive**
complexity rather than cyclomatic — it weights nesting depth and flow breaks,
which is what actually makes code hard to hold in your head, rather than
counting branches, which mostly tells you how many tests it needs.

It is a separate tool from `check_code` (rather than another flag on it)
because it answers a different question and should be reached for
deliberately: when a function is suspected of being tangled, or when a
refactor wants a before-and-after number to justify it. The bundled
`complexity` skill (`harness skills list`) carries the refactoring guidance —
guard clauses first, extract only what you can name.

The same measurement runs at commit time through `prek`, so the ceiling is
enforced rather than merely suggested. See the README's toolchain section.

### `web_search`'s backend

A DuckDuckGo HTML scrape was the obvious no-key option and was tried first;
it hits a bot-detection challenge on essentially every scripted request,
which makes it unusable rather than merely unreliable. `web_search` uses a
self-hosted [SearXNG](https://docs.searxng.org/) when `SEARXNG_URL` is set
(free; `scripts/searxng/up.sh` starts one on `127.0.0.1:8888` with its JSON
API enabled), otherwise Brave's Search API when `BRAVE_API_KEY` is set.
Brave has been metered since February 2026, and Google's Custom Search API
closed to new customers and stops serving on 2027-01-01, so neither is the
default. With no backend configured, `web_search` is left out of every tool
kit: before that, all 61 of its calls across 19 traced runs had returned a
missing-key error. `fetch_url` needs no backend and works for any URL the
model already knows.

`SEARCH_BACKEND=searxng` or `brave` names the backend outright, so a `.env` found in
some parent directory cannot switch it. With `SEARCH_BACKEND=brave`, the key can come
from the macOS Keychain (`security add-generic-password -a harness -s BraveSearch -w <key>`)
instead of `BRAVE_API_KEY`. It is sent in a request header, never in a URL or a trace.
Each run's `run_start` trace event records the backend it used.

SearXNG costs nothing, but it is only as good as the upstream engines that still
answer your machine. On the machine this was built on, DuckDuckGo answered the
first query with a CAPTCHA, and Brave and Google rate-limited it within a day,
whatever the query volume. That left Yahoo, whose results for the same queries
changed within a day (DESIGN-LOG §112.2, §115). If search results look thin,
`docker logs` on the SearXNG container lists which engines are refusing. An
engine the image marks `inactive: true`, such as Mojeek, needs `inactive: false`
in `scripts/searxng/settings.yml`; `disabled: false` alone does not load it.
For results that must be comparable across days, use Brave's API.

## Validation on every write

Every write is checked before the tool call returns, so the model learns it
broke something on the same turn it broke it:

```text
edited core.py (1 replacement)
@@ -12,3 +12,3 @@
-    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))
+    return sorted(counts.items(), key=lambda x: (-x[1], x[0])

*** SYNTAX ERROR — you broke this file, fix it before doing anything else ***
core.py:14:5: SyntaxError '(' was never closed
```

The cost ordering decides what runs automatically: `ast.parse` (~1ms) and Ruff
(~15ms) run after every write; `ty` (~1s) is a deliberate `check_code(types=true)`
call, since cross-file type checking is only meaningful once a step is complete.

## Making the executor accurate

- **A step verify gate.** After a step finishes, its `verify` command runs —
  the only evidence independent of the model's own claim that the step worked.
- **Escalation on failure.** A step that fails verification, or never calls
  `finish_step`, retries once with the failure text and the executor's
  reasoning moved one rung up the ladder (`none → low → medium → high`).
  Paying for high reasoning only on the steps that need it is much cheaper
  than paying for it everywhere.

## Making the executor fast

A diffusion model's wall time is dominated by per-call overhead, not
generation — roughly 80% of executor time is queue and prefill. Everything
here follows from that:

- **Preloading.** A step's declared files are inlined into its prompt, using
  half the step token budget, instead of costing a round trip each.
- **Batched reads and edits.** `read_files` and `edit_files` take a list, so
  the model can act on several files in one call.
- **Dedicated tools over shell.** `check_code` and `run_tests` exist so the
  model doesn't drive `ruff`/`pytest` through `run_bash` in a loop of
  one-command round trips.
- **Prompt caching.** The stable prefix (system prompt, task) gets a cache
  breakpoint on providers that support it.
- **Parallel steps.** `--step-concurrency N` overlaps steps the plan says are
  independent (see below).

## How parallel steps stay safe

The planner declares `depends_on` per step. Two steps run together only when
*both* hold: neither depends on the other (`agent/phases/schedule.py::build_waves`),
and their declared `files` are disjoint (`partition_disjoint`). A step that
declares no files is treated as touching everything and runs alone, since
nothing else can be proven about it. A dependency cycle collapses to plan
order, which is always executable.

Concurrent steps share the run's baseline, read cache and touched list — all
lock-guarded, since they describe the run rather than the step — but each
gets its own `ResourceGuard`, because shell-time and error budgets are
per-step. Concurrency is **off by default**: sequential is the safe reading of
a plan, and a plan whose `depends_on` is wrong will corrupt a run that trusts it.

## The trace

Every model call is appended to `traces/run-*.jsonl`: phase, model, token
counts, dollar cost, latency and measured tok/s, plus every tool call and its
arguments (`agent/trace.py`). It's the record that lets you compare a stronger
planner against the all-Mercury default on the same goal, or the same saved plan across
executors, and it's the same event stream `--tui` subscribes to live.

## Following the run live

Two independent mechanisms make a run legible while it's happening, both on
by default and both off with `--no-narrate`/`--no-animate`:

- **Narration** (`agent/runner.py::_narration`): a short line before each
  state-changing tool call — writing a file, running a command, committing —
  routed through `Trace.say()` like everything else, so it respects `quiet`
  and reaches `--tui`'s log the same way. Only calls worth following along
  with are narrated; reads (`read_file`, `grep`, `list_dir`, ...) and the
  terminal tools (`finish_step`, `submit_plan`, ...) stay silent, since
  narrating those would bury the writes and commands that actually matter.
- **The wait indicator** (`agent/spinner.py` for the CLI, the equivalent
  logic built into `agent/tui.py`'s `RunApp`): a small looping animation
  while a model call is in flight and nothing else is printing — most
  executor turns are all tool calls and no assistant text, so without this
  the terminal looks frozen for however long the call takes. It is not the
  model's own text streaming — a fixed loop, deliberately, since a spinner
  that looked like it was "generating" something would be misleading on a
  turn where nothing is being generated yet. Both are driven by
  `OpenRouterClient`'s `on_call_start`/`on_call_end` hooks (`agent/llm.py`),
  which fire around every blocking call and end early the moment real
  content starts streaming.

The calls/cost/tok-per-s breakdown that used to print automatically at the
end of every run is now opt-in (`--stats` for the CLI; a hidden-by-default
side panel toggled with "s" in `--tui`) — narration already says what
happened without it.

Approval itself collapses to one choice, `--approval-mode` (`manual` /
`auto-python` / `auto-plan`), rather than two booleans (`-y`,
`--auto-approve-python`) a caller had to reason about together — it just
resolves to the same two `Settings` fields before the run starts. In
`--tui`, pressing "p" (or ctrl+p, a priority binding that works even while
the goal bar has focus and would otherwise swallow a bare "p" as typed text)
opens a settings panel (hidden by default, alongside the stats panel) with a
dropdown for the approval mode plus narrate/animate/quick-answers, and a
one-line mode summary stays visible above the input bar at all times. ctrl+a
cycles the approval mode directly, without opening the panel. Since
`Approver` and `ToolContext` are built once per `loop.run()` call, a
dropdown/hotkey change only takes effect on the run *after* it, not the one
already in flight — the CLI's own equivalent is typing `auto` or `python`
(instead of `y`/`n`) at any approval prompt, which raises `Approver`'s trust
level for the remainder of the *current* run, since a plain CLI session has
no other interactive surface to change it from mid-run.

## Safety model

**The path jail does not constrain subprocesses.** `Workspace` refuses any
file path outside the workspace root, symlinks included — but a shell command
is a different matter, because Python build tools search *upward*. See
{doc}`safety` for the full picture and why a nested workspace is refused by default.
