# Command reference

This page reflects **v0.2.0** and is generated from the commands' own definitions
(`harness <command> --help` and `bench <command> --help`), so a flag listed here exists
in that release. When it and `--help` disagree, `--help` is right. Long help texts are cut
at their first sentence or two; `--help` has the rest.

Two commands are installed with the package, plus a dashboard:

- `harness`: the agent (`run`), key setup, preferences, skills, and analysis of saved traces.
- `bench`: the benchmark, which compares whole agent configurations.
- `harness-tui`: the terminal dashboard (needs the `tui` extra).

```bash
harness setup                                   # once: store an OpenRouter or Inception key
harness run -w ~/agent-sandbox "add a --verbose flag to the CLI"
harness run --stop-after plan --out plan.json "goal"   # look at the plan before spending anything
harness run --plan plan.json -w ~/agent-sandbox        # run a saved plan
harness run --review "goal"                     # turn on the review phase (off by default)
```

How to read the tables:

- **Default** is what the flag is when you do not pass it. A blank default means the flag
  is unset and the value comes from a persisted preference (`harness config`), an
  environment variable or a built-in, for example both reasoning levels are `high`
  unless set. A `--x / --no-x` pair shows whether `--x` is on or off by default.
- Every phase uses Mercury (`inception/mercury-2.5`) by default, review is off, and the
  final check is on. `--all-mercury` is therefore a no-op unless you changed a model through
  a preference or environment variable.
- Many flags are measured switches: they exist because they were tried, and a large part of
  them are off because the measurement said so. {doc}`configuration-comparison` lists each
  with the benchmark it was tested on, the DESIGN-LOG section and the outcome.

## The `harness` command

Top level: `--version` prints the version. Commands: `run`, `setup`, `config`, `skills`, `trace`, `sessions`.

### `harness run`

Arguments: `goal` (What the agent should build.)

| flag | default | what it does |
|---|---|---|
| `-w` / `--workspace` `STR` | `.` | Directory the agent may touch. |
| `--executor` `STR` | `inception/mercury-2.5` | Model that executes plan steps (the dLLM under test). |
| `--planner` `STR` | `inception/mercury-2.5` | Model that writes the plan. |
| `--reviewer` `STR` |  | Model that reviews. Defaults to the planner. |
| `--all-mercury` | off | Use the executor model for every phase, as a dLLM-only baseline. |
| `--executor-reasoning` `none\|low\|medium\|high` |  | Mercury's tunable reasoning level for execution. |
| `--planner-reasoning` `none\|low\|medium\|high` |  |  |
| `--plan` `PATH` |  | Skip planning and run a plan saved from an earlier run. |
| `--stdin-plan` | off | Skip planning and accept a plan JSON from stdin instead. |
| `--stop-after` `plan\|execute\|review` | `review` | Stop after this phase. |
| `--max-usd` `FLOAT` | `2.0` | Hard spend cap for the run. |
| `--max-steps` `INT` | `20` |  |
| `--step-budget` `INT` |  | Approx prompt tokens per execute step. Default: a quarter of the executor's context window. |
| `--review-budget` `INT` |  | Approx prompt tokens per review chunk. Default: two thirds of the reviewer's context window. |
| `--step-concurrency` `INT` | `1` | Plan steps to run at once. Only steps with no declared dependency on each other and disjoint files ever overlap. |
| `--verify-steps` / `--no-verify-steps` | on | Run each step's verify command after it finishes. |
| `--escalate` / `--no-escalate` | on | Retry a failed step, telling the next attempt what failed and what the last one already tried, and raising reasoning if it is not already at the top (it is, by default). Off disables the retry itself. |
| `--skills` / `--no-skills` | on | Inject skill reference material into prompts (bundled ones, plus any of your own — see `harness skills`). |
| `--skill-triggers` `STR` | `step` | What an executor step's skill triggers match: 'workspace' (every root file name and the verify command) or 'step' (the step's own files and words). §117. |
| `--skills-dir` `PATH` |  | Use only this directory of skill .md files, instead of the bundled + your-own merge. |
| `--ask` / `--no-ask` | on | Let the planner and executor ask you a clarifying question when genuinely unsure, instead of only guessing. |
| `--narrate` / `--no-narrate` | on | A short line before each write, command, or commit so you can follow along step by step, not just at the start and end of a whole plan step. |
| `--quick-answers` / `--no-quick-answers` | on | For a goal that's really just a question, answer it directly and skip planning (and its narration) entirely. One cheap classification call up front decides which; falls through to the normal pipeline for anything needing real work. |
| `--clarify` / `--no-clarify` | on | Before planning, decide whether the goal is specific enough to build from, and ask up to three questions if it isn't. One cheap call; answers go into the planner's prompt. Needs --ask. |
| `--review-plan` / `--no-review-plan` | off | Show the plan and wait before executing it: Enter to run, 'e' to send it back with feedback, 'q' to stop. Catching a misunderstanding here costs one planning call instead of a whole run. Only ever prompts with a real terminal attached. |
| `--memory` / `--no-memory` | on | Read and write <workspace>/.harness/memory.md, so a run knows what earlier runs on this workspace did and what they left broken. Without it, a follow-up goal like "why did that test fail?" has nothing to refer to. |
| `--review` / `--no-review` | off | Run the review phase (a model reads the finished work and fix rounds act on its findings). Off by default: it is about a fifth of the cost and time, and no suite showed a loss without it (DESIGN-LOG §124.10). The final check still runs either way. |
| `--fix-existing-failures` / `--no-fix-existing-failures` | on | Treat tests that were already failing when the run started as work to do, not as somebody else's problem. |
| `--plan-auditor` `STR` |  | A second model that reviews and may patch the plan once, with no tools (agent/phases/plan_audit.py). Empty: no audit. |
| `--provider-sort` `STR` |  | OpenRouter provider preference: latency, throughput or price. Matters for a model served by many providers (GLM Flash: 0.6s to 16s median first token). |
| `--mercury-provider` `STR` | `openrouter` | Where Mercury calls go: openrouter (default), inception (Inception's own API; Keychain item InceptionLabs) or venice (Keychain item Venice). Other models always go to OpenRouter. `harness setup` persists it. |
| `--mercury-temperature` `FLOAT` |  | Mercury's temperature for every call, 0.5-1.0. Default: none sent, the model's own 1.0 (what every earlier run actually used). |
| `--issue-symbols` / `--no-issue-symbols` | off | Tell the planner where the goal's own identifiers are defined, from a search of the workspace (agent/issue_symbols.py, DESIGN-LOG §128). Off until measured. |
| `--direct` / `--no-direct` | off | Skip the planner: the executor gets one step for the whole goal and finds the files itself; the final check still runs (DESIGN-LOG §129). Off until measured. |
| `--executor-tail` / `--no-executor-tail` | on | Send the step's goal, turns left and key rules after the last message on every executor call (agent/steptail.py, DESIGN-LOG §82). On since §106.2. |
| `--tail-item` `STR` |  | With --executor-tail, a state section to add: errors or ledger. Repeatable. api, inspect and key still exist but were measured and not adopted (DESIGN-LOG §84, §87.2). |
| `--planner-tail` / `--no-planner-tail` | off | The planner's own tail. |
| `--plan-key-facts` / `--no-plan-key-facts` | off | Ask the planner for the one fact each step must not get wrong. |
| `--findings-fix` `STR` |  | With --research-first, how to get convention decisions into findings: |
| `--clarify-context` / `--no-clarify-context` | on | Show the clarify call the code's symbols, README and test setup, and ask only what the code cannot answer. On by default (DESIGN-LOG §92, §101). |
| `--resolve-followups` / `--no-resolve-followups` | on | Tie a short follow-up goal ('fix those issues', 'try again') to the last run, confirmed with one click, instead of asking what it means. DESIGN-LOG §92.3. |
| `--chat-checks` / `--no-chat-checks` | on | With a person present: answer questions they ask back in clarify's answers, and confirm the plan's assumptions before it runs. DESIGN-LOG §111. |
| `--last-word-gates` `STR` | `off` | What the planner's gates do with a plan forced out by the last word: off (skip them, as before), turn (a hold gets one more forced turn), notes (the plan goes through, the hold's message becomes a finding). DESIGN-LOG §88.3. |
| `--mercury-max-tokens` `INT` |  | Output-token cap for Mercury calls. Default 16000 (DESIGN-LOG §95.1); 0 for none. |
| `--mercury-content-stall` `FLOAT` |  | Seconds of silence after a Mercury stream's first delta before the attempt is cut and retried. Default 30 (DESIGN-LOG §95.1); 0 turns it off. |
| `--web-deny` `STR` |  | A URL substring the web tools refuse (fetch_url) or drop from results (web_search), matched anywhere in the URL, even wrapped in another one. Repeatable. For benchmark runs: the benchmark's own sources. |
| `--research-first` / `--no-research-first` | off | Before planning, look things up: the installed libraries' API (inspect_module) first, then skills, then the web. |
| `--numeric-check` / `--no-numeric-check` | off | Before a step finishes, run its code on a case whose answer is known independently (a limit, a symmetry, a conservation law). |
| `--check-after-edit` / `--no-check-after-edit` | on | After a turn that edits files, run the step's tests once and hand the result back with the edits, instead of the model spending a call to ask. On by default (DESIGN-LOG §99: model time -31%, cost -40%). |
| `--combined-fixes` / `--no-combined-fixes` | on | Fix all of a round's findings in one pass (shared reading, edits checked together) instead of one pass per finding. On by default (DESIGN-LOG §101). |
| `--stop-after-idle-fix-round` / `--no-stop-after-idle-fix-round` | off | Do not start a final-check fix round after one that changed no file. Off by default until DESIGN-LOG §124's scout decides it. |
| `--types-after-edit` / `--no-types-after-edit` | on | Report type errors the run introduced in the check after each edit. On by default (DESIGN-LOG §105.2). |
| `--web-research` / `--no-web-research` | off | Aids for finding facts on the web: a research skill and a nudge to open a result before searching again. Off until DESIGN-LOG §116 decides it. |
| `--read-assist` / `--no-read-assist` | on | read_file widens narrow windows to the whole function and answers identical re-reads in one line. On by default (DESIGN-LOG §105.2). |
| `--edit-blocks` / `--no-edit-blocks` | off | Offer edit_blocks (SEARCH/REPLACE text) instead of edit_files. Off: not adopted in DESIGN-LOG §105.2. |
| `--review-presupplied-checks` / `--no-review-presupplied-checks` | on | Hand the reviewer ty's report with the suite's, and drop run_tests from its kit. On by default (DESIGN-LOG §108.2). |
| `--bugfix-evidence` / `--no-bugfix-evidence` | off | Before a step that changed existing source finishes, check once that the plan's reproduction now passes and the run's new tests fail on the old code. Off: dropped in DESIGN-LOG §120.2. |
| `--scaffold` / `--no-scaffold` | on | In an empty workspace, set the project up with uv init and Ruff formatting before planning (DESIGN-LOG §104). |
| `--answer-command` `STR` |  | Answer the run's clarifying questions with this command instead of a person: it gets {"question", "options", "recommended"} as JSON on stdin and prints the answer. |
| `--test-command` `STR` |  | The project's own test command, for a project pytest cannot run, e.g. |
| `--test-label-root` `STR` |  | With --test-command: pass test files under this directory as dotted module labels relative to it (tests/pagination/x.py -> pagination.x). |
| `--draft-style` `STR` | `terse` | How the planner drafts before committing to a plan: 'terse' (chain-of-draft, brief notes; |
| `--plan-probe` `STR` | `repro` | How the planner checks a reported problem before planning its fix. |
| `--test-audit` / `--no-test-audit` | on | After a green run, check in a scratch copy whether the tests can see the change: |
| `--fix-weak-tests` / `--no-fix-weak-tests` | off | When the test audit finds tests that cannot see the change, spend one fix round on them, showing the model what the run changed. |
| `--verify` `STR` |  | A shell command that proves the goal's own claim, checked at the final gate alongside every step's own verify — not instead of it. |
| `--require-tests` / `--no-require-tests` | on | Fail a run that writes behaviour into a workspace with no tests at all. On by default: with nothing to run, the final check has nothing deterministic left to check, and a passing lint stands in for working code. |
| `--check-coverage` / `--no-check-coverage` | on | Check that the tests actually execute the lines the run changed, not just that a green suite exists. The suite runs once either way, under coverage; reading the data costs one extra fast subprocess. |
| `--final-check` / `--no-final-check` | on | After review, re-run every step's verify command and the test suite against the finished code, and fix what fails. Makes no model calls unless something is actually broken. |
| `--stream` / `--no-stream` | on | Stream responses. Needed for time-to-first-token measurement. |
| `--stats` / `--no-stats` | off | Show the calls/cost/tok-per-s breakdown at the end of a run. Off by default — ask for it when you want it, rather than reading past it every time. |
| `--animate` / `--no-animate` | on | A small looping animation while a model call is in flight and nothing else is printing, so the terminal doesn't look frozen. Automatically does nothing when output isn't a real terminal. |
| `--spinner-style` `STR` | `duck` | 'duck' (three ducks paddling across rippled water, bills opening and closing) or 'dots' (a plain single-line spinner). |
| `--cache` / `--no-cache` | on | Send an explicit cache breakpoint on the stable prompt prefix. Only Anthropic models accept one, so this has no effect on the all-Mercury defaults; automatic provider-side caching is separate and unaffected. |
| `--autolint` / `--no-autolint` | on | Run Ruff after each write. Syntax checks always run. |
| `--test-parallel` / `--no-test-parallel` | off | Run the whole-suite baseline and gate checks under pytest-xdist (installed ephemerally). Off by default: it needs test isolation the suite may not have, and helps most on a large suite that is not dominated by one or two slow tests. |
| `--shell-budget` `FLOAT` | `90.0` | Seconds of subprocess time one step may use. |
| `--max-workspace-mb` `FLOAT` | `500.0` | Stop the run if the workspace grows past this. |
| `--max-step-errors` `INT` | `12` | Tool errors in one step before that step is abandoned. |
| `--allow-nested` / `--no-allow-nested` | off | Permit a workspace inside another Python project. Unsafe: the agent's subprocesses can modify the enclosing project's environment. |
| `-y` / `--yes` | off | Auto-approve shell commands. Point this at a throwaway workspace. |
| `--auto-approve-python` | off | Auto-approve run_python calls specifically, without approving shell commands too. |
| `--approval-mode` `manual\|auto-python\|auto-plan` |  | A single choice covering both approval flags at once: 'manual' (prompt for everything — the default), 'auto-python' (same as --auto-approve-python), or 'auto-plan' (same as -y/--yes: |
| `--trace-dir` `STR` | `traces` |  |
| `--failure-log-dir` `STR` | `failed-chats` | A readable copy of any run that fails, is stopped, or times out is saved here — separate from traces/, so a bad run is something you can just open. Pass '' to disable. |
| `--out` `PATH` |  | Write the run summary JSON here. |
| `--tui` | off | Watch the run live in a terminal dashboard instead of scrolling text. |
| `--continue` / `--no-continue` | on | After a successful run, show a suggested next step and offer to keep going with it or a new goal. Only ever prompts with a real terminal attached — a script or pipe gets exactly today's one-shot behavior regardless of this flag. |
| `--goal-loop` | off | After each attempt, ask a cheap judge model whether the goal is actually done — not just that the attempt finished cleanly — and, if not, run again with a narrower instruction. |
| `--max-goal-iterations` `INT` | `3` | Attempts --goal-loop will make before giving up and reporting the last one. |


### `harness setup`

Add your OpenRouter or Inception API key: asks for it, checks it, stores it.

| flag | default | what it does |
|---|---|---|
| `--provider` `STR` |  | openrouter or inception; asked when not given. |
| `--store` `STR` |  | keychain or file. Default: the macOS Keychain where there is one, else credentials.env in the config folder (readable only by you). |
| `--check` / `--no-check` | on | Check the key with the provider first. |


### `harness config`

View or persist default preferences, so you stop retyping the same flags.


#### `harness config show`

What is persisted, and where.


#### `harness config set`

Persist one preference. Applies from your next invocation on.

Arguments: `key` (One of: animate, approval_mode, ask_enabled, auto_approve, auto_approve_python, autolint, cache_prompts, ...), `value`


#### `harness config unset`

Remove one persisted preference, reverting it to the built-in default.

Arguments: `key`


#### `harness config path`

Where preferences are stored, for editing by hand or backing up.


#### `harness config reset`

Delete every persisted preference.

| flag | default | what it does |
|---|---|---|
| `-y` / `--yes` | off | Skip confirmation. |


### `harness skills`

List, add, or scaffold skills — reference material injected into the model's prompt for matching steps.


#### `harness skills list`

Every skill in effect: bundled, plus your own.


#### `harness skills add`

Copy a skill file into your own skills directory.

Arguments: `file` (A skill .md file to copy in.)


#### `harness skills new`

Scaffold a new skill in your own skills directory and print its path.

Arguments: `name` (Skill name, e.g. 'fastapi'.)


#### `harness skills path`

Where your own skills live.


### `harness trace`

Deterministic, model-free analysis over saved traces — run outcomes, undeclared/shadow files, recurring patterns.


#### `harness trace report`

Run outcomes, plus declared-vs-actual and shadow-file issues per run.

Arguments: `trace_dir` (Directory of run-*.jsonl trace files.)

| flag | default | what it does |
|---|---|---|
| `--workspace` / `-w` `DIRECTORY` |  | The project these traces ran against — enables the shadow-file collision check (an undeclared new file whose name already exists elsewhere in the tree). |
| `--failed-only` | off | Only show issues for runs whose final check failed. |
| `--json` | off | Machine-readable output. |


#### `harness trace audit`

Model-in-the-loop audit for defects tracetools' deterministic checks can't see: prompt misinterpretation, test-gaming, and similar judgment calls that need to read a run's actual narration. Traces are audited concurrently and cheaply (a few cents, seconds to tens of seconds for a handful of traces) — see agent/audit.py's module docstring for the measurements behind both defaults.

Arguments: `trace_dir` (Directory of run-*.jsonl trace files.)

| flag | default | what it does |
|---|---|---|
| `--workspace` / `-w` `DIRECTORY` |  | The project these traces ran against — enables the shadow-file and plan-conformance deterministic checks (see `trace report`'s same option), fed to the audit model as pre-computed facts. |
| `--model` `STR` |  | Model to audit with. Defaults to GLM 5.3 Flash — measured against Mercury on real failed traces and found to catch prompt-misinterpretation and test-gaming defects Mercury missed entirely, for a few cents total. |
| `--all` | off | Audit every run, not just failed ones. Off by default: real findings concentrate in failures, and a passing run rarely justifies its own model call. |
| `--concurrency` `INT` | `4` | Traces to audit at once. A directory's audit costs roughly one trace's latency regardless of how many traces there are, up to this cap — raise it for a large bench sweep, lower it if requests start queuing at the provider. |
| `--reasoning-effort` `STR` |  | Passed to the audit model. Defaults to 'low' — measured to cut GLM's per-trace wall time 4-6x and cost 6-11x versus leaving it unset, for the same findings both times (see agent/audit.py's DEFAULT_REASONING_EFFORT). |
| `--json` | off | Machine-readable output. |


#### `harness trace grep`

Which traces mention `pattern`, and where — for spotting the same defect recurring across separate runs (a single trace never shows this).

Arguments: `trace_dir` (Directory of run-*.jsonl trace files.), `pattern` (Literal substring to search for, case-insensitive.)


#### `harness trace sql`

Ask the traces anything, in SQL: no hand-written loop over JSONL.

Arguments: `trace_dir` (Directory searched for run-*.jsonl.), `query` (SQL over `events` (and `results`, if given).)

| flag | default | what it does |
|---|---|---|
| `--results` / `-r` `PATH` |  | Bench results .jsonl; repeatable. |
| `--json` | off | One JSON object per row. |


#### `harness trace after-edit`

What the model did in the turn after an edit that did not finish the step, split by whether the edit turn also ran the tests. `pct_next_check_only` after `edit alone` is the round trip a harness-run check would fold; after `edit + run_tests` it is the model re-running a result it already had.

Arguments: `trace_dir` (Directory searched for run-*.jsonl.)

| flag | default | what it does |
|---|---|---|
| `--phase` `STR` |  | Phase prefix to count; repeatable. Default execute, fix. |


#### `harness trace latency`

Where model time went: model hours with and without retried attempts, retry waits by cause, runaway reasoning (over 30,000 hidden tokens), first-token time by prompt size, and the longest post-first-delta silence where recorded. DESIGN-LOG §95.

Arguments: `trace_dir` (Directory searched for run-*.jsonl.)

| flag | default | what it does |
|---|---|---|
| `--model` `STR` | `mercury` | Substring of the model id; "" for every model. |
| `--by` `STR` | `week` | Period: week or day. |
| `--json` | off | One JSON object per row. |


#### `harness trace holds`

What the model did after each gate held one of its calls: re-sent it at once, reacted first (a test run, a read), or gave up on it. Counts per kind of hold; `--detail` lists every one.

Arguments: `trace_dir` (Directory of run-*.jsonl trace files.)

| flag | default | what it does |
|---|---|---|
| `--detail` | off | One line per hold, not only the counts. |
| `--recursive` | off | Search sub-directories too. |


#### `harness trace leaks`

Which runs reached for a benchmark's own sources: fetches of a denied URL (and how much came back) and searches matching --query. A fetch that returned text may be a leak (read the URLs); one that returned 0 was refused or failed.

Arguments: `trace_dir` (Directory searched recursively.)

| flag | default | what it does |
|---|---|---|
| `--deny` `STR` |  | A benchmark-source URL substring, as for run --web-deny. |
| `--query` `STR` |  | A regex; web searches whose query matches count too. |


### `harness sessions`

Deterministic, model-free analysis of Claude Code's own session transcripts — where the cost goes, what stays in context.


#### `harness sessions report`

Per-session summary, cost by component and context size, a compaction-cap estimate, what stays in context longest, and command/error habits.

Arguments: `directory` (Transcript directory. Default: this working directory's, under ~/.claude/projects.)

| flag | default | what it does |
|---|---|---|
| `--last` `INT` |  | Only the newest N sessions (default: all). |


## Persistable preferences

`harness config set KEY VALUE` accepts exactly these keys (`agent/userconfig.py`): `animate`, `approval_mode`, `ask_enabled`, `auto_approve`, `auto_approve_python`, `autolint`, `cache_prompts`, `check_coverage`, `clarify`, `executor_model`, `executor_reasoning`, `failure_log_dir`, `final_check`, `fix_existing_failures`, `memory`, `mercury_provider`, `narrate`, `planner_model`, `planner_reasoning`, `quick_answers`, `require_tests`, `review_plan`, `reviewer_model`, `show_stats`, `skills_enabled`, `spinner_style`, `step_concurrency`, `stream`. Every other flag is per run.


## The `bench` command

Compare agent configurations on accuracy, speed, and cost.


### `bench list`

Show available configurations and tasks.


### `bench run`

The default `--configs` is `hybrid,mercury-only`. `hybrid` is a retired configuration (Claude Sonnet plans and reviews; see {doc}`benchmarking` for its cost against Mercury), so name your configurations explicitly, e.g. `--configs mercury-only`.

Run a sweep.

| flag | default | what it does |
|---|---|---|
| `--configs` `CONFIGS` | `hybrid,mercury-only` | Comma-separated config names, or 'all'. |
| `--tasks` `TASKS` | `all` | Comma-separated task names, or 'all'. |
| `--repeats` `REPEATS` | `1` | Runs per (config, task). More than one gives you a spread. |
| `--repeat-offset` `REPEAT_OFFSET` | `0` | Number this batch's repeats from N instead of 0, so several batches appended to one --out (run under an hour each) keep distinct repeat indices. |
| `--parallel` `PARALLEL` | `1` | Concurrent runs. The work is network-bound, so 4-8 is reasonable. |
| `--timeout` `TIMEOUT` | `180.0` | Hard wall-clock limit per run, in seconds. Exceeding it kills the run's process and scores it as a timeout rather than waiting for a slow model to finish. 0 disables the limit. |
| `--max-per-model` `MAX_PER_MODEL` | `3` | Concurrent jobs allowed against the same underlying model, independent of --parallel. |
| `--out` `OUT` |  | Where to write the JSONL results. |
| `--plan-only` | off | Stop each run once its plan is accepted (no execution, no scoring): for questions about the plans, such as what the reproduce check did, at a fraction of a full run's cost. |
| `--trace-dir` `TRACE_DIR` | `traces/bench` |  |
| `--keep-workspaces` | off | Do not delete the generated code, so you can read it. |
| `--include-deprecated` | off | Include retired tasks (roman, primes, lru) that sit at a 100% accuracy ceiling for every config tested and no longer discriminate. |
| `-y` / `--yes` | off | Skip the cost confirmation. |
| `--allow-dead-search` | off | Run even when the web-search canary returns nothing (see _search_ready). |


### `bench scout`

A cheap first pass (2 repeats, 300s limit): what fails, and which tasks separate the configs, before any long run.

| flag | default | what it does |
|---|---|---|
| `--configs` `CONFIGS` |  | Comma-separated config names. |
| `--tasks` `TASKS` | `all` | Comma-separated task names, or 'all'. |
| `--repeats` `REPEATS` | `2` |  |
| `--timeout` `TIMEOUT` | `300.0` |  |
| `--parallel` `PARALLEL` | `6` |  |
| `--max-per-model` `MAX_PER_MODEL` | `6` |  |
| `--out` `OUT` |  |  |
| `--plan-only` | off | Stop each run once its plan is accepted (no execution, no scoring): for questions about the plans, such as what the reproduce check did, at a fraction of a full run's cost. |
| `--trace-dir` `TRACE_DIR` | `traces/bench` |  |
| `--keep-workspaces` | off |  |
| `--include-deprecated` | off |  |
| `-y` / `--yes` | off |  |
| `--allow-dead-search` | off | Run even when the web-search canary returns nothing (see _search_ready). |


### `bench external`

Run Aider on the same tasks, for comparison.

| flag | default | what it does |
|---|---|---|
| `--models` `MODELS` | `z-ai/glm-5.3-flash,qwen/qwen3.7-flash` | Comma-separated OpenRouter model ids (no 'openrouter/' prefix). |
| `--tasks` `TASKS` | `all` | Comma-separated task names, or 'all'. |
| `--repeats` `REPEATS` | `1` |  |
| `--parallel` `PARALLEL` | `1` |  |
| `--timeout` `TIMEOUT` | `300.0` |  |
| `--out` `OUT` |  | Where to write the JSONL results. |
| `--keep-workspaces` | off |  |
| `-y` / `--yes` | off | Skip the confirmation. |


### `bench archive`

Copy a sweep's records and traces under results/ and traces/, so its numbers can be re-derived after a scratch directory is cleared.

Arguments: `paths`

| flag | default | what it does |
|---|---|---|
| `--name` `NAME` |  | Archive name: results/NAME.jsonl, traces/NAME/. |
| `--force` | off | Overwrite an existing archive. |


### `bench report`

Re-render results from JSONL files.

Arguments: `paths`

| flag | default | what it does |
|---|---|---|
| `--by` `config\|task\|executor` | `config` |  |
| `--view` `outcome\|diagnostics\|discrimination\|reliability\|failures\|compare\|models\|history\|scout\|repro` | `outcome` | What to print; each view is described below the table. |
| `--include-stalls` | off | Count provider failures (stream stalls, connection resets, 5xx, zero-cost timeouts) as ordinary runs. By default they are left out and named, because a table that mixes them in mostly records who got stalled. |
| `--since` `SINCE` |  | history: ignore runs before this date (YYYY-MM-DD), for a task whose fixture changed. |
| `--a` `A` |  | compare: the baseline config; history: the config |
| `--b` `B` |  | compare: the config being tested |

Views (`--view`): outcome: accuracy, cost and time (the default). diagnostics: what the run did — wasted edits, exhausted loops, nudges, how much it read before its first edit. discrimination: which tasks actually separated the configs and which spent their budget without comparing anything. reliability: does it work every time (pass@k / pass^k), not just on average. failures: how runs ended, by class. compare: --a CONFIG --b CONFIG, with a bootstrap interval and a verdict per metric. models: which model ids and providers actually served each run, so a swap under an unchanged requested id is visible. history: --a CONFIG (default mercury-only), what it normally scores on each task and which tasks are at a ceiling; --since DATE ignores earlier runs. repro: how often the planner's reproduce check ran and what it showed, per config and task.

## Following along in the terminal

The dashboard has its own page, with screenshots: {doc}`tui`.

In `--tui`, press <kbd>p</kbd> (or <kbd>ctrl+p</kbd>, which works even while
the goal/answer bar has focus — see below) to open a settings panel with a
dropdown for approval mode, narrate, animate and quick-answers
so they can be changed without restarting — a compact status line above the
input bar always shows the current mode. A change made mid-run takes effect
on the *next* run (the current plan/execution already has its approver and
context built), not retroactively.

**Keyboard note:** a bare letter binding (`p`, `s`, `c`) only fires while the
bottom bar is *disabled* — while it's focused (before a goal is typed, or
between runs), Textual gives the bar first refusal on every key, so those
letters get typed as text instead. <kbd>ctrl+p</kbd> (open settings) and
<kbd>ctrl+a</kbd> (cycle the approval mode — manual → auto-python →
auto-plan → manual, one key, no panel needed) are priority bindings that
work regardless of focus. `ctrl+p` also replaces Textual's own default
command-palette binding, which the harness doesn't populate with anything —
it used to open an empty, confusing overlay if you tried it.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | the run finished **and** passed its own checks |
| 1 | the harness failed, *or* the work does not pass its own checks — a step failed its verify command, a step was abandoned, or the final check found failing tests. The run still produced output and a trace; it just isn't done |
| 2 | bad invocation (no goal, unknown setting) |
| 130 | interrupted with Ctrl-C |

A run that finishes with failing checks prints what failed before exiting.
This is why exit code 1 is not only for crashes: a run whose steps failed
verification used to exit 0, which made it indistinguishable from a clean one
to anything scripting the harness.
