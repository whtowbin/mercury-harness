# Using the agent

Assumes you finished {doc}`quickstart`. Every flag is in {doc}`cli`; this page is the path
through the ones you need first.

## A first run

Make a workspace **outside** any Python project (the harness refuses a nested one; see
{doc}`safety`), then give the agent a goal:

```bash
mkdir -p ~/agent-sandbox
harness run -w ~/agent-sandbox "build a CLI todo list with JSON persistence and pytest tests"
```

By default Mercury does the planning and the execution, there is no review phase, and a final
check with no model call re-runs every step's verify command, the project's tests and a type
check. The agent asks before it runs shell commands; a run stops at a spend cap of $2.00
(`--max-usd`). Leave the goal off and a terminal prompts you for one.

To look at the plan before anything is built, stop after planning and run the saved plan later:

```bash
harness run -w ~/agent-sandbox --stop-after plan --out plan.json "your goal"
harness run -w ~/agent-sandbox --plan plan.json
```

`--review-plan` shows the plan and waits for your approval first. `--review` adds the
review phase, which costs about a fifth more spend and time (the evidence is in
{doc}`configuration-comparison`).

## Follow-ups in the same workspace

Run `harness run` again with the same `-w`. The agent keeps notes in
`<workspace>/.harness/memory.md` (`--memory`, on by default), so a goal such as "fix those
issues" or "why did that test fail?" refers to the last run, and it confirms the match with one
click instead of guessing. After a successful run in a terminal it also offers a suggested next
step, which you can accept or replace with a new goal (`--continue`).

## Answering clarify questions

Before planning, the agent decides whether the goal is specific enough to build from and asks
up to three questions if it is not. During planning and execution it can also ask when it is
unsure (`--ask`). Both are on by default and need a terminal to answer in. A script or pipe never
gets a prompt.

```bash
harness run --no-clarify -w ~/agent-sandbox "goal"   # never ask before planning
harness run --no-ask -w ~/agent-sandbox "goal"       # never ask at all
```

## The dashboard

With the `tui` extra installed ({doc}`quickstart`), a live dashboard replaces scrolling text, with
the goal prompt docked at the bottom:

```bash
harness run --tui -w ~/agent-sandbox
harness-tui -w ~/agent-sandbox "fix the failing tests"   # same, takes every harness run flag
```

Press <kbd>ctrl+p</kbd> for a settings panel (approval mode, narration, animation).

## Persisted preferences

Flags you type every time can be stored once. They apply from the next invocation, and an
explicit flag still wins.

```bash
harness config set approval_mode auto-python    # run_python without asking, shell still asks
harness config set planner_model anthropic/claude-sonnet-5
harness config show                             # what is persisted, and where
harness config unset planner_model              # back to the built-in default
harness config reset                            # drop everything
harness config path                             # the file, to edit or back up
```

`harness config set` accepts a fixed list of keys, given in {doc}`cli`. Everything else is per run.

## Choosing models and providers

Every phase uses `inception/mercury-2.5` unless you name another model:

```bash
harness run --planner anthropic/claude-sonnet-5 -w ~/agent-sandbox "goal"   # stronger planner
harness run --review --reviewer anthropic/claude-sonnet-5 "goal"            # add a reviewer
```

Any model other than Mercury goes through OpenRouter, so it needs an OpenRouter key. The
measurements on when a stronger planner helps are in {doc}`configuration-comparison`.

`--mercury-provider` chooses where Mercury calls go: `openrouter` (the default) or `inception`
(Inception's own API, needing an Inception key from `harness setup`):

```bash
harness run --mercury-provider inception -w ~/agent-sandbox "goal"
harness config set mercury_provider inception    # make it stick (setup does this for you)
```

## Where traces go

Every run writes a JSONL trace of its model calls and tool calls to `traces/` under the
directory you ran `harness` from (`--trace-dir` changes it). A run that fails, is stopped or
times out also leaves a readable copy in `failed-chats/` (`--failure-log-dir`, and `''` turns it
off), which is the first place to look when something goes wrong. Neither holds your API key.

## Reading a trace

```bash
harness trace report traces/                  # outcome of each run, plus problems per run
harness trace report traces/ --failed-only    # only runs whose final check failed
harness trace grep traces/ "ModuleNotFoundError"   # which runs mention a string, and where
harness trace sql traces/ "SELECT count(*) FROM events"   # any question, in SQL (needs the analysis extra)
```

`harness trace --help` lists the rest (`latency`, `holds`, `leaks`, `audit`). Each takes the
trace directory first.
