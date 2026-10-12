"""Central configuration. Everything tunable lives here or in .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import keystore, userconfig

load_dotenv()
# Keys `harness setup` stored where there is no Keychain. Loaded after .env, and neither
# overrides a variable already set in the environment.
load_dotenv(userconfig.credentials_path())


def _get_keychain_secret(service: str, account: str) -> str | None:
    """A key from this machine's secret store (agent/keystore.py), or None. `account` is
    always "harness"; the parameter stays for the callers and tests that pass it."""
    return keystore.read(service) or None


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# The diffusion model under test. Executes plan steps.
#
# The default for every phase is Mercury, not the original Sonnet-plans/
# Sonnet-reviews hybrid design. Two rounds of bench sweeps agree: on this
# harness's task suite, Mercury doing all three phases (`mercury-only` in
# bench/configs.py) matched or beat the hybrid's accuracy at roughly 1/20th
# the cost and 2-4x the speed — 99-100% accuracy at $0.005/run and ~40s,
# against the hybrid's runs costing 10-20x more and, in the second sweep,
# frequently not finishing inside a generous per-run timeout at all (Sonnet's
# own latency, not a harness bug — see docs/benchmarking.md). A hybrid
# planner/reviewer earns its cost on work that is genuinely ambiguous or
# spans files in ways a single small-context executor cannot see; pass
# --planner/--reviewer explicitly (or `--all-mercury`'s opposite) for that,
# rather than paying for it on every run by default.
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL", "inception/mercury-2.5")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", EXECUTOR_MODEL)
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", PLANNER_MODEL)


def _env_int(name: str, default: int) -> int:
    """An integer from the environment, or the default.

    Deliberately forgiving: this is read at class-definition time, so a typo
    in a .env file (`128k`, or an empty value) would otherwise take down every
    command in the CLI with a raw ValueError traceback before it could say
    what was wrong.
    """
    raw = (os.getenv(name) or "").strip().replace("_", "")
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass
class Budget:
    """Hard stops so a runaway loop cannot burn the whole account."""

    max_steps: int = 20
    # Tool-call iterations allowed inside a single plan step.
    max_iters_per_step: int = 14
    # Tool-call iterations for the reviewer's fix passes.
    max_iters_per_fix: int = 10
    max_review_rounds: int = 2
    # Approximate prompt tokens allowed per execute step, and per review
    # chunk. None means "scale from the model's context window" — the default,
    # because a constant is what a growing workspace outgrows: at a fixed
    # 12,000 a workspace crossing ~1,000 lines silently stopped fitting and the
    # model paged its own files back in at ~1.3s a round trip. An explicit
    # number is used exactly as given, including a smaller one; an earlier
    # version floored these at the default, which quietly turned every
    # lowering into a no-op.
    step_prompt_token_budget: int | None = None
    review_prompt_token_budget: int | None = None
    max_usd: float = 2.00
    # Context windows, used to size the two budgets above. They are separate
    # because the chunks go to different models: review chunks are sent to
    # reviewer_model, and sizing them from the executor's window overflows the
    # reviewer in exactly the hybrid setup this file recommends — a 1M-context
    # executor with a Sonnet reviewer.
    model_context_tokens: int = _env_int("MODEL_CONTEXT_TOKENS", 128_000)
    reviewer_context_tokens: int = _env_int("REVIEWER_CONTEXT_TOKENS", 200_000)

    def step_budget(self) -> int:
        """Tokens a step's prompt may use.

        A quarter of the executor's window by default. The conversation grows
        on top of this as the step runs — run_tool_loop compacts at three
        times this figure — so a quarter leaves room for that and for the
        reply.
        """
        if self.step_prompt_token_budget is not None:
            return self.step_prompt_token_budget
        return max(12_000, self.model_context_tokens // 4)

    def review_budget(self) -> int:
        """Tokens a review chunk may use, sized to the *reviewer's* window.

        Two thirds of it: the reviewer's whole advantage is seeing everything
        at once, and it makes one pass rather than a long tool loop.
        """
        if self.review_prompt_token_budget is not None:
            return self.review_prompt_token_budget
        return max(30_000, self.reviewer_context_tokens * 2 // 3)


@dataclass
class Limits:
    """Resource ceilings, as opposed to Budget's spend ceilings."""

    # Cumulative wall time one step may spend in subprocesses.
    step_shell_budget_s: float = 90.0
    max_workspace_mb: float = 500.0
    max_file_write_mb: float = 5.0
    # Tool errors in one step before that step is abandoned.
    max_step_errors: int = 12


@dataclass
class Settings:
    workspace: str
    goal: str = ""
    executor_model: str = EXECUTOR_MODEL
    planner_model: str = PLANNER_MODEL
    reviewer_model: str = REVIEWER_MODEL
    # Model for agent/relevance.py's LLM-backed judge, when no Jev key is
    # configured. None (the default) uses executor_model -- already the
    # fast, cheap default everywhere else, and relevance scoring is a
    # similarly mechanical, low-reasoning judgement.
    relevance_model: str | None = None
    # Force build_judge() to one backend ("jev" | "llm" | "heuristic")
    # instead of its normal auto-preference. None (the default) is the
    # normal preference order. Exists so a bench sweep or a manual A/B can
    # hold the task and prompt fixed and vary only the judge backend --
    # without this, whichever key happens to be in this machine's Keychain
    # silently decides the backend for every run.
    relevance_backend: str | None = None
    # Execution tool names to include in every step's kit regardless of the
    # step text's own optional-group triggers (see agent/tools/__init__.py's
    # execution_tool_names). Empty by default -- the trigger match already
    # decides this for real use. Exists so a bench sweep can hold a task
    # fixed and force a tool present that the task's own auto-generated step
    # titles would not reliably trigger, to measure the tool's effect in
    # isolation rather than measure the trigger wording.
    force_tools: tuple[str, ...] = ()
    # Mercury exposes tunable reasoning; high for better accuracy across all phases.
    executor_reasoning: str | None = os.getenv("EXECUTOR_REASONING", "high")
    # High reasoning for planner to improve chain-of-draft plan quality.
    planner_reasoning: str | None = os.getenv("PLANNER_REASONING", "high")
    # Optional phase 1b: a second model checks the finished plan against the
    # goal, before execution starts, and can resubmit a corrected plan if it
    # finds a real gap. None (the default) skips this entirely — it exists
    # for a planner that is fast but terser (Mercury) paired with a cheap
    # check from a model that reads more carefully (see agent/phases/plan_audit.py
    # and bench/configs.py's `mercury-plan-glm-audit`), not for routine use.
    plan_auditor_model: str | None = None
    # Cheap by default for the same reason agent/audit.py's trace audit is:
    # this is a narrow alignment check against text already in front of the
    # model, not open-ended discovery, so high reasoning effort has little to
    # find and mostly costs time (DESIGN-LOG §41.1's reasoning_effort finding).
    plan_audit_reasoning: str | None = "low"
    auto_approve: bool = False
    # A narrower trust level than auto_approve: only run_python's own gate is
    # skipped, never run_bash's. Every run_python call already happens after
    # a plan exists (it is only ever offered during execute/fix, not
    # planning), so this is squarely about not re-approving the routine
    # verification snippets a step runs on itself — installs, git, and
    # anything else still go through the normal shell approval.
    auto_approve_python: bool = False
    trace_dir: str = "traces"
    # Run Ruff after every write. Syntax checking is unconditional.
    autolint: bool = True
    # Stream responses. Separates prefill from decode in the timing data.
    stream: bool = True
    # Suppress phase narration. The trace file is written either way.
    quiet: bool = False
    # Send an explicit cache breakpoint on the stable prompt prefix (system
    # prompt plus task), so a phase making a dozen tool calls pays for that
    # prefix once. Only Anthropic models take it -- see llm.CACHEABLE_PREFIXES
    # -- because a provider that does not understand the array-of-blocks
    # content form rejects the request outright rather than ignoring it, so
    # it cannot be sent blindly. With the all-Mercury defaults this therefore
    # changes nothing, and turning it off changes nothing either; the run
    # says so at startup rather than leaving that to be discovered.
    #
    # Automatic provider-side prefix caching is a separate mechanism and is
    # not affected by this flag. It is what actually saves money here today:
    # across 11,699 traced calls Mercury served 26.5% of its prompt tokens
    # from cache and GLM 47.7%, with no breakpoint ever sent.
    cache_prompts: bool = True
    # Reference material injected into phase prompts. Pushed, never fetched:
    # a tool call to fetch guidance costs more than the guidance saves.
    # None means the skills bundled with the package; pass a path to use a
    # different or additional set instead.
    skills_dir: str | None = None
    skills_enabled: bool = True
    skill_token_budget: int = 3_000
    # What an executor step's skill triggers match (DESIGN-LOG §117). "workspace": file-glob
    # triggers see every name at the workspace root and word triggers see the verify command,
    # so a README fires documenting-changes and `uv run pytest` fires uv on most steps. "step":
    # globs see only the step's own files and words skip its verify command (17% fewer skill
    # tokens a step in the replay). "step" is the default since §117.1: no accuracy or
    # resolved loss on fixtures or django, skill tokens a step down 4-8%.
    skill_triggers: str = "step"
    # Let the planner and executor ask a clarifying question instead of
    # guessing. Off by default for unattended runs (bench forces this False).
    ask_enabled: bool = True
    # One cheap, tool-free completion before planning: decide whether the goal
    # is specific enough to build from, and if not, ask up to three questions
    # and fold the answers into the planner's prompt. The planner's own
    # ask_user exists but almost never fires — a model driving a tool loop
    # toward submit_plan has every incentive not to stop — so this gives the
    # question its own call with nothing competing against it. Requires
    # ask_enabled (there is no point asking when nobody can answer); bench
    # forces both False.
    clarify: bool = True
    # The clarify call also sees the code's symbol map, the README's opening and how tests run,
    # with a closing instruction to ask only what the code cannot answer (DESIGN-LOG §92).
    clarify_context: bool = True
    # A short goal that points back ("fix those issues", "try again") is offered the newest
    # remembered run for one click; on yes it carries that run's record and clarify asks
    # nothing (clarify.resolve_follow_up, DESIGN-LOG §92.3).
    resolve_followups: bool = True
    # With a person present (clarify and questions on): the planner answers questions the
    # person asked back in clarify's answers, and the person confirms the plan's assumptions
    # with one keystroke or corrects them (agent/chatflow.py, DESIGN-LOG §111). No extra call.
    chat_checks: bool = True
    # Show the finished plan and wait for a human before executing it, with
    # the option to send it back with feedback instead of accepting it. Off
    # by default because it blocks, and an unattended caller must never be
    # left waiting on a prompt nobody will answer; the CLI and TUI both
    # additionally require a real terminal before honouring it.
    review_plan: bool = False
    # Run phase 3 at all. Off skips review and goes straight to the final gate,
    # which is different from stopping after execute: the gate still runs. The
    # benchmark expressed "no review" as stop_after="execute" and so silently
    # disabled the gate too, making its no-review configuration the most
    # ablated one rather than the least. Off by default since DESIGN-LOG §124.10:
    # review is 18-22% of spend and 20-25% of model time, and no suite shows a
    # loss without it (django and SciCode scouts, the §126 benchmark on django,
    # SciCode and the fixtures, a replay on GAIA's 173 reviewed answers: +1).
    review: bool = False
    # Read and write <workspace>/.harness/memory.md, so a run knows what
    # earlier runs on this workspace did, what they left broken, and what was
    # suggested next. Without it a follow-up goal ("why did that test fail?")
    # is unanswerable, because the only record of the previous run is a trace
    # file nothing reads back. See agent/memory.py.
    memory: bool = True
    # Prompt tokens of recalled memory a run may carry. Past this, the oldest
    # sections are dropped from the prompt, and the file itself is condensed
    # by one cheap call once it grows past several times this.
    memory_token_budget: int = 1_500
    # One cheap completion at the end of a successful run, proposing a
    # concrete next step. Off for unattended runs (bench forces this False) —
    # it costs a small call for something nobody there will read.
    suggest_next_step: bool = True
    # A short line before each state-changing tool call (writing a file,
    # running a command, committing) so a human watching can follow along
    # step by step, not just at the start and end of a whole plan step.
    # Routed through Trace.say() like everything else, so it respects
    # `quiet` the same way — a caller embedding the harness that already
    # sets quiet=True sees no console output either way.
    narrate: bool = True
    # One cheap, tool-free completion tried before planning: for a goal
    # that's really just a question — nothing to build, no files to touch —
    # answer it directly and skip the plan/execute/review pipeline (and its
    # narration) entirely. Falls through to the normal pipeline for
    # anything that looks like real work. Off for unattended runs (bench
    # forces this False, the same as suggest_next_step) — a benchmark is
    # measuring the pipeline itself, not deciding whether to use it.
    quick_answers: bool = True
    # The calls/cost/tok-per-s breakdown _finish() narrates at the end of a
    # run. On by default for direct/library use; the CLI defaults its own
    # --stats flag to off (hold off unless asked) and the TUI always forces
    # this off, since it renders the same numbers in its own stats panel
    # instead of this prose block.
    show_stats: bool = True
    # After a step, run its `verify` command and hand any failure back for a retry.
    verify_steps: bool = True
    # The deterministic gate after review: re-run every step's verify command
    # against the finished code, run the test suite, and type-check the
    # changed files — no model calls at all. Failures become blocker findings
    # and go through the ordinary fix path. This is the only signal in the
    # harness that can contradict a model that believes it succeeded, which
    # is why it defaults on even though it costs wall time.
    final_check: bool = True
    # Whether tests that were already failing when the run started count as
    # work to do. On by default: "the tests are failing, fix them" is most of
    # why anyone points this at a broken repository, and excusing those
    # failures made the agent ignore exactly the job it was given — one real
    # run reported PASS over a suite that could not even be collected. Turn
    # it off for a workspace that merely happens to contain unrelated red
    # tests, where chasing them would spend money on code the goal never
    # mentioned.
    fix_existing_failures: bool = True
    # The project's own test command, for a project pytest cannot run (django's
    # `tests/runtests.py`, found on SWE-bench tasks, DESIGN-LOG §69-§70). A
    # shell-style template; `{targets}` marks where test targets go, otherwise
    # they are appended. Empty means pytest, as before. Its output (unittest's
    # verbose format) is given a pytest-style summary (agent/testcmd.py) so the
    # gates and parsers read it unchanged.
    # Research before planning (agent/research.py, DESIGN-LOG §77): the installed
    # libraries' API through inspect_module first, then skills, then the web, and a
    # one-time hold on a submit_plan that looked nothing up. Opt-in until measured.
    research_first: bool = False
    # Run the code on a case with an independently known answer before a step may finish
    # (agent/numericcheck.py, DESIGN-LOG §81): a prompt block and a hold-once gate on
    # finish_step after an unrun edit. Opt-in until measured.
    numeric_check: bool = False
    # After an execute or fix turn that edits and does not finish, the harness runs the step's
    # tests once (run_tests' default targets) and appends the result to that turn's tool
    # results, so the model need not spend a call asking. 57% of edit turns are followed by a
    # turn that only checks (DESIGN-LOG §90 design C, §98). On by default since §99.3: check-only
    # turns after an edit 77.7% to 32.5%, model time -31%, cost -40%, accuracy not
    # distinguishably lower (vacuous-guard over 12 runs an arm: 0.900 vs 0.983, inside its spread).
    check_after_edit: bool = True
    # Hand all of a round's findings to one fixer loop instead of one loop per finding, so
    # their reading is shared and their edits are checked together. Fix rounds with 2+
    # findings are 19% of rounds but 42% of fix calls; calls grow with findings (7.3 for one,
    # 19.3 for two, 28.4 for three or more) (DESIGN-LOG §100). On by default since §101 (the
    # author's decision): rare on benchmarks, common in app runs (10 of the platformer app's 18
    # fix rounds).
    combined_fixes: bool = True
    # Skip fix round k+1 when round k left every touched file's contents as they were, so a fix
    # round does not run after one that changed nothing (DESIGN-LOG §124.2: 0 of 59 such rounds
    # passed the final check on the 2-5 Oct django and SciCode runs, 4 of 123 in earlier harness
    # versions). Off until §124.4's scout decides it; one rule for every task, not a per-suite cap.
    stop_after_idle_fix_round: bool = False
    # An empty workspace with a Python goal is set up with `uv init`, a [tool.ruff] section (so
    # formatting is on) and pytest and ruff as dev tools before planning (agent/scaffold.py,
    # DESIGN-LOG §104). Bench forces it off: fixtures must start as they are.
    scaffold_new_projects: bool = True
    # `ruff format` on the touched Python files when a step or fix finishes, in projects that
    # configure a formatter only (agent/autoformat.py, DESIGN-LOG §104).
    autoformat: bool = True
    # Offer edit_blocks (one plain-text argument of SEARCH/REPLACE blocks) instead of edit_files
    # and batch_edits_to_file (DESIGN-LOG §104). Off until its A/B (§105).
    edit_blocks: bool = False
    # The after-edit check also reports type errors the run has introduced (the final check's
    # rule, at the edit that caused them). On since §105.2: on semver, runs whose final check
    # found new type errors fell from 6 of 8 to 1 of 8, and fix calls from 26 to 6.
    types_after_edit: bool = True
    # read_file widens a narrow window on a Python file to the function around it and answers an
    # identical re-read of unchanged text in one line (DESIGN-LOG §104). On; off is §105's control.
    read_assist: bool = True
    # Aids for finding facts on the web (DESIGN-LOG §116): the web-research skill, a nudge to open
    # a result after three searches without one or on a rephrased query, and documenting-changes
    # firing only for steps that touch a document rather than for any root *.md. Off until §116's
    # A/B on GAIA decides it.
    web_research: bool = False
    # The reviewer is handed ty's report on this run's changes as well as the suite's output, and
    # its kit has no run_tests: it could only re-run what it was given, on code it cannot change.
    # On since DESIGN-LOG §108.2: review calls 3.15 to 1.33 a run (differs); findings, accuracy and
    # honesty not distinguishably lower. Measured with a Mercury reviewer.
    review_presupplied_checks: bool = True
    # Before an execute step that changed existing source finishes, check once a run that the
    # run's own evidence shows the reported problem going away: the plan's reproduction (when it
    # failed at plan time) no longer fails, and the tests the run added fail on the code as it was
    # before the run. Either finding holds finish_step once (agent/evidence.py). Aimed at django's
    # right-idea-wrong-place failures (11790, 11951), which the run's own checks passed. Off:
    # dropped in DESIGN-LOG §120.2 (fixtures guard failed; no held run moved its fix).
    bugfix_evidence: bool = False
    # OpenRouter provider preference for every call: "latency", "throughput" or
    # "price"; empty is OpenRouter's default (see OpenRouterClient.provider_sort).
    provider_sort: str = ""
    # URL substrings the web tools refuse to fetch or show in search results: a
    # benchmark's own problem, answer and upstream-fix addresses. GLM-planner arms
    # fetched SciCode's gold code and two django tasks' upstream fixes (DESIGN-LOG
    # §80). Empty for ordinary use.
    web_deny: tuple[str, ...] = ()
    # Where Mercury calls go: "openrouter" (default), "inception" (Inception's own
    # API) or "venice". Other models always go to OpenRouter (agent/providers.py,
    # DESIGN-LOG §85). Same model and per-token price on OpenRouter and Inception.
    mercury_provider: str = "openrouter"
    # Mercury's sampling temperature, for every call. None sends none, which is
    # the model's default (1.0 for mercury-2.5) and is what every run so far has
    # actually used: Inception resets anything outside [0.5, 1.0] to 1.0, and
    # every temperature the harness asked for was 0.0-0.3 (DESIGN-LOG §85).
    mercury_temperature: float | None = None
    # Executor prompt arms for Mercury's recency weighting (DESIGN-LOG §82). batch_last
    # (off) moves the batching rule to the end of the system prompt with worked examples;
    # tail_reminder appends the step's goal, the batching rule and the turns left after
    # the last message on every call. The reminder is on since §106.2: with the read
    # assists and the after-edit check it cut execute replies 21% (8.75 to 6.93 a run) at
    # equal accuracy. Measured on Mercury only.
    executor_batch_last: bool = False
    executor_tail_reminder: bool = True
    # With executor_tail_reminder: optional state sections in the tail, by name:
    # "errors" (the last failing test run, and how many runs in a row failed the
    # same way), "ledger" (files edited, test runs, edit misses this step).
    # agent/steptail.py, DESIGN-LOG §83.
    executor_tail_items: tuple[str, ...] = ()
    # The planner's own tail: the goal, the turns left and what submit_plan needs,
    # after the last message on every planning call (DESIGN-LOG §83).
    planner_tail: bool = False
    # submit_plan gets a `key_fact` per step (the one fact the step must not get
    # wrong), repeated in the executor tail when executor_tail_items has "key".
    plan_key_facts: bool = False
    # With research_first, three ways to get convention decisions into `findings`
    # (agent/research.py, DESIGN-LOG §87): the request at the end of the planner's
    # first message; `findings` required; a one-time hold when it has no decision.
    plan_findings_last: bool = False
    plan_findings_required: bool = False
    plan_findings_hold: bool = False
    # What the planner's gates do with a plan forced out by the last word (runner.
    # _last_word), which used to skip them all (DESIGN-LOG §88.3): "off" (skip, as
    # before), "turn" (a hold gets exactly one more forced turn), "notes" (the plan
    # goes through and the hold's message becomes a finding for the executor).
    last_word_gates: str = "off"
    # Output-token cap for every Mercury call that sets none. None sends none, so
    # the provider's own 65,536 applies. 1.1% of SciCode's Mercury calls produced
    # >32k tokens, all reasoning, and none produced 16k-32k (DESIGN-LOG §83).
    # Output-token cap sent on Mercury calls that set none. 16,000 since §95.1:
    # it removed every runaway (~62k hidden reasoning tokens, 77 s a call) at 30%
    # less model time on SciCode, and made no length stop at all. None: no cap.
    mercury_max_tokens: int | None = 16_000
    # Cut a Mercury stream that is silent this many seconds after its first delta,
    # and retry it. 30 s since §95.1 (the author's decision on the scout); it also catches
    # provider stalls that are not runaways, which the cap cannot. None: off.
    mercury_content_stall_s: float | None = 30.0
    # A command that answers the run's questions instead of a person: each question
    # goes to its stdin as JSON ({"question", "options", "recommended"}) and its
    # stdout is the answer (agent/interact.py, command_asker). For scripted runs and
    # the simulated-user evaluation; empty means ask a person as usual.
    answer_command: str = ""
    test_command: str = ""
    # With test_command: test files under this directory are passed to the
    # command as dotted module labels relative to it (`tests/pagination/x.py`
    # -> `pagination.x`), which is what django's runner accepts; it rejects
    # file paths. Empty passes paths unchanged.
    test_label_root: str = ""
    # Show the planner what each already-failing test reports (agent/failsig.py)
    # rather than only its name. A knob so a sweep can price it.
    baseline_digest: bool = True
    # After a green run, check in a scratch copy whether the tests can see the
    # change: do the added tests pass on the pre-run code, do small mutations of
    # the edited functions go unnoticed, did an edited test file lose
    # catching power. Report-only. agent/testcheck.py.
    test_audit: bool = True
    # When the audit finds tests that cannot see the change, spend one fix round
    # on them (and re-audit). Off until a sweep says it helps.
    fix_weak_tests: bool = False
    # Tell the planner not to assert a cause it cannot run for a goal that
    # reports a symptom rather than naming the defect (prompts.py's
    # SYMPTOM_FIRST_PLANNING). Off by default: a 60-run interleaved A/B found
    # it slightly *worse* on both the column it targets and the one it risked
    # (83% vs 90% honest, 96.1% vs 98.3% accuracy, n=30 a side), and it costs
    # 285 words on every planning call. Kept behind the knob rather than
    # deleted, the same way a retired bench config is -- see DESIGN-LOG §45.4.
    symptom_first_planning: bool = False
    # Send every tool schema with `strict: true`, asking the provider to keep
    # arguments to their declared types while decoding. On by default after a
    # 40-run A/B: arguments sent JSON-encoded 0/432 calls vs 9/524, parse
    # errors 0 vs 4, for planner calls ~30% slower at the median (~2s a run);
    # executor calls unchanged (DESIGN-LOG §63.1).
    strict_tool_schemas: bool = True
    # Send a plan back once when the goal renames or removes a name and the
    # plan does not write every file that uses it (agent/callsites.py). On
    # after 80 plan-only runs: plans missing a caller 0/40 vs 6/40, every held
    # plan repaired, planner calls unchanged (DESIGN-LOG §66.1).
    callsite_check: bool = True
    # How the planner checks a reported problem before planning its fix
    # (DESIGN-LOG §54-54.2). Plans that named a cause after one read were the
    # §44 finding; a prompt block and a plan audit (§45) did not help.
    #   "repro" (default): submit_plan carries a `reproduce` snippet asserting
    #     the behaviour the report expects; runner.ReproGate runs it on a
    #     scratch copy first. Fails -> the problem is real, plan accepted, no
    #     extra call. Passes -> the report did not reproduce, plan sent back once.
    #   "require": probe_python in the kit and the first plan held until a probe
    #     ran. Best on wrong reports (no-bug honest 60% -> 88%) but +2.0 calls a
    #     run on every task, so not the default (§54.1).
    #   "offer": probe_python without the hold; almost never used.
    #   "off": none of it.
    # Why "repro" is the default: over 19 tasks (78 runs a side) it passed the
    # rule set before the sweep: accuracy -1.5 points (n.s., within the
    # baseline's historical per-task rates), honest +2.6, calls +0.13 a run.
    # Its benefit is not yet significant: honest on the three wrong-report
    # tasks went 37% -> 53% (n.s.), held back because the planner left
    # `reproduce` empty in 29 of 78 runs. Sending an empty snippet back on a
    # self-declared bugfix (plan_task_kind + repro) cost +1.9 calls and moved
    # honest by 0, so it is opt-in (§54.3). What limits it now is not
    # information but decision: on misreport-slug the planner's own draft
    # says the code is already correct and its plan edits it anyway ("make it
    # more explicit"). Next: act on a passed reproduction at execution, where
    # a plan that edits source after its own reproduction passed is the
    # exact signal; and misreport-discount (intended behaviour reported as a
    # bug, 0-1/15 in every arm), which no reproduction can catch.
    # Withheld on a workspace with no Python source (plan._probe_mode).
    plan_probe: str = "repro"
    # With plan_probe="repro", how submit_plan asks for the reproduction
    # (agent/symptom.py, DESIGN-LOG §65): "snippet" (default, the §54.2 field),
    # "explicit" (asked before the steps, `none: why` instead of an empty
    # string) or "symptom" (the call and its expected value; the harness writes
    # the assertion). Aimed at the wrong-report runs that left the snippet empty.
    repro_form: str = "snippet"
    # Hold, once per run each, a source edit made after the plan's own
    # reproduction passed, and an edit that removes or weakens a test that
    # existed before the run (runner.PremiseGate; DESIGN-LOG §57). Aimed at the
    # two wrong-report failures §54.3 traced: planning a change to code the
    # planner had just shown correct, and deleting the test that pins the
    # behaviour a report called a bug. On by default since §57.1: over 212
    # runs it passed the pre-set rule (accuracy -0.3 points, honest +1.5,
    # calls -0.5, wasted edits -0.01 a run) with no measurable benefit. It
    # reaches few runs: most misreport-discount failures rewrite the pinned
    # test's expected value, which keeps its assertion count and so passes
    # hold (b), and 4 of 5 held edits were re-sent anyway. Next: extend (b) to
    # rewritten assertions, and decide whether once is enough (§57.1).
    premise_gate: bool = True
    # Instead of PremiseGate's once-only hold on a removed or weakened test: hold
    # any edit that removes, weakens or *rewrites* a pre-existing test until it
    # carries a `justification` (agent/testchange.py), show what the test asserts
    # today and why it may be poor, record the reason and print it in the run
    # summary. Off by default; DESIGN-LOG §59.
    test_change_justification: bool = False
    # How the planner drafts before it commits: "terse" (chain-of-draft, brief
    # notes) or "prose" (full sentences and paragraphs, code sketches allowed in
    # the draft only). The draft never reaches the executor, so this only changes
    # the steps written after it. See DESIGN-LOG §48.
    draft_style: str = "terse"
    # Hold a step's first non-test source edit until a test has been run, for a
    # step whose plan names tests (agent/runner.py's _TddGate). Off by default;
    # DESIGN-LOG §49.
    tdd_gate: bool = False
    # "tdd_first" or "legacy": the executor prompt's guidance on order of work.
    # tdd_first by default since DESIGN-LOG §51: §49's arm lifted red-first from
    # 3% to 20% at no measurable cost. "legacy" is the old batch-everything text.
    executor_order: str = "tdd_first"
    # Hand the planner's draft notes (its reading of the request and the
    # failure modes it named) to every executor step. Off by default: the
    # draft has only ever been display text, and whether the executor gains
    # from the planner's reasoning (the architect/editor split, where the same
    # model paired with itself gained 3-5 points on Aider's benchmark) is
    # what this flag's bench arm measures.
    executor_sees_draft: bool = False
    # After a failing test run or a rejected edit, ask for a one-line `why`
    # on the next edit/test call: the failure it read and what the call
    # changes about it. A chain-of-draft note inside a call the model was
    # making anyway, aimed at edit/test cycling (DESIGN-LOG §43). Off by default.
    executor_hypothesis: bool = False
    # Ask the planner for a `task_kind` (bugfix/feature/refactor/tests/docs/
    # other) inside submit_plan: the triage signal without a triage call.
    # Recorded, not routed on, until its agreement with known task categories
    # is measured. Off by default so the default planner schema is unchanged.
    plan_task_kind: bool = False
    # Preload, as read-only context, the workspace files that import each of a
    # step's declared files (tests first, then callers), in whatever preload
    # budget the declared files leave. Aimed at plans that commit to a fix
    # after one read (DESIGN-LOG §44). Deterministic: no model call.
    preload_neighbours: bool = False
    # Add to the planner's first message where the goal's own identifiers are defined
    # (agent/issue_symbols.py, DESIGN-LOG §128): the symbol map covers an alphabetical prefix of a
    # large repository and listed a changed file for 0 of 10 django tasks, the lookup for 8 of 10.
    # Deterministic, no model call, nothing provider-specific. Off until its A/B.
    plan_issue_symbols: bool = False
    # Skip the planner: one step whose intent is the goal, then the same executor, final check
    # and gates (DESIGN-LOG §129). No model call is added; the planner's calls are removed. Off
    # until its A/B, which decides whether a model that can localize alone needs the plan.
    direct: bool = False
    # An explicit shell command that proves the goal's own claim, run as part
    # of the final gate alongside (not instead of) every step's own verify.
    # None by default: this exists for the case a plan's own step-scoped
    # tests can technically pass while missing the goal's specific claim — a
    # step verified with `ruff check` alone, or a test that covers the
    # feature's shape but not the exact scenario the user described. Failing
    # this becomes an ordinary blocker finding, exactly like a step verify
    # failure — no new finding type, no new fix path.
    goal_verify: str | None = None
    # Fail a run that writes or changes behaviour in a workspace that has no
    # tests at all. Six consecutive runs on one project reported PASS over a
    # game with broken scoring and a preferences pane that did nothing: with
    # no tests to run, the deterministic gate had nothing deterministic left
    # to check, and a passing `ruff check` stood in for "it works". Turn it
    # off for a workspace where tests genuinely do not apply.
    require_tests: bool = True
    # Check that the test suite actually executes the lines this run changed,
    # not merely that a suite exists and is green. A real run produced five
    # passing tests about the exact features that were broken, by setting
    # internal state by hand rather than running the new code at all. The
    # suite runs once either way — under `coverage run` — so this costs one
    # extra fast subprocess to read the data, not a second test run.
    check_coverage: bool = True
    # The fraction of a file's changed lines the tests must actually execute
    # before the gate stops complaining. Not zero: `def f(...)` runs at import,
    # so a function nothing ever calls still shows a covered line, and "no
    # lines covered" would have been a rule that could never fire.
    min_changed_coverage: float = 0.5
    # At the gate, let a green whole-suite run stand in for a step's own
    # pytest-selection verify command instead of running both. The two are
    # not independent: every test the selection would run, the suite just
    # ran. One traced run spent 46 of its 441 seconds re-establishing, twice,
    # that `pytest <file> -k <name>` passes inside a suite already green.
    # Deferred rather than dropped — a red suite still runs them, because a
    # failed step verify is what carries the per-step attribution a fix round
    # needs. Turn off to have every verify command run unconditionally.
    skip_subsumed_verifies: bool = True
    # Type-check what a step changed, against the rest of the project, as
    # part of that step's verification. `ty` resolves the whole project to
    # check imports, so it sees a call site in a file the step never opened
    # -- which is how a changed signature shows up as broken callers. Traced
    # runs had the reviewer type-checking 486 times and the executor 16, so
    # this was being found one phase later than it could be.
    type_check_steps: bool = True
    # Narrow a step's whole-suite `verify` to the step's own files. 429 of
    # 960 traced step verifies were a bare `uv run pytest -q` -- the whole
    # suite, run to check one step, which also fails a correct step whenever
    # a later step is the one that writes its tests.
    scope_step_verify: bool = True
    # Run the whole-suite baseline and gate checks under pytest-xdist
    # (`uv run --with pytest-xdist pytest -n auto`), installed ephemerally so
    # the workspace's own pyproject is untouched. Off by default, and this is
    # a real trade rather than a free win: xdist runs tests in separate
    # worker processes, and a suite that shares state across tests --
    # writing to the same file, module-level mutable globals, an assumption
    # about execution order -- can fail under it in ways it never does
    # serially. This was not hypothetical: turning it on against this
    # project's own suite broke two tests that shared a fixture directory on
    # disk, with no change to the code being tested. Worth it for a large
    # suite that is not dominated by one or two slow tests -- parallelism
    # cannot beat wall clock below the slowest single test, so a suite like
    # one traced run's (76 tests, 45s total, one test alone 22.6s of it) sees
    # barely 15% from this, while an evenly-spread suite can see far more.
    # Never combined with coverage measurement in the same invocation: see
    # _run_suite's docstring for why that pairing corrupts rather than helps.
    test_parallel: bool = False
    # Skip a step's verify command when the executor already ran an
    # equivalent pytest invocation that passed, against a workspace that has
    # not changed since. 667 of 960 traced step verifies followed a passing
    # run_tests with nothing mutating in between; on a real project each
    # repeat is twenty seconds for a result that cannot have changed.
    skip_verified_by_tests: bool = True
    # Replay a read-only check's recorded output when the identical call is
    # made again and every byte of every touched file is unchanged. A traced
    # reviewer called run_tests on the same file four times in one phase,
    # twice at 22 seconds, having already been told the suite was green.
    # Turn off to re-run every call for real.
    replay_identical_checks: bool = True
    # The cross-session extension of the same idea, applied to the one check
    # that runs before anything else: collect_baseline's pre-run suite pass.
    # workspace_state() is a cheap (mtime+size, no file reads) digest of the
    # whole workspace, already trusted elsewhere in this codebase to tell
    # "unchanged" from "changed" without reading file contents -- if it
    # matches what a previous run recorded in .harness/test_baseline_cache.json,
    # the workspace cannot have changed since, and the suite's result cannot
    # have changed either. Persisted across runs, unlike replay_identical_checks
    # above, because a baseline is collected once per *run*, not once per
    # call, so there is nothing to reuse from earlier in the same run to
    # begin with. Turn off to always collect a fresh baseline.
    cache_baseline_across_runs: bool = True
    # Ceiling in seconds on one model call including every retry and backoff.
    # Not just one attempt: before this existed, seven attempts at 180s plus
    # six backoffs meant a single call could run for the better part of an
    # hour, and traced benchmark runs did exactly that (3,632s, reported as a
    # 1.4s call). Raise it for a slow model on a long generation; lower it to
    # fail faster against a flaky provider.
    max_call_s: float = 420.0
    # Retry a failed step, feeding the next attempt the failure and what the
    # last one already tried, and turning the executor's reasoning up a level
    # if it is not already at the top. It usually is: `executor_reasoning`
    # defaults to "high", so on a default run the reasoning bump is a no-op and
    # the feedback is the whole mechanism. Turning this off disables the retry
    # itself, not just the bump.
    escalate_on_failure: bool = True
    # Attempts per step, including the first.
    max_step_attempts: int = 2
    # Plan steps to execute at once. 1 is sequential. Only steps that declare no
    # dependency on each other and touch disjoint files ever run together.
    step_concurrency: int = 1
    # Permit a workspace nested inside another Python project. Off by default:
    # build tools search upward, so a nested workspace can modify its host.
    allow_nested: bool = False
    # A readable record of a run that failed, was stopped, or hung, written
    # separately from traces/ so a bad run is something you can just open
    # instead of grepping hundreds of good ones for it. "" or None disables it.
    failure_log_dir: str | None = os.getenv("FAILURE_LOG_DIR", "failed-chats")
    budget: Budget = field(default_factory=Budget)
    limits: Limits = field(default_factory=Limits)
    # Filled in on first api_key access. Not a user-settable field; excluded
    # from repr so a Settings dump never carries a secret into a log.
    _api_key_cache: str = field(default="", repr=False, compare=False)
    _jev_api_key_cache: str = field(default="", repr=False, compare=False)

    @property
    def api_key(self) -> str:
        # Cached after the first read: the Keychain lookup shells out to
        # `security` with a 5s timeout, and this is a property, so a caller
        # that touched it in a loop would pay that repeatedly for a value
        # that cannot change within a run.
        if self._api_key_cache:
            return self._api_key_cache
        # Try Keychain first (most secure), then env var, then .env
        key = _get_keychain_secret("OpenRouter", "harness")
        if not key:
            key = os.getenv("OPENROUTER_API_KEY", "")
        if key:
            object.__setattr__(self, "_api_key_cache", key)
        if not key:
            raise RuntimeError(
                "no OpenRouter API key. Run `harness setup` to add one (or an Inception key "
                "for Mercury only), or set OPENROUTER_API_KEY."
            )
        return key

    @property
    def jev_api_key(self) -> str:
        """TypeSafe/Jev API key, or "" if none is configured.

        Unlike `api_key`, never raises: Jev is optional throughout this
        harness (see agent/relevance.py) — a workspace with no key configured
        falls back to an LLM-backed or purely heuristic judge instead of
        failing. Env var name (`TYPESAFE_API_KEY`) matches what the
        `typesafe_sdk` package itself reads by default, so a key set for the
        raw SDK works here unchanged.
        """
        if self._jev_api_key_cache:
            return self._jev_api_key_cache
        key = _get_keychain_secret("TypeSafe", "harness")
        if not key:
            key = os.getenv("TYPESAFE_API_KEY", "")
        if key:
            object.__setattr__(self, "_jev_api_key_cache", key)
        return key


# Shell commands that never need a human. Matched on the first word only —
# but not on faith alone for the ones that read or write files by argument:
# see approval.py's _escaping_path, which still gates cat/head/tail/wc/stat/
# file/diff/cp/mv/mkdir/touch/tree/find/ls/sed/grep/rg when an argument it
# can classify as a path resolves outside the workspace.
BASH_ALLOWLIST = {
    "ls",
    # Read-only file tests; plans verify answer files with them (DESIGN-LOG §93).
    "test",
    "[",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "find",
    "file",
    "stat",
    "pwd",
    "echo",
    "which",
    "tree",
    "diff",
    "sed",
    "pytest",
    "python",
    "python3",
    "node",
    "npm",
    "npx",
    "git",
    "mkdir",
    "touch",
    "cp",
    "mv",
    # Astral toolchain: uv drives the env, ruff lints and formats, ty type checks.
    "uv",
    "uvx",
    "ruff",
    "ty",
    # agent.dataformats's synthesized verify commands (jq/yq for JSON/YAML/
    # TOML/XML/CSV, xmllint as a fallback for XML) and agent.dataformats's
    # pandoc-based Markdown check. Not classified by _escaping_path's path-
    # arg heuristic below — a traced run's own verify command,
    # `pandoc -f gfm -t plain -o /dev/null README.md`, would misclassify
    # `-o`'s value `/dev/null` as an escaping path if it were, the same
    # reason `_PATH_ARG_COMMANDS` already leaves pytest/ruff/ty/git alone —
    # so these get the allowlist's plain trust, matching that precedent,
    # not the deeper path check. Missing this list entirely is the bug that
    # motivated adding it: `pandoc -f gfm -t plain -o /dev/null README.md`
    # failed with "'pandoc' is not allowlisted" on a real machine where
    # pandoc was genuinely installed, and the run spent 14 iterations
    # trying to "install" a tool that was already there before giving up.
    "pandoc",
    "jq",
    "yq",
    "xmllint",
}
# Never runs, approved or not.
BASH_DENYLIST_SUBSTRINGS = ("rm -rf /", ":(){", "mkfs", "dd if=", "shutdown", "reboot")
