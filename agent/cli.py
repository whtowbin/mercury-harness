"""Command line entry point: `harness`, built on Typer and Rich."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import typer.core
import typer.main
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import awake, config, userconfig
from . import summary as summary_module
from .config import Budget, Limits, Settings
from .loop import model_tok_per_s
from .loop import run as run_agent
from .spinner import Spinner

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A coding-agent harness for Mercury 2.5, a diffusion LLM: it plans, executes "
    "small steps and runs a model-free final check (review is optional, --review). "
    'Run `harness setup` once, then `harness run "goal"`.',
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True)
_prefs = userconfig.load()


def _print_version(value: bool) -> None:
    if value:
        from . import __version__

        console.print(f"mercury-harness {__version__}")
        raise typer.Exit()


@app.callback()
def _app_options(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Print the version."
        ),
    ] = False,
) -> None:
    """The group's own options; each command has its own."""


class Reasoning(StrEnum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


class StopAfter(StrEnum):
    plan = "plan"
    execute = "execute"
    review = "review"


class ApprovalMode(StrEnum):
    """A single choice covering auto_approve and auto_approve_python at
    once — the three tiers of trust the harness actually has, named plainly
    rather than as two independent booleans a user has to reason about
    together. See Settings.auto_approve / .auto_approve_python for what
    each tier actually gates."""

    manual = "manual"
    auto_python = "auto-python"
    auto_plan = "auto-plan"


def _read_approval_mode_pref() -> ApprovalMode | None:
    """The persisted approval mode, if one was set and is still a valid name.

    An unrecognised stored value is ignored rather than raising: a config file
    is hand-editable, and a typo in it should cost a default, not the ability
    to run the tool at all.
    """
    stored = _prefs.get("approval_mode")
    if not stored:
        return None
    try:
        return ApprovalMode(str(stored))
    except ValueError:
        return None


# Read once at import, like every other _prefs.get default in this module —
# Typer evaluates argument defaults at definition time, so a call here would
# be a function call in a default anyway.
_APPROVAL_MODE_PREF = _read_approval_mode_pref()


def _reject_conflicting_flags(
    plan_path: Path | None,
    stdin_plan: bool,
    goal_loop: bool,
    tui: bool,
    draft_style: str,
    plan_probe: str = "off",
) -> None:
    """Exit 2 on flag combinations that cannot mean anything."""
    if plan_path and stdin_plan:
        err_console.print("[red]error:[/] pass --plan or --stdin-plan, not both")
        raise typer.Exit(2)
    if goal_loop and tui:
        err_console.print("[red]error:[/] --goal-loop is not compatible with --tui yet")
        raise typer.Exit(2)
    if draft_style not in ("terse", "prose"):
        err_console.print(
            f"[red]error:[/] --draft-style must be 'terse' or 'prose', not {draft_style!r}"
        )
        raise typer.Exit(2)
    if plan_probe not in ("off", "offer", "require", "repro"):
        err_console.print(
            "[red]error:[/] --plan-probe must be 'off', 'offer', 'require' or 'repro', "
            f"not {plan_probe!r}"
        )
        raise typer.Exit(2)


def _plan_argument(plan_path: Path | None, stdin_plan: bool) -> str | None:
    """How run() is told which saved plan to load: "-" means stdin."""
    if stdin_plan:
        return "-"
    return str(plan_path) if plan_path else None


def _resolve_goal(
    goal: list[str] | None, plan_path: Path | None, stdin_plan: bool, tui: bool
) -> str:
    """The goal text: from the arguments, else asked for interactively. A saved
    plan, a plan on stdin, or the TUI (which asks itself) needs no goal."""
    goal_text = " ".join(goal or []).strip()
    if goal_text or plan_path is not None or stdin_plan or tui:
        return goal_text
    if not sys.stdin.isatty():
        err_console.print(
            "[red]error:[/] give the agent a goal, --plan a saved file, or --stdin-plan"
        )
        raise typer.Exit(2)
    goal_text = typer.prompt("What should the agent build?").strip()
    if not goal_text:
        err_console.print("[red]error:[/] no goal given")
        raise typer.Exit(2)
    return goal_text


def _unless_given(value, field: str):
    """A Mercury limit left unset on the command line keeps the Settings default;
    0 turns it off. Passing the option's own None through would switch a default
    off for every CLI run without anyone asking for it."""
    if value is None:
        return {f.name: f.default for f in dataclasses.fields(Settings)}[field]
    return value or None


@app.command("run")
def run_command(
    goal: Annotated[list[str] | None, typer.Argument(help="What the agent should build.")] = None,
    workspace: Annotated[
        str, typer.Option("-w", "--workspace", help="Directory the agent may touch.")
    ] = ".",
    executor: Annotated[
        str, typer.Option(help="Model that executes plan steps (the dLLM under test).")
    ] = _prefs.get("executor_model", config.EXECUTOR_MODEL),
    planner: Annotated[str, typer.Option(help="Model that writes the plan.")] = _prefs.get(
        "planner_model", config.PLANNER_MODEL
    ),
    reviewer: Annotated[
        str | None, typer.Option(help="Model that reviews. Defaults to the planner.")
    ] = _prefs.get("reviewer_model"),
    all_mercury: Annotated[
        bool,
        typer.Option(
            "--all-mercury", help="Use the executor model for every phase, as a dLLM-only baseline."
        ),
    ] = False,
    executor_reasoning: Annotated[
        Reasoning | None, typer.Option(help="Mercury's tunable reasoning level for execution.")
    ] = None,
    planner_reasoning: Annotated[Reasoning | None, typer.Option(help="")] = None,
    plan_path: Annotated[
        Path | None,
        typer.Option("--plan", help="Skip planning and run a plan saved from an earlier run."),
    ] = None,
    stdin_plan: Annotated[
        bool,
        typer.Option(
            "--stdin-plan", help="Skip planning and accept a plan JSON from stdin instead."
        ),
    ] = False,
    stop_after: Annotated[
        StopAfter, typer.Option(help="Stop after this phase.")
    ] = StopAfter.review,
    max_usd: Annotated[float, typer.Option(help="Hard spend cap for the run.")] = 2.0,
    max_steps: Annotated[int, typer.Option()] = 20,
    step_budget: Annotated[
        int | None,
        typer.Option(
            help="Approx prompt tokens per execute step. Default: a quarter of the "
            "executor's context window."
        ),
    ] = None,
    review_budget: Annotated[
        int | None,
        typer.Option(
            help="Approx prompt tokens per review chunk. Default: two thirds of the "
            "reviewer's context window."
        ),
    ] = None,
    step_concurrency: Annotated[
        int,
        typer.Option(
            help="Plan steps to run at once. Only steps with no declared dependency on each "
            "other and disjoint files ever overlap."
        ),
    ] = 1,
    verify_steps: Annotated[
        bool, typer.Option(help="Run each step's verify command after it finishes.")
    ] = True,
    escalate: Annotated[
        bool,
        typer.Option(
            help="Retry a failed step, telling the next attempt what failed and what the "
            "last one already tried, and raising reasoning if it is not already at the "
            "top (it is, by default). Off disables the retry itself."
        ),
    ] = True,
    skills: Annotated[
        bool,
        typer.Option(
            help="Inject skill reference material into prompts (bundled ones, plus any of "
            "your own — see `harness skills`)."
        ),
    ] = _prefs.get("skills_enabled", True),
    skill_triggers: Annotated[
        str,
        typer.Option(
            "--skill-triggers",
            help="What an executor step's skill triggers match: 'workspace' (every root file "
            "name and the verify command) or 'step' (the step's own files and words). §117.",
        ),
    ] = "step",
    skills_dir: Annotated[
        Path | None,
        typer.Option(
            help="Use only this directory of skill .md files, instead of the bundled + "
            "your-own merge."
        ),
    ] = None,
    ask: Annotated[
        bool,
        typer.Option(
            help="Let the planner and executor ask you a clarifying question when genuinely "
            "unsure, instead of only guessing."
        ),
    ] = _prefs.get("ask_enabled", True),
    narrate: Annotated[
        bool,
        typer.Option(
            "--narrate/--no-narrate",
            help="A short line before each write, command, or commit "
            "so you can follow along step by step, not just at the start and end of a "
            "whole plan step.",
        ),
    ] = _prefs.get("narrate", True),
    quick_answers: Annotated[
        bool,
        typer.Option(
            "--quick-answers/--no-quick-answers",
            help="For a goal that's really just a "
            "question, answer it directly and skip planning (and its narration) "
            "entirely. One cheap classification call up front decides which; falls "
            "through to the normal pipeline for anything needing real work.",
        ),
    ] = _prefs.get("quick_answers", True),
    clarify: Annotated[
        bool,
        typer.Option(
            "--clarify/--no-clarify",
            help="Before planning, decide whether the goal is "
            "specific enough to build from, and ask up to three questions if it isn't. "
            "One cheap call; answers go into the planner's prompt. Needs --ask.",
        ),
    ] = _prefs.get("clarify", True),
    review_plan: Annotated[
        bool,
        typer.Option(
            "--review-plan/--no-review-plan",
            help="Show the plan and wait before executing "
            "it: Enter to run, 'e' to send it back with feedback, 'q' to stop. Catching a "
            "misunderstanding here costs one planning call instead of a whole run. Only "
            "ever prompts with a real terminal attached.",
        ),
    ] = _prefs.get("review_plan", False),
    memory: Annotated[
        bool,
        typer.Option(
            "--memory/--no-memory",
            help="Read and write <workspace>/.harness/memory.md, so a "
            "run knows what earlier runs on this workspace did and what they left "
            'broken. Without it, a follow-up goal like "why did that test fail?" has '
            "nothing to refer to.",
        ),
    ] = _prefs.get("memory", True),
    review: Annotated[
        bool,
        typer.Option(
            "--review/--no-review",
            help="Run the review phase (a model reads the finished work and fix rounds act "
            "on its findings). Off by default: it is about a fifth of the cost and time, and "
            "no suite showed a loss without it (DESIGN-LOG §124.10). The final check still "
            "runs either way.",
        ),
    ] = False,
    fix_existing_failures: Annotated[
        bool,
        typer.Option(
            "--fix-existing-failures/--no-fix-existing-failures",
            help="Treat tests that were already failing when the run started as work to "
            "do, not as somebody else's problem. On by default — \"the tests are "
            'failing, fix them" is most of why you would point this at a broken '
            "repository. Turn it off in a workspace that merely happens to contain "
            "unrelated red tests.",
        ),
    ] = _prefs.get("fix_existing_failures", True),
    plan_auditor: Annotated[
        str,
        typer.Option(
            "--plan-auditor",
            help="A second model that reviews and may patch the plan once, with no tools "
            "(agent/phases/plan_audit.py). Empty: no audit.",
        ),
    ] = "",
    provider_sort: Annotated[
        str,
        typer.Option(
            "--provider-sort",
            help="OpenRouter provider preference: latency, throughput or price. Matters for "
            "a model served by many providers (GLM Flash: 0.6s to 16s median first token).",
        ),
    ] = "",
    mercury_provider: Annotated[
        str,
        typer.Option(
            "--mercury-provider",
            help="Where Mercury calls go: openrouter (default), inception (Inception's own "
            "API; Keychain item InceptionLabs) or venice (Keychain item Venice). Other "
            "models always go to OpenRouter. `harness setup` persists it.",
        ),
    ] = _prefs.get("mercury_provider", "openrouter"),
    mercury_temperature: Annotated[
        float | None,
        typer.Option(
            "--mercury-temperature",
            help="Mercury's temperature for every call, 0.5-1.0. Default: none sent, "
            "the model's own 1.0 (what every earlier run actually used).",
        ),
    ] = None,
    issue_symbols: Annotated[
        bool,
        typer.Option(
            "--issue-symbols/--no-issue-symbols",
            help="Tell the planner where the goal's own identifiers are defined, from a search of "
            "the workspace (agent/issue_symbols.py, DESIGN-LOG §128). Off until measured.",
        ),
    ] = False,
    direct: Annotated[
        bool,
        typer.Option(
            "--direct/--no-direct",
            help="Skip the planner: the executor gets one step for the whole goal and finds the "
            "files itself; the final check still runs (DESIGN-LOG §129). Off until measured.",
        ),
    ] = False,
    executor_tail: Annotated[
        bool,
        typer.Option(
            "--executor-tail/--no-executor-tail",
            help="Send the step's goal, turns left and key rules after the last message on "
            "every executor call (agent/steptail.py, DESIGN-LOG §82). On since §106.2.",
        ),
    ] = True,
    tail_items: Annotated[
        list[str] | None,
        typer.Option(
            "--tail-item",
            help="With --executor-tail, a state section to add: errors or ledger. Repeatable. "
            "api, inspect and key still exist but were measured and not adopted "
            "(DESIGN-LOG §84, §87.2).",
        ),
    ] = None,
    planner_tail: Annotated[
        bool, typer.Option("--planner-tail/--no-planner-tail", help="The planner's own tail.")
    ] = False,
    plan_key_facts: Annotated[
        bool,
        typer.Option(
            "--plan-key-facts/--no-plan-key-facts",
            help="Ask the planner for the one fact each step must not get wrong.",
        ),
    ] = False,
    findings_fix: Annotated[
        list[str] | None,
        typer.Option(
            "--findings-fix",
            help="With --research-first, how to get convention decisions into findings: "
            "last (the request ends the planner's first message), required, or hold "
            "(one hold when findings has no decision). Repeatable. DESIGN-LOG §87. "
            "hold is not recommended: its §87.2 gain did not replicate (§88.6).",
        ),
    ] = None,
    clarify_context: Annotated[
        bool,
        typer.Option(
            "--clarify-context/--no-clarify-context",
            help="Show the clarify call the code's symbols, README and test setup, and ask only "
            "what the code cannot answer. On by default (DESIGN-LOG §92, §101).",
        ),
    ] = True,
    resolve_followups: Annotated[
        bool,
        typer.Option(
            "--resolve-followups/--no-resolve-followups",
            help="Tie a short follow-up goal ('fix those issues', 'try again') to the last run, "
            "confirmed with one click, instead of asking what it means. DESIGN-LOG §92.3.",
        ),
    ] = True,
    chat_checks: Annotated[
        bool,
        typer.Option(
            "--chat-checks/--no-chat-checks",
            help="With a person present: answer questions they ask back in clarify's answers, "
            "and confirm the plan's assumptions before it runs. DESIGN-LOG §111.",
        ),
    ] = True,
    last_word_gates: Annotated[
        str,
        typer.Option(
            "--last-word-gates",
            help="What the planner's gates do with a plan forced out by the last word: off "
            "(skip them, as before), turn (a hold gets one more forced turn), notes (the "
            "plan goes through, the hold's message becomes a finding). DESIGN-LOG §88.3.",
        ),
    ] = "off",
    mercury_max_tokens: Annotated[
        int | None,
        typer.Option(
            "--mercury-max-tokens",
            help="Output-token cap for Mercury calls. Default 16000 (DESIGN-LOG §95.1); "
            "0 for none.",
        ),
    ] = None,
    mercury_content_stall: Annotated[
        float | None,
        typer.Option(
            "--mercury-content-stall",
            help="Seconds of silence after a Mercury stream's first delta before the attempt "
            "is cut and retried. Default 30 (DESIGN-LOG §95.1); 0 turns it off.",
        ),
    ] = None,
    web_deny: Annotated[
        list[str] | None,
        typer.Option(
            "--web-deny",
            help="A URL substring the web tools refuse (fetch_url) or drop from results "
            "(web_search), matched anywhere in the URL, even wrapped in another one. "
            "Repeatable. For benchmark runs: the benchmark's own sources.",
        ),
    ] = None,
    research_first: Annotated[
        bool,
        typer.Option(
            "--research-first/--no-research-first",
            help="Before planning, look things up: the installed libraries' API "
            "(inspect_module) first, then skills, then the web.",
        ),
    ] = False,
    numeric_check: Annotated[
        bool,
        typer.Option(
            "--numeric-check/--no-numeric-check",
            help="Before a step finishes, run its code on a case whose answer is known "
            "independently (a limit, a symmetry, a conservation law).",
        ),
    ] = False,
    check_after_edit: Annotated[
        bool,
        typer.Option(
            "--check-after-edit/--no-check-after-edit",
            help="After a turn that edits files, run the step's tests once and hand the result "
            "back with the edits, instead of the model spending a call to ask. On by default "
            "(DESIGN-LOG §99: model time -31%, cost -40%).",
        ),
    ] = True,
    combined_fixes: Annotated[
        bool,
        typer.Option(
            "--combined-fixes/--no-combined-fixes",
            help="Fix all of a round's findings in one pass (shared reading, edits checked "
            "together) instead of one pass per finding. On by default (DESIGN-LOG §101).",
        ),
    ] = True,
    stop_after_idle_fix_round: Annotated[
        bool,
        typer.Option(
            "--stop-after-idle-fix-round/--no-stop-after-idle-fix-round",
            help="Do not start a final-check fix round after one that changed no file. Off by "
            "default until DESIGN-LOG §124's scout decides it.",
        ),
    ] = False,
    types_after_edit: Annotated[
        bool,
        typer.Option(
            "--types-after-edit/--no-types-after-edit",
            help="Report type errors the run introduced in the check after each edit. On by "
            "default (DESIGN-LOG §105.2).",
        ),
    ] = True,
    web_research: Annotated[
        bool,
        typer.Option(
            "--web-research/--no-web-research",
            help="Aids for finding facts on the web: a research skill and a nudge to open a "
            "result before searching again. Off until DESIGN-LOG §116 decides it.",
        ),
    ] = False,
    read_assist: Annotated[
        bool,
        typer.Option(
            "--read-assist/--no-read-assist",
            help="read_file widens narrow windows to the whole function and answers identical "
            "re-reads in one line. On by default (DESIGN-LOG §105.2).",
        ),
    ] = True,
    edit_blocks: Annotated[
        bool,
        typer.Option(
            "--edit-blocks/--no-edit-blocks",
            help="Offer edit_blocks (SEARCH/REPLACE text) instead of edit_files. Off: not adopted "
            "in DESIGN-LOG §105.2.",
        ),
    ] = False,
    review_presupplied_checks: Annotated[
        bool,
        typer.Option(
            "--review-presupplied-checks/--no-review-presupplied-checks",
            help="Hand the reviewer ty's report with the suite's, and drop run_tests from its "
            "kit. On by default (DESIGN-LOG §108.2).",
        ),
    ] = True,
    bugfix_evidence: Annotated[
        bool,
        typer.Option(
            "--bugfix-evidence/--no-bugfix-evidence",
            help="Before a step that changed existing source finishes, check once that the "
            "plan's reproduction now passes and the run's new tests fail on the old code. Off: "
            "dropped in DESIGN-LOG §120.2.",
        ),
    ] = False,
    scaffold_new_projects: Annotated[
        bool,
        typer.Option(
            "--scaffold/--no-scaffold",
            help="In an empty workspace, set the project up with uv init and Ruff formatting "
            "before planning (DESIGN-LOG §104).",
        ),
    ] = True,
    answer_command: Annotated[
        str,
        typer.Option(
            "--answer-command",
            help="Answer the run's clarifying questions with this command instead of a "
            'person: it gets {"question", "options", "recommended"} as JSON on stdin and '
            "prints the answer.",
        ),
    ] = "",
    test_command: Annotated[
        str,
        typer.Option(
            "--test-command",
            help="The project's own test command, for a project pytest cannot run, e.g. "
            "'python tests/runtests.py --settings=test_sqlite --parallel 1 {targets}'. "
            "{targets} marks where test targets go (appended when absent). Split like a "
            "shell command, so quote a path with spaces. Its unittest-style output is "
            "read like pytest's. Default: pytest.",
        ),
    ] = "",
    test_label_root: Annotated[
        str,
        typer.Option(
            "--test-label-root",
            help="With --test-command: pass test files under this directory as dotted "
            "module labels relative to it (tests/pagination/x.py -> pagination.x).",
        ),
    ] = "",
    draft_style: Annotated[
        str,
        typer.Option(
            "--draft-style",
            help="How the planner drafts before committing to a plan: 'terse' (chain-of-draft, "
            "brief notes; the default) or 'prose' (full sentences and paragraphs, with code "
            "sketches allowed in the draft only). The draft never reaches the executor, so "
            "this only changes the steps written after it. Experimental.",
        ),
    ] = "terse",
    plan_probe: Annotated[
        str,
        typer.Option(
            "--plan-probe",
            help="How the planner checks a reported problem before planning its fix. "
            "'repro' (the default): the plan carries a snippet reproducing the problem, run "
            "on a throwaway copy before the plan is accepted, and the plan is sent back once "
            "if the problem does not show. 'require': the planner may run code and its first "
            "plan is held until it has, which catches more wrong reports at about two extra "
            "model calls a run. 'offer': the tool without the hold. 'off': none. "
            "See DESIGN-LOG §54.",
        ),
    ] = "repro",
    test_audit: Annotated[
        bool,
        typer.Option(
            "--test-audit/--no-test-audit",
            help="After a green run, check in a scratch copy whether the tests can see the "
            "change: do the tests this run added pass on the pre-run code, would small "
            "changes to the edited functions go unnoticed, did an edited test file lose "
            "catching power. Report-only; ~0.4s measured.",
        ),
    ] = True,
    fix_weak_tests: Annotated[
        bool,
        typer.Option(
            "--fix-weak-tests/--no-fix-weak-tests",
            help="When the test audit finds tests that cannot see the change, spend one fix "
            "round on them, showing the model what the run changed. Cut tests-that-miss-"
            "the-change from 11/88 to 4/88 runs and honesty moved the same way, but on a "
            "confirmation sweep it cost ~4 more model calls a run and the honesty gain "
            "was not established, so it is opt-in.",
        ),
    ] = False,
    verify: Annotated[
        str | None,
        typer.Option(
            "--verify",
            help="A shell command that proves the goal's own claim, checked "
            "at the final gate alongside every step's own verify — not instead of "
            "it. For a goal with a specific, checkable claim ('players can't be hit "
            "twice within one second of taking damage'), this ties the gate to that "
            "exact claim rather than trusting a step's own narrower test to have "
            "covered it.",
        ),
    ] = None,
    require_tests: Annotated[
        bool,
        typer.Option(
            "--require-tests/--no-require-tests",
            help="Fail a run that writes behaviour into a workspace with no tests at all. "
            "On by default: with nothing to run, the final check has nothing "
            "deterministic left to check, and a passing lint stands in for working "
            "code.",
        ),
    ] = _prefs.get("require_tests", True),
    check_coverage: Annotated[
        bool,
        typer.Option(
            "--check-coverage/--no-check-coverage",
            help="Check that the tests actually execute the lines the run changed, not just "
            "that a green suite exists. The suite runs once either way, under coverage; "
            "reading the data costs one extra fast subprocess.",
        ),
    ] = _prefs.get("check_coverage", True),
    final_check: Annotated[
        bool,
        typer.Option(
            "--final-check/--no-final-check",
            help="After review, re-run every step's verify "
            "command and the test suite against the finished code, and fix what fails. "
            "Makes no model calls unless something is actually broken.",
        ),
    ] = _prefs.get("final_check", True),
    stream: Annotated[
        bool, typer.Option(help="Stream responses. Needed for time-to-first-token measurement.")
    ] = _prefs.get("stream", True),
    stats: Annotated[
        bool,
        typer.Option(
            "--stats/--no-stats",
            help="Show the calls/cost/tok-per-s breakdown at the end of "
            "a run. Off by default — ask for it when you want it, rather than reading past "
            "it every time.",
        ),
    ] = _prefs.get("show_stats", False),
    animate: Annotated[
        bool,
        typer.Option(
            "--animate/--no-animate",
            help="A small looping animation while a model call is "
            "in flight and nothing else is printing, so the terminal doesn't look frozen. "
            "Automatically does nothing when output isn't a real terminal.",
        ),
    ] = _prefs.get("animate", True),
    spinner_style: Annotated[
        str,
        typer.Option(
            help="'duck' (three ducks paddling across rippled water, bills opening and "
            "closing) or 'dots' (a plain single-line spinner)."
        ),
    ] = _prefs.get("spinner_style", "duck"),
    cache: Annotated[
        bool,
        typer.Option(
            help="Send an explicit cache breakpoint on the stable prompt prefix. "
            "Only Anthropic models accept one, so this has no effect on the "
            "all-Mercury defaults; automatic provider-side caching is separate "
            "and unaffected."
        ),
    ] = _prefs.get("cache_prompts", True),
    autolint: Annotated[
        bool, typer.Option(help="Run Ruff after each write. Syntax checks always run.")
    ] = _prefs.get("autolint", True),
    test_parallel: Annotated[
        bool,
        typer.Option(
            "--test-parallel/--no-test-parallel",
            help="Run the whole-suite baseline and gate checks under pytest-xdist "
            "(installed ephemerally). Off by default: it needs test isolation "
            "the suite may not have, and helps most on a large suite that is "
            "not dominated by one or two slow tests.",
        ),
    ] = _prefs.get("test_parallel", False),
    shell_budget: Annotated[
        float, typer.Option(help="Seconds of subprocess time one step may use.")
    ] = 90.0,
    max_workspace_mb: Annotated[
        float, typer.Option(help="Stop the run if the workspace grows past this.")
    ] = 500.0,
    max_step_errors: Annotated[
        int, typer.Option(help="Tool errors in one step before that step is abandoned.")
    ] = 12,
    allow_nested: Annotated[
        bool,
        typer.Option(
            help="Permit a workspace inside another Python project. Unsafe: the agent's "
            "subprocesses can modify the enclosing project's environment."
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "-y", "--yes", help="Auto-approve shell commands. Point this at a throwaway workspace."
        ),
    ] = _prefs.get("auto_approve", False),
    auto_approve_python: Annotated[
        bool,
        typer.Option(
            "--auto-approve-python",
            help="Auto-approve run_python calls specifically, "
            "without approving shell commands too. Every run_python call already "
            "happens after a plan exists (it's only offered during execute/fix), so "
            "this is about not re-approving the routine verification snippets a step "
            "runs on itself — installs, git, and everything else still gate "
            "normally.",
        ),
    ] = _prefs.get("auto_approve_python", False),
    approval_mode: Annotated[
        ApprovalMode | None,
        typer.Option(
            "--approval-mode",
            help="A single choice covering both approval flags at once: "
            "'manual' (prompt for everything — the default), 'auto-python' (same as "
            "--auto-approve-python), or 'auto-plan' (same as -y/--yes: everything a "
            "step needs runs without asking once a plan exists). Overrides -y and "
            "--auto-approve-python when given; the TUI exposes the same three choices "
            "as a dropdown. Persistable: `harness config set approval_mode auto-python`.",
        ),
    ] = _APPROVAL_MODE_PREF,
    trace_dir: Annotated[str, typer.Option()] = "traces",
    failure_log_dir: Annotated[
        str,
        typer.Option(
            help="A readable copy of any run that fails, is stopped, or times out is saved "
            "here — separate from traces/, so a bad run is something you can just open. "
            "Pass '' to disable."
        ),
    ] = _prefs.get("failure_log_dir", "failed-chats"),
    out: Annotated[Path | None, typer.Option(help="Write the run summary JSON here.")] = None,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui", help="Watch the run live in a terminal dashboard instead of scrolling text."
        ),
    ] = False,
    keep_going: Annotated[
        bool,
        typer.Option(
            "--continue/--no-continue",
            help="After a successful run, show a suggested next "
            "step and offer to keep going with it or a new goal. Only ever prompts with a "
            "real terminal attached — a script or pipe gets exactly today's one-shot "
            "behavior regardless of this flag.",
        ),
    ] = True,
    goal_loop: Annotated[
        bool,
        typer.Option(
            "--goal-loop",
            help="After each attempt, ask a cheap judge model whether the goal "
            "is actually done — not just that the attempt finished cleanly — and, if not, "
            "run again with a narrower instruction. Stops on 'done', on 'blocked' (another "
            "attempt would not help), or after --max-goal-iterations. Each attempt also "
            "self-audits its own trace for harness-level defects into "
            "<workspace>/.harness/feedback.md. Not compatible with --tui yet.",
        ),
    ] = False,
    max_goal_iterations: Annotated[
        int,
        typer.Option(
            help="Attempts --goal-loop will make before giving up and reporting the last one."
        ),
    ] = 3,
) -> None:
    _reject_conflicting_flags(plan_path, stdin_plan, goal_loop, tui, draft_style, plan_probe)
    goal_text = _resolve_goal(goal, plan_path, stdin_plan, tui)

    if approval_mode is not None:
        yes = approval_mode == ApprovalMode.auto_plan
        auto_approve_python = approval_mode in (ApprovalMode.auto_plan, ApprovalMode.auto_python)

    settings = Settings(
        workspace=workspace,
        goal=goal_text,
        executor_model=executor,
        planner_model=executor if all_mercury else planner,
        reviewer_model=executor if all_mercury else (reviewer or planner),
        auto_approve=yes,
        auto_approve_python=auto_approve_python,
        trace_dir=trace_dir,
        failure_log_dir=failure_log_dir or None,
        autolint=autolint,
        test_parallel=test_parallel,
        stream=stream,
        skills_enabled=skills,
        skill_triggers=skill_triggers,
        skills_dir=str(skills_dir) if skills_dir else None,
        ask_enabled=ask,
        clarify=clarify,
        review_plan=review_plan,
        memory=memory,
        review=review,
        final_check=final_check,
        fix_existing_failures=fix_existing_failures,
        research_first=research_first,
        numeric_check=numeric_check,
        check_after_edit=check_after_edit,
        combined_fixes=combined_fixes,
        stop_after_idle_fix_round=stop_after_idle_fix_round,
        scaffold_new_projects=scaffold_new_projects,
        types_after_edit=types_after_edit,
        read_assist=read_assist,
        web_research=web_research,
        edit_blocks=edit_blocks,
        review_presupplied_checks=review_presupplied_checks,
        bugfix_evidence=bugfix_evidence,
        plan_auditor_model=plan_auditor or None,
        provider_sort=provider_sort,
        web_deny=tuple(web_deny or ()),
        mercury_provider=mercury_provider,
        mercury_temperature=mercury_temperature,
        executor_tail_reminder=executor_tail,
        plan_issue_symbols=issue_symbols,
        direct=direct,
        executor_tail_items=tuple(tail_items or ()),
        planner_tail=planner_tail,
        plan_key_facts=plan_key_facts,
        plan_findings_last="last" in (findings_fix or ()),
        plan_findings_required="required" in (findings_fix or ()),
        plan_findings_hold="hold" in (findings_fix or ()),
        last_word_gates=last_word_gates,
        clarify_context=clarify_context,
        resolve_followups=resolve_followups,
        chat_checks=chat_checks,
        mercury_max_tokens=_unless_given(mercury_max_tokens, "mercury_max_tokens"),
        mercury_content_stall_s=_unless_given(mercury_content_stall, "mercury_content_stall_s"),
        answer_command=answer_command,
        test_command=test_command,
        test_label_root=test_label_root,
        test_audit=test_audit,
        fix_weak_tests=fix_weak_tests,
        draft_style=draft_style,
        plan_probe=plan_probe,
        goal_verify=verify,
        require_tests=require_tests,
        check_coverage=check_coverage,
        narrate=narrate,
        quick_answers=quick_answers,
        show_stats=False if tui else stats,
        step_concurrency=step_concurrency,
        verify_steps=verify_steps,
        escalate_on_failure=escalate,
        cache_prompts=cache,
        allow_nested=allow_nested,
        quiet=tui,
        limits=Limits(
            step_shell_budget_s=shell_budget,
            max_workspace_mb=max_workspace_mb,
            max_step_errors=max_step_errors,
        ),
        budget=Budget(
            max_steps=max_steps,
            max_usd=max_usd,
            step_prompt_token_budget=step_budget,
            review_prompt_token_budget=review_budget,
        ),
    )
    if executor_reasoning:
        settings.executor_reasoning = executor_reasoning.value
    if planner_reasoning:
        settings.planner_reasoning = planner_reasoning.value
    settings.suggest_next_step = _anyone_reads_the_suggestion(tui, goal_loop, memory)

    effective_plan_path = _plan_argument(plan_path, stdin_plan)

    # A run left going on a laptop idle-slept mid-call five times in one
    # session; each attempt then sat 219-536s past its bounds (DESIGN-LOG §71).
    awake.keep_awake()
    if tui:
        _require_extra("textual", "tui", "The dashboard")
        from .tui import watch_run

        summary = watch_run(settings, plan_path=effective_plan_path, stop_after=stop_after.value)
    else:
        # loop.run() already echoes goal/workspace/models/phases via
        # Trace.say() while settings.quiet is False (only --tui sets it) —
        # printing our own banner here duplicated every line of it.
        spinner = Spinner(style=spinner_style) if animate else None
        gate = _plan_gate(spinner) if (review_plan and sys.stdin.isatty()) else None
        try:
            if goal_loop:
                from .goal import run_until_done

                summary = run_until_done(
                    settings,
                    max_goal_iterations=max_goal_iterations,
                    plan_path=effective_plan_path,
                    stop_after=stop_after.value,
                    on_call_start=spinner.start if spinner else None,
                    on_call_end=spinner.stop if spinner else None,
                    on_plan=gate,
                )
            else:
                summary = run_agent(
                    settings,
                    plan_path=effective_plan_path,
                    stop_after=stop_after.value,
                    on_call_start=spinner.start if spinner else None,
                    on_call_end=spinner.stop if spinner else None,
                    on_plan=gate,
                )
        except KeyboardInterrupt:
            raise typer.Exit(130) from None
        except Exception as exc:
            err_console.print(f"[red]error:[/] {type(exc).__name__}: {exc}")
            raise typer.Exit(1) from None
        if stats:
            _print_summary(summary)
        _print_session_summary(summary, goal_text)
        _print_outcome(summary)
        if keep_going and summary.get("ok"):
            summary = _interactive_continue(
                settings, summary, stop_after.value, spinner=spinner, show_stats=stats, on_plan=gate
            )

    if out:
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        console.print(f"summary written to [cyan]{out}[/]")

    # Non-zero for work that does not pass its own checks, not only for a
    # harness failure: a run whose steps failed verification used to exit 0,
    # which made it indistinguishable from a clean one to anything scripting
    # the harness.
    if summary.get("error") or not summary.get("ok", True):
        raise typer.Exit(1)


def _plan_gate(spinner: Spinner | None):
    """Show the plan and let the user accept, redirect, or abort it.

    Returned as a callback for loop.run's on_plan: "" runs the plan, "abort"
    stops, anything else is feedback to re-plan with.
    """

    def gate(plan) -> str:
        # The spinner owns the cursor while a call is in flight; a prompt
        # drawn over a half-redrawn animation frame is unreadable.
        if spinner:
            spinner.stop()
        console.print()
        console.print(f"[bold]plan[/] — {len(plan.steps)} step(s)")
        for i, step in enumerate(plan.steps, 1):
            files = f"  [dim]{', '.join(step.files)}[/]" if step.files else ""
            console.print(f"  {i}. [cyan]{step.title}[/]{files}")
            if step.intent:
                console.print(f"     [dim]{step.intent}[/]")
            if step.verify:
                console.print(f"     [dim]verify:[/] {step.verify}")
        answer = typer.prompt(
            "\n[Enter] execute this plan, 'e' to send it back with feedback, 'q' to stop",
            default="",
            show_default=False,
        ).strip()
        if answer.lower() in ("q", "quit", "exit"):
            return "abort"
        if answer.lower() in ("e", "edit", "r", "replan"):
            feedback = typer.prompt(
                "what should the plan do differently?", default="", show_default=False
            ).strip()
            # An empty answer to "what's wrong with it" is not a rejection.
            return feedback or ""
        # Anything else typed is itself the feedback, so the common case of
        # just typing the correction works without the extra 'e' step.
        return answer

    return gate


def _print_session_summary(summary: dict, goal: str) -> None:
    """The end-of-run account: a paragraph, then what was built/failed/next.

    `summary` has no top-level "goal" key (it lives nested at
    `summary["plan"]["goal"]`, when there is a plan at all) -- prefer that,
    since it is the goal actually executed for whichever run produced this
    summary (the refined goal of a --goal-loop attempt, for instance, not
    the original one this call's own `goal` argument still names). Falls
    back to `goal` when there is no plan, e.g. a quick_answer run.
    """
    if not str(summary.get("session_summary") or "").strip():
        return
    plan = summary.get("plan") or {}
    effective_goal = plan.get("goal") or goal
    console.print()
    console.print(
        Panel(
            summary_module.render_rich(effective_goal, summary),
            title="summary",
            border_style="cyan",
            title_align="left",
            padding=(1, 2),
        )
    )


def _print_outcome(summary: dict) -> None:
    """Say plainly whether the work passed its own checks.

    The end of a run used to be a cost table and nothing else, so a run where
    half the steps failed their verify command looked exactly like one where
    none did.
    """
    steps = summary.get("steps") or {}
    check = summary.get("final_check") or {}
    if summary.get("error"):
        return  # already reported through the error path
    problems: list[str] = []
    if steps.get("failed"):
        problems.append(
            f"{len(steps['failed'])} step(s) failed verification: " + ", ".join(steps["failed"])
        )
    if steps.get("abandoned"):
        problems.append(
            f"{len(steps['abandoned'])} step(s) abandoned: " + ", ".join(steps["abandoned"])
        )
    if check and not check.get("ok", True):
        if check.get("tests_ran") and not check.get("tests_passed"):
            problems.append(f"tests failing: {check.get('tests_summary', '')}")
        for failure in check.get("verify_failures") or []:
            problems.append(f"verify fails for [{failure.get('step')}]: {failure.get('command')}")
    if not problems:
        return
    console.print()
    console.print("[bold yellow]this run did not pass its own checks:[/]")
    for problem in problems:
        console.print(f"  [yellow]•[/] {problem}")


def _anyone_reads_the_suggestion(tui: bool, goal_loop: bool, memory: bool) -> bool:
    """Whether the end-of-run next-step suggestion has a reader: the TUI pre-fills it, the goal
    loop's judge and the memory notes read it, and a person at a terminal sees it. A script (the
    SciCode, django and GAIA drivers log stdout to a file and pass --no-memory) has none: there
    it was 2.9% of SciCode spend and 143 of 180 were withheld anyway (DESIGN-LOG §122). bench
    turns it off the same way (bench/runner.py)."""
    return tui or goal_loop or memory or _at_a_terminal()


def _at_a_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _interactive_continue(
    settings: Settings,
    summary: dict,
    stop_after: str,
    spinner: Spinner | None = None,
    show_stats: bool = False,
    on_plan=None,
) -> dict:
    """After a successful run: show the suggestion, offer to keep going.

    Never prompts without a real terminal — a script piping output, or `-y`
    run non-interactively, gets exactly the one-shot behavior it had before
    this existed. Each iteration reuses the same workspace, so the agent is
    building on what it already did, not starting over.
    """
    if not sys.stdin.isatty():
        return summary
    while True:
        suggestion = str(summary.get("suggested_next_step") or "").strip()
        console.print()
        if suggestion:
            console.print(f"[bold]suggested next step:[/] {suggestion}")
            answer = typer.prompt(
                "[Enter] to do that, type a different goal, or 'q' to stop",
                default="",
                show_default=False,
            )
        else:
            answer = typer.prompt("type a new goal, or 'q' to stop", default="", show_default=False)

        stripped = answer.strip()
        if stripped.lower() in ("q", "quit", "exit"):
            return summary
        next_goal = stripped or suggestion
        if not next_goal:
            return summary  # nothing typed and nothing suggested

        settings.goal = next_goal
        console.print(f"\n[bold]continuing:[/] {next_goal}\n")
        try:
            summary = run_agent(
                settings,
                plan_path=None,
                stop_after=stop_after,
                on_call_start=spinner.start if spinner else None,
                on_call_end=spinner.stop if spinner else None,
                on_plan=on_plan,
            )
        except KeyboardInterrupt:
            return summary
        except Exception as exc:
            err_console.print(f"[red]error:[/] {type(exc).__name__}: {exc}")
            return summary
        if show_stats:
            _print_summary(summary)
        _print_session_summary(summary, next_goal)
        _print_outcome(summary)
        if summary.get("error"):
            return summary


def _print_summary(summary: dict) -> None:
    totals = summary.get("totals") or {}
    by_model = totals.get("by_model") or {}
    if not by_model:
        return
    table = Table(title="totals", show_edge=False, header_style="bold")
    table.add_column("model")
    table.add_column("calls", justify="right")
    table.add_column("out tok", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("cost", justify="right")
    for model, slot in by_model.items():
        # model_tok_per_s rather than the same division written out again:
        # this table used to recompute it by hand and had already drifted
        # from the figure every other view of these numbers shows.
        table.add_row(
            model,
            str(slot.get("calls", 0)),
            f"{slot.get('completion_tokens', 0):,}",
            f"{model_tok_per_s(slot):.0f}",
            f"${slot.get('cost_usd', 0.0):.4f}",
        )
    console.print(table)
    console.print(
        f"[bold]total:[/] ${totals.get('cost_usd', 0.0):.4f}  "
        f"[dim]trace:[/] {summary.get('trace_path', '')}"
    )


# -- `harness config ...` --------------------------------------------------


def _require_extra(module: str, extra: str, feature: str) -> None:
    """Exit with the install command when `feature` needs an optional extra that is missing,
    rather than a ModuleNotFoundError from deep inside it. Plain text, unwrapped: Rich would
    read `[tui]` as a markup tag and drop it, and a wrapped URL cannot be pasted."""
    if importlib.util.find_spec(module) is None:
        spec = f"mercury-harness[{extra}] @ git+https://github.com/whtowbin/mercury-harness"
        err_console.print(
            f"error: {feature} needs the {extra} extra. Install it with\n"
            f"  uv tool install '{spec}'\n"
            f"or, in a clone, `uv sync --extra {extra}`.",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )
        raise typer.Exit(2)


def _setup_provider(name: str | None):
    from .setup import KEY_PROVIDERS

    if name:
        if name not in KEY_PROVIDERS:
            err_console.print(f"[red]error:[/] --provider is one of {', '.join(KEY_PROVIDERS)}")
            raise typer.Exit(2)
        return KEY_PROVIDERS[name]
    choices = list(KEY_PROVIDERS.values())
    console.print("Which API should the harness use?")
    for i, p in enumerate(choices, 1):
        console.print(f"  {i}) {p.label}")
    picked = typer.prompt("Choice", default="1").strip()
    if picked not in {str(i) for i in range(1, len(choices) + 1)}:
        err_console.print(f"[red]error:[/] choose 1 to {len(choices)}")
        raise typer.Exit(2)
    return choices[int(picked) - 1]


def _setup_key(provider, check: bool) -> str:
    """Ask for the key until it has a key's shape and, when checked, is not rejected."""
    from .setup import KEY_SHAPE, check_key

    for _ in range(3):
        key = typer.prompt(
            f"Paste your {provider.name} API key (hidden; get one at {provider.signup_url})",
            hide_input=True,
        ).strip()
        if not KEY_SHAPE.fullmatch(key):
            err_console.print("[red]That does not look like an API key.[/] Try again.")
            continue
        if provider.key_hint and not key.startswith(provider.key_hint):
            err_console.print(
                f"[yellow]note:[/] {provider.name} keys usually start {provider.key_hint}"
            )
        if not check:
            return key
        if provider.name == "inception":
            console.print("[dim]Checking with one 1-token request (under $0.00001)...[/]")
        result = check_key(provider, key)
        if result.ok:
            console.print(f"[green]✓[/] {result.detail}")
            return key
        if result.ok is False:
            err_console.print(f"[red]✗[/] {result.detail}. Try again.")
            continue
        err_console.print(f"[yellow]Could not confirm the key:[/] {result.detail}")
        if typer.confirm("Save it anyway?", default=True):
            return key
    err_console.print("[red]error:[/] no usable key after three tries")
    raise typer.Exit(1)


def _setup_target(store: str | None) -> str:
    """Where the key goes: "secure" (the secret store) or "file"; says so when it has to be the
    file and how to get a store."""
    from . import keystore
    from . import setup as key_setup

    has_store = bool(keystore.secret_store())
    target = store or ("secure" if has_store else "file")
    if target not in ("secure", "file"):
        err_console.print("[red]error:[/] --store is secure or file")
        raise typer.Exit(2)
    if target == "secure" and not has_store:
        err_console.print("[red]error:[/] no secret store on this machine; use --store file")
        if hint := key_setup.keyring_hint():
            err_console.print(hint, markup=False, soft_wrap=True)
        raise typer.Exit(2)
    if target == "file" and not store and (hint := key_setup.keyring_hint()):
        console.print("[dim]No secret store here, so the key goes to a file.[/]")
        console.print(hint, markup=False, soft_wrap=True)  # "[keyring]" is not markup
    return target


@app.command("setup")
def setup_command(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="openrouter or inception; asked when not given."),
    ] = None,
    store: Annotated[
        str | None,
        typer.Option(
            "--store",
            help="secure or file. secure: this machine's secret store (the macOS Keychain; on "
            "Linux or Windows the system keyring, with the keyring extra installed). file: "
            "credentials.env in the config folder, readable only by you. Default: secure "
            "where there is a store, else file.",
        ),
    ] = None,
    check: Annotated[
        bool, typer.Option("--check/--no-check", help="Check the key with the provider first.")
    ] = True,
) -> None:
    """Add your OpenRouter or Inception API key: asks for it, checks it, stores it."""
    from . import setup as key_setup

    chosen = _setup_provider(provider)
    target = _setup_target(store)
    where_now = key_setup.existing_key(chosen)
    if where_now and not typer.confirm(
        f"A {chosen.name} key is already set in {where_now}. Replace it?", default=False
    ):
        raise typer.Exit(0)
    key = _setup_key(chosen, check)
    try:
        place = key_setup.store_key(chosen, key, target)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        err_console.print(f"[red]error:[/] could not store the key: {exc}")
        raise typer.Exit(1) from None
    console.print(f"[green]✓[/] saved to {place}")
    if chosen.name == "inception":
        path = key_setup.use_provider_for_mercury("inception")
        console.print(
            f"Mercury calls now go to Inception (mercury_provider in {path}). A model other "
            "than Mercury still needs an OpenRouter key: `harness setup --provider openrouter`."
        )
    elif _prefs.get("mercury_provider", "openrouter") != "openrouter":
        console.print(
            f"[dim]Mercury calls still go to {_prefs.get('mercury_provider')}; "
            "`harness config unset mercury_provider` sends them to OpenRouter.[/]"
        )
    console.print('Next, in your project folder: [bold]harness run "what you want built"[/]')


config_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="View or persist default preferences, so you stop retyping the same flags.",
)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """What is persisted, and where."""
    prefs = userconfig.load()
    if not prefs.values:
        console.print(f"nothing persisted yet. [dim]{userconfig.config_path()}[/]")
        return
    table = Table(show_edge=False, header_style="bold")
    table.add_column("setting")
    table.add_column("value")
    for key, value in sorted(prefs.values.items()):
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"[dim]{userconfig.config_path()}[/]")


@config_app.command("set")
def config_set(
    key: Annotated[
        str, typer.Argument(help="One of: " + ", ".join(sorted(userconfig.PERSISTABLE_FIELDS)))
    ],
    value: Annotated[str, typer.Argument()],
) -> None:
    """Persist one preference. Applies from your next invocation on."""
    try:
        prefs = userconfig.set_value(key, value)
    except ValueError as exc:
        err_console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(2) from None
    console.print(f"{key} = {prefs.get(key)!r}  [dim]{userconfig.config_path()}[/]")


@config_app.command("unset")
def config_unset(key: Annotated[str, typer.Argument()]) -> None:
    """Remove one persisted preference, reverting it to the built-in default."""
    userconfig.unset_value(key)
    console.print(f"{key} unset")


@config_app.command("path")
def config_path_command() -> None:
    """Where preferences are stored, for editing by hand or backing up."""
    console.print(str(userconfig.config_path()))


@config_app.command("reset")
def config_reset(
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Skip confirmation.")] = False,
) -> None:
    """Delete every persisted preference."""
    if not yes and not typer.confirm(f"Delete {userconfig.config_path()}?"):
        raise typer.Exit(1)
    userconfig.reset()
    console.print("preferences reset")


# -- `harness skills ...` ----------------------------------------------

skills_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="List, add, or scaffold skills — reference material "
    "injected into the model's prompt for matching steps.",
)
app.add_typer(skills_app, name="skills")

_SKILL_TEMPLATE = """---
name: {name}
description: One line, shown when this skill is listed.
triggers: [{name}]
phases: [execute, fix]
---

# {name}

Write the guidance here. See the bundled skills (`harness skills list`) for the format
this harness expects, and README.md's "Skills" section for how selection and injection
work.
"""


@skills_app.command("list")
def skills_list() -> None:
    """Every skill in effect: bundled, plus your own."""
    from . import skills as skills_module

    bundled = {s.name for s in skills_module.load(skills_module.DEFAULT_SKILLS_DIR)}
    merged = skills_module.load_merged(userconfig.user_skills_dir())
    if not merged:
        console.print("no skills found")
        return
    table = Table(show_edge=False, header_style="bold")
    table.add_column("name")
    table.add_column("source")
    table.add_column("triggers")
    table.add_column("description")
    for skill in merged:
        source = (
            "yours"
            if skill.name not in bundled
            else (
                "yours (overrides bundled)"
                if skill.path and userconfig.user_skills_dir() in skill.path.parents
                else "bundled"
            )
        )
        table.add_row(
            skill.name, source, ", ".join(skill.triggers) or "(always)", skill.description
        )
    console.print(table)


@skills_app.command("add")
def skills_add(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A skill .md file to copy in.")
    ],
) -> None:
    """Copy a skill file into your own skills directory."""
    from . import skills as skills_module

    text = file.read_text(encoding="utf-8")
    parsed = skills_module.parse(text, file)
    if parsed is None:
        err_console.print(
            f"[red]error:[/] {file} has no valid frontmatter — see "
            "`harness skills new` for the expected shape."
        )
        raise typer.Exit(2)
    target_dir = userconfig.user_skills_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{parsed.name}.md"
    target.write_text(text, encoding="utf-8")
    console.print(f"added [bold]{parsed.name}[/] -> {target}")


@skills_app.command("new")
def skills_new(name: Annotated[str, typer.Argument(help="Skill name, e.g. 'fastapi'.")]) -> None:
    """Scaffold a new skill in your own skills directory and print its path."""
    target_dir = userconfig.user_skills_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.md"
    if target.exists():
        err_console.print(f"[red]error:[/] {target} already exists")
        raise typer.Exit(2)
    target.write_text(_SKILL_TEMPLATE.format(name=name), encoding="utf-8")
    console.print(f"created {target}")
    console.print("edit it, then `harness skills list` to confirm it is picked up.")


@skills_app.command("path")
def skills_path() -> None:
    """Where your own skills live."""
    console.print(str(userconfig.user_skills_dir()))


# -- `harness trace ...` -----------------------------------------------------
# Deterministic, model-free analysis over saved trace JSONL files — see
# agent/tracetools.py. Meant to be run after a session (or any time against a
# project's traces/ directory) rather than wired into every live run: the
# checks it surfaces (declared_vs_actual, new_files_not_in_plan) already run
# live during execution (agent/phases/execute.py) for a *future* run; this is
# for finding the same shapes in traces already on disk, and for the
# cross-trace patterns — the same defect recurring across separate runs —
# that only become visible once you're looking at more than one at a time.

trace_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic, model-free analysis over saved traces — "
    "run outcomes, undeclared/shadow files, recurring patterns.",
)
app.add_typer(trace_app, name="trace")

sessions_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deterministic, model-free analysis of Claude Code's own session transcripts — "
    "where the cost goes, what stays in context.",
)
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("report")
def sessions_report(
    directory: Annotated[
        Path | None,
        typer.Argument(
            file_okay=False,
            help="Transcript directory. Default: this working directory's, "
            "under ~/.claude/projects.",
        ),
    ] = None,
    last: Annotated[
        int | None,
        typer.Option("--last", help="Only the newest N sessions (default: all)."),
    ] = None,
) -> None:
    """Per-session summary, cost by component and context size, a compaction-cap
    estimate, what stays in context longest, and command/error habits."""
    from . import claude_sessions

    target = directory or claude_sessions.default_dir(Path.cwd())
    if not Path(target).is_dir():
        err_console.print(f"[yellow]no transcript directory at {target}[/]")
        raise typer.Exit(1)
    console.print(claude_sessions.report(target, last=last), markup=False, highlight=False)


def _collect_trace_reports(
    tracetools, outcomes: list[dict], trace_dir: Path, workspace: Path | None, failed_only: bool
) -> list[dict]:
    """One row per run: its outcome plus the two per-run issue lists."""
    reports = []
    for row in outcomes:
        if failed_only and row["ok"] is not False:
            continue
        events = tracetools.load_trace(trace_dir / row["trace"])
        unwritten = [d for d in tracetools.declared_vs_actual(events) if d["unwritten"]]
        shadows = [f for f in tracetools.new_files_not_in_plan(events, workspace) if f["shadows"]]
        reports.append({**row, "unwritten_declared_files": unwritten, "shadow_files": shadows})
    return reports


def _print_report_table(reports: list[dict]) -> None:
    table = Table(show_edge=False, header_style="bold")
    for column in ("trace", "ok", "cost", "calls", "issues"):
        table.add_column(column)
    for r in reports:
        issue_count = len(r["unwritten_declared_files"]) + len(r["shadow_files"])
        ok_style = "green" if r["ok"] else ("red" if r["ok"] is False else "dim")
        table.add_row(
            str(r["trace"]),
            f"[{ok_style}]{r['ok']}[/]",
            f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "-",
            str(r["calls"]) if r["calls"] is not None else "-",
            str(issue_count) if issue_count else "[dim]-[/]",
        )
    console.print(table)


def _print_report_details(reports: list[dict], workspace: Path | None) -> None:
    for r in reports:
        if r["shadow_files"]:
            console.print(f"\n[red]shadow files[/] in {r['trace']}:")
            for f in r["shadow_files"]:
                console.print(f"  {f['path']} shadows {', '.join(f['shadows'])}")
        if r["unwritten_declared_files"]:
            console.print(f"\n[yellow]unwritten declared writes[/] in {r['trace']}:")
            for d in r["unwritten_declared_files"]:
                console.print(
                    f"  [{d['step']}] declared but never touched: {', '.join(d['unwritten'])}"
                )

    if not workspace:
        console.print(
            "\n[dim]pass --workspace to also check for shadow-file "
            "collisions against the project's current tree[/]"
        )
    elif any(r["shadow_files"] for r in reports) and len(reports) > 1:
        console.print(
            "\n[dim]note: the shadow-file check compares against the "
            "workspace's CURRENT state, not the state as of when each "
            "trace ran — in a multi-run report, a collision shown "
            "against an older trace may actually have been introduced "
            "by a later one. Trust the most recent trace's report most; "
            "for older ones, check which trace actually created the "
            "colliding path before assuming that run caused it.[/]"
        )


@trace_app.command("report")
def trace_report(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory of run-*.jsonl trace files."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            exists=True,
            file_okay=False,
            help="The project these traces ran against — enables the shadow-file "
            "collision check (an undeclared new file whose name already exists "
            "elsewhere in the tree).",
        ),
    ] = None,
    failed_only: Annotated[
        bool,
        typer.Option("--failed-only", help="Only show issues for runs whose final check failed."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Run outcomes, plus declared-vs-actual and shadow-file issues per run."""
    from . import tracetools

    outcomes = tracetools.run_outcomes(trace_dir)
    if not outcomes:
        err_console.print(f"[yellow]no traces found in {trace_dir}[/]")
        raise typer.Exit(1)

    reports = _collect_trace_reports(tracetools, outcomes, trace_dir, workspace, failed_only)
    if as_json:
        print(json.dumps(reports, indent=2))
        return
    _print_report_table(reports)
    _print_report_details(reports, workspace)


@trace_app.command("audit")
def trace_audit(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory of run-*.jsonl trace files."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            exists=True,
            file_okay=False,
            help="The project these traces ran against — enables the shadow-file "
            "and plan-conformance deterministic checks (see `trace report`'s "
            "same option), fed to the audit model as pre-computed facts.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Model to audit with. Defaults to GLM 5.3 Flash — "
            "measured against Mercury on real failed traces and found to catch "
            "prompt-misinterpretation and test-gaming defects Mercury missed "
            "entirely, for a few cents total. See agent/audit.py's module "
            "docstring for the comparison.",
        ),
    ] = None,
    all_runs: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Audit every run, not just failed ones. Off by default: "
            "real findings concentrate in failures, and a passing run rarely "
            "justifies its own model call.",
        ),
    ] = False,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            help="Traces to audit at once. A directory's audit "
            "costs roughly one trace's latency regardless of how many traces "
            "there are, up to this cap — raise it for a large bench sweep, "
            "lower it if requests start queuing at the provider.",
        ),
    ] = 4,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Passed to the audit model. Defaults to "
            "'low' — measured to cut GLM's per-trace wall time 4-6x and cost "
            "6-11x versus leaving it unset, for the same findings both times "
            "(see agent/audit.py's DEFAULT_REASONING_EFFORT).",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Model-in-the-loop audit for defects tracetools' deterministic checks
    can't see: prompt misinterpretation, test-gaming, and similar judgment
    calls that need to read a run's actual narration. Traces are audited
    concurrently and cheaply (a few cents, seconds to tens of seconds for a
    handful of traces) — see agent/audit.py's module docstring for the
    measurements behind both defaults."""
    from . import audit as audit_module
    from .llm import OpenRouterClient

    settings = Settings(workspace=str(trace_dir))
    client = OpenRouterClient(api_key=settings.api_key, stream=False)
    kwargs: dict = {"failed_only": not all_runs, "concurrency": concurrency}
    if model:
        kwargs["model"] = model
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if workspace:
        kwargs["workspace_root"] = workspace
    results = audit_module.audit_traces(trace_dir, client, **kwargs)

    if not results:
        kind = "runs" if all_runs else "failed runs"
        err_console.print(f"[yellow]no {kind} found in {trace_dir}[/]")
        raise typer.Exit(0)

    if as_json:
        print(json.dumps(results, indent=2))
        return

    total_cost = sum(r["cost_usd"] for r in results)
    for r in results:
        console.print(f"\n[bold]{r['trace']}[/] ({r['model']}, ${r['cost_usd']:.4f})")
        if r.get("error"):
            console.print(f"  [red]audit call failed: {r['error']}[/]")
            continue
        if not r["findings"]:
            console.print("  [dim]no findings[/]")
            continue
        for f in r["findings"]:
            category = f.get("category", "other")
            console.print(f"  [cyan]{category}[/] {f.get('summary', '')}")
            if f.get("evidence"):
                console.print(f"    [dim]{f['evidence']}[/]")
    console.print(f"\n[dim]audit cost: ${total_cost:.4f} across {len(results)} trace(s)[/]")


@trace_app.command("grep")
def trace_grep(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory of run-*.jsonl trace files."),
    ],
    pattern: Annotated[
        str, typer.Argument(help="Literal substring to search for, case-insensitive.")
    ],
) -> None:
    """Which traces mention `pattern`, and where — for spotting the same
    defect recurring across separate runs (a single trace never shows this)."""
    from . import tracetools

    hits = tracetools.grep_traces(trace_dir, pattern)
    if not hits:
        console.print(f"no matches for {pattern!r} in {trace_dir}")
        return
    for name, matches in hits.items():
        console.print(f"\n[bold]{name}[/] ({len(matches)} match(es)):")
        for m in matches[:5]:
            console.print(f"  {m}")
        if len(matches) > 5:
            console.print(f"  [dim]... {len(matches) - 5} more[/]")


@trace_app.command("sql")
def trace_sql(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory searched for run-*.jsonl."),
    ],
    query: Annotated[str, typer.Argument(help="SQL over `events` (and `results`, if given).")],
    results: Annotated[
        list[Path] | None,
        typer.Option("--results", "-r", exists=True, help="Bench results .jsonl; repeatable."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="One JSON object per row.")] = False,
) -> None:
    """Ask the traces anything, in SQL: no hand-written loop over JSONL.

    `events` has one row per trace event: run (the path), file (its name), ts, kind, phase,
    tool, step, iteration, latency_s, prompt_tokens, completion_tokens,
    cost_usd, finish_reason, error, result, command, the lists encoded,
    mistyped and optional, and `data` for any other field
    (`data->>'summary'`). `results` is the bench records; join
    `parse_filename(results.trace_path) = events.file`, which holds however
    TRACE_DIR was written (relative, absolute, a trailing slash).

    A run's `totals` event repeats the sum of its calls' `cost_usd`: sum cost over
    `kind = 'llm_call'` (or read `totals` alone), never over every event, or each run counts
    twice (DESIGN-LOG §107.1).
    """
    from . import tracetools

    _require_extra("duckdb", "analysis", "`harness trace sql`")
    con = tracetools.sql_connection(trace_dir, results)
    rel = con.sql(query)
    if rel is None:
        return
    if as_json:
        columns = rel.columns
        for row in rel.fetchall():
            typer.echo(json.dumps(dict(zip(columns, row, strict=True)), default=str))
        return
    typer.echo(str(rel))


@trace_app.command("after-edit")
def trace_after_edit(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory searched for run-*.jsonl."),
    ],
    phase: Annotated[
        list[str] | None,
        typer.Option("--phase", help="Phase prefix to count; repeatable. Default execute, fix."),
    ] = None,
) -> None:
    """What the model did in the turn after an edit that did not finish the step,
    split by whether the edit turn also ran the tests. `pct_next_check_only`
    after `edit alone` is the round trip a harness-run check would fold; after
    `edit + run_tests` it is the model re-running a result it already had."""
    from . import tracetools

    rows = tracetools.after_edit(trace_dir, tuple(phase) if phase else ("execute", "fix"))
    if not rows:
        console.print(f"no edit turns in {trace_dir}")
        return
    for row in rows:
        typer.echo(json.dumps(row, default=str))


@trace_app.command("latency")
def trace_latency(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory searched for run-*.jsonl."),
    ],
    model: Annotated[
        str, typer.Option("--model", help='Substring of the model id; "" for every model.')
    ] = "mercury",
    by: Annotated[str, typer.Option("--by", help="Period: week or day.")] = "week",
    as_json: Annotated[bool, typer.Option("--json", help="One JSON object per row.")] = False,
) -> None:
    """Where model time went: model hours with and without retried attempts,
    retry waits by cause, runaway reasoning (over 30,000 hidden tokens), first-token
    time by prompt size, and the longest post-first-delta silence where recorded.
    DESIGN-LOG §95."""
    from . import tracetools

    tables = tracetools.latency(trace_dir, model=model, by=by)
    for name, rows in tables.items():
        if as_json:
            for row in rows:
                typer.echo(json.dumps({"table": name, **row}, default=str))
            continue
        typer.echo(f"\n{name}")
        if not rows:
            typer.echo("  (none)")
            continue
        columns = list(rows[0])
        widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in columns}
        typer.echo("  " + "  ".join(k.ljust(widths[k]) for k in columns))
        for row in rows:
            typer.echo("  " + "  ".join(str(row[k]).ljust(widths[k]) for k in columns))


@trace_app.command("holds")
def trace_holds(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory of run-*.jsonl trace files."),
    ],
    detail: Annotated[
        bool, typer.Option("--detail", help="One line per hold, not only the counts.")
    ] = False,
    recursive: Annotated[
        bool, typer.Option("--recursive", help="Search sub-directories too.")
    ] = False,
) -> None:
    """What the model did after each gate held one of its calls: re-sent it at
    once, reacted first (a test run, a read), or gave up on it. Counts per kind
    of hold; `--detail` lists every one."""
    from . import tracetools

    rows = tracetools.holds_in(trace_dir, recursive=recursive)
    if not rows:
        console.print(f"no holds in {trace_dir}")
        return
    for kind, counts in sorted(tracetools.summarize_holds(rows).items()):
        console.print(f"\n[bold]{kind}[/] ({sum(counts.values())}):")
        for label, count in sorted(counts.items(), key=lambda item: -item[1]):
            console.print(f"  {count:4}  {label}")
    if detail:
        console.print()
        for row in rows:
            between = ",".join(row["between"]) or "-"
            paths = ",".join(row["paths"]) or "-"
            console.print(
                f"{row['trace']} {row['kind']} {row['phase']} it{row['iteration']} "
                f"-> {row['response']} between=[{between}] held={paths}"
            )


@trace_app.command("leaks")
def trace_leaks(
    trace_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Directory searched recursively."),
    ],
    deny: Annotated[
        list[str] | None,
        typer.Option("--deny", help="A benchmark-source URL substring, as for run --web-deny."),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option("--query", help="A regex; web searches whose query matches count too."),
    ] = None,
) -> None:
    """Which runs reached for a benchmark's own sources: fetches of a denied URL
    (and how much came back) and searches matching --query. A fetch that
    returned text may be a leak (read the URLs); one that returned 0 was
    refused or failed."""
    from . import tracetools

    if not deny and not query:
        raise typer.BadParameter("give at least one --deny or --query")
    found = tracetools.leaks_in(trace_dir, tuple(deny or ()), query)
    if not found:
        console.print(f"no benchmark-source fetches or searches under {trace_dir}")
        return
    got_text = 0
    for run, hits in found.items():
        shown = sum(h["shown"] for h in hits)
        got_text += shown > 0
        mark = "[red]got text[/]" if shown else "tried"
        console.print(f"\n{run}: {mark} ({len(hits)} calls, {shown:,} chars shown to the model)")
        for hit in hits:
            console.print(
                f"  {hit['phase']} {hit['tool']} shown {hit['shown']:>7,} of {hit['returned']:>9,}"
                f"  {hit['target'][:110]}",
                markup=False,
                highlight=False,
            )
    console.print(
        f"\n{len(found)} runs reached for benchmark sources; {got_text} got text back "
        "(read the URLs: a directory listing is text too)"
    )


def main() -> None:
    app()


def tui_main() -> None:
    """Console-script entry point for the dashboard: `harness-tui [goal] [flags]`.

    Deliberately a rewrite of argv into the existing `run` command rather than
    a second settings-construction path. Every flag `harness run` accepts works
    here unchanged, and there is nothing to keep in sync: the TUI is a way of
    watching a run, not a different kind of run.
    """
    argv = list(sys.argv[1:])
    if "--tui" not in argv:
        argv.append("--tui")
    # The `run` command itself, not the group with "run" prepended: going
    # through the group works, but then --help reads "harness-tui run
    # [OPTIONS]" and advertises a subcommand this executable does not take.
    # TyperGroup, not click.Group: Typer 0.27 vendors click as typer._click and
    # does not install it, so `import click` here would work in this repo's dev
    # environment and crash the installed executable.
    group = typer.main.get_command(app)
    assert isinstance(group, typer.core.TyperGroup), "the harness CLI is a command group"
    group.commands["run"].main(args=argv, prog_name="harness-tui")


if __name__ == "__main__":
    main()
