"""The generic tool-calling loop. Every phase is a configuration of this."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import PurePosixPath
from typing import Any

from . import langtools
from .editcalls import CHANGE_TOOLS, EDIT_TOOLS, edited_paths, string_edits
from .failsig import failed_count
from .failsig import failure_details as _failure_details
from .guards import GuardTripped
from .kitnudges import (
    WIDENING_PHASES,
    KitWatch,
    nudge_about_foreign_lint,
    nudge_about_search_churn,
    nudge_about_shell_duplicates,
)
from .llm import BudgetExceeded, ChatClient, ModelRefused, messages_tokens
from .symptom import snippet_from
from .text import clip, truncate
from .tools import Registry, Tool, ToolContext, audit, tools_named
from .tools.base import content_fingerprint
from .trace import Trace

MAX_TOOL_RESULT_CHARS = 24_000
# Identical calls this many times in one loop means the model is stuck, not working.
REPEAT_LIMIT = 3
# ...and this many means telling it so has not worked either. Past this point
# the call is refused outright rather than dispatched again: a warning
# appended to a result the model has already ignored twice does not stop it
# burning the rest of the step's iterations on the same call, which is
# exactly the loop the warning was meant to break.
REPEAT_HARD_LIMIT = 5
# Consecutive iterations with no growth in ctx.touched before a "you are not
# making progress" nudge fires. Deliberately not the same mechanism as
# REPEAT_LIMIT: REPEAT_LIMIT is blind to a call that is cosmetically
# different every time (_fingerprint hashes the literal arguments, so four
# run_bash calls each with a slightly reworded inline Python probe are four
# distinct fingerprints, never once repeated). A traced fix round exhausted
# its whole 10-iteration budget this way -- four near-identical diagnostic
# run_bash calls, an approval pause it routed around by reissuing the same
# command a different way, one edit, done -- and the edit that finally
# landed made the failure count worse, with no iterations left to notice.
# This tracks progress the same way the rest of the harness already does
# (ctx.touched, via mark_external_writes/before_write), so it needs no new
# classification of which tools "count" as writes.
#
# Progress itself is measured by _touched_fingerprint, not len(ctx.touched):
# an early version compared set *size*, which cannot see a second real edit
# to a file already in the set -- building the scripted test for
# RUN_PYTHON_PROBE_LIMIT below surfaced a run where re-editing the same file
# a step had already touched once left the streak counter climbing exactly
# as if nothing had happened, on a step that was in fact still landing real
# changes. A real content hash of the touched files (bounded by how many a
# step has touched, so affordable unlike workspace_state's whole-tree
# mtime+size walk) changes on a genuine re-edit even though the set of
# touched paths does not -- and unlike mtime+size, it also catches a
# same-length edit two fast tool calls apart, which a coarse filesystem
# clock might not.
NO_PROGRESS_LIMIT = 4
MAX_NO_PROGRESS_NUDGES = 2
# run_python calls importing the same workspace module/class before nudging
# toward trace_python instead. Deliberately not reset by an edit landing in
# between, unlike NO_PROGRESS_LIMIT above -- a live-traced case kept its
# no_progress streak from ever crossing 4 by interleaving a guess-edit
# between each probe (edit, probe, edit, probe...), so a touched-growth
# reset alone missed the exact pattern trace_python exists for: four
# separate run_python calls, each adding one more print() to a still-wrong
# understanding of the same Parser/Tokenizer object, with the guesses in
# between it not actually resolving anything either.
RUN_PYTHON_PROBE_LIMIT = 3
MAX_PROBE_NUDGES = 2
# The same failing test still failing after this many edits between run_tests
# calls. Live-traced (an `expr` run): one test failed for five straight test
# runs while the model edited _term, _expression, _factor, _validate_chars
# and _factor again -- real edits every time, so NO_PROGRESS_LIMIT never
# fired, and no probe tool was involved, so RUN_PYTHON_PROBE_LIMIT didn't
# either. The signal here is the failure itself not changing, not activity.
SAME_FAILURE_EDIT_LIMIT = 2
MAX_TEST_WATCH_NUDGES = 2


def _held_call_message(
    call: dict, held: str, event: str, trace: Trace, phase: str, iteration: int
) -> dict:
    """The tool result standing in for a call a gate refused to run."""
    name, args = _TddGate._args(call)
    # `fp` is the same fingerprint a dispatched call carries in its own tool_call
    # event, so a later call can be matched to this one exactly: whether the model
    # re-sent the held call unchanged is the first thing anyone asks of a hold, and
    # was inferable only from tool order before. `paths` names what it would have
    # edited, which the event otherwise did not say at all.
    trace.event(
        event,
        phase=phase,
        iteration=iteration,
        tool=name,
        fp=_fingerprint(name, (call.get("function") or {}).get("arguments") or "{}"),
        paths=[p for p in _TddGate._edit_paths(name, args) if p],
    )
    return {"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": held}


class _TurnWatchers:
    """The state behind every after-a-turn nudge, kept out of the loop's own
    locals. Each watcher has a counter and a budget of its own; none reads
    another's, so they compose in a fixed order without interacting."""

    def __init__(self) -> None:
        # Tool names from the previous turn, for spotting a round trip that a
        # batched call would have saved.
        self.previous_names: list[str] = []
        self.batching_nudges = 0
        # Whether the previous turn's result carried the harness's after-edit check: an
        # edit after it may depend on it, so it is not a wasted round trip (§108).
        self.previous_checked = False
        # Consecutive iterations with no growth in ctx.touched -- see
        # NO_PROGRESS_LIMIT's comment for the traced failure this catches.
        self.no_progress_streak = 0
        self.no_progress_nudges = 0
        # Per-subject run_python call counts across the whole loop -- see
        # RUN_PYTHON_PROBE_LIMIT's comment for why this is not reset by an edit
        # landing in between, unlike no_progress_streak above.
        self.run_python_subject_counts: Counter[str] = Counter()
        self.probe_nudges = 0
        self.test_watch = _TestWatch()
        self.test_watch_nudges = 0
        # Tools the model is doing by hand (agent/kitnudges.py).
        self.kit_watch = KitWatch()

    def after_turn(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        names: list[str],
        touched_before: Any,
        ctx: ToolContext,
        registry: Registry,
        trace: Trace,
        phase: str,
    ) -> None:
        self.batching_nudges += _nudge_about_batching(
            messages,
            self.previous_names,
            names,
            trace,
            phase,
            budget=MAX_BATCHING_NUDGES - self.batching_nudges,
            edit_blocks="edit_blocks" in registry.names(),
            after_check=self.previous_checked,
        )
        self.previous_names = names

        if _touched_fingerprint(ctx) != touched_before:
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1
            self.no_progress_nudges += _nudge_about_no_progress(
                messages,
                self.no_progress_streak,
                trace,
                phase,
                budget=MAX_NO_PROGRESS_NUDGES - self.no_progress_nudges,
            )

        if (
            phase.startswith(("execute:", "fix:"))
            and self.test_watch_nudges < MAX_TEST_WATCH_NUDGES
        ):
            self.test_watch_nudges += _watch_test_results(
                self.test_watch, tool_calls, messages, ctx, trace, phase
            )

        self.run_python_subject_counts.update(_run_python_subjects(tool_calls))
        self.probe_nudges += _nudge_about_repeated_probing(
            messages,
            self.run_python_subject_counts,
            registry,
            trace,
            phase,
            budget=MAX_PROBE_NUDGES - self.probe_nudges,
        )

        nudge_about_shell_duplicates(self.kit_watch, tool_calls, messages, registry, trace, phase)
        if getattr(ctx, "web_research", False):
            nudge_about_search_churn(self.kit_watch, tool_calls, messages, trace, phase)
        nudge_about_foreign_lint(
            self.kit_watch, tool_calls, messages, ctx.workspace.root, trace, phase
        )


# Tools that can only return an error when the run has no memory store.
_MEMORY_TOOLS = frozenset({"remember", "recall"})


def _without_unavailable(registry: Registry, ctx: ToolContext) -> Registry:
    """`registry` minus tools this run cannot serve.

    Offered with memory off (every bench run), remember/recall returned
    "memory is not available" 84 times across the traces, each a wasted
    round trip, and their schemas rode every call of every loop meanwhile.
    The same reasoning web.available_web_tools applies to web_search, and
    drops ask_user when the run cannot ask anyone (336 tokens a call) and
    audit_previous_runs when no earlier run's trace sits beside this one
    (376 tokens a call, DESIGN-LOG §129).
    """
    names = set(registry.names())
    drop: set[str] = set()
    if ctx.memory is None:
        drop |= _MEMORY_TOOLS
    if not ctx.interactor.enabled:
        drop.add("ask_user")
    if "audit_previous_runs" in names and not audit.previous_trace_files(ctx, 1, ""):
        drop.add("audit_previous_runs")
    drop &= names
    if not drop:
        return registry
    return Registry([t for t in registry.tools() if t.name not in drop])


def over_spend_cap(trace: Trace, max_usd: float | None) -> bool:
    """True once a run has spent its cap. Checked before any optional call --
    the last word, a next-step suggestion, memory compaction -- so the cap
    bounds every model call, not only the ones inside a tool loop."""
    return max_usd is not None and trace.totals.cost_usd >= max_usd


def _within_budget(
    messages: list[dict],
    ctx: ToolContext,
    trace: Trace,
    phase: str,
    max_usd: float | None,
    prompt_token_budget: int | None,
) -> list[dict]:
    """Stop if the spend cap is reached; compact the history if it has outgrown
    the prompt budget. Returns the messages the next call should send."""
    if over_spend_cap(trace, max_usd):
        raise BudgetExceeded(f"spend cap of ${max_usd:.2f} reached during {phase}")
    if prompt_token_budget and messages_tokens(messages) > prompt_token_budget:
        ctx.read_epoch += 1  # compaction drops earlier reads: none is "above" any more
        return _compact(messages, prompt_token_budget, trace, phase, judge=ctx.relevance_judge)
    return messages


def _dispatch_calls(
    tool_calls: list[dict],
    gate: _TddGate | ProbeGate | ReproGate | PremiseGate | GateChain | None,
    messages: list[dict],
    registry: Registry,
    ctx: ToolContext,
    trace: Trace,
    phase: str,
    iteration: int,
    batch_size: int,
    seen_calls: Counter[str],
) -> None:
    """Run one turn's tool calls in order, appending each result to `messages`."""
    for call in tool_calls:
        # GuardTripped propagates out of here uncaught: schedule.py's
        # _guarded() is what actually abandons the step, and a caller with
        # no such handling should fail the phase loudly rather than
        # silently continue on a step that has already shown it is stuck.
        held = gate.hold(call) if gate is not None else None
        if gate is not None and held is not None:
            messages.append(_held_call_message(call, held, gate.event, trace, phase, iteration))
            continue
        result = _run_one_call(call, registry, ctx, trace, phase, iteration, batch_size, seen_calls)
        messages.append(result)
        # A call that failed changed nothing, so no gate learns from it: an edit
        # whose old_string was not found is not an unchecked edit, and a lookup
        # that errored is not a lookup made.
        if gate is not None and not str(result.get("content", "")).startswith("error:"):
            gate.observe(call)


# Tools whose output already tells the model whether its edit worked. A turn that edits
# (editcalls.EDIT_TOOLS) and then runs one of these needs no harness check.
_CHECK_TOOLS = frozenset({"run_tests", "run_python", "run_bash", "check_code", "trace_python"})
AFTER_EDIT_NOTE = (
    "\n\n[harness] After this turn's edits the harness ran the step's tests once (you do "
    "not need to run them again unless you change something):\n"
)
# A fix loop's check is its findings' own (§103), not the step's tests.
AFTER_EDIT_FINDING_NOTE = (
    "\n\n[harness] After this turn's edits the harness re-ran this finding's own check (you "
    "do not need to run it again unless you change something):\n"
)
# A step that has not written its tests yet still hears about the type errors it introduced.
AFTER_EDIT_TYPES_NOTE = (
    "\n\n[harness] After this turn's edits the harness checked the types (this step has no "
    "tests yet, so none ran):\n"
)


def _check_after_edit(
    messages: list[dict],
    names: list[str],
    touched_before: str,
    gate,
    registry: Registry,
    ctx: ToolContext,
    trace: Trace,
    custom: Callable[[ToolContext], str] | None = None,
    types: bool = False,
) -> bool:
    """Settings.check_after_edit: run the step's tests once after a turn that edited the
    workspace and did not check its own edit, and append the result to that turn's last tool
    result, where the next call reads it. Returns True when a result was handed back.

    The pair "a turn that edits, then a turn that only runs the tests" is 57% of edit turns
    across all traces (12,836 of 22,448), each one a model call spent asking whether the edit
    worked; 95% of those runs targeted what the harness could have chosen itself (DESIGN-LOG
    §90 design C, §98). Several edits in one turn get one run. The run does not count for
    `_TddGate` (the model's own test before its first source edit stays required), and does
    count for NumericCheckGate, which only asks that the code ran since the last edit.
    """
    tool = registry.get("run_tests")
    check = custom if custom is not None else (tool.fn if tool is not None else None)
    if check is None or not names:
        return False
    last_edit = max((i for i, name in enumerate(names) if name in EDIT_TOOLS), default=-1)
    if last_edit < 0 or any(name in _CHECK_TOOLS for name in names[last_edit + 1 :]):
        return False
    if _touched_fingerprint(ctx) == touched_before:
        return False  # every edit in the turn failed: nothing new to check
    # A fix round passes its findings' own checks (`custom`): a type-error finding is checked
    # by the type checker, not by tests that already pass. The platformer app's fix round for 5
    # type errors got "PASS" from the tests after every edit, re-read the file each time, and
    # used all 10 turns (DESIGN-LOG §103).
    ctx.test_source = "auto"
    try:
        result = str(check(ctx))
    finally:
        ctx.test_source = ""
    typed = ""
    if types and custom is None:
        from .phases.finalcheck import new_type_errors_text

        typed = new_type_errors_text(ctx)  # Settings.types_after_edit, DESIGN-LOG §104
    # Before a step has written its tests, the test run is noise (§99.1 smoke), but the type
    # errors are not: dropping them with it left 11 of 95 such checks' errors unseen and 8
    # steps finishing verified with them outstanding (§108).
    no_tests = result.startswith("no tests were collected")
    if no_tests:
        note, result = AFTER_EDIT_TYPES_NOTE, typed
    else:
        note = AFTER_EDIT_FINDING_NOTE if custom is not None else AFTER_EDIT_NOTE
        result = f"{result}\n\n{typed}" if typed else result
    delivered = bool(result) and bool(messages) and messages[-1].get("role") == "tool"
    trace.event("after_edit_check", result=result[:300], no_tests=no_tests, delivered=delivered)
    if not delivered:
        return False
    if not no_tests:
        _note_ran(gate)  # the code ran; a type check alone does not run it
    messages[-1]["content"] = str(messages[-1].get("content", "")) + note + result
    return True


def _note_ran(gate) -> None:
    """Tell every gate that tracks "the code ran since the last edit" that it did."""
    for member in getattr(gate, "gates", [gate]):
        ran = getattr(member, "ran", None)
        if callable(ran):
            ran()


def _with_tail(
    messages: list[dict],
    tail: Callable[[int, int, list[dict]], str] | None,
    iteration: int,
    max_iters: int,
    trace: Trace | None = None,
    phase: str = "",
) -> list[dict]:
    """`messages` plus, from the second turn on, `tail`'s text as a last user
    message for this call only. Not appended to the history: it would pile up
    one copy per turn, and the history is the cached prefix of the next call
    (Settings.executor_tail_reminder, DESIGN-LOG §82). Traced, so a review can
    read what the model was told last (DESIGN-LOG §84)."""
    if tail is None or iteration == 0:
        return messages
    text = tail(iteration, max_iters, messages)
    if trace is not None:
        trace.event("tail", phase=phase, iteration=iteration, text=text)
    return [*messages, {"role": "user", "content": text}]


def run_tool_loop(
    client: ChatClient,
    model: str,
    phase: str,
    system: str,
    user: str,
    registry: Registry,
    ctx: ToolContext,
    trace: Trace,
    max_iters: int,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_usd: float | None = None,
    prompt_token_budget: int | None = None,
    seen_calls: Counter[str] | None = None,
    salvage: bool = False,
    tdd_tests: list[str] | None = None,
    gate: ProbeGate | ReproGate | PremiseGate | GateChain | None = None,
    tail: Callable[[int, int, list[dict]], str] | None = None,
    last_word_gates: str = "off",
    check_after_edit: bool = False,
    after_edit_check: Callable[[ToolContext], str] | None = None,
    types_after_edit: bool = False,
) -> tuple[Any, list[dict]]:
    """Drive `model` until it calls a terminal tool, stalls, or runs out of budget.

    Returns (terminal tool result, full message history). Raises GuardTripped if
    the step's resource guard trips (e.g. too many consecutive tool errors) —
    callers that want a step abandoned rather than the whole phase failing
    should catch it at the point they can do that (schedule.py's _guarded()
    does this for execute steps); a caller with no such handling will simply
    fail the phase, which is the correct default over silently continuing.
    """
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    registry = _without_unavailable(registry, ctx)
    ctx.done = False
    ctx.result = None
    ctx.guard.reset_step()
    stalls = 0
    # Fingerprints of calls already made, to notice a model going in circles.
    # A caller may pass its own counter to have it span several loops: an
    # execute step retried at higher reasoning gets a fresh loop but is still
    # the same step, and a model that ran the identical failing command three
    # times per attempt would otherwise never reach the hard limit.
    if seen_calls is None:
        seen_calls = Counter()
    watchers = _TurnWatchers()
    # A new loop starts a new conversation: no earlier read is in it (read_file, §104).
    ctx.read_epoch += 1
    # None unless the caller asked for the red-first gate (see _TddGate) or
    # passed one of its own (the planner's ProbeGate). Both at once are chained,
    # the red-first gate first: an earlier version let tdd_tests replace `gate`,
    # which silently dropped the premise and numeric gates on every first attempt
    # of a tdd_gate run (code review of 2026-09-30, finding 1).
    call_gate: _TddGate | ProbeGate | ReproGate | PremiseGate | GateChain | None = gate
    if tdd_tests is not None:
        call_gate = (
            GateChain(_TddGate(tdd_tests), gate) if gate is not None else _TddGate(tdd_tests)
        )

    for iteration in range(max_iters):
        ctx.cancel.check()
        messages = _within_budget(messages, ctx, trace, phase, max_usd, prompt_token_budget)

        try:
            reply = client.chat(
                model=model,
                messages=_with_tail(messages, tail, iteration, max_iters, trace, phase),
                tools=registry.schemas(),
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )
        except ModelRefused as exc:
            # Not a stall: the run fails here rather than a phase degrading around a
            # model that will refuse every call (the planner's one-step fallback, §127.9).
            trace.event("model_refused", phase=phase, model=model, provider=exc.provider)
            trace.say(f"   ! {exc}")
            raise
        trace.llm_call(phase, model, reply.usage, reply.finish_reason, reply=reply)
        messages.append(reply.as_message())

        if reply.content.strip():
            trace.event(
                "assistant_text", phase=phase, iteration=iteration, text=reply.content[:2000]
            )

        if not reply.tool_calls:
            stalls += 1
            if stalls >= 2:
                trace.event("stalled", phase=phase, iteration=iteration)
                break
            messages.append(_stall_nudge(registry))
            continue

        stalls = 0
        batch_size = len(reply.tool_calls)
        names = _tool_names(reply.tool_calls)
        touched_before = _touched_fingerprint(ctx)
        _dispatch_calls(
            reply.tool_calls,
            call_gate,
            messages,
            registry,
            ctx,
            trace,
            phase,
            iteration,
            batch_size,
            seen_calls,
        )

        watchers.after_turn(
            messages, reply.tool_calls, names, touched_before, ctx, registry, trace, phase
        )

        if ctx.done:
            return ctx.result, messages
        watchers.previous_checked = check_after_edit and _check_after_edit(
            messages,
            names,
            touched_before,
            call_gate,
            registry,
            ctx,
            trace,
            after_edit_check,
            types_after_edit,
        )

    spent = _where_it_went(messages)
    trace.event("loop_exhausted", phase=phase, iterations=max_iters, done=ctx.done, spent_on=spent)
    over_cap = over_spend_cap(trace, max_usd)
    if over_cap:
        trace.event("last_word_skipped", phase=phase, reason="spend cap reached")
    trace.say(f"   ! {phase} used all {max_iters} iterations without finishing — spent on: {spent}")
    return _exhausted_result(
        ctx,
        messages,
        phase,
        salvage,
        trace,
        None
        if over_cap
        else partial(
            _last_word,
            client,
            model,
            messages,
            registry,
            ctx,
            trace,
            phase,
            reasoning_effort,
            temperature,
            gate=call_gate,
            gates_mode=last_word_gates,
        ),
    )


# Calls whose batched form does exactly the same work in one round trip.
# Deliberately only these two: reads and edits compose safely, whereas
# "edit then run the tests" must stay two turns, because the second one's
# whole purpose is to see the result of the first.
_BATCHABLE = {"read_file": "read_files", "edit_file": "edit_files"}
# With Settings.edit_blocks the kit has edit_blocks in place of edit_files (§104).
_BATCHED_EDIT_BLOCKS = {"read_file": "read_files", "edit_file": "edit_blocks"}
# A nudge the model ignores twice is not going to work the third time, and
# every repeat is prompt tokens on each subsequent call of the step.
MAX_BATCHING_NUDGES = 2


def _tool_names(calls: list[dict]) -> list[str]:
    return [((call.get("function") or {}).get("name") or "") for call in calls]


def _batching_nudge(previous: list[str], current: list[str], edit_blocks: bool = False) -> str:
    """Advice for a turn that spent a round trip it did not have to.

    89% of this agent's model time is time to first token, so a round trip
    costs about 1.3s whatever it carries — which makes two read_file calls in
    two turns twice the price of one read_files call for the same bytes.
    Across the traces the model batched on 7.9% of turns and 92% carried a
    single call, and the exhortation to batch has been in EXECUTOR_SYSTEM the
    whole time. Saying it at the moment the round trip is wasted, attached to
    the result that wasted it, is a different thing from saying it once at the
    top of a prompt the model has been reading for eight turns.

    Returns "" unless this turn and the previous one were each a single call
    of the same batchable tool.
    """
    if len(current) != 1 or len(previous) != 1 or current[0] != previous[0]:
        return ""
    batched = (_BATCHED_EDIT_BLOCKS if edit_blocks else _BATCHABLE).get(current[0])
    if batched is None:
        return ""
    return (
        f"\n\n[harness] That is two {current[0]} calls in two turns. "
        f"{batched} does the same work in one: a round trip costs about a "
        "second whatever it carries, so deciding everything you need and "
        "asking once is the single cheapest thing you can do here."
    )


def _nudge_about_batching(
    messages: list[dict],
    previous: list[str],
    current: list[str],
    trace: Trace,
    phase: str,
    budget: int,
    edit_blocks: bool = False,
    after_check: bool = False,
) -> int:
    """Append the batching advice to the turn's last tool result. Returns 1 if
    it did, so the caller can count against its budget.

    It rides on a message that is already being sent, so it costs no round
    trip — only the tokens of the sentence itself, and only on the turns that
    earned it. An edit after a turn whose result carried the after-edit check
    did not earn it: the check is new information, and editing the test, seeing
    it fail, then editing the source is the order the executor is told to follow.
    41 of 71 edit batching nudges fired right after a FAIL check (§108).
    """
    if budget <= 0 or (after_check and current and current[0] in EDIT_TOOLS):
        return 0
    nudge = _batching_nudge(previous, current, edit_blocks)
    if not nudge or not messages or messages[-1].get("role") != "tool":
        return 0
    messages[-1]["content"] = str(messages[-1].get("content", "")) + nudge
    trace.event("batching_nudge", phase=phase, tool=current[0])
    return 1


def _touched_fingerprint(ctx: ToolContext) -> str:
    """Content fingerprint of just the files touched so far this step, not
    the whole workspace -- bounded by how many files this step has touched
    (typically a handful), not everything under root the way
    workspace_state's whole-tree walk is, which is what makes hashing real
    bytes affordable here where it would not be there. Real content, not
    mtime+size: a same-length edit (one literal to another) changes neither,
    and mtime granularity on some filesystems is coarser than two edits a
    fast tool loop can land inside. Unlike comparing len(ctx.touched), this
    changes when an already-touched file is genuinely edited again.
    """
    return content_fingerprint(ctx)


def _nudge_about_no_progress(
    messages: list[dict], streak: int, trace: Trace, phase: str, budget: int
) -> int:
    """Append a "you have not changed anything in a while" nudge once `streak`
    crosses NO_PROGRESS_LIMIT. Returns 1 if it did, so the caller can count
    against its budget — same shape and same reason as _nudge_about_batching:
    rides on a message already being sent, no extra round trip, and capped so
    an ignored nudge does not repeat itself into the ground.

    Unlike REPEAT_LIMIT's exact-fingerprint match, this only needs the streak
    to have just crossed the threshold (not every iteration past it) so the
    same nudge is not appended on every subsequent turn regardless of budget.

    Scoped to execute/fix phases only. A live run surfaced this firing during
    `plan` and `review:N` — both read-only by design (planning explores
    before submit_plan, review_registry has no write tools at all) — where
    "make the edit now" is not just unhelpful, it is wrong advice for what
    that phase is supposed to be doing.
    """
    if not (phase.startswith("execute:") or phase.startswith("fix:")):
        return 0
    if budget <= 0 or streak != NO_PROGRESS_LIMIT:
        return 0
    if not messages or messages[-1].get("role") != "tool":
        return 0
    nudge = (
        f"\n\n[harness] That is {streak} tool calls in a row with no file changed. "
        "If you are still diagnosing, name the specific hypothesis your next call "
        "tests and what result would confirm or rule it out — then act on it. If "
        "you already know what is wrong, make the edit now rather than checking "
        "again. If something is genuinely blocking you (an approval pause, a "
        "missing piece of information), say so and call the terminal tool rather "
        "than repeating the same check."
    )
    messages[-1]["content"] = str(messages[-1].get("content", "")) + nudge
    trace.event("no_progress_nudge", phase=phase, streak=streak)
    return 1


class GateChain:
    """Several gates on one loop: the first that holds a call wins, and every
    gate observes every call that went through. `event` names whichever gate
    held last, for the held-call trace event."""

    def __init__(self, *gates) -> None:
        self.gates = [g for g in gates if g is not None]
        self.event = "gate_held"

    def observe(self, call: dict) -> None:
        for gate in self.gates:
            gate.observe(call)

    def hold(self, call: dict) -> str | None:
        for gate in self.gates:
            held = gate.hold(call)
            if held is not None:
                self.event = gate.event
                return held
        return None


class ProbeGate:
    """Holds the planner's first submit_plan until it has run a probe.

    Offering probe_python was not enough: across 10 live runs of the four
    symptom-reported bench tasks the planner probed on one task and never on
    the other three, so a sweep of the offer alone would compare the default
    with itself. Like _TddGate, this acts once and says how to skip it.
    """

    event = "probe_gate_held"

    def __init__(self) -> None:
        self.probed = False
        self.held = False

    def observe(self, call: dict) -> None:
        if (call.get("function") or {}).get("name") == "probe_python":
            self.probed = True

    def hold(self, call: dict) -> str | None:
        if self.held or self.probed or (call.get("function") or {}).get("name") != "submit_plan":
            return None
        self.held = True
        return (
            "[harness] Plan NOT submitted: you have not run anything yet. Before you name a "
            "cause, use probe_python to reproduce what the goal describes, through the entry "
            "point it names, and look at the actual output. Then send submit_plan again, "
            "revised by what you saw. If the goal names its change outright and there is "
            "nothing to reproduce, re-send the plan unchanged now."
        )


# A snippet that dies on one of these never reached the code under test, so
# its failure reproduces nothing; anything else non-zero (an AssertionError,
# the KeyError the report complained about) counts as the problem showing.
_BROKEN_SNIPPET = re.compile(
    r"\b(SyntaxError|IndentationError|NameError|ModuleNotFoundError|ImportError)\b"
)


class ReproGate:
    """Runs the planner's `reproduce` snippet before its plan is accepted.

    The probe gate (ProbeGate) held every plan once, costing +2.6 planning
    calls on reports that were accurate, and paid off only when the report was
    wrong (DESIGN-LOG §54.1). This moves the check into the plan submission
    itself: a snippet that fails means the problem is real and the plan goes
    through with no extra call; one that passes means the reported problem did
    not show, and only then is the plan sent back, once.
    """

    event = "repro_gate_held"

    def __init__(self, run: Callable[[str], tuple[int, str] | str], trace: Trace) -> None:
        self.run = run
        self.trace = trace
        self.checked = False
        self.asked = False
        # (snippet, output) when the reproduction passed, for PremiseGate.
        self.passed: tuple[str, str] | None = None
        # The snippet when it failed, i.e. the problem was real: EvidenceGate re-runs it
        # before a fix step finishes (Settings.bugfix_evidence, DESIGN-LOG §120).
        self.reproduced: str | None = None

    def observe(self, call: dict) -> None:
        pass

    def hold(self, call: dict) -> str | None:
        fn = call.get("function") or {}
        if fn.get("name") != "submit_plan":
            return None
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        code = snippet_from(args)
        if not code:
            return self._missing(str(args.get("task_kind") or ""))
        if self.checked:
            return None
        self.checked = True
        return self._check(code)

    def _missing(self, task_kind: str) -> str | None:
        """An empty snippet: fine for new work, sent back once for a bugfix.

        Only reachable when submit_plan carries `task_kind` (plan_task_kind on);
        the default "repro" does not ask for it. Built because 29 of 78 §54.2
        runs left `reproduce` empty; measured in §54.3 it fired in 29 of 108
        runs, mostly on accurate reports, for +1.9 calls a run and no change
        in honest, so it stays opt-in.
        """
        if task_kind != "bugfix" or self.asked or self.checked:
            if not (self.asked or self.checked):
                self.trace.event("repro", outcome="none")
            return None
        self.asked = True
        self.trace.event("repro", outcome="missing")
        return (
            "[harness] Plan NOT accepted yet: you classified this goal as a bugfix but left "
            "`reproduce` empty. Write a short snippet asserting the behaviour the report "
            "expects, through the entry point it names; it must fail while the problem "
            "exists. Then send submit_plan again. If nothing is actually reported broken, "
            "set task_kind to what the goal really is instead."
        )

    def _check(self, code: str) -> str | None:
        ran = self.run(code)
        if isinstance(ran, str):
            self.trace.event("repro", outcome="not_run", detail=ran[:200])
            return None
        exit_code, output = ran
        if exit_code != 0 and not _BROKEN_SNIPPET.search(output[-600:]):
            self.trace.event("repro", outcome="reproduced")
            self.reproduced = code
            return None
        outcome = "passed" if exit_code == 0 else "broken"
        self.trace.event("repro", outcome=outcome)
        shown = output or "(no output)"
        if outcome == "passed":
            self.passed = (code, shown)
        if outcome == "broken":
            return (
                "[harness] Plan NOT accepted yet: your `reproduce` snippet failed before it "
                f"reached the behaviour in question:\n{shown}\nFix the snippet so it exercises "
                "what the goal reports, then send submit_plan again."
            )
        return (
            "[harness] Plan NOT accepted yet: your `reproduce` snippet PASSED, so the problem "
            f"the goal reports did not show up in the current code. Output:\n{shown}\n"
            "Either the report is mistaken, and then the draft should say so and the plan "
            "should change no source code, or the snippet does not exercise what the report "
            "describes. Decide which, then send submit_plan again."
        )


def _proposed_contents(ctx: ToolContext, name: str, args: dict) -> dict[str, str | None]:
    """{path: content after this call} for the files an edit call would change,
    None for a deletion. Only as good as a plain string replace, which is what
    the edit tools do."""
    if name == "write_file":
        return {str(args.get("path", "")): str(args.get("content", ""))}
    if name == "delete_file":
        return {str(args.get("path", "")): None}
    after: dict[str, str | None] = {}
    for edit in string_edits(name, args):
        path = edit["path"]
        current = after.get(path) if path in after else ctx.try_read(path)
        old, new = edit["old_string"], edit["new_string"]
        if edit["create"] and current is None:
            after[path] = new + "\n"  # edit_blocks creates the file, as write_file would
            continue
        if current is None or not old or old not in current:
            continue
        after[path] = current.replace(old, new, -1 if edit["replace_all"] else 1)
    return after


# Files that decide how a project's tests run, rather than being tests or code under
# test. A settings module named in the test command (`--settings=test_sqlite`) counts.
_TEST_INFRA_NAMES = frozenset({"conftest.py", "pytest.ini", "tox.ini", "noxfile.py", "runtests.py"})


def is_test_infrastructure(path: str, test_command: str = "", label_root: str = "") -> bool:
    name = PurePosixPath(path).name
    if name in _TEST_INFRA_NAMES:
        return True
    command = test_command or ""
    if path and path in command:
        return True
    for token in command.split():
        if token.startswith("--settings="):
            module = token.split("=", 1)[1].replace(".", "/") + ".py"
            root = label_root.strip("/")
            if path in (module, f"{root}/{module}" if root else module):
                return True
    return False


# Holds one run may spend on edits to pre-existing tests when justification is
# required: the first asks for a reason, the second says why the reason was not
# enough. After that the edit goes through and is recorded as unjustified.
MAX_TEST_CHANGE_HOLDS = 2


class PremiseGate:
    """Holds, once per run each, two edits that act against the run's own evidence.

    Both come from §54.3's traces of the wrong-report tasks:

    - **A source edit after the plan's own reproduction passed.** On
      misreport-slug the planner wrote that the code was already correct and
      then planned a change anyway ("make it more explicit") in 7 of 14 runs.
    - **An edit that removes or weakens a test that existed before the run.**
      49 of 53 dishonest misreport-discount runs deleted the seeded test that
      pins the behaviour the report called a bug. A passing test asserting the
      current behaviour is the codebase saying the behaviour is intended.

    Like _TddGate it acts before the edit lands, once, and says how to proceed
    if the edit is right after all. Run-scoped through ctx.premise_holds.
    """

    def __init__(self, ctx: ToolContext, goal: str = "", justify: bool = False) -> None:
        self.ctx = ctx
        self.event = "premise_gate_held"
        self.goal = goal
        self.justify = justify

    def observe(self, call: dict) -> None:
        pass

    def hold(self, call: dict) -> str | None:
        name, args = _TddGate._args(call)
        if name not in CHANGE_TOOLS:
            return None
        return (
            self._after_passed_repro(name, args)
            or self._test_infrastructure(name, args)
            or self._pinned_tests(name, args)
        )

    def _test_infrastructure(self, name: str, args: dict) -> str | None:
        """Hold, once per file, an edit to the machinery that runs the tests.

        django runs (DESIGN-LOG §72) edited `tests/runtests.py` and the test
        settings module `tests/test_sqlite.py`: changes that decide how every test
        runs, made in passing, which failed the run's own checks and led a goal
        loop to revert a correct fix along with them. The hold names a concrete
        alternative, since holds asking only to reconsider are mostly re-sent
        unchanged (§58). A goal that names the file is asking for the change.
        """
        for path in _TddGate._edit_paths(name, args):
            key = f"infra:{path}"
            if (
                not path
                or key in self.ctx.premise_holds
                or not is_test_infrastructure(path, self.ctx.test_command, self.ctx.test_label_root)
            ):
                continue
            if PurePosixPath(path).name in self.goal or not self._existed_before_run(path):
                continue
            self.ctx.premise_holds.add(key)
            self.event = "test_infrastructure_held"
            return (
                f"[harness] Held, and NOT applied: {path} is part of how this project runs "
                "its tests, not code under test, and the goal does not ask to change it. "
                "Leave it as it is and make the change in the code the goal is about; run "
                "the tests the project's usual way. If the goal truly requires changing "
                "this file, send the same edit again."
            )
        return None

    def _existed_before_run(self, path: str) -> bool:
        """A file this run created is not the project's machinery: 53 of 64 bench
        edits to a conftest.py-like name were write_file creating one."""
        snapshots = self.ctx.baseline.snapshots
        if path in snapshots:
            return snapshots[path] is not None
        return (self.ctx.workspace.root / path).is_file()

    def _after_passed_repro(self, name: str, args: dict) -> str | None:
        repro = self.ctx.repro_passed
        if repro is None or "repro_passed" in self.ctx.premise_holds:
            return None
        paths = [p for p in _TddGate._edit_paths(name, args) if p.endswith(".py")]
        if not any(not langtools.is_test_file(langtools.PYTHON, p) for p in paths):
            return None
        self.ctx.premise_holds.add("repro_passed")
        self.event = "repro_passed_held"
        snippet, output = repro
        return (
            "[harness] Held, and NOT applied: this changes source code, but the plan's own "
            "reproduction of the reported problem PASSED against the unchanged code, so the "
            f"problem was not there.\nSnippet:\n{clip(snippet, 600)}\nOutput:\n"
            f"{clip(output, 600)}\nOnly change source if you now have a failing test or "
            "run that shows a real defect; a refactor to make correct code 'more explicit' "
            "is not what the goal asked for. If the report was mistaken, change nothing and "
            "say so in finish_step. If you do have evidence of a defect, re-send this edit."
        )

    def _changed_tests(self, name: str, args: dict) -> list[tuple[str, str, dict]]:
        """(test id, removed | weakened | rewritten, description) for each test that
        existed before the run and that this call would change."""
        from .testchange import describe_test
        from .testcheck import tests_in

        found: list[tuple[str, str, dict]] = []
        for path, after in _proposed_contents(self.ctx, name, args).items():
            if not langtools.is_test_file(langtools.PYTHON, path):
                continue
            snapshots = self.ctx.baseline.snapshots
            original = snapshots[path] if path in snapshots else self.ctx.try_read(path)
            if original is None:
                continue
            current = self.ctx.try_read(path)
            pinned, before, new = tests_in(original), tests_in(current), tests_in(after)
            for test, (asserts, body) in before.items():
                if test not in pinned:
                    continue
                if test not in new:
                    kind = "removed"
                elif new[test][0] < asserts:
                    kind = "weakened"
                elif new[test][1] != body:
                    kind = "rewritten"
                else:
                    continue
                found.append((f"{path}::{test}", kind, describe_test(current, test)))
        return found

    def _justified_test_change(self, name: str, args: dict) -> str | None:
        """Hold an edit to a pre-existing test until it says why; see testchange.py.

        At most MAX_TEST_CHANGE_HOLDS holds a run, so a legitimate fix is delayed
        by two round trips, never blocked; whatever gets through without a reason
        is recorded as unjustified for the summary.
        """
        from .testchange import check_justification, hold_message

        changed = self._changed_tests(name, args)
        if not changed:
            return None
        text = str(args.get("justification") or "").strip()
        verdict = check_justification(text, self.goal)
        state = self.ctx.test_change_state
        entry = {
            "tests": [c[0] for c in changed],
            "kinds": [c[1] for c in changed],
            "justification": text[:400],
        }
        if verdict.ok:
            state["log"].append({**entry, "basis": verdict.basis})
            self.ctx.trace.event("test_change_justified", basis=verdict.basis, **entry)
            return None
        if state["holds"] >= MAX_TEST_CHANGE_HOLDS:
            state["log"].append({**entry, "basis": "unjustified"})
            self.ctx.trace.event("test_change_unjustified", **entry)
            return None
        second = state["holds"] > 0
        state["holds"] += 1
        self.event = "pinned_test_held"
        return hold_message(changed, verdict.problem, second)

    def _pinned_tests(self, name: str, args: dict) -> str | None:
        if self.justify:
            return self._justified_test_change(name, args)
        if "pinned_test" in self.ctx.premise_holds:
            return None
        from .testcheck import tests_in

        lost: list[str] = []
        for path, after in _proposed_contents(self.ctx, name, args).items():
            if not langtools.is_test_file(langtools.PYTHON, path):
                continue
            snapshots = self.ctx.baseline.snapshots
            original = snapshots[path] if path in snapshots else self.ctx.try_read(path)
            if original is None:
                continue
            before, pinned = tests_in(self.ctx.try_read(path)), tests_in(original)
            new = tests_in(after)
            lost += [
                f"{path}::{test}"
                for test, (asserts, _) in before.items()
                if test in pinned and (test not in new or new[test][0] < asserts)
            ]
        if not lost:
            return None
        self.ctx.premise_holds.add("pinned_test")
        self.event = "pinned_test_held"
        return (
            "[harness] Held, and NOT applied: this edit removes or weakens tests that existed "
            f"before this run: {', '.join(lost[:5])}. A passing test that asserts the current "
            "behaviour is the codebase saying that behaviour is intended. Read what the test "
            "asserts. Unless the goal explicitly asks for that behaviour to change (quote its "
            "words in finish_step), the right fix is to undo the source change that broke the "
            "test, not the test. If the goal does ask for it, re-send this edit."
        )


class _TddGate:
    """Holds a step's first non-test source edit until a test has been run.

    The executor prompt says to write the test, watch it fail, then fix. On 1,285
    traced steps that edited source, 13% ran a failing test before their first
    source edit and 85% ran no test at all, and a nudge appended to the edit's
    result cannot change that: by then the edit has happened. So this acts one
    step earlier, once per step. The first call that would edit non-test source,
    before any test has run in this loop, is not applied; the model is told why
    and can re-send it unchanged straight away. The cost of being wrong (a step
    with nothing a test can observe) is one wasted round trip, and the message
    says how to skip it.

    Only built for a step whose plan names tests, on its first attempt.
    """

    def __init__(self, tests: list[str]) -> None:
        self.tests = tests
        self.event = "tdd_gate_held"
        self.test_run = False
        self.held = False

    @staticmethod
    def _edit_paths(name: str, args: dict) -> list[str]:
        return edited_paths(name, args)

    @staticmethod
    def _args(call: dict) -> tuple[str, dict]:
        fn = call.get("function") or {}
        try:
            parsed = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            parsed = {}
        return str(fn.get("name", "")), parsed if isinstance(parsed, dict) else {}

    def observe(self, call: dict) -> None:
        """Record that a test was run, by the tool or by a shell command."""
        name, args = self._args(call)
        if name == "run_tests" or (name == "run_bash" and "pytest" in str(args.get("command", ""))):
            self.test_run = True

    def hold(self, call: dict) -> str | None:
        """The message to return instead of applying `call`, or None to let it run."""
        if self.held or self.test_run:
            return None
        name, args = self._args(call)
        paths = [p for p in self._edit_paths(name, args) if p.endswith(".py")]
        if not any(not langtools.is_test_file(langtools.PYTHON, p) for p in paths):
            return None
        self.held = True
        planned = "; ".join(self.tests[:4])
        return (
            "[harness] Held, and NOT applied: this edit changes source, and you have not run "
            "any test in this step yet. The step lists tests to write"
            + (f" ({planned})" if planned else "")
            + ". First add or update the test, and run it with run_tests to see it fail for "
            "the reason the step describes; then make this edit. Re-send the same edit call "
            "unchanged after that. If this step genuinely has nothing a test can observe, "
            "re-send it now and say so in finish_step's summary."
        )


class _TestWatch:
    """Tracks what run_tests said last, across one tool loop, to notice a
    failure that edits are not changing (or making worse). State only;
    _watch_test_results decides what to say."""

    def __init__(self) -> None:
        self.ids: frozenset[str] = frozenset()
        self.count = 0
        self.fingerprint: str | None = None
        self.edits_same_failure = 0
        self.nudged_for: frozenset[str] = frozenset()


def _watch_test_results(
    watch: _TestWatch,
    calls: list[dict],
    messages: list[dict],
    ctx: ToolContext,
    trace: Trace,
    phase: str,
) -> int:
    """Append a targeted nudge to a run_tests result when the same failing
    test survives SAME_FAILURE_EDIT_LIMIT edits, or the failed count rose
    after an edit. Returns 1 per nudge sent, so the caller can budget them.
    Nothing here costs a model call."""
    results = messages[-len(calls) :] if calls else []
    sent = 0
    for call, message in zip(calls, results, strict=False):
        text = _test_result_text(call, str(message.get("content", "")))
        if text is None:
            continue
        fingerprint = _touched_fingerprint(ctx)
        edited = watch.fingerprint is not None and fingerprint != watch.fingerprint
        watch.fingerprint = fingerprint
        # A collection error is the worst outcome an edit can have -- nothing
        # ran at all -- and it arrives as its own banner rather than "FAIL", so
        # treating only FAIL as failure meant the one edit that broke the suite
        # outright reset the watcher instead of warning about it.
        broke_collection = text.startswith("COLLECTION ERROR")
        if not text.startswith("FAIL") and not broke_collection:
            watch.ids, watch.count, watch.edits_same_failure = frozenset(), 0, 0
            continue
        ids, count, where = _failure_details(text)
        note = ""
        if edited and broke_collection:
            note = (
                "That edit stopped the suite collecting, so no test ran at all — "
                "this is an import or syntax problem in what you just changed, not "
                "a failing assertion. Use show_changes and fix that first."
            )
            trace.event(
                "failure_count_rose_nudge",
                phase=phase,
                before=watch.count,
                after=count,
                collection_broken=True,
            )
        elif edited and watch.count and count > watch.count:
            note = (
                f"That edit made things worse: {watch.count} failed before it, "
                f"{count} now. Look at what changed (show_changes) before "
                "editing anything else."
            )
            trace.event("failure_count_rose_nudge", phase=phase, before=watch.count, after=count)
        elif ids and ids == watch.ids:
            watch.edits_same_failure += 1 if edited else 0
            if watch.edits_same_failure >= SAME_FAILURE_EDIT_LIMIT and ids != watch.nudged_for:
                watch.nudged_for = ids
                name = sorted(ids)[0]
                note = (
                    f"{name} has now failed after {watch.edits_same_failure} edits "
                    "without its failure changing."
                    + (f" It surfaces at {where}." if where else "")
                    + " Work out what value that code received and why it is wrong "
                    "before editing another function — one more guess at a "
                    "different place is what the last edits were."
                )
                trace.event(
                    "same_failure_nudge",
                    phase=phase,
                    test=name,
                    edits=watch.edits_same_failure,
                    where=where,
                )
        else:
            watch.edits_same_failure = 0
        watch.ids, watch.count = ids, count
        if note:
            message["content"] = str(message.get("content", "")) + f"\n\n[harness] {note}"
            sent += 1
    return sent


_PYTEST_COMMAND = re.compile(r"\bpytest\b")
_PASSED = re.compile(r"\b\d+ passed\b")
_COLLECTION_BROKEN = re.compile(r"error(s)? during collection|^ERROR collecting ", re.MULTILINE)


def _test_result_text(call: dict, content: str) -> str | None:
    """A test run's result in run_tests' own form, or None if `call` ran no tests.

    run_tests already answers "PASS", "FAIL ..." or "COLLECTION ERROR ...". A
    pytest run through run_bash answers `exit=N (Xs)` and raw output, and 504
    of those in the bench traces never reached this watcher, so the
    same-failure and made-it-worse nudges could not see them (DESIGN-LOG §66).
    The verdict comes from pytest's own summary, not the exit code, which a
    pipe (`pytest | tail`) replaces with the last command's.
    """
    fn = call.get("function") or {}
    name = fn.get("name")
    if name == "run_tests":
        return content
    if name != "run_bash":
        return None
    try:
        command = str(json.loads(fn.get("arguments") or "{}").get("command", ""))
    except (json.JSONDecodeError, AttributeError):
        return None
    if not _PYTEST_COMMAND.search(command):
        return None
    if _COLLECTION_BROKEN.search(content):
        return "COLLECTION ERROR\n" + content
    if failed_count(content):
        return "FAIL\n" + content
    # Only a real pytest summary counts as a pass: a command that never ran
    # pytest (not found, a typo, a timeout) would otherwise reset the watcher.
    return "PASS\n" + content if _PASSED.search(content) else None


_IMPORT_SUBJECT_RE = re.compile(r"(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")


def _run_python_subjects(calls: list[dict]) -> set[str]:
    """Which workspace modules/classes this turn's run_python call(s), if
    any, import -- the signal _nudge_about_repeated_probing counts across
    the whole loop. A turn with no run_python call, or one whose code the
    regex finds nothing importable in (a bare expression), contributes
    nothing either way -- this only tracks investigation of workspace code,
    not every run_python call.
    """
    subjects: set[str] = set()
    for call in calls:
        fn = call.get("function") or {}
        if fn.get("name") != "run_python":
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        code = str(args.get("code", ""))
        for match in _IMPORT_SUBJECT_RE.finditer(code):
            name = match.group(1) or match.group(2)
            if name:
                subjects.add(name.split(".")[0])
    return subjects


def _nudge_about_repeated_probing(
    messages: list[dict],
    subject_counts: Counter[str],
    registry: Registry,
    trace: Trace,
    phase: str,
    budget: int,
) -> int:
    """Point at trace_python by name once run_python has probed the same
    workspace subject RUN_PYTHON_PROBE_LIMIT times in this loop.

    Unlike _nudge_about_no_progress, `subject_counts` is never reset by an
    edit landing in between probes -- see RUN_PYTHON_PROBE_LIMIT's comment
    for the live-traced case (probe, guess-edit, probe, guess-edit...) that
    pattern exists to catch. An execute or fix loop not offered trace_python
    gets it added when the nudge fires: gated on the kit, this fired once in
    3,915 bench runs, because the debug trigger is on 0.3% of steps. Any
    other loop (the read-only planner and reviewer) is left alone.
    """
    if budget <= 0:
        return 0
    offered = "trace_python" in registry.names()
    if not offered and not phase.startswith(WIDENING_PHASES):
        return 0
    hit = next((s for s, n in subject_counts.items() if n >= RUN_PYTHON_PROBE_LIMIT), None)
    if hit is None:
        return 0
    if not messages or messages[-1].get("role") != "tool":
        return 0
    if not offered:
        registry.add(tools_named({"trace_python"}))
    # Reset, not delete: three more run_python calls against this same
    # subject earns another nudge, up to `budget` for the whole loop.
    subject_counts[hit] = 0
    nudge = (
        f"\n\n[harness] That is {RUN_PYTHON_PROBE_LIMIT} run_python calls now, each "
        f"adding one more thing to check about {hit!r}. trace_python runs a snippet "
        "once and shows every line it executes plus the locals that changed — likely "
        "faster than another incremental print()."
    )
    messages[-1]["content"] = str(messages[-1].get("content", "")) + nudge
    trace.event("repeated_probe_nudge", phase=phase, subject=hit, streak=RUN_PYTHON_PROBE_LIMIT)
    return 1


def _where_it_went(messages: list[dict]) -> str:
    """What a loop spent its iterations on, as a compact histogram.

    "ran out of iterations" on its own says a budget was reached and nothing
    about why, which leaves the only diagnosis available a manual read of the
    trace. A loop that exhausts has almost always done one of three visible
    things: called one tool over and over, alternated between two, or made
    steady progress and simply needed more room. The counts tell those apart
    at a glance, and repeats are called out because they are the pathological
    case — one traced step called run_tests four times in a row.
    """
    names: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            names.append(((call.get("function") or {}).get("name")) or "?")
    if not names:
        return "no tool calls at all"
    counts = Counter(names)
    listed = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in counts.most_common(6))
    longest, run, previous = 1, 1, None
    for name in names:
        run = run + 1 if name == previous else 1
        longest = max(longest, run)
        previous = name
    if longest >= 3:
        stuck = max(counts, key=lambda n: counts[n])
        listed += f" (up to {longest} calls to {stuck} in a row)"
    return listed


def _last_word_request(tool: str) -> str:
    if tool == "submit_plan":
        # The planner's exploration is the expensive part of planning. On a large
        # repository it ran out of iterations with the right functions found
        # (django-12708: _delete_composed_index, _constraint_names, the
        # introspection code) and the run fell back to a one-step plan holding
        # the goal text and nothing it had learned (DESIGN-LOG §71).
        # "Make the first step settle what is unknown" read as licence to plan only
        # that: django-12708's forced plan was one step, "Add reproduction script",
        # with the root cause named correctly in its draft, and the run finished
        # having fixed nothing (DESIGN-LOG §72).
        return (
            "You have run out of tool-call iterations, so this is your last turn: no "
            "further tools will run. Call submit_plan now with the complete plan for the "
            "goal: the steps that make the change it asks for, naming the files and "
            "functions you found. If something is still uncertain, the executor can check "
            "it inside the step that makes the change; do not submit a plan that only "
            "investigates."
        )
    return (
        "You have run out of tool-call iterations, so this is your last "
        f"turn: no further tools will run. Call {tool} now and "
        "report honestly — what you changed, what you did not get to, and "
        "what was blocking you. If the work is incomplete say so plainly; "
        "an accurate account of an unfinished step is worth more than a "
        "summary that implies it is done."
    )


def _last_word_hold(
    tool_calls: list[dict], gate: Any, mode: str, trace: Trace, phase: str
) -> str | None:
    """What `gate` would say about the last word's calls, or None.

    The last word used to dispatch its terminal call straight past every gate, and
    half of django's research-first planners end there (DESIGN-LOG §88), so the
    research, call-site and findings gates never saw those plans. `mode` "off"
    keeps that; "turn" and "notes" consult the gate and trace what it said.
    """
    if gate is None or mode == "off":
        return None
    for call in tool_calls:
        held = gate.hold(call)
        if held is not None:
            trace.event("last_word_gate", phase=phase, gate=gate.event, mode=mode)
            return held
    return None


def _last_word(
    client,
    model: str,
    messages: list[dict],
    registry: Registry,
    ctx: ToolContext,
    trace: Trace,
    phase: str,
    reasoning_effort: str | None,
    temperature: float | None,
    gate: Any = None,
    gates_mode: str = "off",
) -> None:
    """One final turn, with only the terminal tool on offer.

    Fifteen percent of traced tool loops — 45 of 301 — ended having never
    called their terminal tool, and everything the model had worked out was
    then either discarded or scraped out of its last prose by _salvage. But a
    loop that ran out of iterations has usually not run out of *knowledge*:
    it knows what it changed and what was blocking it, and was simply still
    working when the cap arrived.

    So it is asked, once, for that — with the terminal tool as the only tool
    available, so the answer comes back structured rather than as prose to be
    parsed. One extra call per exhausted loop, on a path that has already
    spent its whole budget, in exchange for a real report instead of a shrug.
    """
    terminal = [t for t in registry.tools() if t.terminal]
    if not terminal:
        return
    messages.append({"role": "user", "content": _last_word_request(terminal[0].name)})
    trace.event("last_word", phase=phase, tool=terminal[0].name)
    only_terminal = Registry(terminal)

    def forced() -> Any:
        reply = client.chat(
            model=model,
            messages=messages,
            tools=only_terminal.schemas(),
            # Required, not merely offered: 32 of 188 traced last words came back as
            # prose, among them a django planner's that left the run a one-step
            # fallback plan (DESIGN-LOG §74). Mercury's endpoint supports it.
            tool_choice={"type": "function", "function": {"name": terminal[0].name}},
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        trace.llm_call(phase, model, reply.usage, reply.finish_reason, reply=reply)
        messages.append(reply.as_message())
        return reply

    reply = forced()
    if not reply.tool_calls and model in getattr(client, "refuses_forced_choice", ()):
        # A model that refuses a forced tool choice may answer in prose; ask once more, plainly.
        messages.append(
            {"role": "user", "content": f"Call {terminal[0].name} now with your answer."}
        )
        reply = forced()
    if not reply.tool_calls:
        return
    held = _last_word_hold(reply.tool_calls, gate, gates_mode, trace, phase)
    if held and gates_mode == "turn":
        # (a) One more forced turn to act on the hold, dispatched without the gate
        # so a plan always comes out: a second hold would leave no turn at all and
        # recreate the one-step fallback the forced call exists to prevent.
        for call in reply.tool_calls:
            messages.append(_held_call_message(call, held, gate.event, trace, phase, -1))
        trace.event("last_word_extra_turn", phase=phase)
        reply = forced()
        if not reply.tool_calls:
            return
    for call in reply.tool_calls:
        messages.append(
            _run_one_call(
                call,
                only_terminal,
                ctx,
                trace,
                phase,
                iteration=-1,
                batch_size=len(reply.tool_calls),
                seen_calls=Counter(),
            )
        )
    if held and gates_mode == "notes" and isinstance(ctx.result, dict):
        # (b) The plan goes through; what the gate would have said reaches the
        # executor as a finding (finalize_plan reads findings with or without
        # research_first).
        note = "planner check not addressed before the plan was forced: " + held[:400]
        ctx.result["findings"] = [*(ctx.result.get("findings") or []), note]
    if ctx.done:
        trace.say("   ran out of iterations; asked it to report what it had")


# A stall costs one extra call (the nudge below) before the loop gives up —
# never a second, unbounded argument about it.
_STALL_TEXT = (
    "You did not call a tool. Continue the work with tool calls, "
    "and call the terminal tool when you are finished."
)


def _stall_nudge(registry: Registry) -> dict:
    """The reply to a turn with no tool call, naming the loop's terminal tool.

    "Continue the work" alone read, to a planner that had just written its plan as text, as
    "go and build it": with no write tools it narrated through ask_user instead, 8-11
    unanswerable questions a run, until the last word. 6 of the 13 recent runs whose first
    planner reply was text did this; 4 of the 131 whose first reply was a call (§108).
    """
    terminal = [t.name for t in registry.tools() if t.terminal]
    if len(terminal) != 1:
        return {"role": "user", "content": _STALL_TEXT}
    return {
        "role": "user",
        "content": (
            f"You did not call a tool. If what you just wrote is your answer, call "
            f"{terminal[0]} with it now; otherwise continue the work with tool calls."
        ),
    }


# Tools whose printed output is the evidence behind a number a run reports.
COMMAND_TOOLS = frozenset({"run_python", "run_bash"})
# Per logged output, in characters: enough for a results table, not a log dump.
COMMAND_LOG_OUTPUT_CHARS = 2_000


def _log_command(ctx: ToolContext, name: str, raw_args: str, output: str) -> None:
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    command = str(args.get("code") or args.get("command") or "")
    with ctx.lock:
        ctx.command_log.append(
            {"tool": name, "command": command[:300], "output": output[:COMMAND_LOG_OUTPUT_CHARS]}
        )


def _run_one_call(
    call: dict,
    registry: Registry,
    ctx: ToolContext,
    trace: Trace,
    phase: str,
    iteration: int,
    batch_size: int,
    seen_calls: Counter[str],
) -> dict:
    """Dispatch one tool call from a batch and return its tool-role message.

    GuardTripped is deliberately not caught here — see the call site. Every
    other path returns a message; nothing here costs a model call, so pulling
    it out of run_tool_loop only changes how the same round trip is read.
    """
    name = (call.get("function") or {}).get("name", "")
    raw_args = (call.get("function") or {}).get("arguments") or "{}"

    fingerprint = _fingerprint(name, raw_args)
    seen_calls[fingerprint] += 1
    repeats = seen_calls[fingerprint]

    if repeats >= REPEAT_HARD_LIMIT:
        trace.event("repeated_call_refused", phase=phase, tool=name, times=repeats)
        return {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": name,
            "content": (
                f"error: refusing to run {name} a {repeats}th time with "
                "identical arguments. The result will not change. Do "
                "something different: read the file you are editing, try "
                "another approach, or call the terminal tool and report "
                "what is blocking you."
            ),
        }

    # After the hard-limit refusal, never before it. A replay that
    # short-circuits the refusal makes it unreachable for exactly the tools
    # it matters most for — run_tests is 434 of 766 traced repeats — and a
    # model stuck on one command then gets unlimited cheap replays and spins
    # to max_iters, which is the loop the refusal exists to break.
    state = _replay_state(ctx, name)
    cached = _cached_result(ctx, name, fingerprint, state, trace, phase, repeats)
    if cached is not None:
        return {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "name": name,
            "content": truncate(cached, MAX_TOOL_RESULT_CHARS),
        }

    try:
        output = _dispatch(
            registry, ctx, name, raw_args, trace, phase, iteration=iteration, batch_size=batch_size
        )
        if output.startswith("error:"):
            output += ctx.guard.record_error() or ""
    except GuardTripped as exc:
        trace.event("guard_tripped", phase=phase, reason=str(exc), errors=ctx.guard.step_errors)
        trace.say(f"  ! {exc}")
        raise

    # Keyed on the state observed *before* dispatch, and stored only if the
    # workspace still looks the same afterwards. Recording against the after
    # state attributed the output to a state it was not computed against:
    # check_code(fix=true) writes, and under step concurrency a sibling step
    # editing a file mid-run would have had its change folded into the key,
    # so the next identical call replayed a pass against a tree that had
    # since broken.
    if not output.startswith("error:") and state is not None:
        _remember_result(ctx, name, fingerprint, output, state, phase)
    if name in COMMAND_TOOLS and not output.startswith("error:"):
        _log_command(ctx, name, raw_args, output)

    if repeats >= REPEAT_LIMIT:
        trace.event("repeated_call", phase=phase, tool=name, times=repeats)
        output += (
            f"\n\n[harness] That is call {repeats} with identical arguments. "
            "Repeating it will not change the result. Read the file, or try a "
            "different approach, or call the terminal tool and report the problem."
        )

    if len(output) > MAX_TOOL_RESULT_CHARS:
        output = truncate(output, MAX_TOOL_RESULT_CHARS)
    return {"role": "tool", "tool_call_id": call.get("id", ""), "name": name, "content": output}


# Tools that only read the workspace and report on it. Running one of these
# twice over byte-identical files cannot produce a different answer, so the
# second run is pure wall time. Deliberately not a general-purpose cache:
# anything that writes, fetches, or asks a person is excluded, and so is
# anything whose result depends on state this fingerprint does not cover.
_REPLAYABLE = frozenset(
    {
        "run_tests",
        "check_code",
        "check_complexity",
        "lint_project",
        "test_project",
        "validate_data_file",
    }
)


def _replay_enabled(ctx: ToolContext) -> bool:
    """Whether this run wants identical read-only checks replayed.

    Mirrors Settings.replay_identical_checks, carried on the context the
    same way `autolint` and `narrate` are. Defaults to the shipped
    behaviour for a bare ToolContext built directly in a test.
    """
    return bool(getattr(ctx, "replay_checks", True))


def _replay_state(ctx: ToolContext, name: str) -> str | None:
    """A snapshot of everything the replay cache treats as "the workspace".

    `None` when this call is not replayable at all, which is also what turns
    the feature off. Computed once per call and passed to both the lookup and
    the store, so the two cannot disagree and the walk is not paid twice.

    A `stat()` manifest of every non-ignored file, not the bytes of
    `ctx.touched`. The touched-file digest this used to hash could not see a
    change it had no entry for: `run_python` writes files without marking
    them touched, and an empty touched list hashes to a constant, so every
    file in the workspace was unmonitored at the start of a run. It also
    cannot see a virtualenv repair — `walk_files` prunes `.venv` — which is
    the case that actively misleads: a `run_tests` that failed with
    ModuleNotFoundError would be replayed after `envfail.repair` fixed the
    environment, telling the model in so many words that nothing had changed
    and re-running would teach it nothing.
    """
    if name not in _REPLAYABLE or not _replay_enabled(ctx):
        return None
    from .tools.base import workspace_state

    return workspace_state(ctx.workspace.root)


def _cached_result(
    ctx: ToolContext,
    name: str,
    fingerprint: str,
    state: str | None,
    trace: Trace,
    phase: str,
    repeats: int,
) -> str | None:
    """A previous identical call's output, when nothing it read has changed.

    The reviewer in one traced run called `run_tests` on the same file four
    times in a single phase — twice at 22 seconds a run, against a workspace
    it had not edited between them, having already been told the suite was
    green. The harness noticed (it appends a "that is call N with identical
    arguments" nudge) but paid for each run first. Noticing after the fact is
    not worth much when the fact cost 22 seconds.

    The equality relied on is exact, not heuristic: same phase, same tool,
    same arguments, and every file in the workspace unchanged in mtime and
    size. The phase is part of the key because a tool name is bound to a
    different function in different registries — execute's `check_code`
    takes `fix=true` and writes, review's read-only binding cannot — so a
    result recorded under one must not be served under the other.
    """
    if repeats < 2 or state is None or not ctx.replay:
        return None
    recorded = ctx.replay.get((phase, fingerprint, state))
    if recorded is None:
        return None
    trace.event("replayed_call", phase=phase, tool=name, times=repeats)
    return recorded + (
        "\n\n[harness] Identical call, and nothing in the workspace has "
        "changed since it last ran, so this is the recorded result rather "
        "than a second run. Re-running it will not tell you anything new."
    )


def _remember_result(
    ctx: ToolContext, name: str, fingerprint: str, output: str, state: str, phase: str
) -> None:
    """Record a replayable tool's output against the state that produced it.

    Stored only if the workspace still matches what it was before the call —
    a tool that wrote something (check_code with `fix`), or a sibling step
    that edited a file while this one ran, means the output describes a state
    that no longer exists and is not safe to replay against either one.
    """
    if _replay_state(ctx, name) != state:
        return
    ctx.replay[(phase, fingerprint, state)] = truncate(output, MAX_TOOL_RESULT_CHARS)


def _exhausted_result(
    ctx: ToolContext,
    messages: list[dict],
    phase: str,
    salvage: bool,
    trace: Trace,
    last_word: Callable[[], None] | None = None,
) -> tuple[Any, list[dict]]:
    """What run_tool_loop returns when max_iters ran out without a terminal call."""
    if ctx.result is not None or not salvage:
        return ctx.result, messages
    # Ask before scraping: a structured report the model wrote on purpose
    # beats the last paragraph of prose it happened to leave behind.
    if last_word is not None:
        last_word()
    if ctx.done:
        return ctx.result, messages
    rescued = _salvage(messages)
    if not rescued:
        return ctx.result, messages
    trace.event("salvaged_result", phase=phase, chars=len(rescued))
    trace.say("   ! ran out of iterations without finishing; keeping the analysis it had reached")
    return {
        "summary": rescued,
        "verified": False,
        "notes": (
            "INCOMPLETE: the model ran out of tool-call iterations "
            "before calling the terminal tool. This is its last "
            "reasoning, not a confirmed result."
        ),
        "incomplete": True,
    }, messages


# Full-width stops, so a non-ASCII salvage is trimmed at a sentence rather
# than kept whole because no ASCII period was ever going to appear in it.
_CJK_ENDS = ("\u3002", "\uff01", "\uff1f")


def _trim_to_last_sentence(text: str) -> str:
    """Drop a trailing half-sentence, unless that would leave nothing.

    A salvaged message is often the one the completion cap cut off, so its
    last line stops mid-clause. One traced run surfaced, as a step's entire
    reported result, "... aren't being printed because the
    `print_feature_importance_table` and" — and that fragment was then
    written verbatim into the workspace's memory file as the durable record
    of what the step built. A cut-off thought reads as a thought; the
    sentences before it are what the model actually established.
    """
    stripped = text.strip()
    # Deliberately not ` or ) or ```: the traced fragment this exists to
    # catch ended "...because the `print_feature_importance_table`", which a
    # backtick terminator would have waved straight through.
    if stripped.endswith((".", "!", "?", *_CJK_ENDS)):
        return stripped
    cut = max(stripped.rfind(mark) for mark in (". ", ".\n", "! ", "? ", ":\n", *_CJK_ENDS))
    trimmed = stripped[: cut + 1].strip() if cut > 0 else ""
    # Only prefer the trimmed form when enough survives to still say something.
    return trimmed if len(trimmed) >= 40 else stripped


def _salvage(messages: list[dict]) -> str:
    """The most recent substantive assistant text in a loop that never finished.

    Losing this was a real failure: a fix phase once worked out exactly why two
    tests failed, wrote the diagnosis out, hit its iteration cap before calling
    finish_step, and the harness discarded the whole thing and reported that
    nothing had happened. A model that ran out of turns mid-thought still knows
    more than the empty result that replaced it — passed on clearly labelled as
    unconfirmed, so nothing downstream mistakes it for a verified outcome.

    Which assistant turn gets taken matters more than it looks. A turn that
    carries tool calls is the model thinking *out loud on its way to an
    action* — "Now I see the issue ... let me fix it" — whereas a turn with
    no tool calls is the model having stopped to state something. Taking the
    most recent text of either kind surfaced the former: a traced run
    reported a step's result as a mid-sentence narration of an edit it was
    about to make, which read to the person who asked for it as a thinking
    step presented as a deliverable. So a tool-free turn wins outright, and
    interleaved reasoning is the fallback rather than the default.
    """
    reasoning = ""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        # Skip the bare "" content that accompanies a pure tool-call turn.
        if len(content) < 40:
            continue
        if not message.get("tool_calls"):
            return _trim_to_last_sentence(truncate(content, 4_000))
        if not reasoning:
            reasoning = content
    return _trim_to_last_sentence(truncate(reasoning, 4_000)) if reasoning else ""


def _fingerprint(name: str, raw_args: str) -> str:
    """Stable identity for a call, insensitive to key order in the arguments."""
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        normalised = json.dumps(parsed, sort_keys=True, default=str)
    except (json.JSONDecodeError, TypeError):
        normalised = str(raw_args)
    digest = hashlib.sha1(f"{name}:{normalised}".encode()).hexdigest()
    return digest[:16]


def _dispatch(
    registry: Registry,
    ctx: ToolContext,
    name: str,
    raw_args: str,
    trace: Trace,
    phase: str,
    iteration: int = 0,
    batch_size: int = 1,
) -> str:
    tool = registry.get(name)
    if tool is None:
        return f"error: unknown tool {name!r}. Available: {', '.join(registry.names())}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError as exc:
        return f"error: arguments were not valid JSON ({exc}). Re-send them as a JSON object."
    if not isinstance(args, dict):
        return "error: arguments must be a JSON object."
    encoded = _encoded_keys(tool, args)
    mistyped = _encoded_keys(tool, args, _SCALAR_TYPES)
    args, parse_errors = _coerce_args(tool, args)

    trace.event(
        "tool_call",
        phase=phase,
        tool=name,
        iteration=iteration,
        batch_size=batch_size,
        args={k: _abbrev(v) for k, v in args.items()},
        fp=_fingerprint(name, raw_args),
        **({"encoded": encoded} if encoded else {}),
        **({"mistyped": mistyped} if mistyped else {}),
    )

    if parse_errors:
        # A declared array/object argument was sent as a JSON string (the model
        # double-encoding it) and that string itself failed to parse — most often
        # an unescaped newline or quote inside a batched edit's old_string/new_string.
        # Surfacing json.loads's own message here, instead of falling through to the
        # tool's generic "wrong type" error, is what lets the model fix its own
        # escaping on retry rather than resending the same malformed payload blind
        # (a real trace: edit_files failed silently twice in a row this way before
        # the model gave up and switched to single edit_file calls).
        # `raw` is the untruncated argument string: the tool_call event above
        # abbreviates every value to 200 characters, which hid the actual defect
        # in all 657 traced submit_plan failures (DESIGN-LOG §63).
        trace.event(
            "tool_arg_parse_error", phase=phase, tool=name, errors=parse_errors, raw=raw_args
        )
        return (
            "error: "
            + "; ".join(parse_errors)
            + "\nRe-send this call with valid JSON: every newline inside a string value "
            "must be written as the two characters backslash-n, not a literal line break, "
            "and every embedded double-quote must be backslash-escaped."
        )

    if ctx.narrate:
        label = _narration(name, args)
        if label:
            trace.say(f"   … {label}")
    started = time.monotonic()
    try:
        result = tool.fn(ctx, **args)
    except TypeError as exc:
        return f"error: bad arguments for {name}: {exc}"
    except Exception as exc:  # tools must never crash the run
        trace.event("tool_error", phase=phase, tool=name, error=repr(exc))
        return f"error: {type(exc).__name__}: {exc}"
    # How long the tool itself took, so `harness trace sql` can say which tools cost
    # more than the model (DESIGN-LOG §91, review Pass 3). Keyed to its tool_call by fp.
    trace.event(
        "tool_done",
        phase=phase,
        tool=name,
        fp=_fingerprint(name, raw_args),
        latency_s=round(time.monotonic() - started, 3),
        ok=not (isinstance(result, str) and result.startswith("error:")),
    )
    # A tool that fails by returning "error: ..." rather than raising — the
    # ordinary path, since tools must never crash the run — left no trace of
    # *why* anywhere: only the (200-char-truncated) input args were ever
    # logged, never the result. That gap is what made a real failure
    # (edit_files silently returning 0/N-applied) something this session had
    # to reconstruct from indirect evidence — timing, narration text, the
    # absence of a following edit_file event — instead of just reading what
    # the tool actually said. Logged only on the failure path: the common
    # case already has its own event kinds (edit_file, write_file, ...) that
    # say what happened without needing the return string duplicated too.
    if isinstance(result, str) and result.startswith("error:"):
        trace.event("tool_call_failed", phase=phase, tool=name, result=result[:2_000])
    return result


_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "on"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "off", ""})


def _coerce_args(tool: Tool, args: dict) -> tuple[dict, list[str]]:
    """Coerce string arguments into the types the tool's own schema declares.

    Models routinely send JSON strings where the schema says boolean or
    integer — a real trace has `run_tests(verbose="true")`. Python treats
    every non-empty string as truthy, so `"false"` means True, and that is
    not a cosmetic problem: `edit_file(replace_all="false")` would replace
    every occurrence in the file instead of one, `check_code(fix="false")`
    would rewrite files the caller did not ask to have rewritten, and
    `finish_step(verified="false")` would mark a step verified that nothing
    had verified.

    Arrays have the same problem and a nastier failure. A traced model sent
    `read_files(paths='["game.py", "main.py"]')` — the list as a JSON string —
    and since a string is iterable, nothing raised: read_files looked for one
    file literally named `["game.py", "main.py"]`, and check_code iterated the
    string character by character and reported thirteen fabricated "E902 No
    such file" diagnostics for `[`, `"`, `d`, `m` and friends, which the model
    then set about fixing. Silent nonsense is worse than a type error, and it
    lands on exactly the plural tools batching depends on.

    Anything that does not look like the declared type is passed through
    untouched, so a genuinely bad argument still reaches the tool and gets
    the ordinary "bad arguments" error rather than being silently reshaped —
    *except* a declared array/object value that looks like JSON (starts with
    `[`/`{`) but fails to parse: that specific shape means the model tried to
    double-encode a batch of edits and mis-escaped something inside it (a real
    trace: `edit_files` failed silently, twice in a row, on a multi-line
    old_string with an unescaped newline). Those are returned as `errors` —
    the caller surfaces json.loads's own message instead of the tool's generic
    "wrong type" error, so the model can see exactly what broke and fix it on
    retry rather than resending the same malformed payload blind.

    Returns (coerced_args, errors) — errors is empty in the ordinary case.
    """
    props = (tool.parameters or {}).get("properties") or {}
    coerced = dict(args)
    errors: list[str] = []
    for key, value in args.items():
        if not isinstance(value, str):
            continue
        spec = props.get(key) or {}
        fixed, err = _coerce_one(value, spec.get("type"), (spec.get("items") or {}).get("type"))
        if err is not None and (tool.name, key) in _NOTES_ARGUMENTS:
            coerced[key] = _salvage_notes(value)
        elif err is not None:
            errors.append(f"{key}: {err}")
        elif fixed is not _UNCHANGED:
            coerced[key] = fixed
    return coerced, errors


# Arguments that are free-text notes, where a list that will not parse is
# still worth more read as text than sent back as an error: submit_plan's
# draft failed to parse in 510 of 657 traced plan rejections, each costing a
# whole planning round trip (DESIGN-LOG §63). Not for anything a tool acts
# on (paths, edits, steps), where reshaping a bad value is silent nonsense.
_NOTES_ARGUMENTS = frozenset({("submit_plan", "draft")})


def _salvage_notes(value: str) -> list[str]:
    """A malformed JSON list of strings, read as the notes it was meant to be."""
    inner = value.strip().removeprefix("[").removesuffix("]").strip()
    parts = re.split(r'"\s*,\s*"', inner)
    notes = [p.strip().strip('"').strip() for p in parts]
    return [n for n in notes if n] or [value]


_STRUCTURED_TYPES = ("array", "object")
_SCALAR_TYPES = ("boolean", "integer", "number")


def _encoded_keys(
    tool: Tool, args: dict, declared: tuple[str, ...] = _STRUCTURED_TYPES
) -> list[str]:
    """Arguments of a `declared` type that the model sent as a string instead.

    Recorded on the tool_call event, arrays/objects as `encoded` and
    booleans/numbers as `mistyped`: both happen far more often than a parse
    failure, so they are what shows whether a strict schema changes how
    Mercury encodes arguments (DESIGN-LOG §63).
    """
    props = (tool.parameters or {}).get("properties") or {}
    return sorted(
        key
        for key, value in args.items()
        if isinstance(value, str) and (props.get(key) or {}).get("type") in declared
    )


# `False` and `0` are valid results, so "leave it alone" cannot be None.
_UNCHANGED = object()


def _coerce_one(
    value: str, declared: str | None, item_type: str | None = None
) -> tuple[Any, str | None]:
    """The string in the type its schema declares, or _UNCHANGED. Second element
    is a human-readable parse error, non-None only when a declared array/object
    value looked like JSON but failed to parse — see _coerce_args."""
    if declared == "boolean":
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True, None
        return (False, None) if lowered in _FALSE_WORDS else (_UNCHANGED, None)
    if declared in ("integer", "number"):
        try:
            return (int(value) if declared == "integer" else float(value)), None
        except ValueError:
            return _UNCHANGED, None
    if declared in ("array", "object"):
        return _parse_structured(value, declared, item_type)
    return _UNCHANGED, None


def _parse_structured(
    value: str, declared: str, item_type: str | None = None
) -> tuple[Any, str | None]:
    """A JSON array/object sent as a string, or (_UNCHANGED, None) to leave it
    alone, or (_UNCHANGED, <message>) when it looked like JSON and failed to parse.

    A bare string where an array *of strings* is declared becomes a one-element
    list: `paths="game.py"` plainly means `["game.py"]`. Not so for an array of
    objects — a string is never a valid object, and wrapping it produced a list
    whose single element then had `.get()` called on it. That crashed three
    real benchmark runs with "AttributeError: 'str' object has no attribute
    'get'" when a model sent submit_plan's `steps` as prose. Leaving it
    untouched (with no error) gets the tool's own "bad arguments" error
    instead, which is recoverable — that path is for a string that was never
    meant to be JSON at all, e.g. a bare filename.

    A string that *does* start with `[`/`{` but fails `json.loads` is a
    different, more diagnosable case: the model was trying to send structured
    data and got the encoding wrong, almost always an unescaped newline or
    quote inside a multi-line string value. That gets a real error message
    instead of silent pass-through, since the tool's own generic type error
    gives the model no way to know *why* its retry will fail the same way.
    """
    text = value.strip()
    if text.startswith(("[", "{")):
        wanted = list if declared == "array" else dict
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            parsed = _repaired(text, wanted)
            if parsed is None:
                return _UNCHANGED, (f"looked like a JSON {declared} but failed to parse: {exc}")
        return (parsed, None) if isinstance(parsed, wanted) else (_UNCHANGED, None)
    if declared == "array" and text and item_type in (None, "string"):
        return [value], None
    return _UNCHANGED, None


_TRAILING_COMMA = re.compile(r",\s*([\]}])")


def _repaired(text: str, wanted: type) -> Any:
    """`text` parsed after the repairs that cannot change what it says, or None.

    Each repair only accepts a result of the declared type: a trailing comma
    before a closing bracket; a complete value followed by nothing but closing
    brackets and punctuation ("Extra data"); a Python-style literal (single
    quotes, True/None) with no double quote anywhere, since Python joins
    adjacent strings and would turn `"a" "b"` into one "ab". Anything that
    would need guessing at content is left to the caller's error.
    """
    candidates: list[Any] = []
    unstrailed = _TRAILING_COMMA.sub(r"\1", text)
    with contextlib.suppress(json.JSONDecodeError):
        candidates.append(json.loads(unstrailed))
    with contextlib.suppress(json.JSONDecodeError):
        head, end = json.JSONDecoder().raw_decode(unstrailed)
        if not unstrailed[end:].strip(" \t\n]},."):
            candidates.append(head)
    if '"' not in text:
        with contextlib.suppress(ValueError, SyntaxError, MemoryError, RecursionError):
            candidates.append(ast.literal_eval(text))
    return next((c for c in candidates if isinstance(c, wanted)), None)


def _abbrev(value: Any, limit: int = 200) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit} chars)"


def _narrate_edit_files(args: dict) -> str:
    edits = args.get("edits") or []
    n = len(edits) if isinstance(edits, list) else 0
    return f"editing {n} file(s)"


def _narrate_check_code(args: dict) -> str:
    return "checking code" + (" (with types)" if args.get("types") else "")


def _narrate_git_commit(args: dict) -> str:
    first_line = str(args.get("message", "")).splitlines()[:1]
    return f"committing: {clip(first_line[0] if first_line else '', 50)}"


# One formatter per tool worth narrating, keyed by name. A flat chain of
# `if name == ...` here scored the same as genuinely nested logic under
# cognitive complexity despite being simple dispatch — the same "replace a
# chain of elif on one value with a dict lookup" the complexity skill
# recommends, applied to itself. Behaviourally identical: same strings, same
# lookup cost either way (dict lookup is if anything cheaper than a 17-way
# if-chain), no tool call gains or loses its narration.
_NARRATORS: dict[str, Callable[[dict], str]] = {
    "write_file": lambda a: f"writing {a.get('path', '?')}",
    "edit_file": lambda a: f"editing {a.get('path', '?')}",
    "edit_files": _narrate_edit_files,
    "edit_blocks": lambda a: f"editing {len(set(edited_paths('edit_blocks', a)))} file(s)",
    "batch_edits_to_file": lambda a: f"editing {a.get('path', '?')}",
    "move_file": lambda a: f"moving {a.get('src', '?')} -> {a.get('dst', '?')}",
    "delete_file": lambda a: f"deleting {a.get('path', '?')}",
    "run_bash": lambda a: f"running: {clip(str(a.get('command', '')), 60)}",
    "run_python": lambda a: "running a Python check",
    "run_tests": lambda a: "running tests",
    "check_code": _narrate_check_code,
    "git_add": lambda a: "staging changes",
    "git_commit": _narrate_git_commit,
    "fetch_url": lambda a: f"fetching {a.get('url', '?')}",
    "web_search": lambda a: f"searching: {a.get('query', '?')}",
    "inspect_module": lambda a: f"inspecting {a.get('module', '?')}",
    "pypi_info": lambda a: f"looking up {a.get('package', '?')} on PyPI",
}


def _narration(name: str, args: dict) -> str | None:
    """A short "here's what's about to happen" line for a tool call, so a
    human watching can follow along step by step rather than seeing nothing
    between a step starting and it finishing. Only for calls worth
    narrating — read-only exploration (read_file, grep, list_dir, ...) and
    the terminal tools (finish_step, submit_plan, submit_findings, which
    already get their own narration in schedule.py/loop.py) return None.
    """
    formatter = _NARRATORS.get(name)
    return formatter(args) if formatter else None


def _tool_exchanges(tail: list[dict]) -> list[list[dict]]:
    """Group a flat message list into (assistant [+ its tool results]) units
    -- the smallest piece compaction can drop without orphaning a tool result
    from the call it answers."""
    exchanges: list[list[dict]] = []
    for message in tail:
        if message.get("role") != "tool" or not exchanges:
            exchanges.append([message])
        else:
            exchanges[-1].append(message)
    return exchanges


def _exchange_text(exchange: list[dict]) -> str:
    parts = []
    for message in exchange:
        content = message.get("content") or ""
        if isinstance(content, str):
            parts.append(content)
        for call in message.get("tool_calls") or []:
            parts.append((call.get("function") or {}).get("arguments", ""))
    return " ".join(parts)[:3_000]


def _current_context(messages: list[dict]) -> str:
    """What "still relevant" is measured against -- the most recent thing the
    model actually said, or failing that, the task it was given."""
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(message.get("content") or "").strip():
            return str(message["content"])
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _compact(messages: list[dict], budget: int, trace: Trace, phase: str, judge=None) -> list[dict]:
    """Drop tool exchanges no longer worth keeping, freeing room for the step
    to continue. With a judge (agent/relevance.py) available, that means the
    least *relevant* exchanges to what the model is doing right now -- not
    simply the oldest, which can drop a still-relevant early read while
    keeping a stale, unrelated recent one. Without a judge (none configured,
    or scoring itself failed), falls back to the original behaviour: drop
    the oldest first.

    Chunking is supposed to prevent this firing at all; when it does, a step
    was too big, which the trace will show either way.
    """
    trace.event("compacted", phase=phase, before_tokens=messages_tokens(messages))
    head, tail = messages[:2], messages[2:]

    if judge is not None and tail:
        try:
            tail = _compact_by_relevance(messages, head, tail, budget, judge)
        except Exception as exc:
            trace.event(
                "compact_relevance_failed", phase=phase, error=f"{type(exc).__name__}: {exc}"
            )
            tail = _compact_fifo(head, tail, budget)
    else:
        tail = _compact_fifo(head, tail, budget)

    elision = {
        "role": "user",
        "content": "[earlier tool output elided to stay inside the context budget]",
    }
    compacted = [*head, elision, *tail]
    judge_usage = {}
    if judge is not None:
        # Only the attributes a concrete judge actually exposes (see
        # JevJudge/LLMJudge's own __init__) -- HeuristicJudge has none of
        # these, and the RelevanceJudge Protocol itself declares no usage
        # contract, so this reads what's there rather than assuming a shape.
        for attr in ("calls", "cost_usd", "input_tokens", "output_tokens"):
            value = getattr(judge, attr, None)
            if value:
                judge_usage[attr] = value
    trace.event(
        "compacted_done", phase=phase, after_tokens=messages_tokens(compacted), **judge_usage
    )
    return compacted


def _compact_fifo(head: list[dict], tail: list[dict], budget: int) -> list[dict]:
    while tail and messages_tokens(head + tail) > budget and len(tail) > 6:
        tail.pop(0)
        # A tool result cannot be orphaned from its assistant call.
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
    return tail


def _compact_by_relevance(
    messages: list[dict], head: list[dict], tail: list[dict], budget: int, judge
) -> list[dict]:
    exchanges = _tool_exchanges(tail)
    # The last exchange is almost always the one the model is mid-thought
    # about; never a candidate for dropping regardless of score.
    protected, scoreable = exchanges[-1:], exchanges[:-1]
    if not scoreable:
        return tail
    # Floor is on *exchanges*, not messages (each is >= 2 messages, unlike
    # the old FIFO path's message-counted floor) -- always keep at least one
    # scoreable exchange alongside the always-protected most recent one, so
    # compaction can still free real room even when there are only a
    # handful of exchanges to begin with.
    floor = max(1, len(scoreable) // 5)
    if len(scoreable) <= floor:
        # Already at the floor -- no score, however it comes back, could
        # cause anything to actually be dropped below. A budget too tight
        # for even the fixed head to fit under would otherwise pay for a
        # judge call on every single tool-loop iteration for no effect at
        # all, live-traced at 13 wasted calls in one bench step.
        return tail
    context = _current_context(messages)
    scores = judge.score(context, [_exchange_text(e) for e in scoreable])
    # Lowest-scored first, so those are the ones popped when still over
    # budget -- ties broken by age (older first), same bias FIFO already had.
    order = sorted(range(len(scoreable)), key=lambda i: (scores[i], i))
    kept = {i for i in range(len(scoreable))}
    flat = [m for e in scoreable for m in e] + [m for e in protected for m in e]
    for i in order:
        if messages_tokens(head + flat) <= budget or len(kept) <= floor:
            break
        kept.discard(i)
        flat = [m for j, e in enumerate(scoreable) if j in kept for m in e] + [
            m for e in protected for m in e
        ]
    return flat
