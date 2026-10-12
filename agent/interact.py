"""Asking the human a clarifying question mid-run.

Approval (approval.py) gates whether a risky action runs. This is a different
axis: the model isn't unsafe, it's uncertain — an ambiguous goal, a choice
between two reasonable designs, a spec that doesn't say what should happen on
some edge case. Guessing silently is how an agent builds the wrong thing
quickly; asking is only useful if there's someone to ask and somewhere to put
the question, which is why this is pluggable exactly like Trace.on_event —
the plain CLI prompts on stdin, the TUI shows a modal, and a non-interactive
run (bench, no TTY, or the user opted out) gets a clear fallback that tells
the model to proceed on its own judgment rather than block forever.
"""

from __future__ import annotations

import inspect
import json
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable

# (question, options, recommended) -> answer. The last two are optional in
# practice; a caller that only handles the first two still satisfies this
# because Interactor passes them positionally only when they are set.
Asker = Callable[..., str]

# One human, one terminal, one question at a time. With --step-concurrency > 1
# several worker threads can reach a question or an approval prompt at the same
# moment, and every way of presenting one assumes it is the only one in flight:
# the stdin path would have two input() calls racing for the same keystrokes,
# and the TUI's bottom bar holds a single "who is waiting" slot that a second
# asker would overwrite, stranding the first thread forever.
#
# Shared between Interactor and Approver rather than held per-object, because
# the collision is over the terminal itself, which they share — a clarifying
# question and an approval prompt racing each other is the same bug as two
# approvals racing.
HUMAN_LOCK = threading.RLock()


# Questions asked during the run (the planner's or executor's ask_user), after clarify's
# batch, which has its own budget (CLARIFY_QUESTIONS). Separate since 2026-10-01: clarify used to
# share this budget, spent all of it on generic questions, and the platformer app's planner's one
# question that mattered was refused (DESIGN-LOG §98, §101). No real run has needed more than 3
# later questions; the one that asked 4 asked "are you available to answer questions?".
#
# A run may ask this many questions before the harness stops relaying them.
# One real trace shows a planner asking seven in a row and never producing a
# plan — and most of them were not questions at all but notes to itself
# ("Run the tests to see failures"), each one blocking on a human who had
# nothing useful to answer. A cap turns a pathological loop into at worst a
# few wasted questions.
MAX_QUESTIONS_PER_RUN = 3
# Clarify's own budget: its questions are asked together, before planning (DESIGN-LOG §101).
CLARIFY_QUESTIONS = 5


class Interactor:
    def __init__(
        self,
        ask: Asker | None = None,
        enabled: bool = True,
        max_questions: int = MAX_QUESTIONS_PER_RUN,
        ask_many: Callable[[list[tuple[str, list[str] | None, str | None]]], list[str]]
        | None = None,
    ):
        # ask=None means "use the default stdin prompt when there is a TTY".
        # A caller (the TUI) that wants full control passes its own callable.
        self._ask = ask
        self.enabled = enabled
        self.max_questions = max_questions
        self.asked = 0
        self.answered = 0
        # A caller that can show several questions at once (the TUI's form) passes this:
        # it takes [(question, options, recommended), ...] and returns one answer each.
        self._ask_many = ask_many
        self.clarify_asked = 0
        # Questions that reached nobody (questions off, no TTY). See _unanswerable.
        self.unanswered = 0

    def ask(
        self, question: str, options: list[str] | None = None, recommended: str | None = None
    ) -> str:
        self.asked += 1
        if not self.enabled:
            return self._unanswerable("clarifying questions are turned off for this run")
        if self.asked > self.max_questions:
            return self._over_limit(self.max_questions)
        with HUMAN_LOCK:
            return self._ask_locked(question, options, recommended)

    def confirm(self, question: str, options: list[str]) -> str:
        """A harness check for the person (agent/chatflow.py), not one of the model's questions:
        it draws on no question budget. The first option is the recommended one, so Enter
        accepts it."""
        if not self.enabled:
            return self._no_answer("questions are turned off for this run")
        with HUMAN_LOCK:
            return self._ask_locked(question, options, options[0] if options else None)

    def ask_many(self, questions: list[tuple[str, list[str] | None]]) -> list[str]:
        """Clarify's questions, asked together and answered in one go, one answer each.

        They draw on clarify's own budget (CLARIFY_QUESTIONS), never on the run's later
        questions. A caller with a form (`ask_many=` to the constructor) shows them all at once;
        otherwise they are asked one after another under one hold of the terminal, so nothing
        else can interleave (DESIGN-LOG §101).
        """
        room = max(0, CLARIFY_QUESTIONS - self.clarify_asked)
        asked, refused = questions[:room], questions[room:]
        self.clarify_asked += len(asked)
        if not self.enabled:
            reason = "clarifying questions are turned off for this run"
            return [self._no_answer(reason) for _ in questions]
        with HUMAN_LOCK:
            if self._ask_many is not None and asked:
                answers = list(self._ask_many([(q, o, None) for q, o in asked]))
                answers += [""] * (len(asked) - len(answers))
                # A skipped question comes back as the "(no answer given)" sentinel.
                self.answered += sum(1 for a in answers if a and not a.startswith("(no answer"))
            else:
                answers = [self._ask_locked(q, o) for q, o in asked]
        return answers + [self._over_limit(CLARIFY_QUESTIONS) for _ in refused]

    def _ask_locked(
        self, question: str, options: list[str] | None, recommended: str | None = None
    ) -> str:
        if self._ask is not None:
            answer = self._call_asker(question, options, recommended)
            self.answered += 1
            return answer
        if not sys.stdin.isatty():
            return self._unanswerable("no TTY is available to ask")

        print(f"\n  agent has a question:\n    {question}")
        if options:
            for i, option in enumerate(options, 1):
                mark = "  (recommended)" if option == recommended else ""
                print(f"    {i}. {option}{mark}")
            hint = (
                "  answer (number, free text, or Enter for the recommendation): "
                if recommended
                else "  answer (number or free text): "
            )
            raw = input(hint).strip()
            if not raw and recommended:
                self.answered += 1
                return recommended
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                self.answered += 1
                return options[int(raw) - 1]
            self.answered += 1
            return raw or self._no_answer("empty answer")
        answer = input("  answer: ").strip()
        self.answered += 1
        return answer or self._no_answer("empty answer")

    def _call_asker(self, question: str, options: list[str] | None, recommended: str | None) -> str:
        """Call the supplied asker, passing `recommended` only if it takes it.

        `recommended` was added after several callers already existed, and an
        asker written against the two-argument shape is still a perfectly
        valid one — the TUI needs the third argument, a test double or an
        embedding caller does not. Checked by signature rather than by
        catching TypeError, because a TypeError raised *inside* the callback
        would otherwise be indistinguishable from a arity mismatch and the
        retry would ask the human the same question twice.
        """
        assert self._ask is not None
        try:
            takes = len(inspect.signature(self._ask).parameters)
        except (TypeError, ValueError):
            takes = 3
        if takes >= 3:
            return self._ask(question, options, recommended)
        return self._ask(question, options)

    @staticmethod
    def _over_limit(limit: int) -> str:
        """The refusal once a person has already answered this run's questions. Unlike
        `_no_answer`, it does not steer toward the conservative option: a person who asked
        for a change, then answered questions about it, wants the change. "Prefer the
        conservative option" kept the platformer app's flagpole off screen (DESIGN-LOG §98).
        Runs with nobody to ask (benchmarks, questions off) keep `_no_answer`'s wording."""
        return (
            f"(no answer available: this run has already asked its {limit}-question limit. "
            "Decide it yourself and state the decision you made in your summary or notes. "
            "Choose the option that delivers what the goal asks for; among those, the most "
            "easily reversed.)"
        )

    def _unanswerable(self, reason: str) -> str:
        """The reply when nobody can answer: the usual fallback the first time, then a plain
        statement that asking again changes nothing.

        Each repeat used to get the same "Proceed on your own best judgment", which a planner
        that had written its plan as text read as permission to keep narrating: 8-11 questions
        in a row, all unanswered, in 6 recent bench runs (DESIGN-LOG §108)."""
        self.unanswered += 1
        if self.unanswered == 1:
            return self._no_answer(reason)
        return (
            f"(no answer available, and none will come: {reason}, and this is question "
            f"{self.unanswered}. Do not ask again. Decide it yourself, note the assumption, "
            "and finish your work with the tool that ends it.)"
        )

    @staticmethod
    def _no_answer(reason: str) -> str:
        return (
            f"(no answer available: {reason}. Proceed on your own best judgment, "
            "state the assumption you made in your summary or notes, and prefer "
            "the more conservative or easily-reversed option when genuinely unsure.)"
        )


# Long enough for a model call behind the command (the simulated user is one), short
# enough that a hung command costs one question rather than the run.
ANSWER_COMMAND_TIMEOUT_S = 120.0


def command_asker(command: str, timeout_s: float = ANSWER_COMMAND_TIMEOUT_S) -> Asker:
    """An asker that puts each question to `command` instead of a person.

    The question goes to the command's stdin as JSON and its stdout is the
    answer. A failed, silent or hung command gives the same no-answer fallback
    as a run with nobody to ask, so the run proceeds on its own judgment.
    """
    argv = shlex.split(command)

    def ask(question: str, options: list[str] | None = None, recommended: str | None = None) -> str:
        payload = json.dumps(
            {"question": question, "options": options or [], "recommended": recommended}
        )
        try:
            done = subprocess.run(
                argv, input=payload, capture_output=True, text=True, timeout=timeout_s
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Interactor._no_answer(f"the answer command failed ({type(exc).__name__})")
        answer = done.stdout.strip()
        if done.returncode != 0 or not answer:
            return Interactor._no_answer(
                f"the answer command gave no answer (exit {done.returncode})"
            )
        return answer

    return ask
