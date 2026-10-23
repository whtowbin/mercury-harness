"""A live terminal dashboard for one harness run, built on Textual.

The agent runs synchronously and can take minutes; watching it meant watching
scrollback. This runs it in a background worker thread and renders live
updates by subscribing to Trace's on_event hook — no change to any agent
internals, the same hook a future headless consumer (a web UI, a log
shipper) could use.

The goal/question input is a single bar docked at the bottom of the screen,
always visible rather than a modal that appears and disappears — what it is
for at any moment (the initial goal, an ask_user question, the next goal
once a run finishes) is communicated by the label above it and its
placeholder text, tracked in `self._mode`. A modal is still used, but only
for the one shape a single-line bar cannot represent: ask_user with a short
list of concrete options to choose from.

The layout follows the chat-style agents people already know (Claude Code
and its peers): a one-line bar across the top with the workspace, the models
and the running cost; a transcript that fills the screen; a strip of ducks
paddling across the window while a model call is in flight; and the prompt
box with a one-line status beneath it. The transcript is a scroll view of
separate widgets rather than a RichLog: model text renders as real markdown
(Textual's Markdown widget, which reflows when the window is resized, where
a RichLog line is fixed at the width it was written), and the view is
anchored to the bottom only until you scroll up, instead of RichLog's
jump-to-the-end on every write. Each tool call is one dim line under the
step it belongs to, the way the agents above show them.

The calls/cost/tok-per-s breakdown lives in a side panel, hidden by default
(ctrl+o toggles it); the top bar already carries the headline numbers.

The workspace panel ("w" toggles it, on the opposite side from settings) is
the one place this UI can change what the agent is actually allowed to
touch, rather than how it behaves: a "Browse…" button opens the host OS's
own native folder picker (agent/native_dialogs.py — Finder's dialog on
macOS, not a recreation of one) so choosing a workspace feels like choosing
one in any other app, with a plain text Input beside it for anyone on a
platform with no native picker, over SSH with no display, or who would
simply rather type a path. Either way the choice goes through the same
`Workspace(path)` constructor the backend itself uses to open a run — so an
invalid path is rejected here, at the moment it is chosen, with the same
message a run would have failed with — and the same nested-project check
(`enclosing_project()`) `loop._check_not_nested` runs at the top of a real
run: a workspace nested inside another Python project asks for confirmation
here, with an explanation of why that matters, instead of only being
discovered as a `NestedWorkspaceError` after the run has already started.
The tree beneath both controls is a live listing of that directory, filtered
to the same IGNORED_DIRS every file tool already excludes, so the panel
shows what the agent's own tools would ever actually see rather than a raw
OS listing cluttered with a virtualenv.

Threading: Trace.event() can fire from several threads at once when
--step-concurrency > 1 (agent/phases/schedule.py runs steps in a
ThreadPoolExecutor). Every UI update from the worker thread — a trace event,
opening the options modal — crosses to the UI thread via `call_from_thread`.
That is deliberately the *only* cross-thread mechanism used here: an earlier
version posted a custom `Message` subclass via `post_message` (Textual's
other documented thread-safe primitive), and it reproducibly deadlocked
Textual's headless test harness — both when posted from the same context and
from a genuine background thread. The cause was never pinned down;
`call_from_thread` sidesteps it entirely and is equally supported for this.
`ask_user` additionally has to *block* the calling thread until a human
answers, which it does with a plain `threading.Event` a worker thread can
wait on without touching Textual's own event loop. The native folder picker
is a subprocess call that can block for as long as a human takes to decide,
so it runs the same way: its own worker thread, its own `call_from_thread`
back into the UI once a path is chosen (or nothing is).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Console, RenderableType
from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Input,
    Label,
    Markdown,
    RadioButton,
    RadioSet,
    Select,
    Static,
    TextArea,
)

from . import native_dialogs
from . import summary as summary_module
from .cancel import CancelToken
from .config import Settings
from .editcalls import EDIT_TOOLS
from .loop import run as run_agent
from .spinner import DUCK_ROWS, duck_scene, with_elapsed
from .workspace import IGNORED_DIRS, Workspace, WorkspaceError

# The transcript drops its oldest entries past this many, so a long session
# does not grow the widget tree without bound.
MAX_LOG_ITEMS = 1500

# A palette of its own rather than Textual's stock blue: warm amber for the
# things you act on, a muted teal for structure, on a near-black ground.
HARNESS_THEME = Theme(
    name="harness",
    primary="#E0A458",
    secondary="#7FB7BE",
    accent="#7FB7BE",
    warning="#E5C07B",
    error="#E06C75",
    success="#98C379",
    background="#15161A",
    surface="#1C1D23",
    panel="#24262E",
    foreground="#D8D8DC",
    dark=True,
)

# What the bottom bar is for right now. "running" disables it; every other
# mode enables it and gives on_input_submitted a different meaning.
Mode = Literal["initial", "running", "asking", "done", "idle"]

# The three tiers of approval trust, named the same way agent.cli's
# --approval-mode is — see Settings.auto_approve / .auto_approve_python for
# what each one actually gates. A dropdown rather than two independent
# switches, since "auto-python" and "auto-plan" are the only two meaningful
# combinations of the two booleans that ever make sense together (there is
# no "approve bash but not python" tier), and presenting them as one choice
# says that directly instead of leaving it to be discovered.
APPROVAL_MODES = [
    ("Manual — prompt for everything", "manual"),
    ("Auto Python — skip run_python prompts only", "auto-python"),
    ("Auto Plan — skip every prompt once a plan exists", "auto-plan"),
]
ON_OFF = [("On", "on"), ("Off", "off")]

# Models offered by the settings panel. A Select needs a fixed list, so this
# is the set the harness and its benchmark suite already name (see
# bench/configs.py) rather than an attempt to mirror everything OpenRouter
# hosts. Whatever a run was actually launched with is prepended at startup,
# so a model passed on the command line is never missing from its own
# dropdown.
KNOWN_MODELS = [
    "inception/mercury-2.5",
    "inception/mercury-3",
    "z-ai/glm-5.3-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-v4-flash",
    "openai/gpt-5-nano",
    "openai/gpt-oss-120b",
    "qwen/qwen3.7-flash",
    "poolside/laguna-s-2.1",
]


def model_options(*current: str) -> list[tuple[str, str]]:
    """The dropdown's options, guaranteeing the in-use models appear."""
    names = list(KNOWN_MODELS)
    for name in current:
        if name and name not in names:
            names.insert(0, name)
    return [(name, name) for name in names]


@dataclass
class RunStats:
    """Counters for one prompt, or for a whole session of them."""

    calls: int = 0
    cost_usd: float = 0.0
    # Wall time inside model calls, and the part of it spent generating
    # rather than queued or prefilling. Decode time is the interesting half
    # for a diffusion model: it is where the speed difference actually shows.
    latency_s: float = 0.0
    decode_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_call(self, record: dict) -> None:
        self.calls += 1
        self.cost_usd += float(record.get("cost_usd") or 0.0)
        latency = float(record.get("latency_s") or 0.0)
        ttft = float(record.get("ttft_s") or 0.0)
        self.latency_s += latency
        self.decode_s += max(0.0, latency - ttft)
        self.prompt_tokens += int(record.get("prompt_tokens") or 0)
        self.completion_tokens += int(record.get("completion_tokens") or 0)

    def reset(self) -> None:
        for name, value in RunStats().__dict__.items():
            setattr(self, name, value)

    @property
    def decode_tok_per_s(self) -> float:
        return self.completion_tokens / self.decode_s if self.decode_s > 0 else 0.0


class PromptBox(TextArea):
    """The bottom bar: multi-line and soft-wrapping, unlike Textual's Input.

    Input is strictly one line, so a long goal scrolled sideways and the
    beginning of what you were typing disappeared — and a detailed goal is
    exactly the kind most worth reading back before sending. TextArea wraps,
    but binds Enter to "insert a newline", so this rebinds Enter to submit
    and puts the newline on shift+Enter, the convention every chat box uses.

    It keeps Input's `value` property, so the ask/approve plumbing that sets
    and clears the bar did not have to learn a second widget's API.
    """

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("shift+enter", "newline", "New line", show=False),
    ]

    class Submitted(Message):
        """Posted when Enter is pressed. Mirrors Input.Submitted's shape."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.text = new_value or ""

    def action_submit(self) -> None:
        if not self.disabled:
            self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")


class ChoiceModal(ModalScreen[str]):
    """ask_user with a short list of concrete options, shown as buttons.

    Buttons rather than a list: a list needs you to work out that it is
    focusable, arrow to a row, and press Enter, which is three pieces of
    knowledge for what is meant to be a single decision. A button is a thing
    you click, and every option is visible at once without navigating.

    "Other…" is always present. The options are the model's guesses at what
    you might want, and a question whose real answer is "neither, do this
    instead" is exactly the kind worth being able to answer.
    """

    DEFAULT_CSS = """
    ChoiceModal { align: center middle; }
    #choice-box { width: 90%; max-width: 120; height: auto; max-height: 80%;
                 border: round $accent; padding: 1 2; background: $surface; }
    #choice-title { text-style: bold; color: $accent; }
    /* height: auto and no cap below max-height on the box: a question is
       free-form text and truncating it is how you get an answer to a
       question nobody could read. It scrolls if it is genuinely enormous. */
    #choice-question { height: auto; margin: 1 0; }
    #choice-buttons { height: auto; margin-top: 1; }
    #choice-buttons Button { width: 100%; margin-bottom: 1; }
    #other-row { height: auto; display: none; }
    #other-input { height: 5; border: round $accent; }
    """

    def __init__(self, question: str, options: list[str], recommended: str | None = None):
        super().__init__()
        self.question = question
        self.options = options
        self.recommended = recommended if recommended in options else None

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-box"):
            yield Label("🤖 the agent has a question", id="choice-title")
            # Text, not markup: a question or an option can hold square
            # brackets ("[fix-bug]", "list[str]"), which markup reads as style
            # tags and silently drops.
            yield Static(Text(self.question), id="choice-question")
            with Vertical(id="choice-buttons"):
                for i, option in enumerate(self.options):
                    recommended = option == self.recommended
                    label = f"{option}  ·  recommended" if recommended else option
                    yield Button(
                        Text(label),
                        id=f"choice-{i}",
                        variant="success" if recommended else "primary",
                    )
                yield Button("Other… (type your own answer)", id="choice-other", variant="default")
            with Vertical(id="other-row"):
                yield Label("your answer (Enter to send):")
                yield PromptBox(id="other-input", soft_wrap=True)

    def on_mount(self) -> None:
        # Focus the recommendation if there is one, so Enter takes it.
        target = "choice-other" if not self.options else f"choice-{self._default_index()}"
        self.query_one(f"#{target}", Button).focus()

    def _default_index(self) -> int:
        if self.recommended in self.options:
            return self.options.index(self.recommended)
        return 0

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "choice-other":
            self._show_other()
            return
        index = int(str(event.button.id).rsplit("-", 1)[-1])
        self.dismiss(self.options[index])

    def _show_other(self) -> None:
        self.query_one("#other-row").styles.display = "block"
        self.query_one("#other-input", PromptBox).focus()

    def on_key(self, event: events.Key) -> None:
        """Typing while a button has focus starts your own answer: a
        correction ("Go ahead, or say what to change") should not need a
        click on "Other…" first."""
        box = self.query_one("#other-input", PromptBox)
        if self.focused is box or not event.is_printable or event.character is None:
            return
        event.stop()
        self._show_other()
        box.insert(event.character)

    def on_prompt_box_submitted(self, event: PromptBox.Submitted) -> None:
        answer = event.value.strip()
        if answer:
            self.dismiss(answer)


# What an unanswered question returns, here and from ask_user's bar: the
# clarify phase drops any answer starting "(no answer", and Approver's gate
# reads it as a refusal.
NO_ANSWER = "(no answer given)"


class QuestionsModal(ModalScreen[list[str]]):
    """Clarify's questions, all shown at once and answered in one go.

    Each question shows its options as a radio list (the recommended one
    preselected) and a free-text box, which wins when filled. Returns one
    answer per question, in order; a question left blank is NO_ANSWER, which
    the clarify phase drops rather than handing the planner as if it were an
    answer. "Skip all" (or Escape) returns NO_ANSWER for every question.

    The buttons sit outside the scrolling list on purpose: a modal whose
    buttons scroll with long content can push them below an 80x24 screen
    (see RunApp._confirm_nested_workspace).
    """

    BINDINGS = [
        Binding("ctrl+s", "send", "Send answers", priority=True),
        Binding("escape", "skip", "Skip all"),
    ]
    DEFAULT_CSS = """
    QuestionsModal { align: center middle; }
    #questions-box { width: 90%; max-width: 120; height: auto; max-height: 90%;
                     border: round $primary; padding: 0 2; background: $surface; }
    #questions-title { text-style: bold; color: $primary; padding: 1 0 0 0; }
    #questions-list { height: auto; max-height: 60vh; }
    .question { height: auto; margin-top: 1; }
    .question-text { text-style: bold; height: auto; }
    .question RadioSet { width: 100%; height: auto; border: none; padding: 0;
                         background: transparent; }
    .question Input { margin-top: 0; }
    #questions-buttons { height: 3; margin: 1 0 0 0; }
    #questions-buttons Button { margin-right: 2; }
    """

    def __init__(self, questions: list[tuple[str, list[str] | None, str | None]]):
        super().__init__()
        self.questions = questions

    def compose(self) -> ComposeResult:
        count = len(self.questions)
        with Vertical(id="questions-box"):
            yield Label(
                f"🤖 before planning, {count} question{'s' if count != 1 else ''}",
                id="questions-title",
            )
            with VerticalScroll(id="questions-list"):
                for i, question in enumerate(self.questions):
                    yield from self._question_block(i, *question)
            with Horizontal(id="questions-buttons"):
                yield Button("Send answers  (ctrl+s)", id="questions-send", variant="primary")
                yield Button("Skip all", id="questions-skip")

    def _question_block(
        self, i: int, question: str, options: list[str] | None, recommended: str | None
    ) -> ComposeResult:
        with Vertical(classes="question"):
            yield Static(f"{i + 1}. {escape(question)}", classes="question-text")
            if options:
                with RadioSet(id=f"q{i}-options"):
                    for option in options:
                        tag = "  · recommended" if option == recommended else ""
                        yield RadioButton(
                            escape(option + tag), value=option == recommended, name=option
                        )
            placeholder = "or type your own answer" if options else "your answer"
            yield Input(placeholder=placeholder, id=f"q{i}-text")

    def on_mount(self) -> None:
        first = self.query(RadioSet).first() if self.questions and self.questions[0][1] else None
        (first or self.query_one("#q0-text", Input)).focus()

    def answers(self) -> list[str]:
        result = []
        for i, (_, options, _) in enumerate(self.questions):
            typed = self.query_one(f"#q{i}-text", Input).value.strip()
            chosen = ""
            if options:
                pressed = self.query_one(f"#q{i}-options", RadioSet).pressed_button
                chosen = (pressed.name or "") if pressed is not None else ""
            result.append(typed or chosen or NO_ANSWER)
        return result

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in a text box moves to the next question; on the last one, sends."""
        index = int(str(event.input.id)[1:].split("-")[0])
        if index + 1 >= len(self.questions):
            self.action_send()
            return
        next_options = self.questions[index + 1][1]
        target = f"#q{index + 1}-options" if next_options else f"#q{index + 1}-text"
        self.query_one(target).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "questions-send":
            self.action_send()
        elif event.button.id == "questions-skip":
            self.action_skip()

    def action_send(self) -> None:
        self.dismiss(self.answers())

    def action_skip(self) -> None:
        self.dismiss([NO_ANSWER] * len(self.questions))


class WorkspaceTree(DirectoryTree):
    """The workspace's file tree, filtered to the same directories every
    file tool in agent/workspace.py already excludes — `.venv`,
    `node_modules`, `__pycache__`, and the rest of IGNORED_DIRS — so this
    panel shows what the agent's own tools would ever actually see, not a
    raw OS listing cluttered with a virtualenv or a build directory it will
    never look inside.
    """

    def filter_paths(self, paths):
        return [p for p in paths if p.name not in IGNORED_DIRS]


# Signals that a message is markdown: a heading, a **bold** pair, a code
# fence, inline `code`, a [link](url), or two or more list items. Trace
# narration is plain text (Trace.say prints it with print()), so anything else
# renders as plain text. Deliberately not "any '#' or '*' anywhere": a shell
# command or a path containing either character must stay plain.
_MARKDOWN_SIGNALS = (
    re.compile(r"(?m)^#{1,6}\s"),
    re.compile(r"\*\*[^*\n]+\*\*"),
    re.compile(r"(?m)^\s*```"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\[[^\]\n]+\]\([^)\s]+\)"),
)
_MARKDOWN_LIST_ITEM = re.compile(r"(?m)^\s*(?:[-*+]|\d+\.)\s+\S")


def _looks_like_markdown(text: str) -> bool:
    if any(signal.search(text) for signal in _MARKDOWN_SIGNALS):
        return True
    return len(_MARKDOWN_LIST_ITEM.findall(text)) >= 2


def _render_narration(text: str) -> Text | str:
    """A trace "log" message, as the transcript should show it.

    Returns the markdown source (a str) when the message is markdown, for
    the transcript to give its own Markdown widget, and plain Text otherwise.

    Never Rich markup. Trace narration is plain text, and step ids appear in
    square brackets ("[fix-bug] Fixed _refill()"), which Text.from_markup
    reads as a style tag and silently drops. An earlier version rendered
    every line through from_markup, so every step id vanished from this
    view while the plain CLI showed it.
    """
    if _looks_like_markdown(text):
        return text
    return Text(text)


# How each tool call reads in the transcript: a verb and its main argument,
# "Edit throttle.py", rather than the function name. Tools missing here show
# their own name.
_TOOL_VERBS = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "multi_edit": "Edit",
    "run_bash": "Bash",
    "run_python": "Python",
    "run_tests": "Tests",
    "glob_files": "Glob",
    "grep": "Search",
    "list_dir": "List",
    "web_search": "Web search",
    "fetch_url": "Fetch",
    "research": "Research",
    "read_files": "Read",
    "check_code": "Check",
}
# Argument names in the order they are worth showing.
_TOOL_ARG_KEYS = ("path", "paths", "file", "command", "cmd", "pattern", "query", "url", "code")
# Terminal and bookkeeping tools. Each already gets its own narration line
# (the plan, the step result, the review verdict), so a tool line would only
# repeat it.
_QUIET_TOOLS = {"submit_plan", "finish_step", "submit_findings", "ask_user", "give_up"}
# Batched edits (edit_files, edit_blocks, ...): every edit tool except the
# two single-file ones. The call's own arguments are clipped in the trace, and
# each file's edit_file event writes its own line (RunApp._render_edit_result).
# Derived from the shared list so a new batch tool is covered when it is added
# there; edit_blocks was missed when it was a list of its own (§106.1).
_SINGLE_FILE_EDIT_TOOLS = frozenset({"edit_file", "write_file"})
_BATCH_EDIT_TOOLS = EDIT_TOOLS - _SINGLE_FILE_EDIT_TOOLS


def tool_label(name: str, args: dict | None) -> str:
    verb = _TOOL_VERBS.get(name, name)
    for key in _TOOL_ARG_KEYS:
        value = (args or {}).get(key)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if key == "paths" and isinstance(value, str):
            value = value.strip("[]").replace('"', "")
        if isinstance(value, str) and value.strip():
            first = value.strip().splitlines()[0]
            return f"{verb} {first[:100]}{'…' if len(first) > 100 else ''}"
    return verb


# The Mac's own clipboard, written directly when the TUI runs on this machine.
# Copying otherwise goes through the terminal (OSC 52), which Ghostty, kitty
# and WezTerm accept but Terminal.app and some editor terminals ignore.
_LOCAL_COPY_COMMAND = ("pbcopy",)


def local_clipboard_command(
    env: Mapping[str, str] | None = None, platform: str | None = None
) -> tuple[str, ...] | None:
    """`pbcopy` on a local macOS session, None anywhere else.

    Never over SSH: pbcopy would fill the remote machine's clipboard, not
    the one in front of the person, and OSC 52 is the route that reaches
    their terminal from there.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    if (platform or sys.platform) != "darwin":
        return None
    if any(environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return None
    return _LOCAL_COPY_COMMAND if shutil.which(_LOCAL_COPY_COMMAND[0]) else None


def copy_with(command: tuple[str, ...], text: str) -> bool:
    """Pipe `text` to a clipboard command. False, never an exception, when it
    fails: the OSC 52 copy has already been sent either way. A UTF-8 locale,
    because pbcopy reads its input in the current one and garbles non-ASCII
    text under the C locale."""
    env = {**os.environ, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}
    try:
        subprocess.run(
            command,
            input=text.encode("utf-8"),
            env=env,
            timeout=2,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _plain(renderable: RenderableType) -> str:
    """The text a renderable shows, for tests and for copying out."""
    if isinstance(renderable, str):
        return Text.from_markup(renderable).plain
    if isinstance(renderable, Text):
        return renderable.plain
    console = Console(width=120, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


class Transcript(VerticalScroll):
    """The conversation: one widget per entry, anchored to the bottom.

    `anchor()` keeps the view at the newest entry until you scroll up, then
    leaves you where you are until you scroll back down. RichLog, which this
    replaces, called scroll_end() on every write whenever auto_scroll was on,
    so reading back through a running session meant being dragged to the
    bottom every time a tool call landed.
    """

    DEFAULT_CSS = """
    Transcript { padding: 0 1; scrollbar-size-vertical: 1; }
    Transcript > * { height: auto; }
    Transcript > .user { background: $panel; border-left: outer $primary;
                         padding: 0 1; margin: 1 0 1 0; }
    Transcript > .gap { margin-top: 1; }
    Transcript > .step { color: $secondary; text-style: bold; margin-top: 1; }
    Transcript > .tool { color: $text-muted; padding-left: 2; }
    Transcript > .thought { color: $text-muted; text-style: italic; padding-left: 2; }
    Transcript > Markdown.thought MarkdownParagraph { margin: 0; }
    Transcript > .result { padding-left: 2; }
    Transcript > .muted { color: $text-muted; }
    Transcript > .ok { color: $success; }
    Transcript > .warn { color: $warning; }
    Transcript > .error { color: $error; text-style: bold; }
    Transcript > Markdown { margin: 0; padding: 0; background: transparent; }
    Transcript Markdown MarkdownFence { margin: 1 0; max-height: 40; }
    Transcript Markdown MarkdownH1, Transcript Markdown MarkdownH2,
    Transcript Markdown MarkdownH3 {
        color: $primary; text-style: bold; background: transparent;
        border: none; margin: 1 0 0 0; padding: 0; content-align: left top;
    }
    """

    def on_mount(self) -> None:
        self.anchor()

    def on_key(self, event) -> None:
        """Clicking the transcript (to select text) moves focus to it; typing
        then goes back to the prompt box instead of being lost."""
        if not event.is_printable or event.character is None:
            return
        bar = self.app.query_one("#goal-input", PromptBox)
        if bar.disabled:
            return
        event.stop()
        bar.focus()
        bar.insert(event.character)

    def write(self, content: RenderableType, classes: str = "") -> Static:
        """Add an entry. A str is this module's own Rich markup; Text and
        other renderables are shown as they are."""
        entry = Static(content, classes=classes)
        entry.plain = _plain(content)  # ty: ignore[unresolved-attribute]
        return self._add(entry)

    def write_markdown(self, source: str, classes: str = "") -> Markdown:
        entry = Markdown(source, classes=classes)
        entry.plain = source  # ty: ignore[unresolved-attribute]
        return self._add(entry)

    def _add(self, entry: Widget) -> Any:
        self.mount(entry)
        overflow = len(self.children) - MAX_LOG_ITEMS
        if overflow > 0:
            self.remove_children(list(self.children[:overflow]))
        return entry

    @property
    def entries(self) -> list[Widget]:
        return list(self.children)

    def plain_text(self) -> str:
        return "\n".join(getattr(child, "plain", "") for child in self.children)


# The model's own commentary is shown up to this long; past it, the full
# text is in the trace.
_ASSISTANT_TEXT_LIMIT = 400

# "-- step 1/2: [fix-bug] Fix refill bug" (schedule.py's step banner).
_STEP_BANNER = re.compile(r"^-- step (\d+/\d+): \[([^\]]+)\] (.*)$")
# "[fix-bug] Fixed _refill() ..." (a step's own report when it finishes).
_STEP_RESULT = re.compile(r"^\[[a-z0-9][a-z0-9-]*\] ")


def _narration_class(line: str) -> str:
    """A style for one narration line, from the marks the runner already uses."""
    if _STEP_RESULT.match(line):
        return "result"
    if line.startswith("✓"):
        return "ok result"
    if line.startswith(("✗", "!")):
        return "warn result"
    if line.endswith((": PASS", ": clean")):
        return "ok"
    if line.endswith(": FAIL"):
        return "warn"
    if line.startswith(("understood as:", "draft:", "retrying", "===", "read cache:", "steps:")):
        return "muted"
    return ""


def _colour_ducks(lines: list[str]) -> Text:
    """The duck strip in colour: yellow ducks, an orange bill, blue water."""
    out = Text()
    for r, line in enumerate(lines):
        if r:
            out.append("\n")
        for ch in line:
            if ch == "~":
                style = "#4F7CAC"
            elif r == 1 and ch in "<>":
                style = "bold #E8833A"
            elif ch == "o" and r == 1:
                style = "bold #FFFFFF"
            elif ch.isspace():
                style = ""
            elif (r == DUCK_ROWS - 1 and ch.isdigit()) or (ch == "s" and r == DUCK_ROWS - 1):
                style = "dim"
            else:
                style = "#F2C14E"
            out.append(ch, style=style)
    return out


class RunApp(App[dict]):
    """Runs one agent session and renders it live. Returns the run summary."""

    TITLE = "harness"
    # Textual's own ctrl+p is normally bound to a command palette we never
    # populate with commands — pressing it opened an empty, useless overlay
    # that looked exactly like a broken "settings panel" from the outside.
    # Disabling it here reclaims ctrl+p for our own settings toggle instead.
    ENABLE_COMMAND_PALETTE = False
    # The plain letters below only fire while the goal/answer bar is
    # *disabled* ("running" mode) — Textual gives a focused Input first
    # refusal on every key it doesn't bind itself, so with the bar focused
    # (the common case: before a goal is typed, or between runs) "p"/"s"/"c"
    # are typed as literal text instead of toggling anything. The ctrl+
    # bindings are `priority=True`, which checks the App's own bindings
    # *before* the focused widget gets a turn — they work no matter what
    # has focus, including a freshly-focused, empty goal bar.
    BINDINGS = [
        Binding("q", "quit_confirm", "Quit", show=False),
        Binding("c", "stop_run", "Stop", show=False),
        Binding("s", "toggle_stats", "Stats", show=False),
        Binding("p", "toggle_settings", "Settings", show=False),
        Binding("w", "toggle_workspace", "Workspace", show=False),
        # The ones that must work even while the prompt box has focus. Quit
        # is here because "q" was swallowed as typed text in every mode where
        # you would actually reach for it — at the goal prompt, at the
        # next-goal prompt, and while answering a question — which made the
        # app look like it could not be closed at all.
        # These are also the footer, which is clickable: the discoverable,
        # mouse-friendly route that a row of big buttons used to provide.
        Binding("ctrl+g", "stop_run", "Stop", priority=True),
        Binding("ctrl+a", "cycle_approval_mode", "Approvals", priority=True),
        Binding("ctrl+p", "toggle_settings", "Settings", priority=True),
        Binding("ctrl+w", "toggle_workspace", "Files", priority=True),
        Binding("ctrl+o", "toggle_stats", "Stats", priority=True),
        Binding("ctrl+q", "quit_confirm", "Quit", priority=True),
        # Priority, so the prompt box's own TextArea bindings for these keys
        # never decide which selection is meant. On macOS the terminal
        # usually keeps ⌘C and ⌘A for itself (Ghostty's defaults do), so
        # selections also copy themselves, see on_text_selected.
        Binding("ctrl+c,super+c", "copy_selection", "Copy", show=False, priority=True),
        Binding("super+a", "select_all_here", "Select all", show=False, priority=True),
    ]
    DEFAULT_CSS = """
    Screen { background: $background; }
    #topbar { height: 1; background: $panel; padding: 0 1; }
    #topbar-left { width: 1fr; }
    #topbar-right { width: auto; color: $text-muted; }
    #body { height: 1fr; }
    #log { width: 1fr; }
    .side { width: 44; background: $surface; padding: 0 1;
            overflow-y: auto; scrollbar-size: 1 1; border-left: tall $panel; }
    #workspace-panel { border-left: none; border-right: tall $panel; width: 40; }
    .side-title { color: $primary; text-style: bold; padding: 1 0 0 0; }
    /* overflow-y: auto on .side -- the settings panel has eleven controls
       and a short terminal fits maybe six, so without it the ones below the
       fold were unreachable. */
    #settings-panel Label { padding: 1 0 0 0; color: $text-muted; }
    #stats-panel Static { padding: 1 0 0 0; }
    #workspace-path { padding: 0 0 1 0; color: $text-muted; }
    #workspace-controls { height: 3; }
    #workspace-controls Button { width: 1fr; }
    #workspace-tree { margin-top: 1; height: 1fr; background: $surface; }
    #wait-indicator { height: 4; padding: 0 1; }
    /* height: auto on the question: an ask_user question is free-form text
       with no length limit, and a clipped question is one the human cannot
       answer. max-height keeps a very long one from pushing the prompt box
       off the screen; #body is 1fr, so the transcript gives up the space. */
    #prompt-label { height: auto; max-height: 14; padding: 0 2; margin-top: 1;
                    color: $warning; }
    #prompt-label.empty { display: none; }
    /* Grows with what you type instead of scrolling a fixed 5 rows. */
    #goal-input { height: auto; min-height: 3; max-height: 12; margin: 0 1;
                  border: round $panel-lighten-2; background: $background; }
    #goal-input:focus { border: round $primary; }
    #goal-input:disabled { border: round $panel; opacity: 0.6; }
    #mode-line { height: auto; max-height: 2; padding: 0 2; color: $text-muted; }
    Footer { background: $background; }
    """

    def __init__(self, settings: Settings, plan_path: str | None, stop_after: str):
        super().__init__()
        self.settings = settings
        self.plan_path = plan_path
        self.stop_after = stop_after
        self.summary: dict | None = None
        # Two tallies, deliberately: `_run` resets with each new goal, while
        # `_session` accumulates across every prompt in this TUI session.
        # "What did this prompt cost" and "what have I spent in here" are
        # different questions, and a panel showing only one answers half.
        self._run = RunStats()
        self._session = RunStats()
        self._prompts = 0
        self._current_step = ""
        self._agent_running = False
        self._mode: Mode = "idle"
        # Set only while a worker thread is blocked in _on_ask waiting on the
        # bottom bar — on_input_submitted checks this rather than trusting
        # _mode == "asking" alone, since the event must exist to signal.
        self._ask_answer: dict[str, str] = {}
        self._ask_done: threading.Event | None = None
        # Fresh per run (see _start): a stop request set on one goal must not
        # carry over and instantly cancel the next one in the continue loop.
        self._cancel = CancelToken()
        # The wait indicator: refcounted like Spinner (agent/spinner.py),
        # since --step-concurrency > 1 means more than one worker thread can
        # have a model call in flight at once — one call finishing must not
        # switch the indicator off while another is still waiting.
        self._waiting_count = 0
        self._waiting = False
        self._wait_frame = 0
        # Whether the indicator currently shows the ducks. Only the change between
        # ducks and blank needs a layout pass (DESIGN-LOG §95).
        self._indicator_shown = False
        self._wait_started = 0.0
        # The TUI's own on/off for the wait animation — independent of
        # narrate/quick_answers/auto_approve*, which are real Settings
        # fields the backend reads; there is no backend equivalent of "is
        # something rendering a spinner" for this one to drive.
        self._show_wait_animation = True
        # Set by a quit request made while a run is still going, so the
        # second request exits instead of asking again.
        self._quit_armed = False
        # A tool line was just written, so the runner's own "… editing x.py"
        # narration for the same call would repeat it (see _render_log_message).
        self._tool_line_pending = False
        # The line each tool call wrote, keyed by (tool, path), so the
        # tool's own result event can add what it did to that same line.
        self._tool_lines: dict[tuple[str, str], Static] = {}

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(self._topbar_left(), id="topbar-left")
            yield Static("", id="topbar-right")
        with Horizontal(id="body"):
            # Workspace first, so it sits on the far left, where an IDE's
            # file tree is.
            with Vertical(id="workspace-panel", classes="side"):
                yield Label("Workspace", classes="side-title")
                yield Static(self.settings.workspace, id="workspace-path")
                with Horizontal(id="workspace-controls"):
                    yield Button("Browse…", id="btn-browse-workspace", variant="primary")
                yield Input(placeholder="or type a path, Enter to set", id="workspace-input")
                yield WorkspaceTree(self.settings.workspace, id="workspace-tree")
            with Vertical(id="stats-panel", classes="side"):
                yield Label("Usage", classes="side-title")
                yield Static(self._stats_text(), id="stats")
            yield Transcript(id="log")
            with Vertical(id="settings-panel", classes="side"):
                yield Label("Settings", classes="side-title")
                yield from self._settings_controls()
        yield Static("", id="wait-indicator")
        yield Static("", id="prompt-label", classes="empty")
        yield PromptBox(id="goal-input", soft_wrap=True)
        yield Static(self._mode_line_text(), id="mode-line")
        yield Footer(compact=True, show_command_palette=False)

    def _settings_controls(self) -> ComposeResult:
        def on_off(label: str, select_id: str, on: bool) -> ComposeResult:
            yield Label(label)
            yield Select(ON_OFF, id=select_id, value="on" if on else "off", allow_blank=False)

        yield Label("Approval mode")
        yield Select(
            APPROVAL_MODES, id="mode-select", value=self._approval_mode(), allow_blank=False
        )
        for label, select_id, model in (
            ("Planner model", "planner-select", self.settings.planner_model),
            ("Executor model", "executor-select", self.settings.executor_model),
            ("Reviewer model", "reviewer-select", self.settings.reviewer_model),
        ):
            yield Label(label)
            yield Select(model_options(model), id=select_id, value=model, allow_blank=False)
        yield from on_off("Narrate steps", "narrate-select", self.settings.narrate)
        yield from on_off("Wait animation", "animate-select", True)
        yield from on_off("Quick answers", "quick-select", self.settings.quick_answers)
        yield from on_off("Clarify before planning", "clarify-select", self.settings.clarify)
        yield from on_off(
            "Review plan before running", "review-plan-select", self.settings.review_plan
        )
        yield from on_off("Final check", "final-check-select", self.settings.final_check)

    @property
    def _transcript(self) -> Transcript:
        return self.query_one("#log", Transcript)

    def on_mount(self) -> None:
        self.register_theme(HARNESS_THEME)
        self.theme = "harness"
        for panel in ("#workspace-panel", "#stats-panel", "#settings-panel", "#wait-indicator"):
            self.query_one(panel).styles.display = "none"
        self.set_interval(0.15, self._tick_wait_indicator)
        if not self.settings.goal.strip() and self.plan_path is None:
            self._transcript.write(
                Text.assemble(
                    ("🦆 harness", "bold"),
                    ("  ·  a coding agent in ", "dim"),
                    (self.settings.workspace, ""),
                ),
                classes="gap",
            )
            self._set_mode(
                "initial",
                "",
                placeholder="What should the agent build?   (shift+Enter for a new line)",
            )
        else:
            self._start()

    def _start(self) -> None:
        self._set_mode("running", placeholder="working…   ctrl+g to stop")
        self._cancel = CancelToken()
        self.query_one("#wait-indicator").styles.display = "block"
        self._transcript.write(Text(self.settings.goal or "(from saved plan)"), classes="user")
        self._refresh_topbar()
        self._run_agent()

    def _topbar_left(self) -> Text:
        workspace = Path(self.settings.workspace).expanduser()
        with contextlib.suppress(ValueError):
            workspace = Path("~") / workspace.resolve().relative_to(Path.home())
        models = {
            self.settings.planner_model,
            self.settings.executor_model,
            self.settings.reviewer_model,
        }
        model_text = (
            next(iter(models)).split("/")[-1]
            if len(models) == 1
            else f"{self.settings.planner_model.split('/')[-1]} → "
            f"{self.settings.executor_model.split('/')[-1]}"
        )
        return Text.assemble(
            ("🦆 harness", "bold"),
            ("  ", ""),
            (str(workspace), "dim"),
            ("  ·  ", "dim"),
            (model_text, "dim"),
        )

    def _refresh_topbar(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#topbar-left", Static).update(self._topbar_left())
            parts = []
            if self._current_step:
                parts.append(self._current_step)
            if self._session.calls:
                parts.append(f"{self._session.calls} calls")
                parts.append(f"${self._session.cost_usd:.4f}")
            self.query_one("#topbar-right", Static).update("  ·  ".join(parts))

    def _refresh_stats(self) -> None:
        """Redraw the stats panel, whether or not it is currently visible,
        and the top bar's headline numbers."""
        self._refresh_topbar()
        try:
            self.query_one("#stats", Static).update(self._stats_text())
        except NoMatches:
            return

    def _stats_text(self) -> str:
        """The stats panel: this prompt on top, the whole session beneath."""

        def block(title: str, stats: RunStats) -> list[str]:
            # "cyan", not a Textual CSS variable like $accent: those are only
            # meaningful inside a DEFAULT_CSS rule, not inside Rich markup
            # embedded in a widget's content. $accent here silently rendered
            # as no style at all — the section headers were plain, unbolded
            # text with no visible bug to notice unless you compared it
            # against what the markup was supposed to produce.
            lines = [f"[bold cyan]{title}[/]"]
            lines.append(f"  calls      [bold]{stats.calls}[/]")
            lines.append(f"  cost       [bold]${stats.cost_usd:.4f}[/]")
            lines.append(f"  model time [bold]{stats.latency_s:.1f}s[/]")
            lines.append(f"  decoding   [bold]{stats.decode_s:.1f}s[/]")
            if stats.decode_tok_per_s:
                lines.append(f"  decode t/s [bold]{stats.decode_tok_per_s:.0f}[/]")
            lines.append(f"  out tokens [bold]{stats.completion_tokens:,}[/]")
            return lines

        parts = block("this prompt", self._run)
        if self._current_step:
            parts.append(f"  step       [bold]{self._current_step}[/]")
        parts.append("")
        parts.extend(block("session total", self._session))
        parts.append(f"  prompts    [bold]{max(1, self._prompts)}[/]")
        return "\n".join(parts)

    # -- the settings panel and its "current flags" summary ---------------

    def _approval_mode(self) -> str:
        if self.settings.auto_approve:
            return "auto-plan"
        if self.settings.auto_approve_python:
            return "auto-python"
        return "manual"

    def _mode_line_text(self) -> str:
        def flag(name: str, on: bool) -> str:
            return f"{name} [{'green' if on else 'red'}]{'on' if on else 'off'}[/]"

        flags = "  ·  ".join(
            [
                flag("narrate", self.settings.narrate),
                flag("quick-answers", self.settings.quick_answers),
                flag("animate", self._show_wait_animation),
            ]
        )
        return f"[#E0A458]⏵⏵[/] [bold]{self._approval_mode()}[/] approvals   [dim]{flags}[/]"

    def on_select_changed(self, event: Select.Changed) -> None:
        """Applied to self.settings immediately, but Approver/ToolContext are
        built once at the top of each loop.run() call — a change made while
        a run is already in progress takes effect on the *next* run (a
        --continue iteration, or the next goal typed at the "done" prompt),
        not retroactively on approval prompts already in flight. Same as a
        CLI flag being fixed for the process's whole invocation; this just
        lets you change it between runs without restarting the TUI.
        """
        select_id = event.select.id
        value = event.value
        if select_id == "mode-select":
            self.settings.auto_approve = value == "auto-plan"
            self.settings.auto_approve_python = value in ("auto-plan", "auto-python")
        elif select_id == "narrate-select":
            self.settings.narrate = value == "on"
        elif select_id == "animate-select":
            self._show_wait_animation = value == "on"
        elif select_id == "quick-select":
            self.settings.quick_answers = value == "on"
        elif select_id == "clarify-select":
            self.settings.clarify = value == "on"
        elif select_id == "review-plan-select":
            self.settings.review_plan = value == "on"
        elif select_id == "final-check-select":
            self.settings.final_check = value == "on"
        elif select_id == "planner-select":
            self.settings.planner_model = str(value)
        elif select_id == "executor-select":
            self.settings.executor_model = str(value)
        elif select_id == "reviewer-select":
            self.settings.reviewer_model = str(value)
        else:
            return
        self.query_one("#mode-line", Static).update(self._mode_line_text())

    # -- the workspace panel: native picker, manual path, live file tree ---

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "workspace-input":
            return
        text = event.value.strip()
        event.input.value = ""
        if text:
            self._apply_workspace(text)

    @work(thread=True)
    def _browse_workspace(self) -> None:
        """Open the OS's native folder picker. Runs in its own worker thread
        for the same reason `_run_agent` and `ask_user` do: the subprocess
        call underneath (see agent/native_dialogs.py) blocks for as long as
        a human takes to decide, and blocking Textual's own event loop for
        that long would freeze the whole UI — the goal bar, the log, every
        other button — until the dialog closed.
        """
        self.call_from_thread(self._set_browse_button_disabled, True)
        try:
            chosen = native_dialogs.choose_folder("Select the agent's workspace")
        finally:
            self.call_from_thread(self._set_browse_button_disabled, False)
        if chosen:
            self.call_from_thread(self._apply_workspace, chosen)
        else:
            self.call_from_thread(self._note_no_folder_chosen)

    def _set_browse_button_disabled(self, disabled: bool) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#btn-browse-workspace", Button).disabled = disabled

    def _note_no_folder_chosen(self) -> None:
        """Covers two indistinguishable outcomes at once (see
        native_dialogs.choose_folder's docstring): the dialog was cancelled,
        or this platform has no native picker this module knows how to
        drive. Either way, saying so beats silence — a button that visibly
        does nothing looks broken rather than merely declined."""
        self._transcript.write(
            "[dim]no folder selected — type a path in the workspace panel instead[/]"
        )

    def _apply_workspace(self, path: str) -> None:
        """Validate and switch the workspace the next run will use.

        Goes through the same `Workspace(path)` construction the backend
        itself uses to open a run, so an invalid path is rejected right here
        — with the same message a run would have failed with — rather than
        accepted now and only discovered broken when the next run actually
        starts. Also runs the same nested-project check
        `loop._check_not_nested` runs at the top of every real run
        (`enclosing_project()`) — added after a real run of this exact
        shape only surfaced it once the run had already started, which is a
        worse place to learn about it than the moment the folder is chosen.
        Applied to `self.settings` immediately either way, same contract as
        `on_select_changed`: it takes effect on the *next* run, not
        retroactively on one already in progress.
        """
        try:
            candidate = Workspace(path)
        except WorkspaceError as exc:
            self._transcript.write(f"[bold red]workspace not set:[/] {escape(str(exc))}")
            return
        enclosing = candidate.enclosing_project()
        if enclosing and not self.settings.allow_nested:
            self._confirm_nested_workspace(candidate.root, enclosing)
            return
        self._set_workspace(candidate.root)

    def _confirm_nested_workspace(self, root: Path, enclosing: Path) -> None:
        """Ask before doing what `loop._check_not_nested` would otherwise
        refuse outright at run time — see `NestedWorkspaceError`'s own
        message, which the log line below and this docstring between them
        say in full; the modal itself says only the short version.

        Why this is actually a problem, not a formality: the path jail
        (`Workspace.resolve()`) only constrains the *file tools* — read_file,
        write_file, and the rest of agent/tools/fs.py. It says nothing about
        where a subprocess ends up. `uv pip install`, `uv sync`, and `uv add`
        all search *upward* from the current directory for the nearest
        `.venv`/`pyproject.toml` when the workspace doesn't have its own —
        so a workspace nested inside another project can have a
        model-issued dependency install land in that *other* project's
        environment, not this one, entirely outside anything the file jail
        was ever able to see or stop. Safe to accept anyway when the
        workspace already owns its own `.venv` (workspace_env() then pins
        uv there instead of letting it search further up) or when the
        outer `.venv`/project found above it is not something worth
        protecting — both real cases, which is why this asks rather than
        refusing outright the way the CLI's default does.

        The modal's own text carries no interpolated paths, deliberately:
        a first version put `root`/`enclosing` inline, and since a real
        path's length is unbounded, a sufficiently long one wrapped the
        question across enough lines to push "Use it anyway" below a plain
        80x24 terminal — present, focusable, and entirely unreachable by a
        click, caught only because the test for clicking it failed at the
        default size and passed once the terminal was made artificially
        large to compensate. Shortening the fixed wording bought some
        margin but not a guarantee, since the paths themselves were still
        inline and still unbounded; leaving them out entirely is the only
        fix that holds regardless of how long a real path is. The full
        detail, paths included, goes to the log instead — which scrolls,
        unlike a modal's buttons.
        """
        self._transcript.write(
            f"[yellow]{escape(str(root))} is inside the Python project at "
            f"{escape(str(enclosing))}[/]"
        )

        def handle(result: str | None) -> None:
            if result == "Use it anyway":
                self.settings.allow_nested = True
                self._set_workspace(root)
            else:
                self._transcript.write("[dim]workspace not changed[/]")

        question = (
            "This workspace is inside another Python project's folder.\n"
            "Commands the agent runs (uv pip install, uv sync, uv add) can "
            "reach and modify that project's environment too, not just "
            "this workspace — the file jail does not stop a subprocess.\n"
            "Safe anyway if this workspace already has its own .venv."
        )
        self.push_screen(
            ChoiceModal(question, ["Use it anyway", "Choose a different folder"]),
            handle,
        )

    def _set_workspace(self, root: Path) -> None:
        self.settings.workspace = str(root)
        self.query_one("#workspace-path", Static).update(self.settings.workspace)
        self.query_one("#workspace-tree", WorkspaceTree).path = self.settings.workspace
        suffix = " [dim](takes effect on the next run)[/]" if self._agent_running else ""
        self._transcript.write(f"[dim]workspace set:[/] {escape(self.settings.workspace)}{suffix}")
        self._refresh_topbar()

    # -- the persistent bottom bar ----------------------------------------

    def _set_mode(
        self, mode: Mode, label: str = "", value: str = "", placeholder: str = ""
    ) -> None:
        self._mode = mode
        bar = self.query_one("#goal-input", PromptBox)
        prompt = self.query_one("#prompt-label", Static)
        prompt.update(label)
        prompt.set_class(not label, "empty")
        bar.value = value
        bar.placeholder = placeholder
        bar.disabled = mode == "running"
        if not bar.disabled:
            bar.focus()

    def on_prompt_box_submitted(self, event: PromptBox.Submitted) -> None:
        text = event.value.strip()
        if self._mode == "initial":
            self._submit_initial_goal(text)
        elif self._mode == "done":
            self._submit_next_goal(text)
        elif self._mode == "asking":
            self._submit_answer(text)
        # "running" and "idle" have nothing bound to Enter.

    def _submit_initial_goal(self, text: str) -> None:
        if not text:
            self.summary = {"error": "no goal given"}
            self.exit(self.summary)
            return
        self.settings.goal = text
        self._start()

    def _submit_next_goal(self, text: str) -> None:
        if not text:
            return  # nothing typed, nothing suggested to fall back on either
        self.settings.goal = text
        self._run.reset()
        self._prompts += 1
        self._current_step = ""
        self.summary = None
        self._refresh_stats()
        self._start()

    def _submit_answer(self, text: str) -> None:
        if self._ask_done is None:
            return
        self._ask_answer["answer"] = text or "(no answer given)"
        self._ask_done.set()
        self._ask_done = None
        # Clear the box and the question. Nothing else did: after answering,
        # the worker thread carries on and never touches the bar again, so
        # the answer sat there as pre-filled text and the question stayed
        # above it — the next prompt would then arrive into a box already
        # holding the previous reply.
        self._set_mode(
            "idle", "[dim]answer sent — the run is continuing[/]", value="", placeholder=""
        )

    # -- background execution -------------------------------------------

    @work(thread=True, exclusive=True)
    def _run_agent(self) -> None:
        self._agent_running = True
        # plan_path only applies to the very first run — a continuation
        # supplies a fresh goal, not a saved plan.
        plan_path = self.plan_path
        self.plan_path = None
        try:
            summary = run_agent(
                self.settings,
                plan_path=plan_path,
                stop_after=self.stop_after,
                on_event=self._on_trace_event,
                on_ask=self._on_ask,
                on_ask_many=self._on_ask_many,
                on_approve=self._on_approve,
                cancel=self._cancel,
                on_call_start=self._wait_start,
                on_call_end=self._wait_stop,
                on_plan=self._on_plan if self.settings.review_plan else None,
            )
        except Exception as exc:  # surfaced to the UI, not swallowed
            summary = {"error": f"{type(exc).__name__}: {exc}"}
        self.summary = summary
        self.call_from_thread(self._finish)

    # -- the "still waiting" indicator, called from the worker thread -----

    def _wait_start(self) -> None:
        def bump() -> None:
            if self._waiting_count == 0:
                self._wait_started = time.time()
            self._waiting_count += 1
            self._waiting = self._waiting_count > 0

        self.call_from_thread(bump)

    def _wait_stop(self) -> None:
        def bump() -> None:
            self._waiting_count = max(0, self._waiting_count - 1)
            self._waiting = self._waiting_count > 0

        self.call_from_thread(bump)

    def _tick_wait_indicator(self) -> None:
        # set_interval keeps firing into app teardown, and by then compose()'s
        # widgets may already be gone — an unguarded query_one raises NoMatches
        # inside Textual's timer callback, which surfaces as the whole app
        # crashing rather than as one skipped animation frame.
        try:
            indicator = self.query_one("#wait-indicator", Static)
        except NoMatches:
            return
        if self._waiting and self._show_wait_animation:
            self._wait_frame += 1
            width = max(1, indicator.content_region.width or self.size.width - 2)
            lines = duck_scene(self._wait_frame, width).split("\n")
            # Past ELAPSED_DISPLAY_THRESHOLD_S (a diffusion model's own
            # prefill routinely runs longer on a healthy call, see llm.py's
            # DIFFUSION_FIRST_BYTE_TIMEOUT_S) a person watching cannot tell
            # "still working" from "stuck" without a number.
            lines[-1] = with_elapsed(lines[-1], time.time() - self._wait_started)
            # Every frame has DUCK_ROWS lines, so after the first one a repaint is
            # enough. update() lays the whole screen out by default, and over a
            # transcript of 1,251 entries that pass, 7 times a second, held a third
            # of a core while nothing else happened (scripts/tui_bench.py, §95).
            indicator.update(_colour_ducks(lines), layout=not self._indicator_shown)
            self._indicator_shown = True
        elif self._indicator_shown:
            indicator.update("")
            self._indicator_shown = False

    def _finish(self) -> None:
        self._agent_running = False
        with contextlib.suppress(NoMatches):
            self.query_one("#wait-indicator").styles.display = "none"
        log = self._transcript
        error = (self.summary or {}).get("error", "")
        if error:
            stopped = error.startswith("stopped")
            style = "yellow" if stopped else "bold red"
            label = "stopped:" if stopped else "FAILED:"
            log.write(f"[{style}]{label}[/] {escape(error)}", classes="gap")
            log_path = (self.summary or {}).get("failure_log_path")
            if log_path:
                log.write(f"[dim]saved:[/] {escape(str(log_path))}")
            self._set_mode("done", "", placeholder="What next?  (ctrl+q to exit)")
            return
        # Not unconditionally green: a run whose steps failed their own verify
        # command, or whose final check came back failing, used to end with
        # exactly the same "done." as a clean one, which is the single most
        # misleading thing the dashboard could say.
        problems = self._outcome_problems()
        if problems:
            log.write("[bold yellow]this run did not pass its own checks:[/]", classes="gap")
            for problem in problems:
                log.write(Text.assemble(("  • ", "yellow"), problem))

        if str((self.summary or {}).get("session_summary") or "").strip():
            # The rich version, not the stored plain one: bold headings and
            # spaced sections are the whole point.
            log.write(
                summary_module.render_rich(self.settings.goal, self.summary or {}), classes="gap"
            )
        if problems:
            log.write("[bold yellow]⚠ finished with failures[/]", classes="gap")
        else:
            log.write("[bold green]✓ done[/]", classes="gap")
        # Shown, never pre-filled. Pre-filling made the machine's idea the
        # default: across seventeen traced runs on one project, ten goals were
        # this suggestion accepted with a keystroke, and the feature the person
        # kept asking for went unbuilt while the agent worked through its own
        # list. A suggestion is a question now, and the box starts empty.
        suggestion = str((self.summary or {}).get("suggested_next_step") or "").strip()
        if suggestion:
            log.write(Text(suggestion, style="italic"), classes="gap muted")
        label = "answer the question above, or type your own next goal" if suggestion else ""
        self._set_mode("done", label, value="", placeholder="What next?  (ctrl+q to exit)")

    # -- trace event -> UI, called from the worker thread -----------------

    def _on_trace_event(self, record: dict[str, Any]) -> None:
        self.call_from_thread(self._render_event, record)

    def _render_event(self, record: dict[str, Any]) -> None:
        kind = record.get("kind", "")
        if kind == "llm_call":
            # Both tallies from one event: the panel shows this prompt's
            # spend next to the session's, and neither is derivable from the
            # other after a reset.
            self._run.add_call(record)
            self._session.add_call(record)
            self._refresh_stats()
        elif kind == "log":
            # detail=True lines (which model, which tools resolved, a phase's
            # own banner, where the trace file lives) still reach the trace
            # file via Trace.say -- read them back there, not here. Keeping
            # them out of this panel is the whole point: the conversation
            # reads like the run's own real narrative, not a settings dump
            # with narrative buried inside it.
            if not record.get("detail"):
                self._render_log_message(str(record.get("message", "")))
        elif kind == "step_attempt":
            self._current_step = str(record.get("step", ""))
            self._refresh_stats()
        elif kind == "tool_call":
            self._render_tool_call(record)
        elif kind == "edit_file":
            self._render_edit_result(record)
        elif kind == "write_file":
            self._render_write_result(record)
        elif kind == "run_tests":
            self._render_tests_result(record)
        elif kind == "assistant_text":
            self._render_assistant_text(record)
        elif kind == "step_end" or kind == "step_abandoned":
            summary = str(record.get("summary") or record.get("reason") or "")
            self._transcript.write(
                Text.assemble((f"[{record.get('step', '?')}] ", "cyan"), summary),
                classes="result",
            )
        elif kind in ("run_failed", "budget_exceeded"):
            message = str(record.get("message") or record.get("error"))
            self._transcript.write(f"[bold red]{kind}:[/] {escape(message)}", classes="gap")

    def _render_tool_call(self, record: dict[str, Any]) -> None:
        tool = str(record.get("tool", "?"))
        self._tool_line_pending = False
        if tool in _QUIET_TOOLS:
            return
        if tool in _BATCH_EDIT_TOOLS:
            # Each file's edit_file event writes its own line; the runner's
            # "… editing N files" narration would only repeat them.
            self._tool_line_pending = True
            return
        args = record.get("args") if isinstance(record.get("args"), dict) else {}
        line = self._transcript.write(
            Text.assemble(("⎿ ", "dim"), tool_label(tool, args)), classes="tool"
        )
        self._tool_lines[(tool, str((args or {}).get("path") or ""))] = line
        self._tool_line_pending = True

    def _finish_tool_line(self, key: tuple[str, str], label: str, detail: Text) -> None:
        """Add what a tool did to the line its call wrote, or write a new
        line when the call wrote none (a batched edit)."""
        if key not in self._tool_lines:
            # The call may spell the path differently ("./x.py"): fall back to
            # that tool's most recent line still waiting for its result.
            pending = [k for k in self._tool_lines if k[0] == key[0]]
            key = pending[-1] if pending else key
        line = self._tool_lines.pop(key, None)
        content = Text.assemble(("⎿ ", "dim"), label, "  ", detail)
        if line is None or not line.is_attached:
            self._transcript.write(content, classes="tool")
            return
        line.update(content)
        line.plain = content.plain  # ty: ignore[unresolved-attribute]

    def _render_edit_result(self, record: dict[str, Any]) -> None:
        path = str(record.get("path", "?"))
        detail = Text()
        if "added" in record:
            detail.append(f"+{record.get('added', 0)}", style="green")
            detail.append(" ")
            detail.append(f"-{record.get('removed', 0)}", style="red")
        preview = str(record.get("preview") or "")
        if preview:
            detail.append(f"\n    {preview}", style="dim italic")
        self._finish_tool_line(("edit_file", path), f"Edit {path}", detail)

    def _render_write_result(self, record: dict[str, Any]) -> None:
        path = str(record.get("path", "?"))
        lines = record.get("lines")
        size = f"{lines} lines" if lines is not None else f"{record.get('bytes', 0)} bytes"
        verb = "overwrote" if record.get("overwrote") else "new file"
        self._finish_tool_line(
            ("write_file", path), f"Write {path}", Text(f"{size}, {verb}", style="dim")
        )

    def _render_tests_result(self, record: dict[str, Any]) -> None:
        """A run_tests event carries no call id, so it completes the most
        recent test run's line, whatever path that call named."""
        passed = record.get("exit_code") == 0
        summary = str(record.get("summary") or ("passed" if passed else "failed"))
        detail = Text(f"→ {summary}", style="green" if passed else "red")
        if record.get("source") == "auto":
            # check_after_edit: the harness ran the step's tests itself after
            # an edit turn. No tool_call precedes it, and it must not complete
            # (or read as) a test run the model asked for.
            self._transcript.write(
                Text.assemble(("⎿ ", "dim"), ("Harness check", "italic"), "  ", detail),
                classes="tool",
            )
            return
        keys = [k for k in self._tool_lines if k[0] == "run_tests"]
        key = keys[-1] if keys else ("run_tests", "")
        line = self._tool_lines.get(key)
        label = getattr(line, "plain", "Tests").removeprefix("⎿ ")
        self._finish_tool_line(key, label, detail)

    def _render_assistant_text(self, record: dict[str, Any]) -> None:
        """What the model said alongside its tool calls ("Let me read the
        original throttle.py to compare..."): the running commentary a chat
        agent shows between its tool lines."""
        text = str(record.get("text") or "").strip()
        if not text:
            return
        if len(text) > _ASSISTANT_TEXT_LIMIT:
            text = text[:_ASSISTANT_TEXT_LIMIT].rstrip() + " …"
        rendered = _render_narration(text)
        if isinstance(rendered, str):
            self._transcript.write_markdown(rendered, classes="thought")
        else:
            self._transcript.write(rendered, classes="thought")

    def _render_log_message(self, message: str) -> None:
        """One trace narration line, placed the way a chat transcript would.

        A leading newline in the narration is its own "new section" signal,
        kept as a gap. The "   … editing x.py" line that follows a tool call
        says the same thing as the tool line already shown, so it is dropped
        when it directly follows one.
        """
        gap = message.startswith("\n")
        body = message.strip("\n")
        if not body.strip():
            return
        stripped = body.strip()
        follows_tool = self._tool_line_pending
        self._tool_line_pending = False
        step = _STEP_BANNER.match(stripped)
        if step:
            number, step_id, title = step.groups()
            self._transcript.write(
                Text.assemble(
                    ("● ", "bold"),
                    (f"Step {number}", "bold"),
                    ("  ", ""),
                    (step_id, "dim"),
                    ("  ", ""),
                    (title, ""),
                ),
                classes="step",
            )
            return
        if stripped.startswith("… "):
            if not follows_tool:
                self._transcript.write(Text.assemble(("⎿ ", "dim"), stripped[2:]), classes="tool")
            return
        classes = " ".join(c for c in ("gap" if gap else "", _narration_class(stripped)) if c)
        rendered = _render_narration(body if "\n" in body else stripped)
        if isinstance(rendered, str):
            self._transcript.write_markdown(rendered, classes=classes)
        else:
            self._transcript.write(rendered, classes=classes)

    # -- ask_user, called from the worker thread — must block -------------

    def _on_ask(
        self, question: str, options: list[str] | None, recommended: str | None = None
    ) -> str:
        if options:
            return self._ask_via_modal(question, options, recommended)
        return self._ask_via_bar(question)

    def _ask_via_bar(self, question: str) -> str:
        return self._prompt_via_bar(f"🤖 {escape(question)}", "your answer")

    def _call_for_attention(self, notice: str | None = None) -> None:
        """Ring the terminal bell, and optionally put a notice on screen, when the
        run starts waiting on a person. A bash approval sat unseen in the bottom bar
        for 90 s of a traced run (DESIGN-LOG §95); the bell is what reaches someone
        looking at another window. UI thread only."""
        self.bell()
        if notice:
            self.notify(notice, title="waiting for you", severity="warning", timeout=15)

    def _prompt_via_bar(
        self, label: str, placeholder: str, notice: str | None = "answer in the bar below"
    ) -> str:
        """Block the calling (worker) thread until the bottom bar is answered.

        Shared by ask_user's free-text case and by approval requests — an
        empty answer here already denies an approval correctly on its own:
        _gate only allows on a "y"/"a"-prefixed answer, so the sentinel
        "(no answer given)" this returns on a bare Enter falls through to
        Denied exactly like typing "n" would.
        """
        done = threading.Event()
        self._ask_answer = {}

        def open_bar() -> None:
            self._ask_done = done
            self._set_mode("asking", label, placeholder=placeholder)
            self._call_for_attention(notice)

        self.call_from_thread(open_bar)
        done.wait()
        return self._ask_answer.get("answer", "(no answer given)")

    def _outcome_problems(self) -> list[str]:
        """Ways this run failed its own checks, as lines to show the user."""
        summary = self.summary or {}
        steps = summary.get("steps") or {}
        check = summary.get("final_check") or {}
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
                problems.append(
                    f"verify fails for [{failure.get('step')}]: {failure.get('command')}"
                )
        return problems

    def _on_plan(self, plan) -> str:
        """Show the plan and block the worker until the bar answers.

        Returns what loop.run's on_plan contract expects: "" to execute the
        plan, "abort" to stop, anything else as feedback to re-plan with.
        Enter alone means "run it" — the common case should cost one
        keystroke — which is why the bar's usual empty-answer sentinel is
        mapped to acceptance here rather than treated as a refusal.
        """

        def show() -> None:
            lines = Text.assemble(("● Plan", "bold"), (f"  {len(plan.steps)} step(s)", "dim"))
            for i, step in enumerate(plan.steps, 1):
                lines.append(f"\n  {i}. ")
                lines.append(step.title, style="cyan")
                if step.files:
                    lines.append(f"  {', '.join(step.files)}", style="dim")
                if step.intent:
                    lines.append(f"\n     {step.intent}", style="dim")
                if step.verify:
                    lines.append("\n     verify: ", style="dim")
                    lines.append(step.verify)
            self._transcript.write(lines, classes="step")

        self.call_from_thread(show)
        answer = self._prompt_via_bar(
            "📋 [bold]plan ready[/] — Enter to run it, or type what to change",
            "Enter = execute, 'q' = stop, or describe what the plan should do differently",
        ).strip()
        if not answer or answer.startswith("(no answer"):
            return ""
        if answer.lower() in ("q", "quit", "exit", "abort"):
            return "abort"
        return answer

    def _on_approve(self, shown: str) -> str:
        return self._prompt_via_bar(
            f"🔒 approve — {escape(shown)}",
            "y / a(lways this) / auto (everything now on) / n  (Enter alone = no, "
            "or ctrl+a to cycle mode)",
        )

    def _on_ask_many(self, questions: list[tuple[str, list[str] | None, str | None]]) -> list[str]:
        """Clarify's questions as one form; blocks the worker until it is
        answered, like _ask_via_modal. One answer per question, in order."""
        holder: dict[str, list[str]] = {}
        done = threading.Event()

        def show() -> None:
            def handle(result: list[str] | None) -> None:
                holder["answers"] = list(result or [NO_ANSWER] * len(questions))
                done.set()

            self.push_screen(QuestionsModal(questions), handle)

        self.call_from_thread(show)
        done.wait()
        answers = holder.get("answers", [])
        return (answers + [NO_ANSWER] * len(questions))[: len(questions)]

    def _ask_via_modal(
        self, question: str, options: list[str], recommended: str | None = None
    ) -> str:
        answer_holder: dict[str, str] = {}
        done = threading.Event()

        def show_modal() -> None:
            def handle_result(result: str | None) -> None:
                answer_holder["answer"] = result or "(no answer given)"
                done.set()

            self.push_screen(ChoiceModal(question, options, recommended), handle_result)
            self._call_for_attention()

        self.call_from_thread(show_modal)
        done.wait()
        return answer_holder.get("answer", "(no answer given)")

    def action_stop_run(self) -> None:
        """Ask the run to stop at its next safe checkpoint.

        Not an instant kill: the worker thread may be mid HTTP-call, which
        cannot be interrupted from here (the same reason bench's --timeout
        runs a job in a whole separate process). Setting the token stops the
        *next* tool-loop iteration or plan step from starting, which is a
        bounded wait now that every streamed call has its own wall-clock
        deadline (see agent/llm.py) rather than a wait of unknown length.
        """
        if not self._agent_running:
            return
        self._cancel.cancel()
        self._transcript.write(
            "[yellow]stopping — finishing the current step, then wrapping up…[/]"
        )

    def action_quit_confirm(self) -> None:
        """Quit, but do not silently abandon a run that is still going.

        The first request while the agent is working asks it to stop and says
        so; the second exits regardless. Without this, quitting mid-run
        dropped a worker thread mid-step with no record of why the run
        ended — and with it, the common case (quitting when nothing is
        running) is still a single keystroke.
        """
        if not self._agent_running or self._quit_armed:
            self.exit(self.summary)
            return
        self._quit_armed = True
        self._cancel.cancel()
        self._transcript.write(
            "[yellow]a run is still going — asking it to stop. "
            "Quit again (ctrl+q) to exit immediately.[/]"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """The one button left: the workspace panel's native folder picker.
        The panels and actions are reached through the (clickable) footer."""
        if event.button.id == "btn-browse-workspace":
            self._browse_workspace()

    def on_text_selected(self, event: events.TextSelected) -> None:
        """Copy a mouse selection in the output as soon as it is made, the
        way the terminal's own copy-on-select does. The terminal cannot copy
        it: the drag went to this app, so the terminal's own selection is
        empty, and ⌘C copies that empty selection without ever reaching
        here (Ghostty binds super+c to its own copy by default)."""
        self.action_copy_selection()

    def action_copy_selection(self) -> None:
        """Copy what is selected: text selected in the transcript first,
        then the prompt box's own selection. Sent through the terminal (OSC
        52) always, and on a local Mac through pbcopy too, so copying works
        in terminals that ignore OSC 52."""
        selected = self.screen.get_selected_text()
        if not selected:
            selected = self.query_one("#goal-input", PromptBox).selected_text
        if selected:
            self.copy_to_clipboard(selected)
            command = self._local_copy_command()
            if command:
                copy_with(command, selected)
            self.notify(f"copied {len(selected):,} characters", timeout=1.5)

    def _local_copy_command(self) -> tuple[str, ...] | None:
        return local_clipboard_command()

    def action_select_all_here(self) -> None:
        """Select all of whatever you are working in: the prompt box's text
        while it has focus, otherwise the whole transcript."""
        bar = self.query_one("#goal-input", PromptBox)
        if self.focused is bar:
            self.screen.clear_selection()
            bar.select_all()
        else:
            self._transcript.text_select_all()
            self.action_copy_selection()

    def action_toggle_stats(self) -> None:
        panel = self.query_one("#stats-panel")
        panel.styles.display = "block" if panel.styles.display == "none" else "none"

    def action_toggle_settings(self) -> None:
        panel = self.query_one("#settings-panel")
        opening = panel.styles.display == "none"
        panel.styles.display = "block" if opening else "none"
        if opening:
            # Land keyboard focus on the first control immediately — without
            # this, opening the panel leaves focus wherever it already was
            # (often nowhere focusable), and a keyboard-only user would have
            # no way to reach a Select at all short of guessing at Tab.
            self.query_one("#mode-select", Select).focus()

    def action_toggle_workspace(self) -> None:
        panel = self.query_one("#workspace-panel")
        opening = panel.styles.display == "none"
        panel.styles.display = "block" if opening else "none"
        if opening:
            self.query_one("#btn-browse-workspace", Button).focus()

    def action_cycle_approval_mode(self) -> None:
        """The one-key version of changing #mode-select: manual -> auto-python
        -> auto-plan -> manual, without needing the settings panel open at
        all. Setting the Select's value (rather than settings fields
        directly) reuses on_select_changed for the actual update and the
        mode-line refresh, the same as a person picking it from the dropdown
        would."""
        order = ["manual", "auto-python", "auto-plan"]
        current = self._approval_mode()
        next_mode = order[(order.index(current) + 1) % len(order)]
        self.query_one("#mode-select", Select).value = next_mode
        self._transcript.write(f"[dim]approval mode →[/] [bold]{next_mode}[/]")


def watch_run(settings: Settings, plan_path: str | None = None, stop_after: str = "review") -> dict:
    """Run the agent inside the TUI and return its summary, same shape as
    agent.loop.run(). Blocks until the run finishes and the user presses q."""
    app = RunApp(settings, plan_path, stop_after)
    app.run()
    return app.summary or {"error": "the TUI closed before the run produced a summary"}
