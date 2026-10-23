"""Human-in-the-loop gate for shell commands and arbitrary code execution.

The allowlist only ever answers "is the leading word ordinarily safe" — it was
never meant to certify an entire command line, because the shell that actually
runs it (`shell=True` in shell.py) does not care what the first word was. A
compound command's later clauses run with exactly the same privilege as its
first: `git status && rm -rf ~/.ssh` executes the whole thing the moment `git`
is recognised, unless something here treats the compound structure itself as
the thing needing approval, independent of the allowlist. Nor does an
allowlisted leading word say anything about *where* it reads or writes:
`sed -i s/x/y/ ~/.ssh/authorized_keys` and `cp app.py ~/.bashrc` are both
single, non-compound, fully allowlisted commands — see `_escaping_path`,
which closes that gap for the commands whose argument grammar this can
confidently classify.

run_python (and ast_grep_rewrite's apply=true) carry the same risk through a
different door: arbitrary Python or a structural rewrite, not a shell string,
so the allowlist/denylist (which pattern-match shell syntax) do not apply —
but they still need the same auto_approve/prompt-or-deny gate, or they are a
zero-friction way to do anything the shell path is trying to guard against.

The prompt-or-deny gate is pluggable, the same way Trace.on_event and
Interactor.ask are: the default is a blocking input() on stdin, which is
exactly right for the plain CLI and exactly wrong for the TUI, which has put
the terminal into raw mode and reads every keystroke into its own widgets —
a stray input() call there blocks forever on a prompt nobody can ever see or
answer, indistinguishable from a hang. `harness run --tui` with no `--yes`
used to deadlock on the first command needing approval for exactly this
reason; the TUI now supplies its own callback that routes the prompt through
the same bottom bar ask_user uses.

`unattended=True` (only ever passed by `loop.py`'s real construction, never
by a test fixture reaching for `auto_approve=True` as a convenience default)
is the same shape of fix for a different deadlock: a bash command that is
neither allowlisted nor workspace-scoped used to run anyway the moment
`auto_approve` was set, trusted silently rather than gated, because nobody
was going to be there to answer a prompt. `unattended=True` turns that into a
denial instead — reported to the model and recorded in the trace, so a human
reviewing the run afterward sees exactly what it wanted to do and why it
could not — rather than a silent, ungated escape hatch that happened to look
identical to deliberate trust. Interactive `auto_approve` (typed as an answer
mid-session, by a human who is by definition present when they type it)
keeps its original meaning unchanged: full trust, no further checks.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from .config import BASH_ALLOWLIST, BASH_DENYLIST_SUBSTRINGS
from .interact import HUMAN_LOCK

# Any of these means "more than one command is about to run": sequencing (;),
# chaining (&&, ||), piping (|), command substitution ($(...) or backticks).
#
# Where they are found matters, and matching them everywhere was a bias this
# codebase chose deliberately on the grounds that a false positive "costs an
# extra approval prompt". Unattended, there is no prompt to pay: it is a hard
# denial. A traced run asked to run
# `uv run python -c 'import xgboost; import lightgbm'` -- one command, the
# semicolons inside a quoted Python string -- and was refused. The same
# string was that step's `verify`, so the step failed for being unverifiable
# rather than for being wrong, and the executor then burned all fourteen of
# its iterations trying to get the check to run at all.
#
# So quoting is honoured, exactly as the shell honours it: single quotes make
# everything literal; double quotes still expand $(...) and backticks but not
# ; | &&. Nothing that was refused before for an unquoted marker is accepted
# now.
_SEPARATORS = re.compile(r"&&|\|\||[;|&]")
_SUBSTITUTIONS = re.compile(r"[`]|\$\(")
# What Approver._wrapped_parts_pass removes before checking the parts: a stderr
# merge, and one final output cut with a numeric count (`| tail`, `| head -40`,
# `| tail -n 20`). Nothing else is removed, so nothing else can hide.
_STDERR_MERGE = re.compile(r"(?<=\s)2>&1(?=\s|$)")
_OUTPUT_CUT = re.compile(r"\s*\|\s*(?:head|tail)(?:\s+(?:-n\s*)?-?\d+)?\s*$")
# A path token whose meaning the shell decides at run time (Approver._resolves_inside_workspace).
_UNPLACEABLE = re.compile(r"[$`{}]|(?:^|[=:])~(?!/|$)")
# Shell syntax a wrapped part may not contain once 2>&1 and the output cut are
# removed: substitution, expansion, redirection, other operators, grouping,
# escapes, globs, a second line.
_RAW_SHELL_SYNTAX = re.compile(r"[`$<>;|&(){}\\*?\[\]\n]")


def _strip_quoted(command: str, keep_double: bool) -> str:
    """`command` with quoted spans blanked out, so a scan sees only the parts
    the shell would treat as syntax.

    `keep_double` keeps double-quoted spans, because the shell still expands
    `$(...)` and backticks inside them -- only single quotes make those
    literal. An unterminated quote is treated as quoting nothing, which keeps
    the scan conservative: a malformed command stays fully visible to it.
    """
    out: list[str] = []
    quote: str | None = None
    for char in command:
        if quote is None and char in ("'", '"'):
            quote = char
            out.append(" ")
            continue
        if quote is not None and char == quote:
            quote = None
            out.append(" ")
            continue
        blanked = quote is not None and not (keep_double and quote == '"')
        out.append(" " if blanked else char)
    if quote is not None:
        return command  # unterminated: scan the whole thing
    return "".join(out)


def is_compound(command: str) -> bool:
    """Whether `command` runs more than one thing, honouring shell quoting."""
    return bool(
        _SEPARATORS.search(_strip_quoted(command, keep_double=False))
        or _SUBSTITUTIONS.search(_strip_quoted(command, keep_double=True))
    )


# Denylist substrings, but tolerant of the whitespace and casing variation a
# model (or a person) writes without thinking: "rm  -rf   /" or "RM -RF /"
# previously matched nothing, because the check was a literal, case-sensitive
# substring test.
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


# Commands whose non-flag arguments are the files/directories they act on.
# For these, an otherwise-allowlisted, non-compound command still gates if
# any such argument resolves outside the workspace: the structured file
# tools (edit_file, write_file, move_file, delete_file) already enforce
# exactly this boundary for the operations these commands duplicate, and a
# bare leading-word check has nothing to say about where they read or write.
_PATH_ARG_COMMANDS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "diff",
        "cp",
        "mv",
        "mkdir",
        "touch",
        "tree",
        "find",
        "ls",
        "test",
        "[",
    }
)
# These take a pattern/script as their first non-flag argument, not a path —
# `sed s/x/y/ file` and `grep pattern file` would otherwise misclassify the
# pattern itself as an escaping path. Skipped unless -e/-f supplied the
# pattern separately, in which case every remaining non-flag token really is
# a path.
_SCRIPT_FIRST_COMMANDS = frozenset({"sed", "grep", "rg"})

# find's own execution primaries: each takes an arbitrary shell command as
# its argument, terminated by a bare `;` or `+`. Their tokens (the command
# name, its own flags stripped of their leading `-`, a quoted argument that
# happens to contain no `/`) are indistinguishable from ordinary relative
# path candidates by the classifier below, and every one of them resolves
# "inside the workspace" the same way a real path would — which is exactly
# how `find . -exec sh -c 'cat ~/.ssh/id_rsa' +` was waved through with no
# gate at all. Rather than trying to parse -exec's own embedded command
# (out of scope, per this module's docstring), a `find` invocation with any
# of these present is deliberately unclassifiable, which routes it through
# the ordinary approval gate like any other command this heuristic has no
# opinion about.
_FIND_EXEC_PRIMARIES = frozenset({"-exec", "-execdir", "-ok", "-okdir"})


def _classify_path_args(head: str, tokens: list[str]) -> list[str] | None:
    """Non-flag arguments to check against the workspace, for a command this
    heuristic can classify — or None for one it cannot, which the caller
    treats as "not confidently safe" rather than guessing.

    Not a parser for each tool's full argument grammar (git subcommands,
    find's -exec, and plenty else are out of scope) — the same deliberate
    under-diagnose bias envfail.py documents for a different kind of guess:
    an argument this cannot confidently classify falls through to the
    ordinary approval gate instead of being waved through.
    """
    if head in _SCRIPT_FIRST_COMMANDS:
        has_script_flag = any(t in ("-e", "-f") or t.startswith(("-e", "-f")) for t in tokens)
        candidates = [t for t in tokens if not t.startswith("-")]
        return candidates if has_script_flag else candidates[1:]
    if head == "find" and any(t in _FIND_EXEC_PRIMARIES for t in tokens):
        return None
    if head in _PATH_ARG_COMMANDS:
        return [t for t in tokens if not t.startswith("-")]
    return []  # not a command this heuristic has an opinion about


class Denied(Exception):
    pass


class Approver:
    def __init__(
        self,
        auto_approve: bool = False,
        ask: Callable[[str], str] | None = None,
        auto_approve_python: bool = False,
        notify: Callable[[str], None] | None = None,
        workspace_root: Path | None = None,
        unattended: bool = False,
    ):
        self.auto_approve = auto_approve
        # A narrower trust level than auto_approve: run_python is the only
        # tool this covers, never run_bash. It exists because run_python's
        # own callers (execution_registry, used only in the execute and fix
        # phases — never planning) mean every call it gates already happens
        # after a plan exists; the frequent prompt it was producing was for
        # the routine "does this work" snippets a step runs on itself, not
        # for anything a plan hadn't already scoped. This is a trust
        # decision, not a safety classifier — it does not inspect the
        # snippet's content, so it is exactly as permissive as -y for
        # Python specifically, while run_bash (installs, git, anything
        # else) still gates normally.
        self.auto_approve_python = auto_approve_python
        # ask=None means "prompt on stdin when there is a TTY" — the plain
        # CLI's default. A caller that wants full control (the TUI) passes
        # its own callable instead: given the "agent wants to ..." text, it
        # returns the same y/a/n-shaped answer a human would type at the
        # stdin prompt.
        self._ask = ask
        # Announces that the run is about to block. Wired to Trace.say by
        # loop.run, so the pause shows up in the console *and* the TUI log
        # rather than only in whatever surface happens to be reading stdin.
        self._notify = notify
        # Keyed on the exact command/snippet, not a leading word: approving
        # one `git` invocation must not blanket-approve every future command
        # that happens to start with `git`, chained clauses included.
        self.session_allowed: set[str] = set()
        # For _escaping_path. None (the default every test fixture gets)
        # means "no workspace to check against", which _escaping_path reads
        # as "cannot classify" — conservative, not permissive: see its
        # docstring.
        # Resolved here so the comparison in _resolves_inside_workspace, whose
        # candidate side is resolved, is like with like: an unresolved root
        # behind a symlink (macOS's /var -> /private/var) denied every path.
        self._workspace_root = Path(workspace_root).resolve() if workspace_root else None
        # See the module docstring's last paragraph. Captured once, at
        # construction, deliberately not re-derived from `self.auto_approve`
        # later: a human typing "auto" mid-session is present by definition
        # when they type it, and must keep today's full-trust meaning, not
        # newly start getting denied.
        self._unattended = unattended

    def check_verify(self, command: str) -> str:
        """Approve a step's verify command; return the exact command to run.

        A plain `a && b` chain -- which the harness itself writes for a step
        whose files span Python and a data or doc format -- is approved one
        piece at a time under the full run_bash rules, and what runs is those
        approved pieces rejoined. check_bash alone refuses every compound
        command when unattended, so such a step's verify could never pass
        there. Any other operator still goes to check_bash whole.
        """
        segments = and_chain_segments(command)
        if segments is None:
            self.check_bash(command)
            return command
        for segment in segments:
            self.check_bash(segment)
        return " && ".join(segments)

    def check_bash(self, command: str) -> None:
        normalized = _normalize(command)
        for bad in BASH_DENYLIST_SUBSTRINGS:
            if _normalize(bad) in normalized:
                raise Denied(f"command blocked by denylist: {bad!r}")

        head = command.strip().split()[0] if command.strip() else ""
        compound = is_compound(command)
        if compound and self._wrapped_parts_pass(command):
            return
        escaping = None
        # A compound command is not made safe by an allowlisted first word —
        # only an explicit, whole-command approval (or interactive
        # auto_approve) covers it.
        if not compound and head in BASH_ALLOWLIST:
            escaping = self._escaping_path(head, command)
            if escaping is None:
                return

        if compound:
            reason = "contains chained/piped commands"
        elif head not in BASH_ALLOWLIST:
            reason = f"'{head}' is not allowlisted"
        else:
            reason = f"would touch {escaping!r}, outside the workspace"
        self._gate(command, f"run:\n    $ {command}", reason, bash=True)

    def _wrapped_parts_pass(self, command: str) -> bool:
        """Whether a compound `command` is only plain commands that would each
        pass alone, joined by `&&`, with at most `2>&1` and one trailing
        `| head` or `| tail` with a numeric count; `cd` may come first if it
        stays inside the workspace, and later parts' paths are then checked
        from there.

        Unattended runs denied all of these, and each denial cost a round trip
        to the same command unwrapped: 20 of 21 denials in Mercury 3's django
        runs, 9 for Mercury 2.5, mostly `runtests.py ... 2>&1 | tail -40` and
        `cd tests && python runtests.py ...` (DESIGN-LOG §131.10).

        The parts checked are the raw text that runs, split on `&&`, not a
        re-quoted copy: a re-quoted part hides a `$(...)` that was in double
        quotes, and an operator shlex does not list (`&>`). Any part with shell
        syntax beyond plain words and quotes is refused outright.
        """
        root = self._workspace_root
        if root is None:
            return False
        inner = _OUTPUT_CUT.sub("", _STDERR_MERGE.sub(" ", command))
        base = root
        for i, part in enumerate(inner.split("&&")):
            if _RAW_SHELL_SYNTAX.search(part):
                return False
            try:
                tokens = shlex.split(part)
            except ValueError:
                return False
            if not tokens:
                return False
            if tokens[0] == "cd":
                if i > 0 or len(tokens) != 2 or tokens[1].startswith("-"):
                    return False
                # No `..` in a cd target: Path.resolve() follows a symlink before applying
                # `..`, the shell applies `..` to the text first, so `a/../../x` with `a` a
                # link deeper into the workspace resolves inside here and lands outside there.
                if ".." in Path(tokens[1]).parts:
                    return False
                if not self._resolves_inside_workspace(root, tokens[1]):
                    return False
                base = (root / Path(tokens[1]).expanduser()).resolve()
                continue
            if tokens[0] not in BASH_ALLOWLIST:
                return False
            if self._escaping_path(tokens[0], part, base) is not None:
                return False
        return True

    def _escaping_path(self, head: str, command: str, base: Path | None = None) -> str | None:
        """None if `head`'s command has no arguments this can confidently
        classify as a path, or every one resolves inside the workspace.
        Otherwise the first escaping argument found, for the denial message.
        """
        if self._workspace_root is None:
            return None
        try:
            tokens = shlex.split(command)[1:]
        except ValueError:
            return command  # unbalanced quoting; cannot classify -> gate
        candidates = _classify_path_args(head, tokens)
        if candidates is None:
            return command
        if head in _PATH_ARG_COMMANDS or head in _SCRIPT_FIRST_COMMANDS:
            # Any argument the shell would expand, not only the ones classified
            # as paths: `grep {x,/etc/passwd}` and `grep x$IFS/etc/passwd`
            # become `grep x /etc/passwd` from the pattern position (§131.10).
            for token in tokens:
                if _UNPLACEABLE.search(token):
                    return token
            # A flag's own value can be a path: `grep --file=/etc/passwd x`.
            candidates = candidates + [
                t.split("=", 1)[1] for t in tokens if t.startswith("-") and "=" in t
            ]
        root = self._workspace_root  # narrowed non-None by the check above
        for token in candidates:
            if not self._resolves_inside_workspace(root, token, base):
                return token
        return None

    def _resolves_inside_workspace(self, root: Path, token: str, base: Path | None = None) -> bool:
        """Whether `token`, as a path relative to `base` (default `root`, the
        workspace; a directory a leading `cd` moved to), lands inside `root`.
        A token the shell would expand cannot be placed without running it, so
        it does not count as inside (§131.10): `$HOME/...`, backticks, braces
        (`{/etc/passwd,x}`), and `~` other than `~` or `~/` (bash reads `~-`
        as OLDPWD and `~+` as PWD; Python's expanduser leaves them as names).
        Checked on the shlex token, which has had quotes and backslashes
        applied as bash applies them; a scan of the raw text was fooled by
        `{"/etc/passwd",x}`. Fails closed on a literal brace or a quoted
        tilde, which no §131 trace used."""
        if _UNPLACEABLE.search(token):
            return False
        try:
            # expanduser() on the token itself, before joining: a real shell
            # expands a leading ~ in its own argument regardless of where it
            # sits in the command, but `(root / "~/x").expanduser()` does not
            # -- expanduser() only ever acts on a path whose *own* first
            # component is "~", and by then it is buried after `root`'s. Not
            # joining an absolute (or ~-expanded) token onto root at all is
            # what a real `cp`/`sed`/etc. would do with it too.
            expanded = Path(token).expanduser()
            candidate = (
                expanded if expanded.is_absolute() else (base or root) / expanded
            ).resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        return candidate == root or root in candidate.parents

    def check_run(self, key: str, description: str) -> None:
        """Approve something that carries run_bash's risk without being shell
        syntax — a Python snippet, or a structural rewrite. No
        allowlist/denylist applies (they pattern-match shell syntax
        specifically); this is the same auto_approve/prompt-or-deny gate on
        its own, plus the narrower auto_approve_python trust level (see
        __init__).

        Deliberately does not pass `bash=True` to `_gate`: `unattended`
        gates run_bash's allowlist/denylist/compound-command/path checks,
        none of which apply here, and it is `auto_approve_python` — a
        decision already made separately per call — not `unattended`, that
        this path's trust should turn on.
        """
        if self.auto_approve_python:
            return
        self._gate(key, description, "is not shell syntax, so no allowlist applies")

    def _gate(self, key: str, shown: str, reason_if_no_tty: str, bash: bool = False) -> None:
        if self._blocked_unattended(key, bash):
            raise Denied(
                f"{reason_if_no_tty}, and this run is unattended, so it cannot "
                "pause to ask. Either the command needs to be allowlisted and "
                "workspace-scoped, or this run needs a human present to approve it."
            )
        if self.auto_approve or key in self.session_allowed:
            return
        # Serialised against every other prompt in the run: with
        # --step-concurrency > 1, two steps can need approval at the same
        # instant, and neither the stdin prompt nor the TUI's single-slot
        # bottom bar can represent two questions at once. Re-checked inside
        # the lock because the answer to the prompt we queued behind may
        # already have covered this exact call ("always", or a switch to
        # auto), in which case asking again would be asking twice for one
        # decision the human has already made.
        with HUMAN_LOCK:
            if self.auto_approve or key in self.session_allowed:
                return
            self._gate_locked(key, shown, reason_if_no_tty)

    def _blocked_unattended(self, key: str, bash: bool) -> bool:
        """True when this call would otherwise be silently waved through by
        `auto_approve` alone, on a run with nobody present to ask, and it is
        the run_bash gate `unattended` is scoped to (see check_run's
        docstring for why check_run never passes bash=True)."""
        return bash and self._unattended and self.auto_approve and key not in self.session_allowed

    def _announce_pause(self, shown: str) -> None:
        """Say plainly that the run has stopped and is waiting for a person.

        This exists because of a run that looked hung. Every shell command in
        it was allowlisted, so no prompt had appeared for its whole length;
        then run_python — which always gates, allowlist or not — printed two
        unstyled lines into a stream of dozens of similar-looking narration
        lines and blocked on input() forever. Worse, the narration for a tool
        is printed *before* the tool runs, so the terminal said "running a
        Python check" while the harness was in fact waiting to be told it
        could. Hence the explicit "nothing has run yet".

        Shows a one-line summary of `shown`, not just its first line: `shown`
        is `"run:\n    $ <command>"` for a shell command or
        `"run this Python:\n<code>"` for a snippet, so the actual content a
        person needs to recognise what is being asked about is always on the
        *second* line, never the first. Taking only the first line used to
        produce "waiting for your approval: run:" with the command itself
        nowhere in it — content-free at the one moment it matters most, since
        this is the line printed *before* the interactive prompt that does
        show the full text, and the only one a narration-only viewer (one
        that never reaches an interactive terminal at all) ever sees.
        """
        summary = " ".join(shown.split())
        if len(summary) > 160:
            summary = summary[:159] + "…"
        if self._notify:
            self._notify(f"\n⏸  PAUSED — waiting for your approval: {summary}")

    def _gate_locked(self, key: str, shown: str, reason_if_no_tty: str) -> None:
        self._announce_pause(shown)
        if self._ask is not None:
            answer = self._ask(shown).strip().lower()
        elif not sys.stdin.isatty():
            raise Denied(
                f"{reason_if_no_tty} and no TTY is available to ask. "
                "Re-run with --yes, or simplify to one allowlisted command."
            )
        else:
            print("\n" + "─" * 66)
            print("  PAUSED — this run needs your approval before it can continue.")
            print("  Nothing has run yet; the line above is what it wants to do.")
            print(f"\n  agent wants to {shown}")
            answer = (
                input(
                    "  allow? [y]es / [a]lways this exact one / [auto] everything from now "
                    "on / [python] auto-approve python from now on / [n]o: "
                )
                .strip()
                .lower()
            )
        # "auto"/"python" change the approval mode for the rest of this run,
        # not just this one call — the only way to raise trust mid-run
        # without restarting, since there is nowhere else in a plain CLI
        # session to type that. Checked before the "a" prefix below so
        # "auto" (which also starts with "a") switches mode rather than
        # merely allowlisting this one command; "always" still falls
        # through to that case unchanged.
        if answer in ("auto", "plan", "auto-plan"):
            self.auto_approve = True
            return
        if answer in ("python", "auto-python"):
            self.auto_approve_python = True
            return
        if answer.startswith("a"):
            self.session_allowed.add(key)
            return
        if answer.startswith("y"):
            return
        raise Denied("user denied it")


# Shell operators shlex splits out with punctuation_chars; only "&&" is allowed
# through check_verify, and only between commands that each pass on their own.
_SHELL_OPERATORS = frozenset({"&&", "||", ";", ";;", "|", "&", "<", ">", ">>", "<<", "(", ")"})


def and_chain_segments(command: str) -> list[str] | None:
    """[`a`, `b`] for a command that is exactly `a && b` (two or more parts, no
    other operator); None for anything else, including a single command."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    if any(t in _SHELL_OPERATORS and t != "&&" for t in tokens):
        return None
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "&&":
            segments.append([])
        else:
            segments[-1].append(token)
    if len(segments) < 2 or any(not s for s in segments):
        return None
    return [shlex.join(s) for s in segments]
