"""Direct unit tests for Approver's answer parsing (agent/approval.py) — in
particular the "auto"/"python" mid-run mode-switch answers, the only way a
plain CLI session (no TUI, no restart) has to raise its own trust level once
a run is already going."""

from __future__ import annotations

import pytest

from agent.approval import Approver, Denied


def _approver(answers: list[str], **kw) -> Approver:
    it = iter(answers)
    approver = Approver(**kw)
    approver._ask = lambda shown: next(it)
    return approver


def test_auto_answer_flips_auto_approve_and_approves_the_current_call():
    approver = _approver(["auto"])
    approver.check_bash("rm somefile.txt")  # not allowlisted -> gates, answers "auto"
    assert approver.auto_approve is True


def test_auto_answer_means_every_later_call_passes_without_asking():
    approver = _approver(["auto"])
    approver.check_bash("rm somefile.txt")
    # a second, different, still-unallowlisted command must not ask again
    approver.check_bash("some-other-unlisted-command")


def test_python_answer_flips_auto_approve_python_only():
    approver = _approver(["python"])
    approver.check_run("key1", "run some python")
    assert approver.auto_approve_python is True
    assert approver.auto_approve is False


def test_python_answer_does_not_auto_approve_bash():
    approver = _approver(["python", "n"])
    approver.check_run("key1", "run some python")
    assert approver.auto_approve_python is True
    # run_bash is a different gate entirely; auto_approve_python must not
    # leak into it — the second queued answer ("n") is consumed and denies.
    try:
        approver.check_bash("some-unlisted-command")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_always_still_only_allowlists_the_exact_command():
    """ "always" starts with "a", same as "auto" — must not be swallowed by
    the new auto/plan/auto-plan branch, which is checked first."""
    approver = _approver(["always"])
    approver.check_bash("rm somefile.txt")
    assert approver.auto_approve is False
    assert "rm somefile.txt" in approver.session_allowed
    # a different unlisted command must still ask (none queued -> StopIteration)
    try:
        approver.check_bash("a-different-unlisted-command")
        asked_again = False
    except StopIteration:
        asked_again = True
    assert asked_again


def test_plain_no_still_denies():
    approver = _approver(["n"])
    try:
        approver.check_bash("rm somefile.txt")
        raised = False
    except Denied:
        raised = True
    assert raised


# -- one human, one question at a time -----------------------------------


def test_two_threads_needing_approval_do_not_prompt_at_the_same_time():
    """With --step-concurrency > 1 two steps can need approval at the same
    instant. Neither the stdin prompt nor the TUI's single-slot bottom bar can
    represent two questions at once, so they must serialise."""
    import threading
    import time

    overlapping = []
    inside = []
    lock = threading.Lock()

    def slow_ask(shown: str) -> str:
        with lock:
            inside.append(1)
            overlapping.append(len(inside))
        time.sleep(0.05)
        with lock:
            inside.pop()
        return "y"

    approver = Approver()
    approver._ask = slow_ask
    threads = [
        threading.Thread(target=approver.check_bash, args=(f"rm file{i}.txt",)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlapping == [1, 1, 1, 1], "prompts overlapped instead of serialising"


def test_a_second_thread_does_not_re_ask_what_the_first_already_allowed():
    """Both threads want the same command. The one that queued behind should
    find it already approved rather than asking the human twice for one
    decision they have already made."""
    import threading

    asked = []

    def ask_once(shown: str) -> str:
        asked.append(shown)
        return "a"  # always-allow this exact command

    approver = Approver()
    approver._ask = ask_once
    threads = [
        threading.Thread(target=approver.check_bash, args=("rm same.txt",)) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(asked) == 1


def test_switching_to_auto_mid_run_stops_later_threads_asking():
    import threading

    asked = []

    def ask(shown: str) -> str:
        asked.append(shown)
        return "auto"

    approver = Approver()
    approver._ask = ask
    threads = [
        threading.Thread(target=approver.check_bash, args=(f"rm f{i}.txt",)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(asked) == 1
    assert approver.auto_approve is True


# -- a pause the human can actually notice --------------------------------


def test_a_gate_announces_the_pause_before_blocking():
    """A traced run looked hung: every shell command in it was allowlisted so
    no prompt had ever appeared, then run_python gated silently and the
    narration above it still read "running a Python check"."""
    notices: list[str] = []
    approver = _approver(["y"], notify=notices.append)
    approver.check_run("print(1)", "run this Python:\nprint(1)")
    assert notices, "the run must say out loud that it stopped"
    assert "PAUSED" in notices[0]
    assert "approval" in notices[0]


def test_the_pause_notice_names_what_is_being_asked_about():
    """A real traced pause said only "waiting for your approval: run:" --
    `shown` is "run:\\n    $ <command>", and taking just its first line drops
    the command onto a second line the notice never reached. Strict now, no
    "or 'run:'" fallback: the actual command must be there."""
    notices: list[str] = []
    approver = _approver(["y"], notify=notices.append)
    approver.check_bash("rm somefile.txt")
    assert "rm somefile.txt" in notices[0]


def test_the_pause_notice_names_a_python_snippet_too():
    notices: list[str] = []
    approver = _approver(["y"], notify=notices.append)
    approver.check_run("some-key", "run this Python:\nprint('hello world')")
    assert "hello world" in notices[0]


def test_a_long_command_is_clipped_not_dropped():
    notices: list[str] = []
    approver = _approver(["y"], notify=notices.append)
    long_command = "rm " + "x" * 300  # "rm" is not allowlisted -> always gates
    approver.check_bash(long_command)
    assert "rm" in notices[0]
    assert len(notices[0]) < 250


def test_nothing_is_announced_when_nothing_will_block():
    """auto-approve must stay silent — the announcement is about a stop."""
    notices: list[str] = []
    approver = Approver(auto_approve=True, notify=notices.append)
    approver.check_bash("rm somefile.txt")
    assert notices == []


def test_an_already_allowed_command_does_not_announce_again():
    notices: list[str] = []
    approver = _approver(["a"], notify=notices.append)
    approver.check_bash("rm same.txt")
    approver.check_bash("rm same.txt")
    assert len(notices) == 1, "the second call was already covered"


def test_auto_approve_python_never_announces():
    notices: list[str] = []
    approver = Approver(auto_approve_python=True, notify=notices.append)
    approver.check_run("print(1)", "run this Python:\nprint(1)")
    assert notices == []


# -- path-scoping: an allowlisted command still gates on where it points --


def test_allowlisted_write_command_with_in_workspace_path_needs_no_approval(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver.check_bash("touch notes.txt")  # relative, inside the workspace; must not raise


def test_allowlisted_write_command_with_an_escaping_path_still_gates(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver._ask = lambda shown: "n"
    try:
        approver.check_bash("cp secret.txt ~/leaked.txt")
        raised = False
    except Denied:
        raised = True
    assert raised, "cp is allowlisted, but ~/leaked.txt is outside the workspace"


def test_sed_script_argument_is_not_mistaken_for_a_path(tmp_path):
    """sed's first non-flag token is its script, not a file — misclassifying
    it would gate every ordinary in-place edit for no reason."""
    approver = Approver(workspace_root=tmp_path)
    approver.check_bash("sed s/x/y/ notes.txt")  # must not raise


def test_sed_dash_e_means_every_remaining_token_is_a_real_path(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    try:
        approver.check_bash("sed -e s/x/y/ /etc/hosts")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_grep_pattern_argument_is_not_mistaken_for_a_path(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver.check_bash("grep TODO notes.txt")  # must not raise


def test_unparseable_quoting_falls_through_to_the_ordinary_gate(tmp_path):
    """Cannot confidently classify the arguments -> gate rather than guess
    it is safe."""
    approver = Approver(workspace_root=tmp_path)
    try:
        approver.check_bash("cp 'unterminated")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_no_workspace_root_means_path_scoping_is_skipped_not_enforced(tmp_path):
    """The default every existing test fixture gets: no workspace_root means
    'cannot check', read conservatively as 'nothing to add', not as a new
    reason to deny — unchanged behaviour for every Approver() that does not
    opt into this."""
    approver = Approver()
    approver.check_bash("cp secret.txt ~/leaked.txt")  # must not raise


# -- find's -exec/-execdir/-ok: an embedded command, not a path -------------
# `find`'s own execution primaries take an arbitrary shell command as their
# argument. Its tokens (command name, flags with the leading `-` stripped,
# a quoted argument with no `/` in it) look exactly like ordinary relative
# path candidates to the classifier, and every one resolves "inside the
# workspace" the same way a real path would -- so
# `find . -exec sh -c 'cat ~/.ssh/id_rsa' +` was previously waved through
# with no gate at all, in attended or unattended mode alike.


def test_find_exec_is_not_silently_approved(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver._ask = lambda shown: "n"
    try:
        approver.check_bash("find . -exec sh -c 'cat ~/.ssh/id_rsa' +")
        raised = False
    except Denied:
        raised = True
    assert raised, "find -exec embeds an arbitrary command; it must not be waved through"


def test_find_execdir_is_not_silently_approved(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver._ask = lambda shown: "n"
    try:
        approver.check_bash("find . -execdir rm -rf {} +")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_find_ok_is_not_silently_approved(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver._ask = lambda shown: "n"
    try:
        approver.check_bash("find . -ok cat {} ;")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_find_exec_denies_even_under_blanket_auto_approve_when_unattended(tmp_path):
    """The scenario that made this a real gap, not a theoretical one: an
    unattended run with auto_approve=True previously ran this with nobody
    ever asked."""
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    try:
        approver.check_bash("find . -exec sh -c 'cat ~/.ssh/id_rsa' +")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_ordinary_find_without_exec_is_unaffected(tmp_path):
    approver = Approver(workspace_root=tmp_path)
    approver.check_bash("find . -name '*.py'")  # must not raise


# -- unattended: real runs stop trusting anything that would need a prompt -


def test_unattended_denies_a_non_allowlisted_command_without_ever_asking():
    asked = []
    approver = Approver(
        auto_approve=True, unattended=True, ask=lambda shown: asked.append(shown) or "y"
    )
    try:
        approver.check_bash("some-unlisted-command")
        raised = False
    except Denied:
        raised = True
    assert raised
    assert asked == [], "an unattended run must never block on a prompt"


def test_unattended_denies_a_compound_command():
    approver = Approver(auto_approve=True, unattended=True)
    try:
        approver.check_bash("git status && rm -rf ~")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_unattended_denies_a_path_escaping_allowlisted_command(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    try:
        approver.check_bash("cp secret.txt ~/leaked.txt")
        raised = False
    except Denied:
        raised = True
    assert raised


def test_unattended_still_runs_an_allowlisted_workspace_safe_command_silently(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("git status")  # no _ask configured; must not raise or block


def test_unattended_does_not_change_check_run_bench_relies_on(tmp_path):
    """bench sets auto_approve=True (so a real run sets unattended=True too)
    but never auto_approve_python=True. run_python and ast_grep_rewrite's
    apply=true must still bypass silently through check_run exactly as
    today, or every bench run breaks on its first run_python call — that
    trust decision belongs to auto_approve_python alone, not to bash's
    allowlist/path-scoping, which is what `unattended` gates."""
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_run("some-key", "run some python")  # must not raise


def test_unattended_has_no_effect_when_auto_approve_is_off():
    """unattended alone means nothing: it only ever gates the same silent
    bypass `auto_approve` would otherwise grant."""
    approver = Approver(auto_approve=False, unattended=True, ask=lambda shown: "y")
    approver.check_bash("some-unlisted-command")  # asks and gets "y"; must not raise


def test_a_human_typing_auto_mid_session_keeps_full_trust_not_unattended_denial():
    """Typing 'auto' means a human is present at that exact moment — it must
    keep today's full-bypass meaning, not start being denied as if this were
    an unattended run from the start."""
    approver = _approver(["auto"])
    approver.check_bash("rm somefile.txt")
    assert approver.auto_approve is True
    approver.check_bash("some-other-unlisted-command")  # must not raise either


# -- data-format/prose verify tools: the traced pandoc regression -----------
# A real run's synthesized verify command, `pandoc -f gfm -t plain -o
# /dev/null README.md`, failed with "'pandoc' is not allowlisted" on a
# machine where pandoc was genuinely installed -- BASH_ALLOWLIST had never
# been updated when agent.dataformats started generating commands that use
# it (and jq/yq/xmllint alongside it). The run spent 14 iterations trying to
# "install" a tool that was already there, then gave up.


def test_the_traced_pandoc_verify_command_is_now_allowlisted(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("pandoc -f gfm -t plain -o /dev/null README.md")  # must not raise


def test_yq_data_format_check_is_allowlisted(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("yq '.' config.json")  # must not raise


def test_jq_data_format_check_is_allowlisted(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("jq empty config.json")  # must not raise


def test_xmllint_is_allowlisted(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("xmllint --noout config.xml")  # must not raise


def test_npx_is_allowlisted_for_eslint(tmp_path):
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("npx --no-install eslint src/a.js")  # must not raise


def test_pandocs_dev_null_output_argument_is_not_misread_as_an_escaping_path(tmp_path):
    """pandoc is deliberately left out of the path-arg heuristic
    (_PATH_ARG_COMMANDS): it can't tell -o's *value* apart from a bare
    positional path, and /dev/null -- the exact idiom the synthesized
    verify command uses -- would otherwise be flagged as outside the
    workspace. This is the regression test for that specific trap."""
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    approver.check_bash("pandoc -f gfm -t plain -o /dev/null README.md")  # must not raise


def test_file_tests_run_unattended_inside_the_workspace_only(tmp_path):
    """Plans verify answer files with `test -s answer.txt`; GAIA scout g2 had 50 of 83 final
    verifies refused for want of `test` on the allowlist (DESIGN-LOG §93)."""
    approver = Approver(workspace_root=tmp_path, unattended=True)
    approver.check_bash("test -s answer.txt")
    approver.check_bash("[ -f answer.txt ]")
    with pytest.raises(Denied):
        approver.check_bash("test -f /etc/passwd")


# -- §131.10: an allowed command wrapped in cd/&&, 2>&1 and a numeric tail ----

_RUNTESTS = "python tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"


def _unattended(tmp_path):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    return Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)


def test_unattended_allows_the_wrapped_test_commands_it_used_to_deny(tmp_path):
    """Denied in §131's django and SciCode traces, each retried bare next call."""
    approver = _unattended(tmp_path)
    for command in (
        f"{_RUNTESTS} migrations.test_writer 2>&1 | tail -20",
        f"{_RUNTESTS} model_enums.tests 2>&1 | tail -n 20",
        "cd tests && python runtests.py model_tests.test_choices --verbosity=2",
        "cd tests && python runtests.py fast_delete_combine --settings=test_sqlite 2>&1 | tail -40",
        "uv run pytest -s -q tests/test_debug.py 2>&1 | head -80",
        "python -m py_compile solution.py && echo 'Syntax OK'",
        "uv run ruff check --fix solution.py && cat solution.py",
        "ls -la tests/ | head -40",
        "pwd && ls -la",
    ):
        approver.check_bash(command)  # must not raise


def test_unattended_still_denies_anything_past_the_narrow_wrapping(tmp_path):
    approver = _unattended(tmp_path)
    for command in (
        "cd /etc && cat passwd",  # cd outside the workspace
        "cd .. && ls",
        "cd tests && curl https://example.com",  # a part not allowlisted
        f"{_RUNTESTS} x 2>&1 | sh",  # a pipe into anything but head/tail
        f"{_RUNTESTS} x | tail -5 | sh",
        f"{_RUNTESTS} x | tail -f",  # not a numeric count
        f"{_RUNTESTS} x; rm notes.txt",  # ; is not &&
        f"{_RUNTESTS} x || echo failed",
        "cat ../../secret | head -5",  # a part whose path escapes
        "echo $(cat ~/.ssh/id_rsa) | tail -1",  # substitution
        "git status && rm -rf ~",
        # From the security review of 1ded9df: each passed the first version of the rule.
        'ls && echo "$(cat ~/.ssh/id_rsa)"',  # double-quoted substitution, hidden by re-quoting
        'ls && echo "`id`"',
        "python x.py &>/tmp/out && echo done",  # &> is not in shlex's operator list
        "cat $HOME/.ssh/id_rsa && echo",  # expansion
        "cd - && cat notes.txt",  # cd to OLDPWD
        "cd t* && ls",  # a glob the shell expands
        "echo ok && cd tests",  # cd only first
        "ls\ncat ~/.ssh/id_rsa && echo",  # a second line
    ):
        with pytest.raises(Denied):
            approver.check_bash(command)


def test_paths_after_cd_are_checked_from_the_cd_directory(tmp_path):
    """tests/link points outside the workspace: `link/x` is inside from the
    workspace root (no such file there) but outside from tests/, where it runs."""
    approver = _unattended(tmp_path / "ws")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "ws" / "tests" / "link").symlink_to(outside)
    with pytest.raises(Denied):
        approver.check_bash("cd tests && cat link/secret.txt")
    approver.check_bash("cd tests && cat lines.txt")  # an ordinary path from there still passes


def test_a_cd_target_with_dot_dot_through_a_symlink_is_denied(tmp_path):
    """Python resolves the symlink before `..`, the shell applies `..` to the text first:
    with ws/a -> ws/deep/dir, `a/../../x` is ws/x to Path.resolve() but the shell lands in
    the workspace's parent. Checked against a real /bin/sh, then the approver."""
    import subprocess

    ws = tmp_path / "ws"
    approver = _unattended(ws)
    (ws / "deep" / "dir").mkdir(parents=True)
    (ws / "x").mkdir()
    (ws / "x" / "secret.txt").write_text("INSIDE")
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "secret.txt").write_text("OUTSIDE")
    (ws / "a").symlink_to(ws / "deep" / "dir")
    command = "cd a/../../x && cat secret.txt"
    shell = subprocess.run(["/bin/sh", "-c", command], cwd=ws, capture_output=True, text=True)
    assert shell.stdout == "OUTSIDE"  # the differential is real
    with pytest.raises(Denied):
        approver.check_bash(command)
    approver.check_bash("cd x && cat secret.txt")  # a plain subdirectory still passes


def test_a_wrapped_command_runs_through_run_bash_unattended(tmp_path):
    """The approval is only useful if run_bash then runs it: a real shell, a
    real workspace, cd into a subdirectory and the output cut."""
    from agent.tools import ToolContext
    from agent.tools.shell import run_bash
    from agent.trace import Trace
    from agent.workspace import Workspace

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "lines.txt").write_text("".join(f"line {i}\n" for i in range(50)))
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True, unattended=True, workspace_root=tmp_path),
        trace=Trace(str(tmp_path / "traces"), run_id="t", echo=False),
    )
    out = run_bash(ctx, "cd tests && cat lines.txt 2>&1 | tail -3")
    assert "line 49" in out and "line 46" not in out


def test_a_path_the_shell_would_expand_is_not_taken_as_inside(tmp_path):
    """Before §131.10 `cat $HOME/.ssh/id_rsa` passed: `$HOME/...` read as a
    folder inside the workspace."""
    approver = Approver(auto_approve=True, unattended=True, workspace_root=tmp_path)
    for command in ("cat $HOME/.ssh/id_rsa", "cp notes.txt ${HOME}/x", "cat `echo ~`/.ssh/id_rsa"):
        with pytest.raises(Denied):
            approver.check_bash(command)
    approver.check_bash("cat notes.txt")  # an ordinary relative path still passes


def test_brace_and_tilde_expansion_python_reads_differently_from_bash(tmp_path):
    """Second review of §131.10: bash expands `~-` (OLDPWD), `~+` (PWD) and
    `{a,b}`; Python's expanduser and Path take them as names in the workspace."""
    approver = _unattended(tmp_path)
    for command in (
        "cd ~- && ls",
        "cat ~+/../secret",
        "cat {/etc/passwd,notes.txt}",
        "cp notes.txt {/tmp/x,y}",
        # Third review: quoting and escapes inside the braces fooled a raw-text scan.
        'cat {"/etc/passwd",x}',
        "cat {/etc/passwd,\\ x}",
        'cp notes.txt {"/abs/path",}',
        "cd tests && cat {../../x,y}",
    ):
        with pytest.raises(Denied):
            approver.check_bash(command)
    for command in ("find . -type f -name '*.py'", "cat tests/lines.txt", "cd tests && ls"):
        approver.check_bash(command)  # must not raise


def test_expansion_from_a_pattern_or_flag_position_cannot_add_a_path(tmp_path):
    """Fourth review: grep's pattern and every flag skipped the path check, so
    brace or $IFS expansion there could add `/etc/passwd` as a file argument;
    and a flag's own value was never checked. 10 of 1,511 run_bash calls in
    all bench traces had a path command with any of these characters."""
    approver = _unattended(tmp_path)
    for command in (
        "grep {x,/etc/passwd}",
        "grep x$IFS/etc/passwd",
        "sed -n {p,/etc/passwd}",
        "cat --{x,/etc/passwd}",
        "grep --file=/etc/passwd x",
        "diff --from-file=/etc/passwd notes.txt",
        "grep 'end$' notes.txt",  # fails closed: a literal $ cannot be told from an expansion
    ):
        with pytest.raises(Denied):
            approver.check_bash(command)
    for command in (
        "grep -n TODO notes.txt",
        "grep --color=never TODO notes.txt",
        "sed -n 1,5p notes.txt",
    ):
        approver.check_bash(command)  # must not raise
