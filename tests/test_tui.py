"""The Textual live-run dashboard.

Every test here runs in-process, as a real `async def` pytest-asyncio test.
That wasn't always true: this file used to run each test in its own
`python3` subprocess, because running more than one headless Textual
`App.run_test()` session in the same process was found to reliably deadlock
while first debugging this file, cause never pinned down.

That finding no longer reproduces. Investigated directly, not assumed: once
the whole codebase's cross-thread UI updates were switched to
`call_from_thread` everywhere (see `RunApp`'s own docstring) and no
`post_message`-from-a-background-thread call remained, this file's full set
of test bodies -- run as genuine in-process pytest-asyncio tests,
sequentially, repeated three times, and under `pytest -n auto` -- passed
cleanly every time, zero deadlocks. A pilot batch of the highest-risk shapes
(a real thread via `call_from_thread`, a blocking `ask_user` wait, a pushed
modal) held up the same way before the rest of the file converted. Working
theory: the two deadlocks this file's history mentions were never two
separate Textual bugs, but one and the same stale-cross-thread-messaging
hazard, and fixing the `post_message` one incidentally fixed this one too --
nobody re-tested the "two apps" claim afterward because the subprocess
workaround was already in place. Not certain (the original cause was never
pinned down either), but tested, not guessed.

Converting surfaced a second, genuine bug along the way, not just a speed
number. Several tests wait for a background thread with
`Thread.join(timeout=5)`, and every one of those threads calls into the app
via `call_from_thread` (`_on_trace_event`/`_on_ask`/`_on_approve` all route
through it). `call_from_thread` schedules its callback on the app's own
event loop via `asyncio.run_coroutine_threadsafe` and then blocks the
*worker* thread on `future.result()` -- which needs that event loop, running
on *this* thread, to actually dispatch the callback. `Thread.join()` is a
synchronous, non-yielding wait: while it blocks, this thread's event loop
cannot run anything, including the very callback the worker is waiting on.
Confirmed directly, not inferred: a version with `timeout=5` always took
almost exactly 5.17s -- the join doesn't succeed, it *expires*, and the
`await` right after is what actually lets the stuck callback run; a version
with no timeout at all never returned, full stop, watched with a watchdog
thread over a real wall clock. `join_thread()` below is the real fix, not a
faster band-aid: it polls with an actual `await asyncio.sleep(...)` between
checks, which yields control back to the loop on every iteration, so the
callback runs almost immediately instead of after a manufactured wait.
Measured: 0.011s instead of ~5.2s per affected call site.

One instance of the same insight had already been worked out independently,
for the native-folder-picker tests, before this file's full conversion --
its own comment said as much ("Not a bare t.join(): the worker's own
call_from_thread calls need this same event loop to keep running while it
waits"). It just hadn't been generalized to the other call sites shaped the
same way. The lesson that matters here is that one: a concurrency fix found
once needs an explicit sweep for the same shape elsewhere, not just a fix
in the one place it was noticed.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.config import Settings
from agent.tui import ChoiceModal, PromptBox, RunApp, Transcript


async def join_thread(t: threading.Thread, timeout: float = 5.0) -> None:
    """Wait for a background thread without blocking the event loop it may
    need. See this module's own docstring for why `Thread.join()` here is
    not just slower but a genuine deadlock risk, and why polling with a real
    `await` between checks is the actual fix, not a workaround."""
    step = 0.01
    waited = 0.0
    while t.is_alive() and waited < timeout:
        await asyncio.sleep(step)
        waited += step


@pytest.fixture(autouse=True)
def _no_real_pbcopy(monkeypatch):
    """Copying in these tests must not overwrite the real clipboard of
    whoever runs them; the local-copy tests opt back in with a stand-in."""
    monkeypatch.setattr(RunApp, "_local_copy_command", lambda self: None)


class StubbedApp(RunApp):
    # The real agent worker is replaced by a no-op in every test here: these
    # tests are about the UI wiring, not about running a real (paid) agent.
    def _run_agent(self):
        pass


def make_app(**kw) -> StubbedApp:
    kw.setdefault("goal", "test goal")
    settings = Settings(workspace=".", **kw)
    return StubbedApp(settings, None, "review")


# -- basic mount --------------------------------------------------------

# -- markdown-vs-Rich-markup detection, tested directly (pure functions, no
# App instance -- these never touch run_test(), so they carry none of this
# file's own subprocess-isolation requirement) -----------------------------


def test_looks_like_markdown_detects_a_heading():
    from agent.tui import _looks_like_markdown

    assert _looks_like_markdown("## Summary\n\nsome text")


def test_looks_like_markdown_detects_bold():
    from agent.tui import _looks_like_markdown

    assert _looks_like_markdown("here are **the results**")


def test_looks_like_markdown_is_false_for_ordinary_narration():
    from agent.tui import _looks_like_markdown

    assert not _looks_like_markdown("[bold]plan[/] — 3 steps")
    assert not _looks_like_markdown("running: grep -n '#TODO' *.py")
    assert not _looks_like_markdown("understood as: build a thing")


@pytest.mark.asyncio
async def test_on_approve_empty_submit_denies():
    """An empty bar submit must resolve to something Approver's _gate treats
    as a denial (not "y"/"a"-prefixed) — never leave the worker hanging."""
    import threading

    app = make_app()
    async with app.run_test() as pilot:
        result = {}

        def worker():
            result["answer"] = app._on_approve("run: rm scratch.py")

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        await pilot.press("enter")  # submit with the bar left empty
        await join_thread(t)
        assert not result["answer"].lower().startswith(("y", "a"))


# -- stopping a run in progress ('c') ---------------------------------------


@pytest.mark.asyncio
async def test_stop_binding_is_a_noop_before_a_run_starts():
    """Nothing running yet (the initial-goal prompt) — 'c' must not raise or
    leave a stray cancelled token behind for the run that's about to start."""
    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "initial"
        await pilot.press("c")
        await pilot.pause()
        assert app._cancel.requested is False


@pytest.mark.asyncio
async def test_stop_request_is_visible_in_the_log():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_running = True
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        await pilot.press("c")
        await pilot.pause()
        assert len(log.entries) > before


@pytest.mark.asyncio
async def test_a_stopped_run_shows_stopped_not_failed():
    import types

    app = make_app()

    def fake_run_agent(self):
        app.summary = {"error": "stopped: stopped by user", "failure_log_path": ""}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "done"

        text = app.query_one("#log", Transcript).plain_text()
        assert "stopped:" in text
        assert "FAILED" not in text


# -- ask_user via the ChoiceModal (with options) ---------------------------

# -- prompting for a missing initial goal ----------------------------------


@pytest.mark.asyncio
async def test_no_goal_at_launch_shows_the_prompt_in_the_bar():
    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "initial"
        bar = app.query_one("#goal-input", PromptBox)
        assert "what should the agent build" in str(bar.placeholder).lower()


@pytest.mark.asyncio
async def test_answering_the_initial_prompt_starts_the_run():
    import types

    from agent.tui import PromptBox

    app = make_app(goal="")
    calls = []

    def fake_run_agent(self):
        calls.append(app.settings.goal)
        app.summary = {"suggested_next_step": ""}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "build a thing"
        await pilot.press("enter")
        await pilot.pause()
        assert calls == ["build a thing"]


@pytest.mark.asyncio
async def test_an_empty_initial_answer_exits_cleanly_without_starting_a_run():
    import types

    app = make_app(goal="")
    calls = []
    app._run_agent = types.MethodType(lambda self: calls.append(1), app)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")  # submit with the bar left empty
        await pilot.pause()
        assert calls == [], "an empty answer must not start a run"
        assert app.summary == {"error": "no goal given"}


@pytest.mark.asyncio
async def test_a_goal_already_given_at_launch_skips_the_prompt():
    import types

    app = make_app()  # make_app's default goal is "test goal"
    calls = []
    app._run_agent = types.MethodType(lambda self: calls.append(app.settings.goal), app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "running"
        assert calls == ["test goal"]


@pytest.mark.asyncio
async def test_a_plan_path_with_no_goal_skips_the_prompt_too():
    import types

    from agent.config import Settings

    settings = Settings(workspace=".", goal="")
    app = StubbedApp(settings, "/some/plan.json", "review")
    calls = []
    app._run_agent = types.MethodType(lambda self: calls.append(app.plan_path), app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "running"
        assert calls == ["/some/plan.json"]


# -- the "next goal" bar after a run finishes -------------------------------


@pytest.mark.asyncio
async def test_done_shows_the_suggestion_but_leaves_the_bar_empty():
    """Pre-filling made the machine's idea the default: across seventeen traced
    runs, ten goals were the previous run's suggestion accepted with a
    keystroke, while the feature the person kept asking for went unbuilt."""
    import types

    from agent.tui import PromptBox

    app = make_app()

    def fake_run_agent(self):
        app.summary = {"suggested_next_step": "Should errors be retried or surfaced?"}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "done"
        bar = app.query_one("#goal-input", PromptBox)
        assert bar.value == "", f"the box must start empty, got {bar.value!r}"
        assert bar.disabled is False
        text = app.query_one("#log", Transcript).plain_text()
        assert "Should errors be retried" in text, "the question is still shown"


@pytest.mark.asyncio
async def test_pressing_enter_on_an_empty_bar_starts_nothing():
    """With no pre-fill there is nothing to accept by reflex; a new run needs a
    goal the person actually typed."""
    import types

    app = make_app()
    calls = []

    def fake_run_agent(self):
        calls.append(app.settings.goal)
        app.summary = {"suggested_next_step": "Should errors be retried or surfaced?"}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert calls == ["test goal"]
        await pilot.press("enter")
        await pilot.pause()
        assert calls == ["test goal"], "an empty bar must not start a run"
        for ch in "make the bird flap":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()
        assert calls == ["test goal", "make the bird flap"]


@pytest.mark.asyncio
async def test_the_next_goal_can_be_overridden_with_different_text():
    import types

    from agent.tui import PromptBox

    app = make_app()
    calls = []

    def fake_run_agent(self):
        calls.append(app.settings.goal)
        app.summary = {"suggested_next_step": "the model's suggestion"}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "a completely different goal"
        await pilot.press("enter")
        await pilot.pause()
        assert calls == ["test goal", "a completely different goal"]


@pytest.mark.asyncio
async def test_an_empty_submit_in_done_mode_does_nothing():
    """No suggestion, nothing typed: staying idle beats erroring or re-running."""
    import types

    app = make_app()
    calls = []

    def fake_run_agent(self):
        calls.append(1)
        if len(calls) == 1:
            app.summary = {"suggested_next_step": ""}
            app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert calls == [1]
        await pilot.press("enter")  # bar is empty, no suggestion either
        await pilot.pause()
        assert calls == [1], "an empty submit with no suggestion must not start a run"


@pytest.mark.asyncio
async def test_continuing_resets_stats_for_the_next_run():
    import types

    from textual.widgets import Static

    app = make_app()

    def fake_run_agent(self):
        app.summary = {"suggested_next_step": ""}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        app._render_event({"kind": "llm_call", "cost_usd": 1.2345})
        await pilot.pause()
        assert "1.2345" in str(app.query_one("#stats", Static).content)

        from agent.tui import PromptBox

        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "next goal"
        await pilot.press("enter")
        await pilot.pause()
        assert "0.0000" in str(app.query_one("#stats", Static).content)


@pytest.mark.asyncio
async def test_a_failed_run_still_offers_the_bar_for_a_new_goal():
    import types

    from agent.tui import PromptBox

    app = make_app()

    def fake_run_agent(self):
        app.summary = {"error": "boom"}
        app._finish()

    app._run_agent = types.MethodType(fake_run_agent, app)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "done"
        assert app.query_one("#goal-input", PromptBox).disabled is False


# -- the stats side panel ('s') ----------------------------------------------


@pytest.mark.asyncio
async def test_stats_panel_is_hidden_by_default():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one("#stats-panel")
        assert panel.styles.display == "none"


@pytest.mark.asyncio
async def test_pressing_s_shows_the_stats_panel():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        panel = app.query_one("#stats-panel")
        assert panel.styles.display == "block"


@pytest.mark.asyncio
async def test_pressing_s_twice_hides_it_again():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.press("s")
        await pilot.pause()
        panel = app.query_one("#stats-panel")
        assert panel.styles.display == "none"


@pytest.mark.asyncio
async def test_the_stats_panel_still_has_the_live_numbers_once_shown():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "llm_call", "cost_usd": 1.5})
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        stats = app.query_one("#stats", Static)
        assert "1.5000" in str(stats.content)


# -- the "still waiting" indicator -------------------------------------------


@pytest.mark.asyncio
async def test_wait_indicator_is_blank_when_nothing_is_waiting():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.4)
        indicator = app.query_one("#wait-indicator", Static)
        assert str(indicator.content).strip() == ""


@pytest.mark.asyncio
async def test_wait_indicator_animates_while_a_call_is_in_flight():
    import threading

    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        threading.Thread(target=app._wait_start).start()
        await pilot.pause(0.4)  # a few 0.15s ticks
        indicator = app.query_one("#wait-indicator", Static)
        assert str(indicator.content).strip() != ""
        threading.Thread(target=app._wait_stop).start()
        await pilot.pause(0.4)
        assert str(indicator.content).strip() == ""


@pytest.mark.asyncio
async def test_wait_indicator_refcounts_across_concurrent_calls():
    """--step-concurrency > 1 means more than one worker thread can have a
    call in flight; one finishing must not turn the indicator off while
    another is still waiting."""
    import threading

    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        threading.Thread(target=app._wait_start).start()
        threading.Thread(target=app._wait_start).start()
        await pilot.pause(0.1)
        threading.Thread(target=app._wait_stop).start()  # one of two calls finished
        await pilot.pause(0.4)
        indicator = app.query_one("#wait-indicator", Static)
        assert str(indicator.content).strip() != "", "the other call is still waiting"
        threading.Thread(target=app._wait_stop).start()
        await pilot.pause(0.4)
        assert str(indicator.content).strip() == ""


# -- the settings panel ('p') and its dropdowns ------------------------------


@pytest.mark.asyncio
async def test_settings_panel_is_hidden_by_default():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        assert panel.styles.display == "none"


@pytest.mark.asyncio
async def test_pressing_p_shows_the_settings_panel():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        assert panel.styles.display == "block"


@pytest.mark.asyncio
async def test_pressing_p_twice_hides_it_again():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.press("p")
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        assert panel.styles.display == "none"


@pytest.mark.asyncio
async def test_mode_select_starts_on_manual_by_default():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one("#mode-select", Select)
        assert select.value == "manual"


@pytest.mark.asyncio
async def test_mode_select_reflects_settings_already_set_before_launch():
    from textual.widgets import Select

    app = make_app(auto_approve_python=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one("#mode-select", Select)
        assert select.value == "auto-python"


@pytest.mark.asyncio
async def test_changing_the_mode_select_updates_settings_immediately():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one("#mode-select", Select)
        select.value = "auto-plan"
        await pilot.pause()
        assert app.settings.auto_approve is True
        assert app.settings.auto_approve_python is True


@pytest.mark.asyncio
async def test_changing_the_mode_select_updates_the_mode_line():
    from textual.widgets import Select, Static

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#mode-select", Select).value = "auto-python"
        await pilot.pause()
        mode_line = str(app.query_one("#mode-line", Static).content)
        assert "auto-python" in mode_line


@pytest.mark.asyncio
async def test_narrate_select_toggles_settings_narrate():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.settings.narrate is True
        app.query_one("#narrate-select", Select).value = "off"
        await pilot.pause()
        assert app.settings.narrate is False


@pytest.mark.asyncio
async def test_quick_select_toggles_settings_quick_answers():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.settings.quick_answers is True
        app.query_one("#quick-select", Select).value = "off"
        await pilot.pause()
        assert app.settings.quick_answers is False


@pytest.mark.asyncio
async def test_animate_select_toggles_the_wait_animation_flag():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._show_wait_animation is True
        app.query_one("#animate-select", Select).value = "off"
        await pilot.pause()
        assert app._show_wait_animation is False


@pytest.mark.asyncio
async def test_animate_off_keeps_the_indicator_blank_even_while_waiting():
    import threading

    from textual.widgets import Select, Static

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#animate-select", Select).value = "off"
        await pilot.pause()
        threading.Thread(target=app._wait_start).start()
        await pilot.pause(0.4)
        indicator = app.query_one("#wait-indicator", Static)
        assert str(indicator.content).strip() == ""
        threading.Thread(target=app._wait_stop).start()
        await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_mode_line_is_visible_from_the_start():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        mode_line = str(app.query_one("#mode-line", Static).content)
        assert "manual" in mode_line
        assert "narrate" in mode_line


# -- reliable hotkeys: ctrl+p / ctrl+a work regardless of what has focus -----


@pytest.mark.asyncio
async def test_plain_p_is_swallowed_as_text_when_the_goal_bar_is_focused():
    """The known, documented limitation this next test's ctrl+p fixes —
    kept here so a regression in either direction is caught."""
    from agent.tui import PromptBox

    app = make_app(goal="")  # initial mode: the bar has focus
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        assert panel.styles.display == "none"
        assert app.query_one("#goal-input", PromptBox).value == "p"


@pytest.mark.asyncio
async def test_ctrl_p_opens_settings_even_with_the_goal_bar_focused():
    app = make_app(goal="")  # initial mode: the bar has focus
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        assert panel.styles.display == "block"


@pytest.mark.asyncio
async def test_opening_settings_focuses_the_mode_select():
    from textual.widgets import Select

    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.focused, Select)
        assert app.focused.id == "mode-select"


@pytest.mark.asyncio
async def test_command_palette_is_disabled_so_ctrl_p_is_free_for_settings():
    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.use_command_palette is False


@pytest.mark.asyncio
async def test_ctrl_a_cycles_the_approval_mode_even_with_the_bar_focused():
    from textual.widgets import Select

    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._approval_mode() == "manual"
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert app._approval_mode() == "auto-python"
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert app._approval_mode() == "auto-plan"
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert app._approval_mode() == "manual"
        assert app.query_one("#mode-select", Select).value == "manual"


@pytest.mark.asyncio
async def test_ctrl_a_updates_the_mode_line():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+a")
        await pilot.pause()
        mode_line = str(app.query_one("#mode-line", Static).content)
        assert "auto-python" in mode_line


@pytest.mark.asyncio
async def test_clicking_the_mode_select_opens_its_overlay():
    """Regression guard for the reported 'nothing happens when I click the
    dropdown' bug — a real mouse click (not a programmatic .value=) must
    expand the Select once the panel is open."""
    from textual.widgets import Select

    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        select = app.query_one("#mode-select", Select)
        await pilot.click(select)
        await pilot.pause()
        assert select.expanded is True


# -- the new settings dropdowns ------------------------------------------


@pytest.mark.asyncio
async def test_clarify_select_toggles_the_setting():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.settings.clarify is True
        app.query_one("#clarify-select", Select).value = "off"
        await pilot.pause()
        assert app.settings.clarify is False


@pytest.mark.asyncio
async def test_final_check_select_toggles_the_setting():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.settings.final_check is True
        app.query_one("#final-check-select", Select).value = "off"
        await pilot.pause()
        assert app.settings.final_check is False


@pytest.mark.asyncio
async def test_review_plan_select_toggles_the_setting():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.settings.review_plan is False
        app.query_one("#review-plan-select", Select).value = "on"
        await pilot.pause()
        assert app.settings.review_plan is True


@pytest.mark.asyncio
async def test_the_selects_reflect_settings_given_before_launch():
    from textual.widgets import Select

    app = make_app(clarify=False, final_check=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#clarify-select", Select).value == "off"
        assert app.query_one("#final-check-select", Select).value == "off"


# -- the run's completion state tells the truth --------------------------


@pytest.mark.asyncio
async def test_a_clean_run_still_says_done():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.summary = {
            "ok": True,
            "steps": {"failed": [], "abandoned": []},
            "final_check": {"ok": True},
        }
        app._finish()
        await pilot.pause()
        text = app.query_one("#log", Transcript).plain_text()
        assert "✓ done" in text
        assert "did not pass" not in text


@pytest.mark.asyncio
async def test_a_run_with_failed_steps_does_not_claim_success():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.summary = {
            "ok": False,
            "steps": {"failed": ["s2"], "abandoned": []},
            "final_check": None,
        }
        app._finish()
        await pilot.pause()
        text = app.query_one("#log", Transcript).plain_text()
        assert "did not pass its own checks" in text
        assert "finished with failures" in text


@pytest.mark.asyncio
async def test_a_failing_final_check_is_shown_in_the_log():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.summary = {
            "ok": False,
            "steps": {"failed": [], "abandoned": []},
            "final_check": {
                "ok": False,
                "tests_ran": True,
                "tests_passed": False,
                "tests_summary": "2 failed",
                "verify_failures": [],
            },
        }
        app._finish()
        await pilot.pause()
        text = app.query_one("#log", Transcript).plain_text()
        assert "2 failed" in text


@pytest.mark.asyncio
async def test_outcome_problems_is_empty_for_a_clean_summary():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.summary = {
            "ok": True,
            "steps": {"failed": [], "abandoned": []},
            "final_check": {"ok": True},
        }
        assert app._outcome_problems() == []


# -- the stats panel: per prompt and per session -------------------------


@pytest.mark.asyncio
async def test_the_workspace_and_stats_panels_render_left_of_the_log():
    """Compose order, which is render order: workspace (the file browser,
    styled like an IDE's leftmost panel) first, then stats, then the log,
    then settings on the far side."""
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        order = [c.id for c in app.query_one("#body").children]
        assert order[0] == "workspace-panel", order
        assert order.index("workspace-panel") < order.index("stats-panel") < order.index("log")


@pytest.mark.asyncio
async def test_a_model_call_adds_to_both_the_run_and_session_tallies():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._render_event(
            {
                "kind": "llm_call",
                "cost_usd": 0.01,
                "latency_s": 2.0,
                "ttft_s": 0.5,
                "completion_tokens": 100,
            }
        )
        assert app._run.calls == 1
        assert app._session.calls == 1
        assert abs(app._run.cost_usd - 0.01) < 1e-9
        assert abs(app._run.decode_s - 1.5) < 1e-9


@pytest.mark.asyncio
async def test_a_new_prompt_resets_the_run_tally_but_not_the_session():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._render_event(
            {
                "kind": "llm_call",
                "cost_usd": 0.02,
                "latency_s": 1.0,
                "ttft_s": 0.2,
                "completion_tokens": 50,
            }
        )
        app._submit_next_goal("another goal")
        await pilot.pause()
        assert app._run.calls == 0, "this prompt's tally should start clean"
        assert app._session.calls == 1, "the session total must survive"
        assert abs(app._session.cost_usd - 0.02) < 1e-9


@pytest.mark.asyncio
async def test_the_stats_text_shows_both_sections():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app._stats_text()
        assert "this prompt" in text
        assert "session total" in text
        assert "decoding" in text
        assert "model time" in text


# -- clickable buttons ----------------------------------------------------


async def click_footer(pilot, app, action: str) -> None:
    """Click the footer key bound to `action`: the mouse route to every
    panel and action, now that the row of big buttons is gone."""
    from textual.widgets._footer import FooterKey

    keys = [k for k in app.query(FooterKey) if k.action == action]
    assert keys, f"no footer key for {action}"
    await pilot.click(keys[0])


@pytest.mark.asyncio
async def test_every_action_is_reachable_from_the_footer():
    """The buttons existed because a binding you have to already know about
    is not discoverable. The footer now carries that job: every action has
    a visible, clickable key there."""
    from textual.widgets import Button
    from textual.widgets._footer import FooterKey

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        actions = {k.action for k in app.query(FooterKey)}
        assert actions >= {
            "toggle_workspace",
            "toggle_settings",
            "toggle_stats",
            "cycle_approval_mode",
            "stop_run",
            "quit_confirm",
        }
        assert {b.id for b in app.query(Button)} == {"btn-browse-workspace"}


def test_workspace_tree_filters_ignored_directories():
    """A pure method, no app to mount — the same IGNORED_DIRS every file
    tool in agent/workspace.py already excludes."""
    from pathlib import Path

    from agent.tui import WorkspaceTree

    paths = [Path("a.py"), Path(".venv"), Path("node_modules"), Path("src")]
    filtered = WorkspaceTree.filter_paths(None, paths)
    assert {p.name for p in filtered} == {"a.py", "src"}


# -- the workspace panel ('w'), native picker, and manual path entry --------


@pytest.mark.asyncio
async def test_workspace_panel_is_hidden_by_default():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one("#workspace-panel")
        assert panel.styles.display == "none"


@pytest.mark.asyncio
async def test_pressing_w_shows_the_workspace_panel():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("w")
        await pilot.pause()
        panel = app.query_one("#workspace-panel")
        assert panel.styles.display == "block"


@pytest.mark.asyncio
async def test_pressing_w_twice_hides_it_again():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("w")
        await pilot.press("w")
        await pilot.pause()
        panel = app.query_one("#workspace-panel")
        assert panel.styles.display == "none"


@pytest.mark.asyncio
async def test_clicking_workspace_opens_the_panel():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await click_footer(pilot, app, "toggle_workspace")
        await pilot.pause()
        assert app.query_one("#workspace-panel").styles.display == "block"


@pytest.mark.asyncio
async def test_the_tree_is_rooted_at_the_configured_workspace():
    import os

    from agent.tui import WorkspaceTree

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        tree = app.query_one("#workspace-tree", WorkspaceTree)
        assert str(tree.path) == os.getcwd() or str(tree.path) == "."


@pytest.mark.asyncio
async def test_a_valid_manual_path_updates_settings_and_the_tree():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    from agent.tui import WorkspaceTree

    workspace_dir = str(Path(tempfile.mkdtemp()).resolve())
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = workspace_dir
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.workspace == workspace_dir
        assert str(app.query_one("#workspace-tree", WorkspaceTree).path) == workspace_dir


@pytest.mark.asyncio
async def test_an_invalid_manual_path_is_rejected_not_applied():
    from textual.widgets import Input

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        before = app.settings.workspace
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = "/definitely/not/a/real/path/zzzzz"
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.workspace == before
        log = app.query_one("#log", Transcript)
        text = log.plain_text()
        assert "workspace not set" in text


@pytest.mark.asyncio
async def test_the_manual_path_field_clears_after_submitting():
    import tempfile

    from textual.widgets import Input

    workspace_dir = tempfile.mkdtemp()
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = workspace_dir
        await pilot.press("enter")
        await pilot.pause()
        assert field.value == ""


@pytest.mark.asyncio
async def test_the_native_picker_result_becomes_the_workspace():
    """_browse_workspace's real logic, called directly in a plain thread —
    the same way test_ask_without_options_* tests _on_ask directly rather
    than through Textual's own worker scheduling, so the test controls
    exactly when it starts and joins."""
    import tempfile
    import threading
    from pathlib import Path

    from textual.widgets import Button

    import agent.native_dialogs as native_dialogs

    chosen_dir = str(Path(tempfile.mkdtemp()).resolve())
    native_dialogs.choose_folder = lambda prompt="": chosen_dir
    app = make_app()
    async with app.run_test():
        t = threading.Thread(target=app._browse_workspace.__wrapped__, args=(app,))
        t.start()
        await join_thread(t)
        assert not t.is_alive(), "the worker never returned"
        assert app.settings.workspace == chosen_dir
        assert app.query_one("#btn-browse-workspace", Button).disabled is False


@pytest.mark.asyncio
async def test_a_cancelled_picker_leaves_the_workspace_unchanged():
    import threading

    import agent.native_dialogs as native_dialogs

    native_dialogs.choose_folder = lambda prompt="": None
    app = make_app()
    async with app.run_test():
        before = app.settings.workspace
        t = threading.Thread(target=app._browse_workspace.__wrapped__, args=(app,))
        t.start()
        await join_thread(t)
        assert not t.is_alive(), "the worker never returned"
        assert app.settings.workspace == before
        log = app.query_one("#log", Transcript)
        text = log.plain_text()
        assert "no folder selected" in text


# -- nested-project confirmation, the same check loop._check_not_nested runs -


@pytest.mark.asyncio
async def test_a_nested_path_is_not_applied_without_confirmation():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    outer = Path(tempfile.mkdtemp())
    (outer / ".venv").mkdir()
    nested = outer / "inner"
    nested.mkdir()
    before = None
    app = make_app()
    async with app.run_test() as pilot:
        before = app.settings.workspace
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = str(nested)
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.workspace == before, "must not apply until confirmed"
        assert len(app.screen_stack) > 1, "a confirmation modal should be showing"


@pytest.mark.asyncio
async def test_the_nested_confirmation_explains_why_it_matters():
    """The path itself is deliberately not on the modal (see
    _confirm_nested_workspace's docstring: an unbounded-length real path
    inline there is what pushed a button off an ordinary terminal in the
    first version) -- it goes to the log instead, which scrolls."""
    import tempfile
    from pathlib import Path

    from textual.widgets import Input, Static

    outer = Path(tempfile.mkdtemp())
    (outer / ".venv").mkdir()
    nested = outer / "inner"
    nested.mkdir()
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = str(nested)
        await pilot.press("enter")
        await pilot.pause()
        question = str(app.screen.query_one("#choice-question", Static).content)
        assert "reach and modify" in question
        log = app.query_one("#log", Transcript)
        log_text = log.plain_text()
        assert str(outer) in log_text


@pytest.mark.asyncio
async def test_the_confirmation_buttons_stay_reachable_on_a_plain_terminal():
    """The regression this whole section exists to pin: a button present in
    the widget tree but positioned past the visible screen is unreachable
    by a real click, and looks identical to "working" unless something
    actually clicks it at a realistic terminal size."""
    import tempfile
    from pathlib import Path

    from textual.widgets import Button, Input

    outer = Path(tempfile.mkdtemp())
    (outer / ".venv").mkdir()
    nested = outer / "inner"
    nested.mkdir()
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = str(nested)
        await pilot.press("enter")
        await pilot.pause()
        for button_id in ("choice-0", "choice-1"):
            region = app.screen.query_one(f"#{button_id}", Button).region
            assert region.y + region.height <= 24, (
                f"{button_id} at {region} extends past an 80x24 terminal"
            )


@pytest.mark.asyncio
async def test_using_a_nested_workspace_anyway_applies_it_and_sets_allow_nested():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    outer = Path(tempfile.mkdtemp())
    (outer / ".venv").mkdir()
    nested = outer / "inner"
    nested.mkdir()
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = str(nested)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.click("#choice-0")  # "Use it anyway"
        await pilot.pause()
        assert app.settings.workspace == str(nested.resolve())
        assert app.settings.allow_nested is True


@pytest.mark.asyncio
async def test_choosing_a_different_folder_leaves_the_workspace_unchanged():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    outer = Path(tempfile.mkdtemp())
    (outer / ".venv").mkdir()
    nested = outer / "inner"
    nested.mkdir()
    app = make_app()
    async with app.run_test() as pilot:
        before = app.settings.workspace
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = str(nested)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.click("#choice-1")  # "Choose a different folder"
        await pilot.pause()
        assert app.settings.workspace == before
        assert app.settings.allow_nested is False
        log = app.query_one("#log", Transcript)
        text = log.plain_text()
        assert "workspace not changed" in text


@pytest.mark.asyncio
async def test_allow_nested_already_set_skips_the_confirmation():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    outer = Path(tempfile.mkdtemp())
    (outer / ".venv").mkdir()
    nested = outer / "inner"
    nested.mkdir()
    app = make_app(allow_nested=True)
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = str(nested)
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.workspace == str(nested.resolve())
        assert len(app.screen_stack) == 1, "no confirmation needed a second time"


@pytest.mark.asyncio
async def test_a_non_nested_workspace_never_shows_the_confirmation():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    workspace_dir = str(Path(tempfile.mkdtemp()).resolve())
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = workspace_dir
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.workspace == workspace_dir
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_setting_the_workspace_while_a_run_is_in_progress_notes_next_run():
    import tempfile
    from pathlib import Path

    from textual.widgets import Input

    workspace_dir = str(Path(tempfile.mkdtemp()).resolve())
    app = make_app()
    async with app.run_test() as pilot:
        app._agent_running = True
        await pilot.press("w")
        await pilot.pause()
        field = app.query_one("#workspace-input", Input)
        field.focus()
        field.value = workspace_dir
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings.workspace == workspace_dir
        log = app.query_one("#log", Transcript)
        text = log.plain_text()
        assert "next run" in text


@pytest.mark.asyncio
async def test_clicking_settings_opens_the_panel():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        await click_footer(pilot, app, "toggle_settings")
        await pilot.pause()
        assert app.query_one("#settings-panel").styles.display == "block"


@pytest.mark.asyncio
async def test_clicking_mode_cycles_the_approval_mode():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._approval_mode() == "manual"
        await click_footer(pilot, app, "cycle_approval_mode")
        await pilot.pause()
        assert app._approval_mode() == "auto-python"


@pytest.mark.asyncio
async def test_clicking_stats_toggles_the_panel():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        before = app.query_one("#stats-panel").styles.display
        await click_footer(pilot, app, "toggle_stats")
        await pilot.pause()
        assert app.query_one("#stats-panel").styles.display != before


# -- quit, which used to be unreachable ----------------------------------


@pytest.mark.asyncio
async def test_ctrl_q_quits_even_with_the_prompt_box_focused():
    """Bare "q" is typed as text in every mode where you would reach for it."""
    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_bare_q_is_still_typed_as_text_when_the_box_has_focus():
    from agent.tui import PromptBox

    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
        assert app.query_one("#goal-input", PromptBox).value == "q"
        assert app.is_running


@pytest.mark.asyncio
async def test_quitting_mid_run_asks_once_before_abandoning_it():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_running = True
        app.action_quit_confirm()
        await pilot.pause()
        assert app.is_running, "the first request should not exit outright"
        assert app._quit_armed
        assert app._cancel.requested, "it should have asked the run to stop"
        app.action_quit_confirm()
        await pilot.pause()
        assert not app.is_running, "the second request exits"


@pytest.mark.asyncio
async def test_quitting_with_nothing_running_exits_immediately():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_running = False
        app.action_quit_confirm()
        await pilot.pause()
        assert not app.is_running


# -- model preference ----------------------------------------------------


@pytest.mark.asyncio
async def test_the_three_model_selects_start_on_the_configured_models():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#planner-select", Select).value == app.settings.planner_model
        assert app.query_one("#executor-select", Select).value == app.settings.executor_model
        assert app.query_one("#reviewer-select", Select).value == app.settings.reviewer_model


@pytest.mark.asyncio
async def test_changing_the_executor_select_updates_settings():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#executor-select", Select).value = "openai/gpt-5-nano"
        await pilot.pause()
        assert app.settings.executor_model == "openai/gpt-5-nano"


@pytest.mark.asyncio
async def test_a_model_given_on_the_command_line_appears_in_its_dropdown():
    """A Select cannot hold a value absent from its options."""
    from textual.widgets import Select

    app = make_app(executor_model="some/unlisted-model")
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one("#executor-select", Select)
        assert select.value == "some/unlisted-model"


# -- the wrapping prompt box ---------------------------------------------


@pytest.mark.asyncio
async def test_the_prompt_box_soft_wraps():
    from agent.tui import PromptBox

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#goal-input", PromptBox).soft_wrap is True


@pytest.mark.asyncio
async def test_enter_submits_rather_than_inserting_a_newline():
    from agent.tui import PromptBox

    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "a goal"
        await pilot.press("enter")
        await pilot.pause()
        assert "\n" not in bar.value
        assert app._mode == "running"


@pytest.mark.asyncio
async def test_a_long_goal_is_kept_whole_rather_than_scrolled_away():
    from agent.tui import PromptBox

    app = make_app(goal="")
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one("#goal-input", PromptBox)
        long_goal = "build " + "a very detailed thing " * 12
        bar.value = long_goal
        await pilot.pause()
        assert bar.value == long_goal


# -- the session summary reaches the log ---------------------------------


@pytest.mark.asyncio
async def test_the_session_summary_is_written_to_the_log_at_the_end():
    """Rendered from the structured summary, not from the stored plain-text
    mirror: the log gets headings and bullets, and the step id survives the
    trip through Rich (it is square-bracketed, which Rich reads as a tag)."""
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.summary = {
            "ok": True,
            "steps": {"total": 1, "failed": [], "abandoned": [], "unverified": []},
            "final_check": {"ok": True},
            "plan": {"steps": [{"id": "do-thing", "files": ["a.py"], "summary": "Did the thing."}]},
            "changed_files": ["a.py"],
            "session_summary": "Goal: do a thing. 1 step ran, all verified.",
        }
        app._finish()
        await pilot.pause()
        text = app.query_one("#log", Transcript).plain_text()
        assert "Built" in text, text[-500:]
        assert "Did the thing." in text, text[-500:]
        assert "[do-thing]" in text, text[-500:]


# -- long text wraps rather than being clipped ----------------------------


@pytest.mark.asyncio
async def test_a_long_question_grows_the_prompt_label_instead_of_clipping():
    app = make_app()
    async with app.run_test(size=(60, 40)) as pilot:
        await pilot.pause()
        long_q = "word " * 40
        app._set_mode("asking", f"agent: {long_q}", placeholder="answer")
        await pilot.pause()
        label = app.query_one("#prompt-label")
        assert label.region.height > 1, "a long question must wrap, not clip to one line"


@pytest.mark.asyncio
async def test_a_short_label_does_not_waste_extra_height():
    app = make_app()
    async with app.run_test(size=(60, 40)) as pilot:
        await pilot.pause()
        app._set_mode("asking", "short", placeholder="answer")
        await pilot.pause()
        label = app.query_one("#prompt-label")
        assert label.region.height == 1


@pytest.mark.asyncio
async def test_the_mode_line_can_grow_for_a_narrow_terminal():
    app = make_app()
    async with app.run_test(size=(40, 40)) as pilot:
        await pilot.pause()
        line = app.query_one("#mode-line")
        assert line.styles.height.value != 1 or line.region.height >= 1


@pytest.mark.asyncio
async def test_stats_section_headers_use_a_real_rich_color_not_a_css_variable():
    """$accent is a Textual CSS variable, meaningless inside Rich markup
    embedded in widget content -- it silently rendered as no style at all."""
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app._stats_text()
        assert r"\$accent" not in text
        assert "[bold cyan]this prompt" in text


# -- options as buttons, with a recommendation and an escape hatch --------


@pytest.mark.asyncio
async def test_every_option_gets_its_own_button():
    from textual.app import App
    from textual.widgets import Button

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("which?", ["alpha", "beta", "gamma"]))
        await pilot.pause()
        labels = [str(b.label) for b in app.screen.query(Button)]
        assert any("alpha" in s for s in labels)
        assert any("beta" in s for s in labels)
        assert any("gamma" in s for s in labels)


@pytest.mark.asyncio
async def test_there_is_always_an_other_option():
    """The listed options are the model's guesses; "neither, do this instead"
    has to be answerable."""
    from textual.app import App
    from textual.widgets import Button

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("which?", ["alpha", "beta"]))
        await pilot.pause()
        assert app.screen.query_one("#choice-other", Button) is not None


@pytest.mark.asyncio
async def test_the_recommended_option_is_marked_and_the_others_are_not():
    from textual.app import App
    from textual.widgets import Button

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("which?", ["alpha", "beta"], recommended="beta"))
        await pilot.pause()
        marked = str(app.screen.query_one("#choice-1", Button).label)
        plain = str(app.screen.query_one("#choice-0", Button).label)
        assert "recommended" in marked
        assert "recommended" not in plain


@pytest.mark.asyncio
async def test_no_recommendation_marks_nothing():
    """Inventing a recommendation the model did not make would be worse than
    showing none."""
    from textual.app import App
    from textual.widgets import Button

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("which?", ["alpha", "beta"]))
        await pilot.pause()
        labels = [str(b.label) for b in app.screen.query(Button)]
        assert not any("recommended" in s for s in labels)


@pytest.mark.asyncio
async def test_a_recommendation_that_is_not_an_option_is_ignored():
    from textual.app import App
    from textual.widgets import Button

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("which?", ["alpha"], recommended="not-an-option"))
        await pilot.pause()
        labels = [str(b.label) for b in app.screen.query(Button)]
        assert not any("recommended" in s for s in labels)


@pytest.mark.asyncio
async def test_other_reveals_a_box_and_returns_what_was_typed():
    import threading

    from agent.tui import PromptBox

    app = make_app()
    async with app.run_test() as pilot:
        result = {}

        def worker():
            result["answer"] = app._on_ask("which?", ["alpha", "beta"])

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        await pilot.click("#choice-other")
        await pilot.pause()
        box = app.screen.query_one("#other-input", PromptBox)
        box.value = "neither, use gamma"
        await pilot.press("enter")
        await join_thread(t)
        assert result["answer"] == "neither, use gamma"


@pytest.mark.asyncio
async def test_the_other_box_is_hidden_until_asked_for():
    from textual.app import App

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("which?", ["alpha"]))
        await pilot.pause()
        assert app.screen.query_one("#other-row").styles.display == "none"


# -- the bar clears after an answer --------------------------------------


@pytest.mark.asyncio
async def test_answering_a_question_clears_the_box():
    """It used to keep the typed answer, so the next question arrived into a
    box already holding the previous reply."""
    import threading

    from agent.tui import PromptBox

    app = make_app()
    async with app.run_test() as pilot:

        def worker():
            app._on_ask("which design?", None)

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "design A"
        await pilot.press("enter")
        await join_thread(t)
        await pilot.pause()
        assert bar.value == "", f"box still holds {bar.value!r}"


@pytest.mark.asyncio
async def test_the_question_stops_being_shown_once_it_is_answered():
    import threading

    from textual.widgets import Static

    from agent.tui import PromptBox

    app = make_app()
    async with app.run_test() as pilot:

        def worker():
            app._on_ask("which design?", None)

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        app.query_one("#goal-input", PromptBox).value = "design A"
        await pilot.press("enter")
        await join_thread(t)
        await pilot.pause()
        label = str(app.query_one("#prompt-label", Static).content)
        assert "which design?" not in label


@pytest.mark.asyncio
async def test_a_long_question_is_not_capped_at_four_lines():
    app = make_app()
    async with app.run_test(size=(56, 44)) as pilot:
        await pilot.pause()
        app._set_mode("asking", "word " * 90, placeholder="answer")
        await pilot.pause()
        height = app.query_one("#prompt-label").region.height
        assert height > 4, f"a long question wrapped to only {height} rows"


@pytest.mark.asyncio
async def test_the_settings_panel_scrolls_when_it_does_not_fit():
    """Eleven controls in a short terminal left the ones below the fold
    simply unreachable."""
    app = make_app()
    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        assert panel.allow_vertical_scroll
        assert panel.max_scroll_y > 0, "nothing to scroll: the panel already fits"


@pytest.mark.asyncio
async def test_the_last_setting_is_reachable_by_scrolling():
    from textual.widgets import Select

    app = make_app()
    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        panel = app.query_one("#settings-panel")
        last = app.query_one("#final-check-select", Select)
        assert last.region.bottom > panel.region.bottom, "fits already; test is moot"
        panel.scroll_end(animate=False)
        await pilot.pause()
        assert last.region.bottom <= panel.region.bottom + 1


# -- migrated in the initial in-process pilot batch, kept here for its own
# section grouping ----------------------------------------------------------

# -- basic mount ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_app_mounts_and_shows_the_goal():
    app = make_app()
    async with app.run_test():
        log = app.query_one("#log", Transcript)
        text = log.plain_text()
        assert "test goal" in text


@pytest.mark.asyncio
async def test_the_bottom_input_bar_is_present_from_the_start():
    """The whole point of this redesign: the bar exists immediately, not
    only after pressing a key to summon a modal."""
    app = make_app()
    async with app.run_test():
        bar = app.query_one("#goal-input", PromptBox)
        assert bar is not None


@pytest.mark.asyncio
async def test_the_bar_is_disabled_while_a_run_is_in_progress():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "running"
        assert app.query_one("#goal-input", PromptBox).disabled is True


# -- trace event routing -----------------------------------------------------


@pytest.mark.asyncio
async def test_direct_render_event_writes_to_the_log():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "log", "message": "hello"})
        await pilot.pause()
        assert len(log.entries) > before


@pytest.mark.asyncio
async def test_a_detail_log_line_is_not_written_to_the_panel():
    """detail=True is Trace.say's signal that a line is setup/diagnostic
    narration (which models, which tools, a phase banner) meant for the
    trace file, not the live conversation -- see trace.py's own docstring
    for say(). The TUI's log panel must honour that, not just the console."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "log", "message": "goal: build a thing", "detail": True})
        await pilot.pause()
        assert len(log.entries) == before


@pytest.mark.asyncio
async def test_a_missing_detail_field_still_shows_the_line():
    """Older trace files (or any other producer of a "log" event) have no
    "detail" key at all -- record.get("detail") must read that as False,
    not silently swallow every line that predates this field."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "log", "message": "understood as: build a thing"})
        await pilot.pause()
        assert len(log.entries) > before


@pytest.mark.asyncio
async def test_a_markdown_heading_in_a_log_line_is_rendered_not_shown_literally():
    """A real bug: a model's own text can reach the log verbatim (a
    quick-answer's direct reply, in the traced case) and be markdown, but
    the log rendered every line with Text.from_markup, which only
    understands Rich's own [tag] bracket syntax -- a markdown heading came
    out as a line that literally starts with two unstyled '#' characters."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "log", "message": "## Summary\n\nsome text"})
        await pilot.pause()
        heading = log.entries[-1].query_one("MarkdownH2")
        assert heading._content.plain == "Summary"


@pytest.mark.asyncio
async def test_markdown_bold_in_a_log_line_is_actually_bold():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "log", "message": "here are **the results**:"})
        await pilot.pause()
        paragraph = log.entries[-1].query_one("MarkdownParagraph")._content
        assert paragraph.plain == "here are the results:"
        strong = [paragraph.plain[sp.start : sp.end] for sp in paragraph.spans]
        assert "the results" in strong, paragraph.spans


@pytest.mark.asyncio
async def test_a_step_id_in_square_brackets_survives_rendering():
    """A real bug: narration went through Text.from_markup, which reads the
    step id in "[fix-bug] Fixed _refill()" as a Rich style tag and drops it,
    so every step id vanished from this view while the plain CLI showed it.
    Trace narration is plain text; none of it is Rich markup."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "log", "message": "   [fix-bug] Fixed _refill()"})
        app._render_event({"kind": "log", "message": "   ✗ [add-tests] 3 new type error(s)"})
        await pilot.pause()
        shown = [str(entry.render()) for entry in log.entries[-2:]]
        assert shown[0] == "[fix-bug] Fixed _refill()"
        assert "[add-tests]" in shown[1]


@pytest.mark.asyncio
async def test_a_hash_or_asterisk_in_an_ordinary_line_does_not_trigger_markdown():
    """Narrow on purpose: a shell command or path containing '#' or '*'
    (a comment, a glob) must not flip into markdown rendering just because
    one of those characters appears somewhere in the line."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "log", "message": "running: grep -n '#TODO' *.py"})
        await pilot.pause()
        text = log.plain_text()
        assert "#TODO" in text and "*.py" in text, text


@pytest.mark.asyncio
async def test_on_trace_event_from_a_real_thread_reaches_the_log():
    # The path a real worker thread actually uses: _on_trace_event calling
    # call_from_thread, not a direct method call from the same context. The
    # highest-risk shape in this batch: real cross-thread Textual traffic is
    # exactly what the original deadlock investigation was chasing.
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)

        def worker():
            app._on_trace_event({"kind": "log", "message": "from a real thread"})

        t = threading.Thread(target=worker)
        t.start()
        await join_thread(t)
        await pilot.pause()
        assert len(log.entries) > before, "_on_trace_event did not reach the log"


@pytest.mark.asyncio
async def test_llm_call_event_updates_the_stats_bar():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "llm_call", "cost_usd": 0.0523})
        await pilot.pause()
        stats = app.query_one("#stats", Static)
        assert "0.0523" in str(stats.content)


@pytest.mark.asyncio
async def test_step_attempt_event_updates_the_current_step():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "step_attempt", "step": "build-thing"})
        await pilot.pause()
        stats = app.query_one("#stats", Static)
        assert "build-thing" in str(stats.content)


@pytest.mark.asyncio
async def test_unknown_event_kinds_are_ignored_without_raising():
    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "something_new_and_unhandled"})
        await pilot.pause()


# -- ask_user via the bottom bar (no options) --------------------------------


@pytest.mark.asyncio
async def test_ask_without_options_uses_the_bar_and_blocks_until_answered():
    app = make_app()
    async with app.run_test() as pilot:
        result = {}

        def worker():
            result["answer"] = app._on_ask("which design?", None)

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        assert app._mode == "asking"
        bar = app.query_one("#goal-input", PromptBox)
        assert bar.disabled is False
        bar.value = "design A"
        await pilot.press("enter")
        await join_thread(t)
        assert not t.is_alive(), "the worker never unblocked"
        assert result["answer"] == "design A"


@pytest.mark.asyncio
async def test_ask_without_options_shows_the_question_in_the_label():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:

        def worker():
            app._on_ask("what should the timeout be?", None)

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        label = str(app.query_one("#prompt-label", Static).content)
        assert "what should the timeout be" in label
        # unblock so the thread can exit cleanly
        app._ask_answer["answer"] = "x"
        app._ask_done.set()
        await join_thread(t)


@pytest.mark.asyncio
async def test_ask_without_options_empty_submit_gives_no_answer_given():
    app = make_app()
    async with app.run_test() as pilot:
        result = {}

        def worker():
            result["answer"] = app._on_ask("anything?", None)

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        await pilot.press("enter")  # submit with the bar left empty
        await join_thread(t)
        assert "no answer given" in result["answer"]


# -- approval requests, via the same bottom bar ------------------------------
# The bug: Approver's default gate is a blocking input() on stdin, which
# hangs forever once Textual owns the terminal (nothing can ever satisfy that
# read, and the prompt is invisible behind the alt-screen). _on_approve routes
# it through the bar instead, the same as ask_user.


@pytest.mark.asyncio
async def test_on_approve_uses_the_bar_and_returns_the_answer():
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        result = {}

        def worker():
            result["answer"] = app._on_approve("run:\n    $ rm scratch.py")

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        assert app._mode == "asking"
        label = str(app.query_one("#prompt-label", Static).content)
        assert "approve" in label.lower()
        assert "rm scratch.py" in label
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "y"
        await pilot.press("enter")
        await join_thread(t)
        assert not t.is_alive(), "the worker never unblocked"
        assert result["answer"] == "y"


# -- stopping a run in progress ('c') ----------------------------------------


@pytest.mark.asyncio
async def test_stop_binding_sets_the_cancel_token_while_running():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._mode == "running"
        # The stubbed _run_agent is a no-op and never sets this itself -- a
        # real run does, as the first thing its worker thread does.
        app._agent_running = True
        assert app._cancel.requested is False
        await pilot.press("c")
        await pilot.pause()
        assert app._cancel.requested is True


# -- ask_user with options: the modal ----------------------------------------


@pytest.mark.asyncio
async def test_ask_with_options_shows_a_modal_and_returns_the_choice():
    app = make_app()
    async with app.run_test() as pilot:
        result = {}

        def worker():
            result["answer"] = app._on_ask("which?", ["alpha", "beta"])

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        assert len(app.screen_stack) == 2, "the choice modal should be open"
        await pilot.click("#choice-1")
        await join_thread(t)
        assert result["answer"] == "beta"


@pytest.mark.asyncio
async def test_choice_modal_renders_the_question():
    from textual.app import App
    from textual.widgets import Static

    class HostApp(App):
        pass

    app = HostApp()
    async with app.run_test() as pilot:
        await app.push_screen(ChoiceModal("is this ok?", ["yes", "no"]))
        await pilot.pause()
        static = app.screen.query_one("#choice-question", Static)
        assert "is this ok?" in str(static.content)


# -- the transcript: scrolling, tool lines, steps ---------------------------


async def _fill(app, pilot, count: int) -> Transcript:
    log = app.query_one("#log", Transcript)
    for i in range(count):
        app._render_event({"kind": "log", "message": f"line {i}"})
    await pilot.pause()
    await pilot.pause()
    return log


@pytest.mark.asyncio
async def test_the_transcript_follows_new_output_while_at_the_bottom():
    app = make_app()
    async with app.run_test() as pilot:
        log = await _fill(app, pilot, 80)
        assert log.max_scroll_y > 0, "needs enough lines to scroll"
        assert log.scroll_y == log.max_scroll_y


@pytest.mark.asyncio
async def test_scrolling_up_is_not_undone_by_new_output():
    """The reported bug: RichLog.write() scrolled to the end on every write
    while auto_scroll was on, so reading back through a running session
    meant being dragged to the bottom each time a tool call landed."""
    app = make_app()
    async with app.run_test() as pilot:
        log = await _fill(app, pilot, 80)
        log.action_page_up()
        await pilot.pause(0.5)  # page_up animates; let it land first
        reading_at = log.scroll_y
        assert reading_at < log.max_scroll_y
        for i in range(20):
            app._render_event({"kind": "log", "message": f"more {i}"})
        await pilot.pause()
        await pilot.pause()
        assert log.scroll_y == reading_at


@pytest.mark.asyncio
async def test_scrolling_back_to_the_bottom_follows_again():
    app = make_app()
    async with app.run_test() as pilot:
        log = await _fill(app, pilot, 80)
        log.action_page_up()
        await pilot.pause()
        log.action_scroll_end()
        await pilot.pause()
        for i in range(20):
            app._render_event({"kind": "log", "message": f"more {i}"})
        await pilot.pause()
        await pilot.pause()
        assert log.scroll_y == log.max_scroll_y


@pytest.mark.asyncio
async def test_a_tool_call_is_one_line_and_its_narration_is_not_repeated():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "tool_call", "tool": "edit_file", "args": {"path": "a.py"}})
        app._render_event({"kind": "log", "message": "   … editing a.py"})
        await pilot.pause()
        added = log.entries[before:]
        assert [str(e.render()) for e in added] == ["⎿ Edit a.py"]


@pytest.mark.asyncio
async def test_narration_with_no_tool_call_before_it_is_still_shown():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "log", "message": "   … ran the suite once for this review"})
        await pilot.pause()
        assert str(log.entries[-1].render()) == "⎿ ran the suite once for this review"


@pytest.mark.asyncio
async def test_bookkeeping_tools_get_no_line_of_their_own():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        for tool in ("finish_step", "submit_plan", "submit_findings"):
            app._render_event({"kind": "tool_call", "tool": tool, "args": {}})
        await pilot.pause()
        assert len(log.entries) == before


@pytest.mark.asyncio
async def test_a_step_banner_becomes_a_step_heading():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event(
            {"kind": "log", "message": "\n-- step 1/2: [fix-bug] Fix refill bug and add clear"}
        )
        await pilot.pause()
        entry = log.entries[-1]
        assert entry.has_class("step")
        assert str(entry.render()) == "● Step 1/2  fix-bug  Fix refill bug and add clear"


def test_tool_label_uses_a_verb_and_the_main_argument():
    from agent.tui import tool_label

    assert tool_label("read_file", {"path": "a.py"}) == "Read a.py"
    assert tool_label("run_bash", {"command": "ls -la\necho hi"}) == "Bash ls -la"
    assert tool_label("some_new_tool", {}) == "some_new_tool"
    assert tool_label("run_bash", {"command": "x" * 300}).endswith("…")


def test_looks_like_markdown_detects_code_lists_and_links():
    from agent.tui import _looks_like_markdown

    assert _looks_like_markdown("```python\nprint(1)\n```")
    assert _looks_like_markdown("- one\n- two")
    assert _looks_like_markdown("the `available()` method")
    assert _looks_like_markdown("see [the docs](https://example.com)")
    assert not _looks_like_markdown("- just one dash item")
    assert not _looks_like_markdown("steps: 2 step(s), all verified")


@pytest.mark.asyncio
async def test_the_transcript_drops_its_oldest_entries_past_the_cap(monkeypatch):
    import agent.tui

    monkeypatch.setattr(agent.tui, "MAX_LOG_ITEMS", 10)
    app = make_app()
    async with app.run_test() as pilot:
        log = await _fill(app, pilot, 25)
        assert len(log.entries) == 10
        assert log.plain_text().splitlines()[-1] == "line 24"


@pytest.mark.asyncio
async def test_the_wait_indicator_draws_ducks_across_its_full_width():
    import time

    from textual.widgets import Static

    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        indicator = app.query_one("#wait-indicator", Static)
        app._waiting_count, app._waiting = 1, True
        app._wait_started = time.time() - 20
        app._wait_frame = 60  # mid-swim
        await pilot.pause(0.2)
        lines = str(indicator.render()).split("\n")
        assert len(lines) == 4
        assert any("( o)" in line for line in lines)
        assert lines[-1].rstrip().endswith("s"), "elapsed shown on the water line"
        assert len(lines[-1]) == indicator.content_region.width


@pytest.mark.asyncio
async def test_the_duck_strip_is_hidden_once_the_run_finishes():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#wait-indicator").styles.display == "block"
        app.summary = {"steps": {"failed": [], "abandoned": []}}
        app._finish()
        await pilot.pause()
        assert app.query_one("#wait-indicator").styles.display == "none"


# -- tool results on their tool's line, the model's commentary --------------


@pytest.mark.asyncio
async def test_an_edit_result_completes_its_own_tool_line():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "tool_call", "tool": "edit_file", "args": {"path": "a.py"}})
        app._render_event({"kind": "log", "message": "   … editing a.py"})
        app._render_event(
            {"kind": "edit_file", "path": "a.py", "added": 2, "removed": 1, "preview": "y = 20"}
        )
        await pilot.pause()
        added = log.entries[before:]
        assert len(added) == 1
        assert str(added[0].render()) == "⎿ Edit a.py  +2 -1\n    y = 20"


@pytest.mark.asyncio
async def test_a_batched_edit_gets_one_line_per_file():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "tool_call", "tool": "edit_files", "args": {"edits": "[..."}})
        app._render_event({"kind": "log", "message": "   … editing 2 files"})
        for path in ("a.py", "b.py"):
            app._render_event(
                {"kind": "edit_file", "path": path, "added": 1, "removed": 0, "preview": "p"}
            )
        await pilot.pause()
        shown = [str(e.render()).splitlines()[0] for e in log.entries[before:]]
        assert shown == ["⎿ Edit a.py  +1 -0", "⎿ Edit b.py  +1 -0"]


@pytest.mark.asyncio
async def test_a_write_and_a_test_run_show_their_results():
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "tool_call", "tool": "write_file", "args": {"path": "t.py"}})
        app._render_event(
            {"kind": "write_file", "path": "t.py", "bytes": 90, "overwrote": False, "lines": 12}
        )
        app._render_event({"kind": "tool_call", "tool": "run_tests", "args": {"path": "t.py"}})
        app._render_event(
            {"kind": "run_tests", "path": "t.py", "exit_code": 1, "summary": "1 failed, 2 passed"}
        )
        await pilot.pause()
        shown = [str(e.render()) for e in log.entries[-2:]]
        assert shown == ["⎿ Write t.py  12 lines, new file", "⎿ Tests t.py  → 1 failed, 2 passed"]


@pytest.mark.asyncio
async def test_real_edit_tool_events_render_in_the_transcript(tmp_path):
    """End to end with the real edit tool: its trace events, delivered the
    way a run delivers them, complete the tool's own line."""
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.tools.fs import edit_file
    from agent.trace import Trace
    from agent.workspace import Workspace

    events: list[dict] = []
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False, on_event=events.append),
    )
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    edit_file(ctx, "m.py", "    return 1\n", "    return 2\n")
    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "tool_call", "tool": "edit_file", "args": {"path": "m.py"}})
        for event in events:
            app._render_event(event)
        await pilot.pause()
        last = app.query_one("#log", Transcript).entries[-1]
        assert str(last.render()) == "⎿ Edit m.py  +1 -1\n    return 2"


@pytest.mark.asyncio
async def test_the_models_commentary_is_shown_and_clipped():
    from agent.tui import _ASSISTANT_TEXT_LIMIT

    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "assistant_text", "text": "\n\nLet me read throttle.py\n\n"})
        app._render_event({"kind": "assistant_text", "text": "   "})
        app._render_event({"kind": "assistant_text", "text": "x" * (_ASSISTANT_TEXT_LIMIT * 2)})
        await pilot.pause()
        added = log.entries[before:]
        assert len(added) == 2
        assert str(added[0].render()) == "Let me read throttle.py"
        assert added[0].has_class("thought")
        assert str(added[1].render()).endswith(" …")


# -- copying and select-all follow where you are working ---------------------


async def _drag_over(app, pilot, entry) -> None:
    r = entry.region
    await pilot.mouse_down(offset=(r.x, r.y))
    await pilot.hover(offset=(r.x + r.width - 1, r.y + r.height - 1))
    await pilot.mouse_up(offset=(r.x + r.width - 1, r.y + r.height - 1))
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["ctrl+c", "super+c"])
async def test_copy_takes_transcript_text_even_while_the_prompt_box_has_focus(key):
    """Copying must take the transcript's selection even though the prompt
    box, a TextArea with its own copy bindings, has focus nearly all the
    time. (This already worked before the app-level binding: TextArea skips
    the key when its own selection is empty. Kept as a guard.)"""
    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "log", "message": "model output to copy"})
        await pilot.pause()
        entry = app.query_one("#log", Transcript).entries[-1]
        await _drag_over(app, pilot, entry)
        app.query_one("#goal-input", PromptBox).focus()
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()
        assert app.clipboard == "model output to copy"


@pytest.mark.asyncio
async def test_copy_falls_back_to_the_prompt_boxs_own_selection():
    app = make_app(goal="")
    async with app.run_test() as pilot:
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "my draft goal"
        bar.focus()
        bar.select_all()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.clipboard == "my draft goal"


@pytest.mark.asyncio
async def test_select_all_in_the_prompt_box_selects_only_its_text():
    app = make_app(goal="")
    async with app.run_test() as pilot:
        app._render_event({"kind": "log", "message": "model output"})
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "only this"
        bar.focus()
        await pilot.press("super+a")
        await pilot.pause()
        assert bar.selected_text == "only this"
        assert not app.screen.get_selected_text()


@pytest.mark.asyncio
async def test_select_all_in_the_transcript_selects_the_transcript():
    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "log", "message": "first output"})
        app._render_event({"kind": "log", "message": "second output"})
        await pilot.pause()
        app.query_one("#log", Transcript).focus()
        await pilot.press("super+a")
        await pilot.pause()
        selected = app.screen.get_selected_text() or ""
        assert "first output" in selected and "second output" in selected


@pytest.mark.asyncio
async def test_typing_in_the_transcript_goes_to_the_prompt_box():
    app = make_app(goal="")
    async with app.run_test() as pilot:
        app.query_one("#log", Transcript).focus()
        await pilot.press("h", "i")
        await pilot.pause()
        bar = app.query_one("#goal-input", PromptBox)
        assert app.focused is bar
        assert bar.value == "hi"


@pytest.mark.asyncio
async def test_a_mouse_selection_in_the_output_is_copied_on_release():
    """Ghostty keeps ⌘C for its own copy, which copies its own (empty)
    selection because the drag went to the app, so ⌘C never reaches this
    app. Copying on release, as Ghostty's copy-on-select does natively,
    puts the text on the clipboard with no key at all."""
    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "log", "message": "select me"})
        await pilot.pause()
        await _drag_over(app, pilot, app.query_one("#log", Transcript).entries[-1])
        await pilot.pause()
        assert app.clipboard == "select me"


# -- copying straight to the Mac's clipboard (pbcopy), not only via OSC 52 ---


def test_pbcopy_is_used_only_on_a_local_mac():
    from agent.tui import _LOCAL_COPY_COMMAND, local_clipboard_command

    assert _LOCAL_COPY_COMMAND == ("pbcopy",)

    assert local_clipboard_command(env={}, platform="linux") is None
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        assert local_clipboard_command(env={name: "x"}, platform="darwin") is None


def test_pbcopy_is_found_on_this_mac():
    import shutil
    import sys

    from agent.tui import local_clipboard_command

    if sys.platform != "darwin" or not shutil.which("pbcopy"):
        pytest.skip("macOS only")
    assert local_clipboard_command(env={}, platform="darwin") == ("pbcopy",)


def test_copy_with_pipes_unicode_text_to_a_real_process(tmp_path):
    """A real subprocess standing in for pbcopy (the real one would overwrite
    whoever runs the tests' clipboard)."""
    from agent.tui import copy_with

    out = tmp_path / "clip"
    text = "naïve café → ✓ 🦆\nsecond line"
    assert copy_with(("sh", "-c", f"cat > '{out}'"), text)
    assert out.read_text(encoding="utf-8") == text


def test_a_failing_clipboard_command_is_not_an_error(tmp_path):
    from agent.tui import copy_with

    assert copy_with(("sh", "-c", "exit 3"), "x") is False
    assert copy_with((str(tmp_path / "no-such-command"),), "x") is False


@pytest.mark.asyncio
async def test_a_selection_reaches_the_local_clipboard_command(tmp_path, monkeypatch):
    out = tmp_path / "clip"
    monkeypatch.setattr(RunApp, "_local_copy_command", lambda self: ("sh", "-c", f"cat > '{out}'"))
    app = make_app()
    async with app.run_test() as pilot:
        app._render_event({"kind": "log", "message": "copy me locally"})
        await pilot.pause()
        await _drag_over(app, pilot, app.query_one("#log", Transcript).entries[-1])
        await pilot.pause()
        assert app.clipboard == "copy me locally", "OSC 52 still sent"
        assert out.read_text() == "copy me locally"


@pytest.mark.asyncio
async def test_the_wait_indicator_lays_out_only_when_it_appears_or_goes(monkeypatch):
    """Each tick used to call Static.update, which lays the whole screen out; over a
    long transcript that held a third of a core while idle (DESIGN-LOG §95,
    scripts/tui_bench.py). Frames are the same height, so only the first needs it."""
    from textual.widgets import Static

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        indicator = app.query_one("#wait-indicator", Static)
        calls: list[bool] = []
        original = indicator.update

        def record(content="", *, layout=True):
            calls.append(layout)
            original(content, layout=layout)

        monkeypatch.setattr(indicator, "update", record)
        app._waiting = False
        app._tick_wait_indicator()
        app._tick_wait_indicator()
        assert calls == []  # blank and already blank: nothing to do
        app._waiting = True
        app._show_wait_animation = True
        app._wait_started = 0.0
        app._tick_wait_indicator()
        app._tick_wait_indicator()
        app._tick_wait_indicator()
        assert calls == [True, False, False]
        app._waiting = False
        app._tick_wait_indicator()
        app._tick_wait_indicator()
        assert calls == [True, False, False, True]


@pytest.mark.asyncio
async def test_a_prompt_that_waits_on_a_person_rings_the_bell_and_says_so(monkeypatch):
    """A bash approval sat unseen in the bottom bar for 90 s of a traced run
    (DESIGN-LOG §95). Every wait on a person now rings the terminal bell; a bar
    prompt also puts a notice on screen."""
    app = make_app()
    async with app.run_test() as pilot:
        bells: list[int] = []
        monkeypatch.setattr(app, "bell", lambda: bells.append(1))
        result = {}

        def worker():
            result["answer"] = app._on_approve("run:\n    $ rm scratch.py")

        t = threading.Thread(target=worker)
        t.start()
        await pilot.pause()
        assert bells == [1]
        assert any("waiting for you" in (n.title or "") for n in app._notifications)
        bar = app.query_one("#goal-input", PromptBox)
        bar.value = "n"
        await pilot.press("enter")
        await join_thread(t)
        assert result["answer"] == "n"

        def asker():
            result["choice"] = app._on_ask("Which one?", ["left", "right"])

        t2 = threading.Thread(target=asker)
        t2.start()
        await pilot.pause()
        assert bells == [1, 1]  # the modal rings too
        app.screen.dismiss("left")
        await join_thread(t2)
        assert result["choice"] == "left"


@pytest.mark.asyncio
async def test_a_harness_test_run_gets_its_own_labelled_line():
    """check_after_edit (DESIGN-LOG §99.4) runs the step's tests itself after an
    edit turn: a run_tests event with source="auto" and no tool_call before
    it. It must say the harness ran it, and leave the model's own open test
    line for the model's own result."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        app._render_event({"kind": "tool_call", "tool": "run_tests", "args": {"path": "t.py"}})
        app._render_event(
            {
                "kind": "run_tests",
                "path": "t.py",
                "exit_code": 0,
                "summary": "2 passed",
                "source": "auto",
            }
        )
        app._render_event({"kind": "after_edit_check", "result": "2 passed"})
        app._render_event(
            {"kind": "run_tests", "path": "t.py", "exit_code": 1, "summary": "1 failed"}
        )
        await pilot.pause()
        shown = [str(e.render()) for e in log.entries[-2:]]
        assert shown == ["⎿ Tests t.py  → 1 failed", "⎿ Harness check  → 2 passed"]


@pytest.mark.asyncio
async def test_the_real_harness_check_event_is_labelled(tmp_path):
    """End to end: the real run_tests tool, run the way check_after_edit runs
    it (ctx.test_source = "auto"), on a real one-test project."""
    from agent.approval import Approver
    from agent.tools import ToolContext
    from agent.tools.checks import run_tests
    from agent.trace import Trace
    from agent.workspace import Workspace

    (tmp_path / "test_one.py").write_text("def test_one():\n    assert True\n")
    events: list[dict] = []
    ctx = ToolContext(
        workspace=Workspace(tmp_path),
        approver=Approver(auto_approve=True),
        trace=Trace(str(tmp_path / "t"), echo=False, on_event=events.append),
    )
    ctx.test_source = "auto"
    run_tests(ctx)
    auto = [e for e in events if e["kind"] == "run_tests"]
    assert auto and auto[-1].get("source") == "auto", events
    app = make_app()
    async with app.run_test() as pilot:
        for event in events:
            app._render_event(event)
        await pilot.pause()
        last = str(app.query_one("#log", Transcript).entries[-1].render())
        assert last.startswith("⎿ Harness check  → "), last


# -- clarify's questions as one form (DESIGN-LOG §101, §102) -----------------

NO = "(no answer given)"
QUESTIONS = [
    ("Which storage backend?", ["SQLite", "Postgres"], "SQLite"),
    ("Keep the old API?", ["yes", "no"], None),
    ("Anything else the plan should know?", None, None),
]


async def _open_form(app, pilot, questions=QUESTIONS):
    from agent.tui import QuestionsModal

    results: list = []
    app.push_screen(QuestionsModal(questions), results.append)
    await pilot.pause()
    return results


@pytest.mark.asyncio
async def test_the_form_shows_every_question_with_the_recommendation_chosen():
    from textual.widgets import RadioSet

    app = make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        results = await _open_form(app, pilot)
        screen = app.screen
        texts = [str(w.render()) for w in screen.query(".question-text")]
        assert texts == [f"{i}. {q}" for i, (q, _, _) in enumerate(QUESTIONS, 1)]
        assert screen.query_one("#q0-options", RadioSet).pressed_button.name == "SQLite"
        assert screen.query_one("#q1-options", RadioSet).pressed_button is None
        screen.action_send()
        await pilot.pause()
        assert results == [["SQLite", NO, NO]]


@pytest.mark.asyncio
async def test_typed_text_wins_over_the_chosen_option_and_blanks_stay_blank():
    from textual.widgets import Input, RadioSet

    app = make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        results = await _open_form(app, pilot)
        screen = app.screen
        screen.query_one("#q0-text", Input).value = "DuckDB, actually"
        list(screen.query_one("#q1-options", RadioSet).query("RadioButton"))[1].value = True
        screen.query_one("#q2-text", Input).value = "  ship it by Friday  "
        await pilot.click("#questions-send", offset=(4, 1))
        await pilot.pause()
        assert results == [["DuckDB, actually", "no", "ship it by Friday"]]


@pytest.mark.asyncio
async def test_skip_all_answers_nothing():
    app = make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        results = await _open_form(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert results == [[NO, NO, NO]]


@pytest.mark.asyncio
async def test_enter_moves_to_the_next_question_and_sends_from_the_last():
    from textual.widgets import Input

    app = make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        results = await _open_form(app, pilot)
        screen = app.screen
        screen.query_one("#q0-text", Input).focus()
        await pilot.press("enter")
        assert screen.focused is screen.query_one("#q1-options")
        screen.query_one("#q2-text", Input).focus()
        await pilot.press("x", "enter")
        await pilot.pause()
        assert results == [["SQLite", NO, "x"]]


@pytest.mark.asyncio
async def test_the_buttons_stay_on_screen_with_five_long_questions():
    long = "a question long enough to wrap across several lines of a narrow terminal " * 3
    questions = [(f"{long} ({i})", ["one", "two", "three"], "two") for i in range(5)]
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _open_form(app, pilot, questions)
        for button in ("#questions-send", "#questions-skip"):
            region = app.screen.query_one(button).region
            assert region.height > 0 and region.bottom <= 24, (button, region)


@pytest.mark.asyncio
async def test_a_real_interactor_gets_its_answers_from_the_form():
    """End to end: the Interactor's ask_many, called from a worker thread as
    the clarify phase calls it, blocks on the TUI form and gets one answer
    per question back."""
    from agent.interact import Interactor
    from agent.tui import QuestionsModal

    app = make_app()
    async with app.run_test(size=(100, 40)) as pilot:
        interactor = Interactor(ask=lambda q, o=None, r=None: "unused", ask_many=app._on_ask_many)
        got: dict = {}

        def worker():
            got["answers"] = interactor.ask_many([(q, o) for q, o, _ in QUESTIONS])

        t = threading.Thread(target=worker, daemon=True)  # a failure must not hang the run
        t.start()
        for _ in range(100):
            await pilot.pause(0.02)
            if isinstance(app.screen, QuestionsModal):
                break
        assert isinstance(app.screen, QuestionsModal)
        await pilot.pause()  # let the form finish mounting before clicking
        from textual.widgets import Input

        app.screen.query_one("#q0-text", Input).value = "SQLite"
        app.screen.query_one("#q2-text", Input).value = "nothing else"
        await pilot.click("#questions-send", offset=(4, 1))
        await join_thread(t)
        assert not isinstance(app.screen, QuestionsModal), "form still open"
        assert not t.is_alive(), "worker still blocked"
        assert got["answers"] == ["SQLite", NO, "nothing else"]


@pytest.mark.asyncio
async def test_a_run_started_from_the_tui_gets_the_form(monkeypatch):
    import agent.tui

    captured: dict = {}

    def fake_run(settings, **kwargs):
        captured.update(kwargs)
        return {"steps": {"failed": [], "abandoned": []}}

    monkeypatch.setattr(agent.tui, "run_agent", fake_run)
    app = RunApp(Settings(workspace=".", goal="g"), None, "review")
    async with app.run_test() as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if captured:
                break
        assert captured.get("on_ask_many") == app._on_ask_many


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["edit_files", "edit_blocks", "batch_edits_to_file"])
async def test_every_batch_edit_tool_gets_one_line_per_file_and_no_bare_line(tool):
    """edit_blocks was missing from the TUI's own batch list, so a run with
    --edit-blocks showed a bare "⎿ edit_blocks" line above the per-file ones."""
    app = make_app()
    async with app.run_test() as pilot:
        log = app.query_one("#log", Transcript)
        before = len(log.entries)
        app._render_event({"kind": "tool_call", "tool": tool, "args": {"blocks": "..."}})
        app._render_event({"kind": "log", "message": "   … editing 1 file"})
        app._render_event(
            {"kind": "edit_file", "path": "a.py", "added": 1, "removed": 1, "preview": "x"}
        )
        await pilot.pause()
        shown = [str(e.render()).splitlines()[0] for e in log.entries[before:]]
        assert shown == ["⎿ Edit a.py  +1 -1"]


# -- square brackets in questions survive; typing starts a correction --------

BRACKETED = "Before it starts, the plan assumes:\n- [fix-bug] returns list[str]\nGo ahead?"


@pytest.mark.asyncio
async def test_a_question_with_square_brackets_is_shown_whole():
    """A real bug: the question went to Static as markup, so "[fix-bug]" and
    "[str]" were read as style tags and dropped from the text shown."""
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(ChoiceModal(BRACKETED, ["keep [x] as is"], "keep [x] as is"))
        await pilot.pause()
        assert str(app.screen.query_one("#choice-question").render()) == BRACKETED
        label = str(app.screen.query_one("#choice-0").label)
        assert label.startswith("keep [x] as is"), label


@pytest.mark.asyncio
async def test_the_one_option_confirmation_accepts_with_enter():
    results: list = []
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(ChoiceModal(BRACKETED, ["Go ahead"], "Go ahead"), results.append)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert results == ["Go ahead"]


@pytest.mark.asyncio
async def test_typing_in_the_choice_modal_starts_a_correction():
    results: list = []
    app = make_app()
    async with app.run_test(size=(100, 30)) as pilot:
        app.push_screen(ChoiceModal(BRACKETED, ["Go ahead"], "Go ahead"), results.append)
        await pilot.pause()
        await pilot.press("u", "s", "e", "space", "x")
        await pilot.press("enter")
        await pilot.pause()
        assert results == ["use x"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, text",
    [("_on_approve", "run_bash: pytest t.py::test_x[a-b]"), ("_ask_via_bar", "keep list[str]?")],
)
async def test_brackets_survive_in_the_prompt_bar(method, text):
    """The bar's label is markup, so an approval showing a parametrized test
    id, or a question about list[str], lost its bracketed part."""
    app = make_app()
    async with app.run_test() as pilot:
        got: dict = {}
        worker = threading.Thread(
            target=lambda: got.setdefault("answer", getattr(app, method)(text)), daemon=True
        )
        worker.start()
        for _ in range(100):
            await pilot.pause(0.02)
            if app._mode == "asking":
                break
        assert text in str(app.query_one("#prompt-label").render())
        app._submit_answer("n")
        await join_thread(worker)
        assert got["answer"] == "n"
