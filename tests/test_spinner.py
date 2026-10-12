"""Spinner: the CLI's "still working" indicator."""

from __future__ import annotations

import io
import itertools
import re
import time

from agent.spinner import DOTS_FRAMES, Spinner


class FakeTTY(io.StringIO):
    """A stream that reports itself as a real terminal, so Spinner treats
    it as enabled — a plain StringIO/pipe (what pytest captures by default)
    reports isatty() False, which would make every test here a no-op."""

    def isatty(self) -> bool:
        return True


def test_disabled_on_a_non_tty_stream():
    spinner = Spinner(stream=io.StringIO())  # isatty() is False
    assert spinner.enabled is False
    spinner.start()
    assert spinner._thread is None  # never started
    spinner.stop()  # must not raise on a stop with nothing running


def test_enabled_on_a_tty_stream():
    spinner = Spinner(stream=FakeTTY())
    assert spinner.enabled is True


def test_start_then_stop_leaves_no_thread_running():
    spinner = Spinner(stream=FakeTTY(), interval=0.01)
    spinner.start()
    time.sleep(0.05)
    assert spinner._thread is not None
    spinner.stop()
    assert spinner._thread is None


def test_stop_without_start_is_a_safe_noop():
    spinner = Spinner(stream=FakeTTY())
    spinner.stop()  # must not raise


def test_double_start_only_spawns_one_thread():
    spinner = Spinner(stream=FakeTTY(), interval=0.01)
    spinner.start()
    first_thread = spinner._thread
    spinner.start()  # refcounted, not a second animation
    assert spinner._thread is first_thread
    spinner.stop()
    assert spinner._thread is first_thread  # still running: one stop left
    spinner.stop()
    assert spinner._thread is None


def test_the_animation_actually_draws_something():
    # dots: a single-line style, so a frame appears in the output verbatim —
    # duck's frames are multi-line and get ANSI codes interleaved between
    # lines on the wire, so they're covered separately (test below).
    stream = FakeTTY()
    spinner = Spinner(style="dots", stream=stream, interval=0.01)
    spinner.start()
    time.sleep(0.05)
    spinner.stop()
    assert any(frame in stream.getvalue() for frame in DOTS_FRAMES)


def test_stopping_clears_the_line_leaving_no_visible_frame():
    stream = FakeTTY()
    spinner = Spinner(style="dots", stream=stream, interval=0.01)
    spinner.start()
    time.sleep(0.05)
    spinner.stop()
    # Everything written after the very last clear sequence must be blank —
    # no stray frame characters left on screen once stopped. Strip ANSI
    # escapes first: their own terminator bytes are letters (e.g. the "A" in
    # "\x1b[1A"), not visible frame content.
    clear = "\r\x1b[K\n" * spinner._lines
    tail = stream.getvalue().rsplit(clear, 1)[-1]
    visible = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", tail)
    assert not any(ch.isalnum() for ch in visible)


def test_duck_style_writes_each_line_with_its_own_clear_sequence():
    """The multi-line style: every embedded newline in a frame must be
    followed by an ANSI clear-to-end-of-line, or a shorter later frame
    would leave stray characters from a longer earlier one."""
    stream = FakeTTY()
    spinner = Spinner(style="duck", stream=stream, interval=0.001)
    spinner.start()
    time.sleep(0.2)  # enough ticks for the first duck to paddle on screen
    spinner.stop()
    output = stream.getvalue()
    assert "\x1b[K" in output
    assert "( o)" in output, "the duck's head should be drawn"


def test_duck_scene_is_always_the_same_shape():
    """The redraw moves the cursor up a fixed count, so every frame must be
    DUCK_ROWS lines, each exactly the width asked for, at every point of the
    swim and at every width, however narrow."""
    from agent.spinner import DUCK_ROWS, duck_scene

    for width in (1, 5, 20, 80, 211):
        for tick in range(0, 700, 7):
            lines = duck_scene(tick, width).split("\n")
            assert len(lines) == DUCK_ROWS
            assert {len(line) for line in lines} == {width}, (width, tick)


def test_ducks_swim_off_one_side_and_come_back_facing_the_other_way():
    from agent.spinner import FLOCK_WIDTH, duck_position, duck_scene

    width = 60
    leg = width + FLOCK_WIDTH
    x, facing_right = duck_position(0, width)
    assert facing_right and x == -FLOCK_WIDTH, "starts fully off the left edge"
    assert "( o)>" in duck_scene(leg // 2, width)
    assert duck_scene(leg, width).split("\n")[1].strip() == "", "gone off the right"
    x, facing_right = duck_position(leg, width)
    assert not facing_right and x == width, "turned round out of sight"
    assert "<(o )" in duck_scene(leg + leg // 2, width)
    assert duck_position(2 * leg, width) == (-FLOCK_WIDTH, True), "and round again"


def test_a_bill_stays_open_about_a_second_and_opens_every_few_seconds():
    """The old frames flipped every bill on every 0.15s tick, too fast to
    read as anything but flicker. Asked for since: open for a full second,
    and more often than the first redesign's once every ~4s."""
    from agent.spinner import DUCK_COUNT, duck_bill_open

    tick_s = 0.15  # Spinner's and the TUI's default interval
    for duck in range(DUCK_COUNT):
        states = [duck_bill_open(t, duck) for t in range(200)]
        runs = [(k, len(list(g))) for k, g in itertools.groupby(states)][1:-1]
        open_s = [n * tick_s for is_open, n in runs if is_open]
        cycle_s = [(a + b) * tick_s for (_, a), (_, b) in itertools.pairwise(runs) if _ is True]
        assert open_s and min(open_s) >= 1.0, open_s
        assert max(cycle_s) <= 3.0, cycle_s


def test_the_ducks_do_not_quack_in_unison():
    from agent.spinner import DUCK_COUNT, duck_bill_open

    patterns = {tuple(duck_bill_open(t, d) for t in range(60)) for d in range(DUCK_COUNT)}
    assert len(patterns) == DUCK_COUNT


def test_an_open_bill_is_drawn():
    from agent.spinner import duck_bill_open, duck_scene

    tick = next(t for t in range(60, 200) if duck_bill_open(t, 0))
    assert "( o)<" in duck_scene(tick, 200)


def test_elapsed_time_overwrites_the_end_of_the_line_without_growing_it():
    from agent.spinner import with_elapsed

    line = "~" * 40
    assert with_elapsed(line, 2.0) == line, "short waits show no number"
    shown = with_elapsed(line, 31.4)
    assert len(shown) == len(line)
    assert shown.endswith(" 31s ")


def test_unknown_style_falls_back_to_dots():
    spinner = Spinner(style="not-a-real-style", stream=FakeTTY())
    assert spinner.frames == DOTS_FRAMES


def test_short_waits_show_no_elapsed_time():
    """Below the threshold, every call is inside the range that needs no
    explanation -- printing "3s" there would just be noise."""
    stream = FakeTTY()
    spinner = Spinner(style="dots", stream=stream, interval=0.01)
    spinner.start()
    time.sleep(0.05)
    spinner.stop()
    assert "(" not in stream.getvalue()


def test_long_waits_show_elapsed_seconds():
    """Past the threshold -- which a diffusion model's own healthy prefill
    routinely exceeds, see llm.py's DIFFUSION_FIRST_BYTE_TIMEOUT_S -- a
    person watching cannot tell "still working" from "stuck" without a
    number."""
    stream = FakeTTY()
    spinner = Spinner(style="dots", stream=stream, interval=0.01)
    spinner.start()
    spinner._started = time.time() - 30  # simulate a long-running call
    time.sleep(0.05)
    spinner.stop()
    assert re.search(r"\(3\ds\)", stream.getvalue())
