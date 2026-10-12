"""A lightweight "still working" indicator for the plain CLI.

Most executor turns are all tool calls and no assistant text — nothing
streams, so the terminal looks frozen for however many seconds a model call
takes. This draws a small looping animation in place while a blocking call
is in flight, and clears it the moment there's something real to show
instead (the first streamed token, or the call finishing).

Deliberately not the model's own text: a spinner that looked like it was
"generating" something would be actively misleading on a tool-only turn,
where nothing is being generated yet. It is a fixed, silent loop — the
thing genuinely true right now is just "still waiting."
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import TextIO

# A compact rotating dot, the safe default — one line, works in any terminal.
DOTS_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# Three ducks swimming across the whole width of the window. They paddle off
# one edge, turn round out of sight, and swim back on facing the other way.
# The art is the old generic ASCII duck (a round head, a bill, a tucked wing,
# a belly sitting in the water), not copied from any one artwork.
#
# The bill opens for about a second every few seconds, each duck out of step
# with the others. An earlier version flipped every bill on every 0.15s tick,
# too fast to read as anything but flicker.
#
# The scene depends on the width it is drawn into, so it is a function of
# (tick, width) rather than a fixed list of frames. Every frame is
# DUCK_ROWS lines, each exactly `width` columns, so a redraw that moves the
# cursor up a fixed count stays correct whatever frame is on screen.
_DUCK_RIGHT = ("    __  ", "___( o)>", "\\ <_. ) ", " `---'  ")
_MIRROR = str.maketrans("()<>/\\`'", ")(><\\/'`")
_DUCK_LEFT = tuple(row[::-1].translate(_MIRROR) for row in _DUCK_RIGHT)
DUCK_WIDTH = len(_DUCK_RIGHT[0])
DUCK_ROWS = len(_DUCK_RIGHT)
DUCK_COUNT = 3
DUCK_GAP = 4
FLOCK_WIDTH = DUCK_COUNT * DUCK_WIDTH + (DUCK_COUNT - 1) * DUCK_GAP
# Ticks between one duck's quacks, and how many ticks the bill stays open.
# At the 0.15s default interval: open for ~1s, every ~2.7s.
QUACK_EVERY = 18
QUACK_TICKS = 7
_WATER = "~ ~~  ~~~ ~  ~~ ~~~  ~ ~~ "


def _duck_art(facing_right: bool, bill_open: bool) -> tuple[str, ...]:
    rows = list(_DUCK_RIGHT if facing_right else _DUCK_LEFT)
    if bill_open:
        bill_col = DUCK_WIDTH - 1 if facing_right else 0
        row = rows[1]
        rows[1] = row[:bill_col] + ("<" if facing_right else ">") + row[bill_col + 1 :]
    return tuple(rows)


def duck_bill_open(tick: int, index: int) -> bool:
    """Whether duck `index`'s bill is open on this tick; the three ducks are
    a third of a cycle apart, so their quacks overlap without matching."""
    return (tick + index * (QUACK_EVERY // DUCK_COUNT)) % QUACK_EVERY < QUACK_TICKS


def duck_position(tick: int, width: int) -> tuple[int, bool]:
    """Where the flock's left edge is on this tick, and which way it faces.

    One leg runs from fully off the left edge to fully off the right edge;
    the flock turns while out of sight and the return leg runs back.
    """
    leg = width + FLOCK_WIDTH
    phase = tick % (2 * leg)
    if phase < leg:
        return phase - FLOCK_WIDTH, True
    return width - (phase - leg), False


def duck_scene(tick: int, width: int) -> str:
    """One frame of the swimming ducks, DUCK_ROWS lines of exactly `width`."""
    width = max(1, width)
    x, facing_right = duck_position(tick, width)
    # The water drifts slowly against the direction of travel.
    drift = tick // 3 if facing_right else -(tick // 3)
    water = "".join(_WATER[(i + drift) % len(_WATER)] for i in range(width))
    canvas = [[" "] * width for _ in range(DUCK_ROWS - 1)] + [list(water)]
    for i in range(DUCK_COUNT):
        left = x + i * (DUCK_WIDTH + DUCK_GAP)
        bill_open = duck_bill_open(tick, i)
        for r, row in enumerate(_duck_art(facing_right, bill_open)):
            for c, ch in enumerate(row):
                if ch != " " and 0 <= left + c < width:
                    canvas[r][left + c] = ch
    return "\n".join("".join(row) for row in canvas)


STYLES = {"dots": DOTS_FRAMES}

# Below this, a call is well inside the range every model normally answers
# in, and printing "3s" would just be noise. Past it, silence has gone on
# long enough that a person watching genuinely cannot tell "still working"
# from "stuck" without a number -- true even for a healthy call: a diffusion
# model's prefill routinely runs past this on its own (see llm.py's
# DIFFUSION_FIRST_BYTE_TIMEOUT_S) before it ever produces a token.
ELAPSED_DISPLAY_THRESHOLD_S = 8.0


def with_elapsed(line: str, elapsed: float) -> str:
    """`line` with the wait so far written over its right end, once the wait
    is long enough to be worth a number. Overwrites rather than appends, so a
    line drawn to the full terminal width never wraps."""
    if elapsed < ELAPSED_DISPLAY_THRESHOLD_S:
        return line
    tag = f" {elapsed:.0f}s "
    if len(line) <= len(tag):
        return line
    return line[: len(line) - len(tag)] + tag


class Spinner:
    """Refcounted so concurrent steps (--step-concurrency > 1) sharing one
    Spinner don't have one call's stop() clear another's still-in-flight
    animation — start()/stop() nest safely, only the outermost pair actually
    draws or clears anything.
    """

    def __init__(self, style: str = "duck", interval: float = 0.15, stream: TextIO | None = None):
        # The ducks are drawn to the terminal's width on every frame (see
        # duck_scene); every other style is a fixed list of frames.
        self.ducks = style == "duck"
        self.frames = [] if self.ducks else STYLES.get(style, DOTS_FRAMES)
        self.interval = interval
        self.stream = stream if stream is not None else sys.stdout
        self._lines = DUCK_ROWS if self.ducks else self.frames[0].count("\n") + 1
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active = 0
        self._started = 0.0

    @property
    def enabled(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active += 1
            if self._active > 1 or self._thread is not None:
                return
            self._stop.clear()
            self._started = time.time()
            self._write("\n" * self._lines)  # reserve the vertical space
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1
            if self._active > 0:
                return
            thread, self._thread = self._thread, None
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=1)
        self._clear()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            self._draw(self._frame(i))
            i += 1
            self._stop.wait(self.interval)

    def _frame(self, tick: int) -> str:
        if self.ducks:
            # One column short of the terminal: writing the last column puts
            # some terminals into a pending-wrap state that breaks the redraw.
            width = shutil.get_terminal_size(fallback=(80, 24)).columns - 1
            return duck_scene(tick, width)
        return self.frames[tick % len(self.frames)]

    def _draw(self, frame: str) -> None:
        lines = frame.split("\n")
        elapsed = time.time() - self._started
        if self.ducks:
            lines[-1] = with_elapsed(lines[-1], elapsed)
        elif elapsed >= ELAPSED_DISPLAY_THRESHOLD_S:
            lines[-1] += f"  ({elapsed:.0f}s)"
        out = f"\x1b[{self._lines}A"
        for line in lines:
            out += f"\r{line}\x1b[K\n"
        self._write(out)

    def _clear(self) -> None:
        out = f"\x1b[{self._lines}A"
        out += "\r\x1b[K\n" * self._lines
        out += f"\x1b[{self._lines}A"
        self._write(out)

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            pass
