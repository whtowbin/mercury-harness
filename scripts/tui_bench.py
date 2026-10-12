"""How much the TUI slows the agent: time per trace event, and idle CPU (DESIGN-LOG §95).

  uv run python scripts/tui_bench.py [--events 2000]

The agent runs on a worker thread and hands every trace event to the UI with
`call_from_thread`, which blocks until the UI has rendered it (agent/tui.py
`_on_trace_event`). This pushes synthetic events through that same call from a real
thread, against the real RunApp in Textual's headless test driver, and reports the
agent-side wait per event by how many transcript entries already exist, with the wait
indicator animating and idle. It also measures the process's CPU time while the app
sits idle for a few seconds, animation on and off.

No model calls and no workspace changes. Textual's headless driver computes layout but
writes nothing to a terminal, so terminal painting cost is not in these numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import threading
import time

from agent.config import Settings
from agent.tui import RunApp


class _BenchApp(RunApp):
    def _run_agent(self):  # the real agent never starts
        pass


def _events(n: int) -> list[dict]:
    """A rough mix of what a run emits: narration, markdown, tool calls and results,
    model calls, and detail lines the panel does not show."""
    kinds = []
    for i in range(n):
        match i % 8:
            case 0:
                kinds.append({"kind": "log", "message": f"step {i}: reading the parser"})
            case 1:
                kinds.append({"kind": "log", "message": "## Plan\n\n- **one** thing\n- `two`"})
            case 2:
                kinds.append(
                    {"kind": "tool_call", "tool": "read_file", "args": {"path": f"f{i}.py"}}
                )
            case 3:
                kinds.append({"kind": "edit_file", "path": f"f{i}.py", "ok": True, "added": 3})
            case 4:
                kinds.append({"kind": "llm_call", "cost_usd": 0.0004, "latency_s": 1.2})
            case 5:
                kinds.append({"kind": "log", "message": "resolved tools", "detail": True})
            case 6:
                kinds.append({"kind": "run_tests", "summary": "3 passed", "exit_code": 0})
            case _:
                kinds.append({"kind": "step_attempt", "step": f"s{i}"})
    return kinds


def _buckets(waits: list[float], n: int) -> list[tuple[str, float, float, float]]:
    out = []
    for lo in (0, 400, 1400, max(0, n - 100)):
        chunk = waits[lo : lo + 100]
        if len(chunk) < 10:
            continue
        ms = sorted(w * 1000 for w in chunk)
        out.append(
            (
                f"events {lo}-{lo + len(chunk)}",
                statistics.median(ms),
                ms[int(0.9 * (len(ms) - 1))],
                ms[-1],
            )
        )
    return out


async def _idle_cpu(seconds: float = 3.0) -> float:
    cpu0, wall0 = time.process_time(), time.perf_counter()
    await asyncio.sleep(seconds)
    return (time.process_time() - cpu0) / (time.perf_counter() - wall0)


async def _run(n: int, animate: bool, ticker: bool = True) -> tuple[list[float], float, float, int]:
    app = _BenchApp(Settings(workspace=".", goal="bench"), None, "review")
    if not ticker:
        app._tick_wait_indicator = lambda: None  # ty: ignore[invalid-assignment]
    waits: list[float] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._waiting = animate
        app._show_wait_animation = animate
        empty_cpu = await _idle_cpu()

        def produce() -> None:
            for record in _events(n):
                t0 = time.perf_counter()
                app._on_trace_event(record)
                waits.append(time.perf_counter() - t0)

        thread = threading.Thread(target=produce)
        thread.start()
        while thread.is_alive():
            await asyncio.sleep(0.01)
        thread.join()
        # Settle first: right after a burst the UI is still mounting and rendering what
        # it was handed, which is backlog, not the cost of sitting idle.
        await asyncio.sleep(10.0)
        full_cpu = await _idle_cpu()
        entries = len(app.query_one("#log").children)
    return waits, empty_cpu, full_cpu, entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=2000)
    n = ap.parse_args().events
    conditions = (
        ("animation on (a call in flight)", True, True),
        ("animation off (idle)", False, True),
        ("0.15 s indicator timer disabled", False, False),
    )
    for label, animate, ticker in conditions:
        waits, empty_cpu, full_cpu, entries = asyncio.run(_run(n, animate, ticker))
        print(
            f"\n{label}: {n} events, {entries} transcript entries kept, "
            f"total agent-side wait {sum(waits):.2f} s"
        )
        print(
            f"  CPU while idle: {100 * empty_cpu:.1f}% of a core with an empty transcript, "
            f"{100 * full_cpu:.1f}% with {entries} entries (after a 10 s settle)"
        )
        print(f"  {'':<16}{'median ms':>10}{'p90 ms':>10}{'max ms':>10}")
        for name, med, p90, mx in _buckets(waits, n):
            print(f"  {name:<16}{med:>10.2f}{p90:>10.2f}{mx:>10.2f}")


if __name__ == "__main__":
    main()
