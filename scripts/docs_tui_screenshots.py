"""Screenshots of the dashboard (`harness-tui`) for docs/tui.md, from a recorded run: no model call.

  uv run python scripts/docs_tui_screenshots.py [TRACE.jsonl] [--out docs/images/tui]

The real agent worker is replaced by a no-op (as in tests/test_tui.py) and the events of a
recorded run are fed to the dashboard the way the live run would deliver them, so what is shown
is the dashboard's own rendering of a real run. Paths in the trace (the benchmark's temporary
workspace, this repository) are replaced by a demo project folder, `~/projects/expr-demo`, and
the folder the screenshots list is a throwaway copy holding just the run's two files. Output is
SVG (Textual's own screenshot format), one file per screen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
from pathlib import Path

from rich.text import Text
from textual.widgets import Static

from agent.config import Settings
from agent.tui import QuestionsModal, RunApp

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = REPO / "traces" / "bench" / "p1-fixtures-rest" / "run-20261002-033550-90dd35.jsonl"
SHOWN_WORKSPACE = "~/projects/expr-demo"
SIZE = (124, 38)


class DemoApp(RunApp):
    """The dashboard with no agent behind it, showing a demo path instead of the real one."""

    def _run_agent(self):
        pass

    def _topbar_left(self) -> Text:
        text = super()._topbar_left()
        return Text(str(text).replace(str(self.settings.workspace), SHOWN_WORKSPACE), style="")


def _scrub(value, replacements: list[tuple[str, str]]):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_scrub(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, replacements) for k, v in value.items()}
    return value


def load_events(trace: Path) -> tuple[str, list[dict]]:
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    start = next(e for e in events if e["kind"] == "run_start")
    workspace = start.get("workspace", "")
    replacements = [(workspace, "expr-demo"), (str(Path.home()), "~")]
    # The macOS temp folder appears in two spellings (/private/var/... and /var/...).
    replacements += [(re.sub(r"^/private", "", workspace), "expr-demo")]
    events = [_scrub(e, replacements) for e in events]
    return start["goal"], [e for e in events if e["kind"] != "run_start"]


def demo_workspace(root: Path) -> Path:
    ws = root / "expr-demo"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "expr.py").write_text('"""Evaluate arithmetic expressions."""\n')
    (ws / "test_expr.py").write_text("from expr import evaluate\n")
    return ws


async def shoot(trace: Path, out: Path) -> list[Path]:
    goal, events = load_events(trace)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        ws = demo_workspace(Path(tmp))
        app = DemoApp(Settings(workspace=str(ws), goal=goal), None, "review")
        async with app.run_test(size=SIZE) as pilot:
            app.query_one("#workspace-path", Static).update(SHOWN_WORKSPACE)

            def save(name: str) -> None:
                path = out / f"{name}.svg"
                app.save_screenshot(filename=path.name, path=str(out))
                written.append(path)

            # 1. Mid-run: the plan written and the first step under way.
            cut = next(i for i, e in enumerate(events) if e["kind"] == "write_file")
            for event in events[: cut + 1]:
                app._render_event(event)
            await pilot.pause()
            save("01-running")

            # 2. The rest of the run, then the usage panel.
            for event in events[cut + 1 :]:
                app._render_event(event)
            await pilot.pause()
            await pilot.press("ctrl+o")
            await pilot.pause()
            save("02-usage-panel")
            await pilot.press("ctrl+o")

            # 3. Settings.
            await pilot.press("ctrl+p")
            await pilot.pause()
            save("03-settings-panel")
            await pilot.press("ctrl+p")

            # 4. Workspace files.
            await pilot.press("ctrl+w")
            await pilot.pause(0.3)
            save("04-workspace-panel")
            await pilot.press("ctrl+w")

            # 5. An approval waiting in the bottom bar (manual mode), as _prompt_via_bar opens it.
            command = "python -c 'from expr import evaluate; print(evaluate(\"2+3*4\"))'"
            app._set_mode(
                "asking",
                f"🔒 approve — run_python: {command}",
                placeholder="y / a(lways this) / auto (everything now on) / n  (Enter alone = no, "
                "or ctrl+a to cycle mode)",
            )
            await pilot.pause()
            save("05-approval")

            # 6. Clarifying questions before planning (the clarify step's form).
            app._set_mode("running", placeholder="working…   ctrl+g to stop")
            app.push_screen(
                QuestionsModal(
                    [
                        (
                            "Should a bare number with a trailing dot, like '3.', be accepted?",
                            ["Yes, as 3.0", "No, raise ValueError"],
                            "Yes, as 3.0",
                        ),
                        ("Anything else the parser should reject?", None, None),
                    ]
                )
            )
            await pilot.pause()
            save("06-questions")
            app.pop_screen()

            # 7. Finished: the outcome and the bar for the next goal.
            app.summary = {
                "steps": {"passed": ["write-tests", "write-expr"], "failed": []},
                "final_check": {"ok": True},
                "suggested_next_step": "Support exponentiation (**) with right associativity?",
            }
            app._finish()
            await pilot.pause()
            save("07-done")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "images" / "tui")
    args = ap.parse_args()
    for path in asyncio.run(shoot(args.trace, args.out)):
        print(path)


if __name__ == "__main__":
    main()
