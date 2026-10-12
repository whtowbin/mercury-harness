"""Goose 1.52.0 (block-goose-cli): the developer extension (shell and file tools), headless
(DESIGN-LOG §109). Provider and model come from GOOSE_PROVIDER/GOOSE_MODEL. Usage is read from
Goose's session store (~/.local/share/goose/sessions/sessions.db) by the session name the driver
sets; cost is tokens times OpenRouter's list price, cached input at the cached price. Goose's
shell has network access, so a django run's leak check is done afterwards on its messages
(`harness trace leaks` cannot read them; see `leaks()` here)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from agents import Stats, price

VERSION = "1.52.0"
DB = Path(
    os.environ.get("GOOSE_SESSIONS_DB", Path.home() / ".local/share/goose/sessions/sessions.db")
)
MAX_TURNS = os.environ.get("GOOSE_MAX_TURNS", "60")


def session_name(out: Path) -> str:
    return f"s109-{out.parent.parent.name}-{out.name}"


def argv(goal: str, ws: Path, out: Path, model: str, reasoning: str) -> list[str]:
    return [
        "goose", "run", "--no-profile", "--with-builtin", "developer",
        "--max-turns", MAX_TURNS, "-n", session_name(out), "-q", "-t", goal,
    ]  # fmt: skip


def env(ws: Path, out: Path) -> dict[str, str]:
    model = os.environ.get("AGENT_MODEL", "inception/mercury-2.5")
    return {
        "GOOSE_PROVIDER": "openrouter",
        "GOOSE_MODEL": model.removeprefix("openrouter/"),
        "GOOSE_DISABLE_KEYRING": "1",
        "GOOSE_MODE": "auto",
        "PATH": f"{ws / '.venv' / 'bin'}:/opt/homebrew/bin:{os.environ.get('PATH', '')}",
        "VIRTUAL_ENV": str(ws / ".venv"),
    }


def _row(name: str) -> sqlite3.Row | None:
    if not DB.exists():
        return None
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            "select * from sessions where name = ? order by created_at desc limit 1", (name,)
        ).fetchone()
    finally:
        con.close()


def stats(out: Path) -> Stats:
    row = _row(session_name(out))
    if row is None:
        return Stats(status="no session")
    model = os.environ.get("AGENT_MODEL") or "inception/mercury-2.5"
    in_t = int(row["accumulated_input_tokens"] or 0)
    out_t = int(row["accumulated_output_tokens"] or 0)
    cached = int(row["accumulated_cache_read_tokens"] or 0)
    calls = _calls(row["id"])
    log = out / "stdout.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    status = "crashed" if "Error" in text[-500:] or "panicked" in text else "done"
    cost = price(model, in_t, out_t, cached)
    s = Stats(calls, cost, in_t, out_t, 0, status, "tokens x list price")
    s.extra = {"cached_tokens": cached}
    return s


def _calls(session_id: str) -> int | None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute(
            "select count(*) from messages where session_id = ? and role = 'assistant'",
            (session_id,),
        ).fetchone()[0]
    except sqlite3.Error:
        return None
    finally:
        con.close()


def leaks(out: Path, patterns: tuple[str, ...]) -> list[str]:
    """Message texts in this run's session that mention a denied source."""
    row = _row(session_name(out))
    if row is None:
        return []
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select content_json from messages where session_id = ?", (row["id"],)
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [r[0][:200] for r in rows if any(p in (r[0] or "") for p in patterns)]
