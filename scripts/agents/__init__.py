"""One interface for running another coding agent on a prepared workspace (DESIGN-LOG §109).

Each agent module exposes:
  argv(goal, ws, out, model, reasoning) -> list[str]   the command to run in `ws`
  env(ws, out) -> dict                                  environment additions for that command
  stats(out) -> Stats                                   calls, tokens and cost after the run

`out` is a per-run directory the driver creates; stdout goes to out/stdout.log. Cost is the
provider's own figure where the agent records one (mini, Aider) and otherwise tokens times
OpenRouter's list price (Goose, Codex), cached input at the cached price: Codex cached 94% of its
django input, so pricing it all at list overstated its cost about 6x (DESIGN-LOG §122). The
suites (django, SciCode, the fixtures) share these modules, so a new agent is one file.

Every suite launches an agent through `sandboxed()`: the agent runs model-chosen shell commands
without asking (mini's `-y`, Goose's auto mode, Aider's `--yes-always`), so it runs under
`sandbox-exec` with the home folder unreadable except its workspace, its output folder, the
tools it needs and its own state, and with an allowlisted environment instead of the caller's
(a release-prep item; the same rules as `scripts/swebench/iso.py` gives harness arms).
Network is not restricted here: mini's shell gets a dead proxy, Codex's own sandbox denies it,
and Goose's shell has the web, as before.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DEAD_PROXY = "http://127.0.0.1:9"
PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
# OpenRouter list prices, $ per million tokens (input, cached input, output), read 2 Oct 2026
# from /api/v1/models. Mercury's cached price is agent/providers.py's; Goose's own session cost
# agrees with it to the cent. None: no cached price known, so cached input is priced as input.
PRICES = {
    "mercury-2.5": (0.04, 0.004, 0.15),
    "glm-5.3-flash": (0.15, None, 0.50),
    "claude-haiku-4.5": (1.00, None, 5.00),
}
AGENTS = ("mini", "aider", "goose", "codex", "claude")

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
HOME = Path.home()
# What every agent may still read in the home folder: uv's interpreters, cache and tools (the
# workspaces' venvs and `uvx` resolve there), git's config and macOS preferences.
HOME_READ = [
    ".cache/uv", ".local/share/uv", ".local/bin", ".config/uv", ".gitconfig", ".config/git",
    "Library/Preferences",
]  # fmt: skip
HOME_WRITE = [".cache/uv"]
# Each agent's own state, read and written: its config, session store and logs. Goose's session
# store is also where the driver reads its usage afterwards (goose.DB).
AGENT_STATE = {
    "mini": ["Library/Application Support/mini-swe-agent"],
    "aider": [".aider"],
    "goose": [".config/goose", ".local/share/goose", ".local/state/goose"],
    "codex": [".codex"],
}
# Passed through from the caller's environment; everything else (other API keys, tokens, the
# harness's own settings) is dropped.
ENV_KEEP = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "TZ", "__CF_USER_TEXT_ENCODING",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
)  # fmt: skip
ENV_KEEP_PREFIXES = ("LC_", "UV_")


@dataclass
class Stats:
    calls: int | None = None  # None when the agent reports usage per turn, not per call
    cost: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    reasoning_tokens: int = 0
    status: str = ""
    cost_source: str = ""  # "reported" or "tokens x list price"
    extra: dict = field(default_factory=dict)


def price(model: str, in_tokens: int, out_tokens: int, cached_tokens: int = 0) -> float:
    """Dollars for a run. `in_tokens` includes `cached_tokens`, as Codex and Goose count them."""
    for key, (pin, pcached, pout) in PRICES.items():
        if key in model:
            cached = min(cached_tokens, in_tokens)
            pcached = pin if pcached is None else pcached
            return ((in_tokens - cached) * pin + cached * pcached + out_tokens * pout) / 1e6
    raise KeyError(f"no list price for {model!r}; add it to agents.PRICES")


def api_key() -> str:
    env = os.environ.get("OPENROUTER_API_KEY", "")
    if env:
        return env
    r = subprocess.run(
        ["security", "find-generic-password", "-a", "harness", "-s", "OpenRouter", "-w"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def sandbox_profile(rw: Iterable[Path], ro: Iterable[Path], home: Path = HOME) -> str:
    """sandbox-exec rules: the home folder and the session scratch folders denied, then the
    allowed paths (the last matching rule wins). File metadata stays readable, so paths still
    resolve. Paths are resolved, since the rules match real paths, not symlinks."""

    def sub(p: Path) -> str:
        return f'(subpath "{p.resolve()}")'

    scratch = [Path(f"/private/tmp/claude-{os.getuid()}"), Path(f"/tmp/claude-{os.getuid()}")]
    read = [home / p for p in HOME_READ] + list(ro)
    lines = [
        "(version 1)",
        "(allow default)",
        f"(deny file-read-data file-read-xattr file-write* {sub(home)})",
        "(deny file-read-data file-read-xattr file-write* " + " ".join(map(sub, scratch)) + ")",
        "(allow file-read-data file-read-xattr " + " ".join(map(sub, read)) + ")",
        "(allow file-write* " + " ".join(sub(home / p) for p in HOME_WRITE) + ")",
        "(allow file-read-data file-read-xattr file-write* " + " ".join(map(sub, rw)) + ")",
    ]
    return "\n".join(lines) + "\n"


def sandboxed(
    agent,
    argv: list[str],
    ws: Path,
    out: Path,
    key: str,
    extra_env: dict[str, str],
    ro: Iterable[Path] = (),
    rw: Iterable[Path] = (),
    home: Path = HOME,
) -> tuple[list[str], dict[str, str]]:
    """The command and environment that run `agent` (a module from `load`) inside a sandbox.
    Writes the profile to out/sandbox.sb and gives the agent out/tmp as its TMPDIR. `ro` adds
    paths the agent may read, such as the harness's .venv for the fixtures' pytest and ruff, and
    `rw` paths it may also write outside `ws` and `out`. Aider's own venv and Goose's session
    store come from the driver module (VENV, DB)."""
    name = agent.__name__.rsplit(".", 1)[-1]
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    writable = [ws, out, *rw, *(home / p for p in AGENT_STATE.get(name, []))]
    if (db := getattr(agent, "DB", None)) is not None:
        writable.append(Path(db).parent)
    read = list(ro)
    if (venv := getattr(agent, "VENV", None)) is not None:
        read.append(Path(venv))
    sb = out / "sandbox.sb"
    sb.write_text(sandbox_profile(writable, read, home))
    env = {k: v for k, v in os.environ.items() if k in ENV_KEEP or k.startswith(ENV_KEEP_PREFIXES)}
    env.update(OPENROUTER_API_KEY=key, TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    # `uvx` (mini-swe-agent) makes its tool environment and a temporary file under uv's tools
    # folder; the first §126 smoke run died there. Not the user's ~/.local/share/uv/tools, whose
    # binaries the user later runs: a folder of the run's own.
    env.update(UV_TOOL_DIR=str(out / "uv-tools"), UV_TOOL_BIN_DIR=str(out / "uv-tools-bin"))
    env.update(extra_env)
    return [str(SANDBOX_EXEC), "-f", str(sb), *argv], env


def load(name: str):
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; one of {AGENTS}")
    return importlib.import_module(f"agents.{name}")
