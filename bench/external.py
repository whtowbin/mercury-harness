"""Running Aider on the same tasks, scored the same way, for comparison.

The point of this module is narrow: everything that makes a bench.runner
record comparable — `metrics.score`, `flatten`, the field names a report row
reads — is reused unmodified. Only the part that actually differs between
harnesses (how the agent is invoked) is new code. If this module scored
things its own way, a table built from both would be comparing numbers that
merely look alike.

Requires the `aider` console script on PATH — install pinned below Python 3.13,
since aider's own dependency chain does not currently run cleanly above that:

    uv tool install aider-chat --python 3.12

(A bare `uv tool install aider-chat` on this machine resolves scipy against
Python 3.14 and tries to build it from source, which needs a Fortran
toolchain nobody has; pinning to 3.12 sidesteps both that and a separate
crash where 3.13 removed the stdlib `audioop` module aider's optional voice
feature imports at startup.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from . import metrics
from .runner import _bare_record, _seed, flatten
from .tasks import Task

AIDER_TIMEOUT_S = 300


def _api_key() -> str:
    """The same OpenRouter key the harness itself uses, from Keychain or env.

    Aider reads its key from the environment, not from this project's own
    config module, so the subprocess needs it set explicitly — this is that
    lookup, not a second credential.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    from agent.keystore import read

    return read("OpenRouter")


def run_aider(
    model: str,
    task: Task,
    repeat: int = 0,
    timeout_s: float = AIDER_TIMEOUT_S,
    keep_workspace: bool = False,
) -> dict[str, Any]:
    """One Aider run against one task, scored like a bench.runner record.

    `model` is an OpenRouter model id without the `openrouter/` prefix — the
    same strings bench/configs.py already uses (GLM_FLASH, QWEN_FLASH) — so a
    result can sit in the same table next to this harness's own rows for the
    identical model.
    """
    from .configs import AgentConfig  # local import: avoids a cycle at module load

    config = AgentConfig(
        name=f"aider-{model.split('/')[-1]}",
        planner=model,
        executor=model,
        reviewer=model,
        group="external",
    )
    workspace = Path(tempfile.mkdtemp(prefix=f"aider-{config.name}-{task.name}-"))
    started = time.time()
    key = _api_key()
    if not key:
        record = _bare_record(config, task, repeat, 0.0)
        record["agent_error"] = "no OpenRouter API key available for the aider subprocess"
        record["scores"] = {"error": "not run: no API key"}
        return flatten(record)

    _seed(task, workspace)
    if not shutil.which("aider"):
        record = _bare_record(config, task, repeat, time.time() - started)
        record["agent_error"] = "aider is not on PATH (uv tool install aider-chat --python 3.12)"
        record["scores"] = {"error": "not run: aider missing"}
        shutil.rmtree(workspace, ignore_errors=True)
        return flatten(record)

    targets = [f.name for f in sorted(workspace.glob("*.py")) if not f.name.startswith("test_")]
    env = {**os.environ, "OPENROUTER_API_KEY": key}
    cmd = [
        "aider",
        "--model",
        f"openrouter/{model}",
        "--yes-always",
        "--no-auto-commits",
        "--no-check-update",
        "--no-show-model-warnings",
        "--no-stream",
        "--no-gitignore",
        "--message",
        task.goal,
        *targets,
    ]
    agent_error = ""
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            agent_error = f"aider exited {proc.returncode}: {proc.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        timed_out = True
        agent_error = f"AiderTimeout: exceeded the {timeout_s:.0f}s wall-clock limit"
    except (OSError, subprocess.SubprocessError) as exc:
        agent_error = f"{type(exc).__name__}: {exc}"

    wall = time.time() - started
    record = {
        "config": config.name,
        "task": task.name,
        "repeat": repeat,
        "executor": model,
        "planner": model,
        "reviewer": model,
        "review_enabled": False,  # aider has no separate review phase
        "started_at": started,
        "wall_s": wall,
        "timed_out": timed_out,
        "agent_error": agent_error,
        # Aider's own cost/token accounting is parsed from its chat log rather
        # than fetched here, to keep this module to one subprocess call; a
        # sweep that wants those numbers can read them from the transcript
        # Aider leaves in the workspace (.aider.chat.history.md).
        "cost_usd": 0.0,
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "model_latency_s": wall,
        "steps": 1,
        "findings": 0,
        "trace_path": "",
        "by_model": {},
    }
    try:
        record["scores"] = metrics.score(task, workspace)
    except Exception as exc:
        # A scoring crash is a data point, not a reason to abandon the run.
        record["scores"] = {"error": f"{type(exc).__name__}: {exc}"}

    record["workspace"] = str(workspace) if keep_workspace else ""
    if not keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return flatten(record)
