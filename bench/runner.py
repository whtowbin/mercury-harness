"""Running one configuration against one task, and scoring the result.

Each run gets its own throwaway workspace under the system temp directory, which
matters for more than hygiene: the harness refuses a workspace nested inside a
Python project, because build tools search upward and can modify their host.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from agent.config import Budget, Limits, Settings
from agent.loop import run as run_agent

from . import diagnostics, metrics
from .configs import AgentConfig
from .tasks import Task


def _seed(task: Task, workspace: Path) -> None:
    """Copy a task's starting files into the workspace, if it has any.

    This is what turns a task into editing existing code rather than writing
    from nothing: the agent finds a working, importable module already there —
    bugs included — instead of an empty directory.
    """
    seed_dir = task.seed_dir
    if seed_dir is None:
        return
    for path in seed_dir.rglob("*"):
        if path.is_file():
            target = workspace / path.relative_to(seed_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


# AgentConfig and Settings name a few knobs differently on purpose (a bench
# config reads better as `planner`/`skills` than `planner_model`/
# `skills_enabled`); those need an explicit rename. Everything else that
# exists on both, under the same name, is forwarded automatically below —
# which is the actual fix: Settings has grown fields since AgentConfig was
# written (allow_nested, quiet, skills_dir, trace_dir, ...) that a hand-kept
# field list here silently never reached. A same-named field can no longer
# go missing this way; a differently-named one still needs a line below, and
# test_bench.py asserts every same-named field actually arrives.
_RENAMED = {
    "planner": "planner_model",
    "executor": "executor_model",
    "reviewer": "reviewer_model",
    "skills": "skills_enabled",
    "plan_auditor": "plan_auditor_model",
}


def _settings_for(config: AgentConfig, task: Task, workspace: Path, trace_dir: str) -> Settings:
    forwarded = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(Settings)
        if f.name not in _RENAMED.values() and hasattr(config, f.name)
    }
    renamed = {
        settings_name: getattr(config, config_name)
        for config_name, settings_name in _RENAMED.items()
    }
    return Settings(
        workspace=str(workspace),
        goal=task.goal,
        trace_dir=trace_dir,
        # The benchmark is unattended, so shell commands and clarifying
        # questions cannot wait on a human.
        auto_approve=True,
        quiet=True,
        ask_enabled=False,
        suggest_next_step=False,
        # A bench task is measuring the plan/execute/review pipeline itself
        # (that's what the whole sweep is comparing); a quick-answer
        # short-circuit deciding not to run it would be measuring the
        # classifier instead, on tasks that all genuinely need the pipeline.
        quick_answers=False,
        # A fixture's workspace is the task's starting state; scaffolding an empty one would
        # add files and a formatter config that the configs being compared did not choose.
        scaffold_new_projects=False,
        # Same reasoning as ask_enabled: there is nobody to answer a
        # clarifying question in a sweep, and a bench task's goal is fixed
        # input that must not vary between configs being compared.
        clarify=False,
        # Memory would make a sweep measure the wrong thing entirely: every
        # repeat of a task would start knowing what the previous repeat did,
        # so repeat 3 is not the same experiment as repeat 1, and two configs
        # run in different orders would not be comparable. Each bench job must
        # be an independent trial.
        memory=False,
        # A bench fixture's workspace starts clean, and a sweep must measure
        # the task it set rather than whatever a seed file happens to leave
        # red; scoring is the hidden suite's job, not the gate's.
        fix_existing_failures=False,
        # Scoring is the hidden suite's job here, and several fixtures are
        # deliberately "write this one function" tasks; failing them for not
        # bringing their own tests would measure the wrong thing.
        require_tests=False,
        # Same reasoning, and it would add a subprocess to every bench job.
        check_coverage=False,
        # A failed bench job already gets a full traceback and its own trace
        # file under trace_dir; a second copy scattered into the project's
        # working directory on every failing job in a large sweep is noise,
        # not a debugging aid the way it is for a single interactive run.
        failure_log_dir=None,
        budget=Budget(max_usd=task.max_usd, step_prompt_token_budget=config.judge_step_budget),
        limits=Limits(),
        **forwarded,
        **renamed,
    )


def _worker(
    config_name: str,
    task_name: str,
    repeat: int,
    trace_dir: str,
    keep_workspace: bool,
    stop_after: str,
    queue,
) -> None:
    """Runs in a separate process, so the parent can kill it on a timeout.

    A hung HTTP call cannot be interrupted from inside the same process — the
    request library blocks the interpreter, threads included. A subprocess is
    the only thing the parent can unconditionally terminate.
    """
    from . import configs as configs_module
    from . import tasks as tasks_module

    config = configs_module.get(config_name)
    task = tasks_module.get(task_name)
    queue.put(run_one(config, task, repeat, trace_dir, keep_workspace, stop_after))


def _sleep_forever(*_args) -> None:
    """A worker with exactly one behaviour: never return.

    Used only to test the kill path below without needing a slow model — a
    spawned child re-imports this module fresh, so a target must be real and
    top-level, not something monkeypatched in the parent test process.
    """
    while True:
        time.sleep(1)


def _run_with_hard_timeout(
    target, args: tuple, timeout_s: float, ctx: mp.context.SpawnContext | None = None
) -> tuple[bool, Any, int | None]:
    """Run `target(*args, queue)` in a spawned process, killed at timeout_s.

    Returns (finished_in_time, result_or_none, exitcode). A hung HTTP call
    cannot be interrupted from inside the same process — the request library
    blocks the interpreter, threads included — so a subprocess is the only
    thing the parent can unconditionally terminate.
    """
    ctx = ctx or mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            # Bounded even here. SIGKILL cannot be blocked, but the *reap* is a
            # separate step, and a scheduler that can still stall forever on it
            # is not the "strict timeout" it claims to be. A run of this harness
            # measured a 2260s wall time on a job whose limit was 120s: this
            # unbounded join was why. If the OS has not reaped the process by
            # now, stop waiting — the sweep's own deadline matters more than a
            # clean wait, and an orphaned process is a much smaller problem
            # than the timeout feature silently not working.
            proc.join(10)
        return False, None, proc.exitcode

    result = queue.get() if not queue.empty() else None
    return True, result, proc.exitcode


def run_one_with_timeout(
    config: AgentConfig,
    task: Task,
    repeat: int = 0,
    trace_dir: str = "traces/bench",
    keep_workspace: bool = False,
    timeout_s: float | None = None,
    stop_after: str = "review",
) -> dict[str, Any]:
    """run_one, killed at a hard wall-clock limit instead of running to completion.

    Without this, one very slow model turns a sweep into a wait of unknown
    length — GLM 5.3 Flash took over 1000s for a single run in an earlier
    sweep. A timed-out run is scored as a failure rather than silently
    vanishing from the results.
    """
    if not timeout_s or timeout_s <= 0:
        return run_one(config, task, repeat, trace_dir, keep_workspace, stop_after)

    started, started_mono = time.time(), time.monotonic()
    finished, result, exitcode = _run_with_hard_timeout(
        _worker, (config.name, task.name, repeat, trace_dir, keep_workspace, stop_after), timeout_s
    )
    wall = time.time() - started
    # The limit is enforced on the monotonic clock, so a run the machine slept through
    # is killed with wall time well past the limit (scout-2: 639s against 300s).
    slept = round(max(0.0, wall - (time.monotonic() - started_mono)), 1)

    if not finished:
        return flatten(_timeout_record(config, task, repeat, wall, timeout_s) | {"slept_s": slept})
    if result is None:
        return flatten(_crash_record(config, task, repeat, wall, exitcode) | {"slept_s": slept})
    return result


def _timeout_record(
    config: AgentConfig, task: Task, repeat: int, wall: float, timeout_s: float
) -> dict[str, Any]:
    record = _bare_record(config, task, repeat, wall)
    record["agent_error"] = f"BenchTimeout: exceeded the {timeout_s:.0f}s wall-clock limit"
    record["timed_out"] = True
    record["scores"] = {"error": "run was killed before finishing; nothing to score"}
    return record


def _crash_record(
    config: AgentConfig, task: Task, repeat: int, wall: float, exitcode: int | None
) -> dict[str, Any]:
    record = _bare_record(config, task, repeat, wall)
    record["agent_error"] = f"worker process exited unexpectedly (code {exitcode})"
    record["scores"] = {"error": "worker crashed before producing a result"}
    return record


def _bare_record(config: AgentConfig, task: Task, repeat: int, wall: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "config": config.name,
        "task": task.name,
        "repeat": repeat,
        "executor": config.executor,
        "planner": config.planner,
        "reviewer": config.reviewer,
        "review_enabled": config.review,
        "started_at": time.time() - wall,
        "wall_s": wall,
        "workspace": "",
        "timed_out": False,
    }
    record.update(_empty_totals())
    return record


def run_one(
    config: AgentConfig,
    task: Task,
    repeat: int = 0,
    trace_dir: str = "traces/bench",
    keep_workspace: bool = False,
    stop_after: str = "review",
) -> dict[str, Any]:
    """Run the agent once and return a flat record of what happened.

    `stop_after="plan"` (`bench run/scout --plan-only`) stops once a plan is accepted:
    for questions about the plans themselves, at a fraction of a full run's cost. The
    workspace is not scored, since nothing was executed.
    """
    workspace = Path(tempfile.mkdtemp(prefix=f"bench-{config.name}-{task.name}-"))
    record: dict[str, Any] = {
        "config": config.name,
        "task": task.name,
        "repeat": repeat,
        "executor": config.executor,
        "planner": config.planner,
        "reviewer": config.reviewer,
        "review_enabled": config.review,
        "started_at": time.time(),
    }
    started = time.time()
    # Monotonic time stops while the machine sleeps; wall time does not. The gap is
    # how long this run spent asleep, which a laptop with its lid closed does for
    # minutes at a time with every in-flight call stalled (DESIGN-LOG §61.2).
    started_mono = time.monotonic()
    try:
        _seed(task, workspace)
        settings = _settings_for(config, task, workspace, trace_dir)
        # Otherwise always run to the end: whether review happens is
        # settings.review, not where the run stops. Expressing it as
        # stop_after="execute" also skipped the final gate, which made the
        # no-review configuration a strictly smaller agent than the no-gate one
        # and its timings incomparable with everything else.
        summary = run_agent(settings, stop_after="plan" if stop_after == "plan" else "review")
        record["plan_only"] = stop_after == "plan"
        # A run that failed part-way still reports what it spent and produced.
        record["agent_error"] = summary.get("error", "")
        record.update(_from_summary(summary))
    except Exception as exc:
        # A crashed run is a data point, not a reason to abandon the sweep.
        record["agent_error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-1500:]
        record.update(_empty_totals())
    record["wall_s"] = time.time() - started
    record["slept_s"] = round(max(0.0, record["wall_s"] - (time.monotonic() - started_mono)), 1)

    try:
        record["scores"] = (
            {"error": "planning only (--plan-only): nothing was executed to score"}
            if stop_after == "plan"
            else metrics.score(task, workspace)
        )
    except Exception as exc:
        record["scores"] = {"error": f"{type(exc).__name__}: {exc}"}

    record.setdefault("timed_out", False)
    record["workspace"] = str(workspace) if keep_workspace else ""
    if not keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return flatten(record)


def _empty_totals() -> dict[str, Any]:
    return {
        "cost_usd": 0.0,
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "model_latency_s": 0.0,
        "steps": 0,
        "findings": 0,
        "trace_path": "",
        "by_model": {},
    }


def _from_summary(summary: dict) -> dict[str, Any]:
    totals = summary.get("totals") or {}
    plan = summary.get("plan") or {}
    return {
        "cost_usd": float(totals.get("cost_usd", 0.0)),
        "calls": int(totals.get("calls", 0)),
        "prompt_tokens": int(totals.get("prompt_tokens", 0)),
        "completion_tokens": int(totals.get("completion_tokens", 0)),
        "cached_tokens": int(totals.get("cached_tokens", 0)),
        "reasoning_tokens": int(totals.get("reasoning_tokens", 0)),
        "model_latency_s": float(totals.get("latency_s", 0.0)),
        "steps": len(plan.get("steps") or []),
        "findings": len(summary.get("findings") or []),
        "trace_path": diagnostics.portable_path(summary.get("trace_path", "")),
        "by_model": totals.get("by_model") or {},
        **_audit_columns(summary.get("test_audit")),
    }


def _audit_columns(audit: dict | None) -> dict[str, Any]:
    """The harness's own post-run test audit (agent/testcheck.py), as columns.

    Recorded next to bench/integrity.py's verdict on the same run so the two
    can be compared: integrity restores the *seeded* bug and knows the answer,
    the audit restores the pre-run source and knows nothing, and how often they
    agree is what says whether the audit is worth trusting on a real project.
    """
    audit = audit or {}
    return {
        "audit_ran": bool(audit.get("ran")),
        "audit_pass_on_old": audit.get("added_tests_pass_on_old_code"),
        "audit_mutants": int(audit.get("mutants_tried", 0)),
        "audit_survivors": len(audit.get("survivors") or []),
        "audit_seconds": float(audit.get("seconds", 0.0)),
        "audit_repaired": audit.get("repaired"),
        "audit_findings_before": len(audit.get("findings_before") or []),
    }


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    """Lift the nested scores into top-level columns the report can use."""
    scores = record.get("scores") or {}
    accuracy = scores.get("accuracy") or {}
    execution = scores.get("execution") or {}
    quality = scores.get("quality") or {}

    record["accuracy"] = float(accuracy.get("ratio", 0.0))
    record["tests_total"] = int(accuracy.get("total", 0))
    record["tests_passed"] = int(accuracy.get("passed", 0))
    record["accuracy_error"] = accuracy.get("error_message", "")
    record["exec_seconds"] = execution.get("seconds")
    record["exec_error"] = execution.get("error", "")
    record["delivered"] = bool(quality.get("delivered", False))
    record["ruff_issues"] = int(quality.get("ruff_issues", 0))
    record["ty_issues"] = int(quality.get("ty_issues", 0))
    record["missing_files"] = quality.get("missing_files") or []

    # Whether the result was reached honestly — see bench/integrity.py. These
    # are the columns that separate "fixed the bug" from "stopped the tests
    # noticing it", which accuracy alone cannot do.
    honesty = scores.get("integrity") or {}
    record["honest"] = bool(honesty.get("honest", True))
    record["broke_seeded_tests"] = list(honesty.get("broken_seeded_tests") or [])
    record["removed_seeded_tests"] = list(honesty.get("seeded_tests_removed") or [])
    record["skipped_seeded_tests"] = list(honesty.get("seeded_tests_skipped") or [])
    record["changed_signatures"] = list(honesty.get("signature_changes") or [])
    record["own_tests_ran"] = bool(honesty.get("own_tests_ran", False))
    record["own_tests_miss_the_bug"] = bool(honesty.get("own_tests_miss_the_bug", False))
    record["weakened_seeded_tests"] = list(honesty.get("seeded_tests_weakened") or [])
    record["source_lines_changed"] = int(honesty.get("source_lines_changed", 0))
    record["changed_correct_code"] = bool(honesty.get("changed_correct_code", False))

    # Process signals from the run's own trace: what the outcome columns
    # cannot say when every config passes everything (see bench/diagnostics.py).
    # A bug in the counters must cost a column, not the run: this once raised
    # inside the worker and turned every job of a sweep into a crash record.
    try:
        record.update(
            {
                f"diag_{k}": v
                for k, v in diagnostics.from_trace(record.get("trace_path") or "").as_dict().items()
            }
        )
    except Exception as exc:
        record["diag_error"] = f"{type(exc).__name__}: {exc}"

    record["served_models"] = diagnostics.served_models_from_trace(record.get("trace_path") or "")
    record["plan_task_kind"] = diagnostics.task_kind_from_trace(record.get("trace_path") or "")

    # Executor timing specifically, so the dLLM can be compared inside a hybrid run.
    slot = (record.get("by_model") or {}).get(record.get("executor", "")) or {}
    record["executor_calls"] = int(slot.get("calls", 0))
    record["executor_out_tokens"] = int(slot.get("completion_tokens", 0))
    record["executor_reasoning_tokens"] = int(slot.get("reasoning_tokens", 0))
    record["executor_latency_s"] = float(slot.get("latency_s", 0.0))
    record["executor_decode_s"] = float(slot.get("decode_s", 0.0))
    record["executor_ttft_s"] = float(slot.get("ttft_s", 0.0))
    return record


class ResultLog:
    """Append-only JSONL of every run, written as the sweep progresses."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> list[dict]:
        records = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
