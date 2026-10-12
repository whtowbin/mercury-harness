"""Benchmark command line.

uv run bench list
uv run bench run --configs hybrid,mercury-only --tasks roman,lru --repeats 3
uv run bench report results/bench-*.jsonl --by task
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent import awake
from agent.tools import web

from . import archive as archive_module
from . import configs, diagnostics, report, stalls, tasks
from . import scout as scout_module
from .external import run_aider
from .runner import ResultLog, run_one, run_one_with_timeout


class ModelGate:
    """Caps concurrent jobs per underlying model, independent of --parallel.

    --parallel bounds how many *jobs* run at once; it says nothing about how
    many of them hit the same model. A sweep with several Sonnet-based configs
    can put more concurrent load on one account's Sonnet quota than the
    account can actually serve, and the jobs queue at the provider rather than
    failing outright — indistinguishable from a hang until --timeout kills
    them having made zero calls. This gates job *starts* in the process that
    schedules them, which works whether the job itself runs in a thread or
    (via --timeout) a spawned subprocess: either way, the parent decides when
    it begins.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._lock = threading.Lock()
        self._semaphores: dict[str, threading.Semaphore] = {}

    def _semaphore(self, model: str) -> threading.Semaphore:
        with self._lock:
            if model not in self._semaphores:
                self._semaphores[model] = threading.Semaphore(self.limit)
            return self._semaphores[model]

    def run(self, models: set[str], fn, *args):
        if self.limit <= 0:
            return fn(*args)
        held = [self._semaphore(m) for m in sorted(models)]
        for sem in held:
            sem.acquire()
        try:
            return fn(*args)
        finally:
            for sem in held:
                sem.release()


def _expand(raw: str, available: list[str], what: str) -> list[str]:
    if raw.strip() in ("all", "*"):
        return available
    if what == "config" and raw.strip() in configs.groups():
        return configs.in_group(raw.strip())
    chosen = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in chosen if item not in available]
    if unknown:
        raise SystemExit(f"unknown {what}: {', '.join(unknown)}\nAvailable: {', '.join(available)}")
    return chosen


def _record_status(record: dict) -> str:
    if record.get("timed_out"):
        return "TIMEOUT"
    if record.get("agent_error"):
        return "ERROR"
    if record.get("plan_only"):
        # Nothing was executed, so there is no accuracy to show; the check's
        # outcome is what a planning-only run is for.
        return f"plan ({report.repro_outcome(record) or 'no check'})"
    return f"{record.get('accuracy', 0.0) * 100:.0f}%"


def _confirmed(yes: bool) -> bool:
    """Ask before spending; --yes skips the question."""
    if yes:
        return True
    if input("proceed? [y/N]: ").strip().lower().startswith("y"):
        return True
    print("cancelled")
    return False


def _search_ready(allow_dead: bool) -> bool:
    """Check web search with a canary query before a sweep spends anything.

    Every upstream engine of the local SearXNG was blocking this machine's IP for two
    days of sweeps; each search returned 200 with nothing, and "research" arms ran
    with no web at all (DESIGN-LOG §89.1). Any config may search (the executor's web
    tools are keyword-triggered), so this does not try to guess which arms would.
    """
    ok, detail = web.search_health()
    if ok is None:
        print(f"note: {detail}; web_search will say so to the model")
        return True
    if ok:
        return True
    if allow_dead:
        print(f"warning: web search is not working ({detail}); running anyway")
        return True
    print(
        f"error: web search is not working: {detail}.\n"
        "Fix the backend (scripts/searxng/) or pass --allow-dead-search to run without it.",
        file=sys.stderr,
    )
    return False


ACCOUNT_FAILURE_LIMIT = 3


class _AccountGuard:
    """Stops a sweep once the account is refusing every request.

    An account that is out of credits fails each remaining run in about two
    seconds, so a sweep left alone burns through its whole job list producing
    records that say nothing (a 192-run sweep lost 69 runs this way, after an
    hour of good ones). Three refusals in a row is not chance.
    """

    def __init__(self, limit: int = ACCOUNT_FAILURE_LIMIT) -> None:
        self.limit = limit
        self.streak = 0

    def __call__(self, record: dict) -> str | None:
        if stalls.is_account_failure(record):
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.limit:
            return (
                f"{self.streak} runs in a row were refused with HTTP 401/402/403: the account is "
                "out of credits or the key is invalid. Stopping instead of burning the job list."
            )
        return None


def _land(record: dict | None, log, announce, stop) -> str | None:
    """Log and announce one finished record; the reason to stop, if it trips `stop`.
    A skipped job (None) leaves no record."""
    if record is None:
        return None
    log.append(record)
    announce(record)
    return stop(record) if stop else None


def _run_parallel(jobs, run, parallel: int, log, announce, stop) -> str | None:
    # Runs are independent: separate workspaces, separate traces, and the time is
    # nearly all spent waiting on the network.
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        landed: set = set()
        for future in as_completed(futures):
            if future.cancelled():
                continue
            landed.add(future)
            reason = _land(future.result(), log, announce, stop)
            if reason:
                for pending in futures:
                    pending.cancel()
                # Runs already underway still finish; their records are kept, not
                # thrown away with the ones that never started. Only those: a run
                # that finished before the stop was logged when it landed, and
                # logging it again duplicated it in the results file (the
                # call-site scout's 23 lines held 12 runs).
                for running in futures:
                    if running not in landed and not running.cancelled():
                        _land(running.result(), log, announce, None)
                return reason
    return None


def _run_jobs(jobs, run, parallel: int, log, announce, stop=None) -> str | None:
    """Run every job, logging and announcing each record as it lands.

    `stop(record)` may return a reason to stop scheduling; the reason is returned.
    Jobs already running finish; jobs not yet started are dropped.
    """
    if parallel > 1:
        return _run_parallel(jobs, run, parallel, log, announce, stop)
    for job in jobs:
        reason = _land(run(*job), log, announce, stop)
        if reason:
            return reason
    return None


def _print_sweep_tables(records: list[dict], task_count: int) -> None:
    completed, stalled = stalls.split(records)
    text = stalls.note(records, stalled)
    if text:
        print(text)
    print(report.render(completed))
    if task_count > 1:
        print(report.render_by_task(completed))


def _config_flags(config) -> list[str]:
    flags = []
    if config.step_concurrency > 1:
        flags.append(f"concurrency={config.step_concurrency}")
    if not config.verify_steps:
        flags.append("no-verify")
    if not config.review:
        flags.append("no-review")
    if not config.autolint:
        flags.append("no-lint")
    return flags


def _task_flags(task) -> list[str]:
    flags = []
    if task.perf_script:
        flags.append("timed workload")
    if task.seed_dir:
        flags.append("edits existing code")
    return flags


def _bracketed(flags: list[str]) -> str:
    return f"  [{', '.join(flags)}]" if flags else ""


def cmd_list(_args) -> int:
    for group in configs.groups():
        print(f"configurations [{group}]:")
        for name in configs.in_group(group):
            config = configs.get(name)
            print(f"  {name:<28} {config.description}{_bracketed(_config_flags(config))}")
        print()
    print("Pass a group name to --configs to run all of it.")
    print("\ntasks:")
    for name in tasks.names():
        task = tasks.get(name)
        print(f"  {name:<18} {task.notes}{_bracketed(_task_flags(task))}")
    retired = [n for n in tasks.all_names() if n not in tasks.names()]
    if retired:
        print(
            f"\nretired (100% ceiling for every config tested; --include-deprecated to run): "
            f"{', '.join(retired)}"
        )
    return 0


def cmd_run(args) -> int:
    chosen_configs = _expand(args.configs, configs.names(), "config")
    task_pool = tasks.all_names() if args.include_deprecated else tasks.names()
    chosen_tasks = _expand(args.tasks, task_pool, "task")
    if args.repeats < 3:
        print(
            f"note: --repeats {args.repeats} is too few to trust a cost or accuracy "
            "delta between configs; the harness has seen 40-65% run-to-run cost "
            "variance on identical inputs. Use --repeats 5+ before drawing a conclusion."
        )

    jobs = plan_jobs(chosen_configs, chosen_tasks, args.repeats, args.repeat_offset)
    estimated = sum(
        tasks.get(t).max_usd
        for _ in chosen_configs
        for t in chosen_tasks
        for _ in range(args.repeats)
    )
    out_path = Path(args.out or f"results/bench-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    log = ResultLog(out_path)

    print(
        f"{len(jobs)} runs: {len(chosen_configs)} config(s) x "
        f"{len(chosen_tasks)} task(s) x {args.repeats} repeat(s)"
    )
    print(f"worst-case spend if every run hits its cap: ${estimated:.2f}")
    print(f"results: {out_path}")
    if not _confirmed(args.yes):
        return 1
    if not _search_ready(args.allow_dead_search):
        return 2

    stopped = _execute(jobs, args, log, stop=_AccountGuard())
    _print_sweep_tables(log.records, len(chosen_tasks))
    print(f"\nresults written to {out_path}")
    if stopped:
        print(f"\nSWEEP STOPPED: {stopped}", file=sys.stderr)
        return 3
    return 0


def _keep_awake() -> subprocess.Popen | None:
    """See agent/awake.py: shared with `harness run` since §71."""
    return awake.keep_awake()


def _execute(jobs, args, log, stop, skip=None) -> str | None:
    """Run `jobs`, announcing each; returns the reason `stop` gave, if any.

    `skip(config_name)` is asked before each job starts; a job it refuses is not run
    and leaves no record (the scout uses it to drop a config that timed out).
    """
    _keep_awake()
    started = time.time()
    completed = 0
    spend = 0.0

    def announce(record: dict) -> None:
        nonlocal completed, spend
        completed += 1
        spend += float(record.get("cost_usd", 0.0))
        status = _record_status(record)
        exec_seconds = record.get("exec_seconds")
        timing = f" exec {exec_seconds:.3f}s" if exec_seconds is not None else ""
        print(
            f"[{completed}/{len(jobs)}] {record['config']}/{record['task']}"
            f"#{record['repeat']}  {status}  "
            f"${record.get('cost_usd', 0.0):.4f}  "
            f"{record.get('wall_s', 0.0):.0f}s{timing}"
            + (f"  {record['agent_error']}" if record.get("agent_error") else ""),
            flush=True,
        )

    gate = ModelGate(args.max_per_model)

    stop_after = "plan" if getattr(args, "plan_only", False) else "review"

    def run(config, task, repeat):
        if skip is not None and skip(config.name):
            return None

        def go():
            if args.timeout:
                return run_one_with_timeout(
                    config,
                    task,
                    repeat,
                    args.trace_dir,
                    args.keep_workspaces,
                    args.timeout,
                    stop_after,
                )
            return run_one(config, task, repeat, args.trace_dir, args.keep_workspaces, stop_after)

        return gate.run(config.models, go)

    stopped = _run_jobs(jobs, run, args.parallel, log, announce, stop=stop)
    print(f"\ndone in {time.time() - started:.0f}s, spent ${spend:.4f}\n")
    return stopped


def cmd_scout(args) -> int:
    """A cheap first pass: 2 repeats per arm per task, a 300s limit, read for failures
    and for which tasks separate the arms. See bench/scout.py."""
    chosen_configs = _expand(args.configs, configs.names(), "config")
    task_pool = tasks.all_names() if args.include_deprecated else tasks.names()
    chosen_tasks = _expand(args.tasks, task_pool, "task")
    jobs = plan_jobs(chosen_configs, chosen_tasks, args.repeats)
    out_path = Path(args.out or f"results/scout-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    log = ResultLog(out_path)
    print(
        f"scout: {len(jobs)} runs ({len(chosen_configs)} config(s) x {len(chosen_tasks)} "
        f"task(s) x {args.repeats}), {args.timeout:.0f}s limit per run\nresults: {out_path}"
    )
    if not _confirmed(args.yes):
        return 1
    if not _search_ready(args.allow_dead_search):
        return 2
    guard = scout_module.ScoutGuard()
    stopped = _execute(jobs, args, log, stop=guard, skip=guard.skip)
    if args.plan_only:
        # Planning-only runs are not scored, so the accuracy readings would call every
        # greenfield task a floor; what they are for is the plans' reproduce check.
        print(scout_module.render_failures(log.records, guard.dropped))
        print("\n== reproduce check (planning only; nothing was executed or scored) ==")
        print(report.render_repro(log.records))
        print(f"results written to {out_path}")
        return 3 if stopped else 0
    print(scout_module.render(log.records, guard.dropped))
    keep = scout_module.surviving_tasks(log.records)
    print(f"\nfor the long run: --tasks {','.join(keep) if keep else '(none survived)'}")
    print(f"results written to {out_path}")
    if stopped:
        print(f"\nSCOUT STOPPED: {stopped}", file=sys.stderr)
        return 3
    return 0


def cmd_external(args) -> int:
    """Run Aider on the same tasks, for a row that sits next to this
    harness's own in one table. See bench/external.py for why the record
    shape is identical rather than a second scoring path."""
    chosen_tasks = _expand(args.tasks, tasks.names(), "task")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    jobs = [(m, t, r) for m in models for t in chosen_tasks for r in range(args.repeats)]
    out_path = Path(args.out or f"results/aider-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    log = ResultLog(out_path)

    print(
        f"{len(jobs)} aider runs: {len(models)} model(s) x {len(chosen_tasks)} task(s) "
        f"x {args.repeats} repeat(s)"
    )
    print(f"results: {out_path}")
    if not _confirmed(args.yes):
        return 1

    started = time.time()
    completed = 0

    def announce(record: dict) -> None:
        nonlocal completed
        completed += 1
        status = _record_status(record)
        print(
            f"[{completed}/{len(jobs)}] {record['config']}/{record['task']}"
            f"#{record['repeat']}  {status}  {record.get('wall_s', 0.0):.0f}s"
            + (f"  {record['agent_error']}" if record.get("agent_error") else ""),
            flush=True,
        )

    def run(model, task_name, repeat):
        return run_aider(model, tasks.get(task_name), repeat, args.timeout, args.keep_workspaces)

    _run_jobs(jobs, run, args.parallel, log, announce)

    print(f"\ndone in {time.time() - started:.0f}s\n")
    _print_sweep_tables(log.records, len(chosen_tasks))
    print(f"\nresults written to {out_path}")
    return 0


def plan_jobs(
    config_names: list[str], task_names: list[str], repeats: int, repeat_offset: int = 0
) -> list[tuple]:
    """Every (config, task, repeat) job, ordered so the configs being compared
    run side by side.

    Repeat-major, then task, then config innermost. The natural nesting, config
    outermost, runs one arm to completion before the next begins, so the arms
    share no stretch of time at all, and a provider whose latency drifts 2.3x
    over an hour hands the whole difference to whichever arm ran second. Every
    sweep this project called "interleaved" before this function existed was in
    fact run back-to-back: 0% time overlap between arms, measured from each
    record's `started_at`.

    `repeat_offset` numbers a later batch of the same sweep from where the last
    one stopped. A long sweep is best run as several batches of one or two
    repeats each, appended to one `--out` (the log is append-only), so a provider
    stall lands across every arm instead of on one block of tasks; without an
    offset each batch would number its repeats from 0 again.
    """
    return [
        (configs.get(c), tasks.get(t), r)
        for r in range(repeat_offset, repeat_offset + repeats)
        for t in task_names
        for c in config_names
    ]


def _backfill_diagnostics(records: list[dict]) -> None:
    """Records written before bench/diagnostics.py existed carry a trace path
    but no counters; reading the trace now costs one file scan and makes a view
    work on every sweep already on disk rather than only future ones."""
    wanted = diagnostics.Diagnostics().as_dict()
    for record in records:
        # Any missing counter, not just the first one ever added: a record
        # written before the plan-quality counters existed has the older diag_*
        # columns, so keying on one of those left the newer columns silently
        # zero in the view.
        if record.get("trace_path") and any(f"diag_{k}" not in record for k in wanted):
            record.update(
                {
                    f"diag_{k}": v
                    for k, v in diagnostics.from_trace(record["trace_path"]).as_dict().items()
                }
            )


def _view_diagnostics(records: list[dict], args) -> str | None:
    _backfill_diagnostics(records)
    return report.render_diagnostics(records, by=args.by)


def _view_discrimination(records: list[dict], args) -> str | None:
    return report.render_discrimination(records)


def _view_reliability(records: list[dict], args) -> str | None:
    _backfill_diagnostics(records)
    return report.render_reliability(records, by=args.by)


def _view_failures(records: list[dict], args) -> str | None:
    _backfill_diagnostics(records)
    return report.render_failures(records, by=args.by)


def _view_models(records: list[dict], args) -> str | None:
    for record in records:
        if "served_models" not in record and record.get("trace_path"):
            record["served_models"] = diagnostics.served_models_from_trace(record["trace_path"])
    return report.render_models(records)


def _view_compare(records: list[dict], args) -> str | None:
    if not (args.a and args.b):
        print("--view compare needs --a and --b (config names)", file=sys.stderr)
        return None
    _backfill_diagnostics(records)
    return report.render_compare(records, args.a, args.b)


def _view_history(records: list[dict], args) -> str | None:
    since = 0.0
    if args.since:
        try:
            since = time.mktime(time.strptime(args.since, "%Y-%m-%d"))
        except ValueError:
            print("--since must look like 2026-09-24", file=sys.stderr)
            return None
    return report.render_history(records, args.a or "mercury-only", since)


def _view_summary(records: list[dict], args) -> str | None:
    text = report.render(records, by=args.by)
    if args.by == "config" and len({r.get("task") for r in records}) > 1:
        text += "\n" + report.render_by_task(records)
    return text


# Views that must see provider failures: `failures` labels them, `models` lists
# what actually served each run, stalled ones included.
_SHOWS_STALLS = frozenset({"failures", "models", "scout"})

# A view returns the text to print, or None after it has already explained a
# usage error itself.
_REPORT_VIEWS = {
    "diagnostics": _view_diagnostics,
    "discrimination": _view_discrimination,
    "reliability": _view_reliability,
    "failures": _view_failures,
    "models": _view_models,
    "compare": _view_compare,
    "history": _view_history,
    "scout": lambda records, args: scout_module.render(records),
    "repro": lambda records, args: report.render_repro(records),
}


def _expand_path(pattern: str) -> list[Path]:
    """The files a report argument names: a glob relative to the working
    directory, or an absolute one. `Path().glob` refuses an absolute pattern on
    Python 3.14, so an absolute path used to crash the report outright."""
    path = Path(pattern).expanduser()
    if path.is_absolute():
        anchor = Path(path.anchor)
        relative = str(path.relative_to(anchor))
        return sorted(anchor.glob(relative)) or [path]
    return sorted(Path().glob(pattern)) or [path]


def cmd_archive(args) -> int:
    try:
        result = archive_module.archive(
            [Path(p).expanduser() for p in args.paths], args.name, force=args.force
        )
    except archive_module.ArchiveError as exc:
        print(f"archive: {exc}", file=sys.stderr)
        return 1
    print(
        f"archived {result.records} record(s) to {result.results_path}; "
        f"{result.traces_copied} trace(s) copied to {result.traces_path}"
        + (f", {result.traces_missing} missing" if result.traces_missing else "")
    )
    return 0


def _backfill_diagnostics(records: list[dict]) -> None:
    """Diagnostics a row was written without (an older bench, or a frozen worktree's), read
    from its own trace now, so a metric added later can still be compared (DESIGN-LOG §105).
    Only missing keys are filled; a row's recorded values are never replaced."""
    names = [f"diag_{f.name}" for f in dataclasses.fields(diagnostics.Diagnostics)]
    for record in records:
        if all(name in record for name in names) or not record.get("trace_path"):
            continue
        try:
            fresh = diagnostics.from_trace(record["trace_path"]).as_dict()
        except OSError:
            continue
        for key, value in fresh.items():
            record.setdefault(f"diag_{key}", value)


def cmd_report(args) -> int:
    records: list[dict] = []
    for pattern in args.paths:
        matches = _expand_path(pattern)
        for path in matches:
            if path.is_file():
                records.extend(ResultLog.load(path))
    if not records:
        print("no results found", file=sys.stderr)
        return 1
    _backfill_diagnostics(records)
    # Provider failures are left out of every view except the two whose job is to
    # show them, and the exclusion is said first, so a table can never silently
    # drop runs. `--include-stalls` restores the old behaviour.
    prefix = ""
    if args.view not in _SHOWS_STALLS and not args.include_stalls:
        completed, stalled = stalls.split(records)
        prefix = stalls.note(records, stalled, by=args.by)
        records = completed
        if not records:
            print(prefix + "every run died of a provider failure; nothing to report.")
            return 1
    text = _REPORT_VIEWS.get(args.view, _view_summary)(records, args)
    if text is None:
        return 1
    print(prefix + text if prefix else text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench", description="Compare agent configurations on accuracy, speed, and cost."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show available configurations and tasks.").set_defaults(
        func=cmd_list
    )

    run = sub.add_parser("run", help="Run a sweep.")
    run.add_argument(
        "--configs", default="hybrid,mercury-only", help="Comma-separated config names, or 'all'."
    )
    run.add_argument("--tasks", default="all", help="Comma-separated task names, or 'all'.")
    run.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per (config, task). More than one gives you a spread.",
    )
    run.add_argument(
        "--repeat-offset",
        type=int,
        default=0,
        help="Number this batch's repeats from N instead of 0, so several batches "
        "appended to one --out (run under an hour each) keep distinct repeat indices.",
    )
    run.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Concurrent runs. The work is network-bound, so 4-8 is reasonable.",
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Hard wall-clock limit per run, in seconds. Exceeding it kills the "
        "run's process and scores it as a timeout rather than waiting for a "
        "slow model to finish. 0 disables the limit.",
    )
    run.add_argument(
        "--max-per-model",
        type=int,
        default=3,
        help="Concurrent jobs allowed against the same underlying model, "
        "independent of --parallel. Configs that share a model (e.g. a "
        "hybrid's planner and a same-model-only baseline) queue for a slot "
        "rather than all firing at once and starving each other at the "
        "provider. 0 disables the cap.",
    )
    run.add_argument("--out", help="Where to write the JSONL results.")
    run.add_argument(
        "--plan-only",
        action="store_true",
        help="Stop each run once its plan is accepted (no execution, no scoring): for "
        "questions about the plans, such as what the reproduce check did, at a "
        "fraction of a full run's cost.",
    )
    run.add_argument("--trace-dir", default="traces/bench")
    run.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Do not delete the generated code, so you can read it.",
    )
    run.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include retired tasks (roman, primes, lru) that sit at a 100%% "
        "accuracy ceiling for every config tested and no longer discriminate.",
    )
    run.add_argument("-y", "--yes", action="store_true", help="Skip the cost confirmation.")
    run.add_argument(
        "--allow-dead-search",
        action="store_true",
        help="Run even when the web-search canary returns nothing (see _search_ready).",
    )
    run.set_defaults(func=cmd_run)

    sc = sub.add_parser(
        "scout",
        help="A cheap first pass (2 repeats, 300s limit): what fails, and which tasks "
        "separate the configs, before any long run.",
    )
    sc.add_argument("--configs", required=True, help="Comma-separated config names.")
    sc.add_argument("--tasks", default="all", help="Comma-separated task names, or 'all'.")
    sc.add_argument("--repeats", type=int, default=scout_module.SCOUT_REPEATS)
    sc.add_argument("--timeout", type=float, default=scout_module.SCOUT_TIMEOUT_S)
    sc.add_argument("--parallel", type=int, default=6)
    sc.add_argument("--max-per-model", type=int, default=6)
    sc.add_argument("--out", default="")
    sc.add_argument(
        "--plan-only",
        action="store_true",
        help="Stop each run once its plan is accepted (no execution, no scoring): for "
        "questions about the plans, such as what the reproduce check did, at a "
        "fraction of a full run's cost.",
    )
    sc.add_argument("--trace-dir", default="traces/bench")
    sc.add_argument("--keep-workspaces", action="store_true")
    sc.add_argument("--include-deprecated", action="store_true")
    sc.add_argument("-y", "--yes", action="store_true")
    sc.add_argument(
        "--allow-dead-search",
        action="store_true",
        help="Run even when the web-search canary returns nothing (see _search_ready).",
    )
    sc.set_defaults(func=cmd_scout)

    ext = sub.add_parser("external", help="Run Aider on the same tasks, for comparison.")
    ext.add_argument(
        "--models",
        default="z-ai/glm-5.3-flash,qwen/qwen3.7-flash",
        help="Comma-separated OpenRouter model ids (no 'openrouter/' prefix).",
    )
    ext.add_argument("--tasks", default="all", help="Comma-separated task names, or 'all'.")
    ext.add_argument("--repeats", type=int, default=1)
    ext.add_argument("--parallel", type=int, default=1)
    ext.add_argument("--timeout", type=float, default=300.0)
    ext.add_argument("--out", help="Where to write the JSONL results.")
    ext.add_argument("--keep-workspaces", action="store_true")
    ext.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation.")
    ext.set_defaults(func=cmd_external)

    arc = sub.add_parser(
        "archive",
        help="Copy a sweep's records and traces under results/ and traces/, so its numbers "
        "can be re-derived after a scratch directory is cleared.",
    )
    arc.add_argument("paths", nargs="+", help="Result JSONL file(s) to archive together.")
    arc.add_argument(
        "--name", required=True, help="Archive name: results/NAME.jsonl, traces/NAME/."
    )
    arc.add_argument("--force", action="store_true", help="Overwrite an existing archive.")
    arc.set_defaults(func=cmd_archive)

    rep = sub.add_parser("report", help="Re-render results from JSONL files.")
    rep.add_argument("paths", nargs="+")
    rep.add_argument("--by", default="config", choices=["config", "task", "executor"])
    rep.add_argument(
        "--view",
        default="outcome",
        choices=[
            "outcome",
            "diagnostics",
            "discrimination",
            "reliability",
            "failures",
            "compare",
            "models",
            "history",
            "scout",
            "repro",
        ],
        help="outcome: accuracy, cost and time (the default). diagnostics: "
        "what the run did — wasted edits, exhausted loops, nudges, how "
        "much it read before its first edit. discrimination: which tasks "
        "actually separated the configs and which spent their budget "
        "without comparing anything. reliability: does it work every "
        "time (pass@k / pass^k), not just on average. failures: how "
        "runs ended, by class. compare: --a CONFIG --b CONFIG, with a "
        "bootstrap interval and a verdict per metric. models: which "
        "model ids and providers actually served each run, so a "
        "swap under an unchanged requested id is visible. history: --a CONFIG "
        "(default mercury-only), what it normally scores on each task and which tasks are "
        "at a ceiling; --since DATE ignores earlier runs. repro: how often the "
        "planner's reproduce check ran and what it showed, per config and task.",
    )
    rep.add_argument(
        "--include-stalls",
        action="store_true",
        help="Count provider failures (stream stalls, connection resets, 5xx, zero-cost "
        "timeouts) as ordinary runs. By default they are left out and named, because a "
        "table that mixes them in mostly records who got stalled.",
    )
    rep.add_argument(
        "--since",
        default="",
        help="history: ignore runs before this date (YYYY-MM-DD), for a task whose fixture "
        "changed.",
    )
    rep.add_argument("--a", default="", help="compare: the baseline config; history: the config")
    rep.add_argument("--b", default="", help="compare: the config being tested")
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
