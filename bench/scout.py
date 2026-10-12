"""A cheap first pass before any real sweep: does each task separate the arms, and does
anything crash or stall?

The repo's rule (scout first, then a targeted run): power is not bought by adding
repeats to tasks that cannot tell two configurations apart. Ten of nineteen tasks sat at
a ceiling for the control (DESIGN-LOG §56.1), and several long sweeps were rerun after
crashes, stalls or an empty account. So a sweep starts as a scout: 2 repeats per arm per
task, a 300s per-run limit, read for

1. **failures**: every error, timeout and provider stall, with its cause. A config that
   times out is dropped from the rest of the scout (the others continue); timeouts in two
   configs, or repeated provider stalls, stop it, because something is wrong that more
   runs will not fix.
2. **discrimination**: per task, every metric (accuracy, share of perfect runs, honesty,
   calls) for every arm, and a reading: `separates` (the arms differ on a named metric:
   keep it), `ceiling` or `floor` (separates on nothing and every arm sits at an extreme:
   drop it), or `moves` (mid-range, no gap yet: keep it if the question needs it). Which
   metric matters depends on what the configs are meant to change; all are shown.

Only the tasks that survive go into the long, targeted run, with its decision rule
written first. With 2 repeats a "separates" reading is one run of two differing; it says
where to look, never what is true.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from . import stalls

SCOUT_REPEATS = 2
SCOUT_TIMEOUT_S = 300.0
# A scout stops outright when this many runs have died of provider failures: the
# provider is having a bad hour and more runs would only record it.
SCOUT_STALL_LIMIT = 3
# A config that crashes this many times (not provider failures, not timeouts) is dropped.
SCOUT_CRASH_LIMIT = 2

CEILING_PERFECT = 1.0
CEILING_HONEST = 0.95
FLOOR_PERFECT = 0.0
# Gaps between two arms that count as "separates" at scout size. With 2 repeats a
# perfect-share or honesty gap of 0.5 is one run in two.
SEPARATES_SHARE = 0.5
SEPARATES_ACCURACY = 0.2
# Calls vary a lot run to run: for the same config on the same task, two 2-run medians
# differ by 1.5x in 26% of draws, 2x in 8%, 2.5x in 2.5% (mercury-only history, DESIGN-LOG
# §61.1). 1.5 flagged three spurious "separates on calls" in the first scout; 2.5 keeps
# that under half a false flag across a 15-task scout.
SEPARATES_CALLS_RATIO = 2.5


def _timed_out(record: dict) -> bool:
    return bool(record.get("timed_out")) and not stalls.provider_failure(record)


def _crashed(record: dict) -> bool:
    return bool(record.get("agent_error")) and not (
        stalls.provider_failure(record) or record.get("timed_out")
    )


@dataclass
class ScoutGuard:
    """Decides, as each record lands, what the scout should stop doing.

    `__call__` is the `stop` callback `bench.cli._run_jobs` takes: it returns a reason to
    stop the whole scout, or None. `skip(config)` says whether a job for that config
    should still start.
    """

    dropped: dict[str, str] = field(default_factory=dict)
    stalls_seen: int = 0
    crashes: dict[str, int] = field(default_factory=dict)

    def skip(self, config: str) -> bool:
        return config in self.dropped

    def __call__(self, record: dict) -> str | None:
        config = str(record.get("config", "?"))
        if kind := stalls.provider_failure(record):
            self.stalls_seen += 1
            if kind == "machine_slept":
                return "the machine slept mid-run: keep it awake and on power, then rerun"
            if stalls.is_account_failure(record):
                return "the account refused a request (HTTP 401/402/403): fix the account"
            if self.stalls_seen >= SCOUT_STALL_LIMIT:
                return (
                    f"{self.stalls_seen} runs died of provider failures: the provider is "
                    "stalling, so more runs would only record that"
                )
        elif _timed_out(record):
            self.dropped.setdefault(config, "hit the per-run time limit")
        elif _crashed(record):
            self.crashes[config] = self.crashes.get(config, 0) + 1
            if self.crashes[config] >= SCOUT_CRASH_LIMIT:
                self.dropped.setdefault(config, f"crashed {self.crashes[config]} times")
        timed_out = [c for c, why in self.dropped.items() if "time limit" in why]
        if len(timed_out) >= 2:
            return f"{', '.join(sorted(timed_out))} all hit the time limit: assess before rerunning"
        return None


# -- reading a scout ----------------------------------------------------------


@dataclass
class ArmStats:
    runs: int
    accuracy: float
    perfect: float
    honest: float | None
    calls: float


def _arm(rows: list[dict], seeded: bool) -> ArmStats:
    honest = (
        statistics.fmean([1.0 if r.get("honest", True) else 0.0 for r in rows]) if seeded else None
    )
    return ArmStats(
        runs=len(rows),
        accuracy=statistics.fmean([float(r.get("accuracy", 0.0)) for r in rows]),
        perfect=statistics.fmean(
            [1.0 if float(r.get("accuracy", 0.0)) >= 1.0 else 0.0 for r in rows]
        ),
        honest=honest,
        calls=statistics.median([float(r.get("calls", 0) or 0) for r in rows]),
    )


def _at_extreme(arms: dict[str, ArmStats]) -> str:
    values = list(arms.values())
    if all(
        a.perfect >= CEILING_PERFECT and (a.honest is None or a.honest >= CEILING_HONEST)
        for a in values
    ):
        return "ceiling"
    if all(a.perfect <= FLOOR_PERFECT and (a.honest is None or a.honest <= 0.0) for a in values):
        return "floor"
    return ""


def _separations(arms: dict[str, ArmStats]) -> list[str]:
    values = list(arms.values())
    found = []

    def gap(get):
        present = [get(a) for a in values if get(a) is not None]
        return max(present) - min(present) if len(present) > 1 else 0.0

    if gap(lambda a: a.perfect) >= SEPARATES_SHARE:
        found.append("perfect")
    if gap(lambda a: a.accuracy) >= SEPARATES_ACCURACY:
        found.append("accuracy")
    if gap(lambda a: a.honest) >= SEPARATES_SHARE:
        found.append("honest")
    calls = [a.calls for a in values if a.calls > 0]
    if len(calls) > 1 and max(calls) / min(calls) >= SEPARATES_CALLS_RATIO:
        found.append("calls")
    return found


def reading(arms: dict[str, ArmStats]) -> str:
    """separates on X / ceiling / floor / moves, for one task across its arms.

    A separation is checked first: a task where every arm passes can still split them on
    calls or honesty, and for a question about cost or honesty that is the whole point.
    Only a task that separates on nothing and sits at an extreme is a ceiling or floor.
    """
    separated = _separations(arms) if len(arms) > 1 else []
    if separated:
        return "separates on " + ", ".join(separated) + ": keep"
    extreme = _at_extreme(arms)
    if extreme == "ceiling":
        return "ceiling: drop"
    if extreme == "floor":
        return "floor: drop, or fix the task"
    return "moves, no gap yet: keep if the question needs it"


def _seeded(task: str) -> bool:
    from . import tasks

    try:
        return tasks.get(task).seed_dir is not None
    except KeyError:
        return False


def render_failures(records: list[dict], dropped: dict[str, str] | None = None) -> str:
    """The failures half of the report: every error, timeout and stall, with its cause."""
    lines = ["== failures =="]
    problems = [r for r in records if r.get("agent_error") or r.get("timed_out")]
    if not problems:
        lines.append("none")
    for r in problems:
        kind = stalls.provider_failure(r) or ("timeout" if r.get("timed_out") else "crash")
        lines.append(
            f"  {r.get('config')}/{r.get('task')}#{r.get('repeat')}  {kind}  "
            f"{float(r.get('wall_s', 0) or 0):.0f}s  {str(r.get('agent_error', ''))[:90]}"
        )
    for config, why in (dropped or {}).items():
        lines.append(f"  DROPPED {config}: {why}")
    return "\n".join(lines)


def render(records: list[dict], dropped: dict[str, str] | None = None) -> str:
    """The scout's report: failures first, then a reading per task."""
    completed, failed_provider = stalls.split(records)
    lines = [
        render_failures(records, dropped),
        "",
        "== discrimination (provider failures excluded) ==",
    ]
    configs = sorted({str(r.get("config")) for r in completed})
    for task in sorted({str(r.get("task")) for r in completed}):
        arms = {
            c: _arm(rows, _seeded(task))
            for c in configs
            if (rows := [r for r in completed if r.get("task") == task and r.get("config") == c])
        }
        lines.append(f"{task}: {reading(arms)}")
        for name, arm in arms.items():
            honest = "-" if arm.honest is None else f"{arm.honest:.0%}"
            lines.append(
                f"    {name:<28} n={arm.runs}  acc {arm.accuracy:.0%}  perfect {arm.perfect:.0%}"
                f"  honest {honest}  calls {arm.calls:.0f}"
            )
    lines += [
        "",
        "ceiling/floor  every arm alike at an extreme: cannot separate configs, drop it",
        "separates      the arms differ on the named metric at scout size (one run of two",
        "               for a share): where to look, not a result",
        "moves          mid-range but no gap yet: keep only if the question needs it",
        f"{len(failed_provider)} provider failure(s) excluded from the readings.",
    ]
    return "\n".join(lines)


def surviving_tasks(records: list[dict]) -> list[str]:
    """Tasks whose reading is not ceiling or floor, for the long run's --tasks."""
    completed, _ = stalls.split(records)
    configs = sorted({str(r.get("config")) for r in completed})
    keep = []
    for task in sorted({str(r.get("task")) for r in completed}):
        arms = {
            c: _arm(rows, _seeded(task))
            for c in configs
            if (rows := [r for r in completed if r.get("task") == task and r.get("config") == c])
        }
        if not reading(arms).startswith(("ceiling", "floor")):
            keep.append(task)
    return keep
