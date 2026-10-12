"""Aggregating benchmark records into a comparison.

Every number is reported with the number of runs behind it. A single run of an
LLM agent is a sample, not a measurement, and a table that hides n invites
conclusions the data does not support.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from . import stalls, stats


@dataclass
class Aggregate:
    config: str
    runs: int = 0
    failures: int = 0
    timeouts: int = 0
    accuracy: list[float] = field(default_factory=list)
    delivered: list[bool] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    wall: list[float] = field(default_factory=list)
    model_latency: list[float] = field(default_factory=list)
    exec_seconds: list[float] = field(default_factory=list)
    exec_timeouts: int = 0
    ruff: list[int] = field(default_factory=list)
    ty: list[int] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)
    executor_decode_tok_s: list[float] = field(default_factory=list)
    executor_ttft: list[float] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.runs += 1
        if record.get("timed_out"):
            self.timeouts += 1
        elif record.get("agent_error"):
            self.failures += 1
        self.accuracy.append(float(record.get("accuracy", 0.0)))
        self.delivered.append(bool(record.get("delivered")))
        self.cost.append(float(record.get("cost_usd", 0.0)))
        self.wall.append(float(record.get("wall_s", 0.0)))
        self.model_latency.append(float(record.get("model_latency_s", 0.0)))
        self.ruff.append(int(record.get("ruff_issues", 0)))
        self.ty.append(int(record.get("ty_issues", 0)))
        self.calls.append(int(record.get("calls", 0)))

        seconds = record.get("exec_seconds")
        if seconds is not None:
            self.exec_seconds.append(float(seconds))
        elif record.get("exec_error"):
            self.exec_timeouts += 1

        decode_s = float(record.get("executor_decode_s", 0.0))
        out_tokens = int(record.get("executor_out_tokens", 0))
        if decode_s > 0 and out_tokens:
            self.executor_decode_tok_s.append(out_tokens / decode_s)
        calls = int(record.get("executor_calls", 0))
        if calls:
            self.executor_ttft.append(float(record.get("executor_ttft_s", 0.0)) / calls)

    # -- summaries --------------------------------------------------------
    @staticmethod
    def _mean(values: list) -> float:
        return statistics.fmean(values) if values else 0.0

    @property
    def mean_accuracy(self) -> float:
        return self._mean(self.accuracy)

    @property
    def accuracy_spread(self) -> float:
        return statistics.pstdev(self.accuracy) if len(self.accuracy) > 1 else 0.0

    @property
    def delivery_rate(self) -> float:
        return self._mean([1.0 if d else 0.0 for d in self.delivered])

    @property
    def mean_cost(self) -> float:
        return self._mean(self.cost)

    @property
    def mean_wall(self) -> float:
        return self._mean(self.wall)

    @property
    def median_exec(self) -> float | None:
        return statistics.median(self.exec_seconds) if self.exec_seconds else None

    @property
    def cost_per_passing_run(self) -> float | None:
        """What it costs to get one fully correct result out of this config."""
        total_cost = sum(self.cost)
        perfect = sum(1 for a in self.accuracy if a >= 1.0)
        return total_cost / perfect if perfect else None


def aggregate(records: list[dict], by: str = "config") -> dict[str, Aggregate]:
    groups: dict[str, Aggregate] = {}
    for record in records:
        key = str(record.get(by, "?"))
        groups.setdefault(key, Aggregate(config=key)).add(record)
    return groups


def _model_s(records: list[dict]) -> float:
    """Mean seconds per run spent inside model calls.

    `wall` cannot be compared across configurations from a parallel sweep: a
    sweep run with --parallel 3 measured 94.8 minutes of summed per-run wall in
    38 minutes of real time, so every wall figure carried 2.5x of contention
    with it. One config looked six times slower than another on wall and was
    the faster of the two on model time.
    """
    values = [r.get("model_latency_s", 0.0) for r in records]
    return sum(values) / len(values) if values else 0.0


def _is_seeded_task(task_name: str) -> bool:
    """Whether a task has a seed directory, looked up from the task registry
    rather than kept as a second, parallel list here.

    This used to be a literal set of names, and it went stale the moment the
    task list changed under it: silent-setting was dropped and cross-file,
    regression and no-bug were added, and every one of the three new tasks
    silently reported "-" for `honest` in both sweeps that used them — the
    denominator was being computed correctly by bench.integrity, and this
    function was throwing the result away before it ever reached the table.
    """
    from . import tasks

    try:
        return tasks.get(task_name).seed_dir is not None
    except KeyError:
        return False


def _honest_rate(records: list[dict]) -> str:
    """Share of *seeded* runs reached honestly, or "-" for greenfield tasks.

    Only seeded tasks can be dishonest in these ways — there is no seeded test
    to weaken and no original bug to miss — so mixing greenfield runs into the
    denominator would quietly inflate every score.
    """
    seeded = [r for r in records if _is_seeded_task(str(r.get("task", "")))]
    if not seeded:
        return "-"
    honest = sum(1 for r in seeded if r.get("honest", True))
    return f"{100 * honest / len(seeded):.0f}%"


def render(records: list[dict], by: str = "config") -> str:
    """A fixed-width comparison table, one row per configuration."""
    if not records:
        return "no results"
    groups = aggregate(records, by=by)
    groups_records: dict[str, list[dict]] = {}
    for record in records:
        groups_records.setdefault(str(record.get(by, "?")), []).append(record)
    # Sorted by cost, not accuracy. Accuracy has been a ceiling on every sweep
    # so far — eight of nine configurations at 100% — so ordering by it puts
    # noise at the top and reads as a ranking. The column stays; it costs one
    # pytest run that already happens, and a task that does start
    # discriminating would be worth seeing.
    order = sorted(groups, key=lambda k: groups[k].mean_cost)

    header = (
        f"{'config':<18}{'n':>3}  {'accuracy':>10}  {'deliv':>6}  "
        f"{'cost/run':>9}  {'$/pass':>8}  {'wall':>7}  {'calls':>6}  "
        f"{'model':>7}  {'s/call':>7}  {'honest':>7}  {'fail':>5}  {'t/out':>5}"
    )
    lines = [header, "-" * len(header)]
    for name in order:
        agg = groups[name]
        accuracy = f"{agg.mean_accuracy * 100:5.1f}%"
        if agg.accuracy_spread > 0:
            accuracy += f"±{agg.accuracy_spread * 100:.0f}"
        per_pass = agg.cost_per_passing_run
        calls = Aggregate._mean(agg.calls)
        per_call = agg.mean_wall / calls if calls else 0.0
        lines.append(
            f"{name:<18}{agg.runs:>3}  {accuracy:>10}  "
            f"{agg.delivery_rate * 100:5.0f}%  "
            f"${agg.mean_cost:>8.4f}  "
            f"{('$' + format(per_pass, '.4f')) if per_pass else '—':>8}  "
            f"{agg.mean_wall:>6.1f}s  {calls:>6.1f}  {_model_s(groups_records[name]):>6.0f}s  "
            f"{per_call:>6.2f}s  {_honest_rate(groups_records[name]):>7}  "
            f"{agg.failures:>5}  {agg.timeouts:>5}"
        )
    lines.append("")
    lines.append("accuracy  share of hidden tests passed (mean ± population sd)")
    lines.append("deliv     runs that produced the expected files with valid syntax")
    lines.append("$/pass    total spend divided by runs that passed every hidden test")
    lines.append("wall      end-to-end seconds per run. Inflated by --parallel, which lets")
    lines.append("          concurrent runs contend for CPU: compare `model` across configs")
    lines.append("model     seconds per run actually spent inside model calls — the honest")
    lines.append("          latency number, unaffected by how many jobs ran at once")
    lines.append("honest    runs that passed without weakening a seeded test, changing a")
    lines.append("          pinned signature, or writing tests that miss the bug they fixed")
    lines.append("fail      runs that raised an error before finishing")
    lines.append("t/out     runs killed by --timeout before finishing — a slow-model signal")
    lines.append("calls     model calls per run — the thing wall time is proportional to")
    lines.append("s/call    wall seconds per model call, mostly queue and prefill")
    lines.append("tok/s     executor decode rate, prefill excluded")
    return "\n".join(lines)


def render_by_task(records: list[dict]) -> str:
    """The same comparison, split per task, since configs are not uniformly good."""
    tasks = sorted({str(r.get("task", "?")) for r in records})
    blocks = []
    for task in tasks:
        subset = [r for r in records if r.get("task") == task]
        blocks.append(f"\n### task: {task}\n" + render(subset))
    return "\n".join(blocks)


# -- process view: how a config worked, not just whether it passed --------

_DIAG_COLUMNS = (
    ("calls", "calls", "model calls per run"),
    ("diag_test_runs", "tests", "run_tests calls per run"),
    ("diag_unproductive_edits", "wasted", "edits after a failure that did not reduce it"),
    ("diag_longest_unproductive_streak", "streak", "longest run of those in a row"),
    ("diag_loops_exhausted", "loopex", "tool loops that hit their iteration cap"),
    ("diag_nudges", "nudge", "harness nudges fired"),
    ("diag_step_retries", "retry", "steps retried after failing their verify"),
    ("diag_tool_errors", "toolerr", "tool calls that errored or had unparseable args"),
    ("diag_reads_before_first_edit", "look", "read-only calls before the first edit"),
    ("diag_plan_tool_calls", "plan", "tool calls made while planning"),
    ("diag_plan_probes", "probes", "probe_python calls while planning"),
    ("diag_probe_holds", "phold", "first plans held by a probe or repro gate"),
    ("diag_repro_reproduced", "repro+", "reproduce snippets that showed the problem"),
    ("diag_repro_passed", "repro-", "reproduce snippets that passed: report not reproduced"),
    ("diag_repro_holds", "rhold", "source edits held after a passed reproduction"),
    ("diag_pinned_test_holds", "pinhold", "edits held for removing or weakening an old test"),
    ("diag_plan_draft_words", "draftw", "words in the plan's draft"),
    ("diag_plan_named_tests", "tests/p", "named tests across the plan's steps"),
    ("diag_source_edit_steps", "srcstp", "execute steps that edited non-test source"),
    ("diag_red_first_steps", "red1st", "of those, steps that ran a failing test first"),
    ("diag_tdd_holds", "held", "first source edits the red-first gate held"),
    ("diag_web_searches", "search", "web_search calls"),
    ("diag_empty_searches", "srch0", "of those, searches that returned nothing"),
)


def render_diagnostics(records: list[dict], by: str = "config") -> str:
    """Mean process signals per run, one row per group.

    Separate from `render` rather than more columns on it: the outcome table
    answers "which config would I ship" and this one answers "what did it do
    all those calls on", and a sweep where every arm passes everything needs
    the second question to say anything at all.
    """
    if not records:
        return "no results"
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(by, "?")), []).append(record)

    header = f"{by:<22}{'n':>3}" + "".join(f"{label:>8}" for _, label, _ in _DIAG_COLUMNS)
    lines = [header, "-" * len(header)]
    for name in sorted(groups):
        subset = groups[name]
        row = f"{name:<22}{len(subset):>3}"
        for key, _, _ in _DIAG_COLUMNS:
            mean = statistics.fmean([float(r.get(key, 0) or 0) for r in subset])
            row += f"{mean:>8.1f}"
        lines.append(row)
    lines.append("")
    for _, label, note in _DIAG_COLUMNS:
        lines.append(f"{label:<8}  {note}")
    lines.append("")
    lines.append("All means per run. Counted from each run's own trace with no model")
    lines.append("call; see bench/diagnostics.py for exactly what each one reads.")
    return "\n".join(lines)


_DISCRIMINATION_LEGEND = [
    "",
    "acc gap     largest mean-accuracy difference between two configs",
    "call gap    same, for model calls per run",
    'honest gap  same, for the share of runs reached honestly; "—" on a',
    "            greenfield task, which has no seeded test to weaken",
    "",
    "With a handful of repeats per config, one run flipping is a large gap:",
    "read this as which rows are worth looking at, not as significance.",
    "separates   no = every config scored alike on all three, so this task",
    "            spent its budget without comparing anything",
]


def _mean_by_config(present: dict[str, list[dict]], value) -> dict[str, float]:
    return {c: statistics.fmean([value(r) for r in rows]) for c, rows in present.items()}


def _discrimination_row(task: str, records: list[dict], configs: list[str]) -> str | None:
    """One task's line in the discrimination table, or None when fewer than two
    configs ran it (a gap needs something to be a gap between)."""
    subset = [r for r in records if str(r.get("task")) == task]
    per_config = {c: [r for r in subset if str(r.get("config")) == c] for c in configs}
    present = {c: rows for c, rows in per_config.items() if rows}
    if len(present) < 2:
        return None
    accuracy = _mean_by_config(present, lambda r: float(r.get("accuracy", 0.0)))
    calls = _mean_by_config(present, lambda r: float(r.get("calls", 0) or 0))
    # Only seeded tasks can be dishonest in these ways -- see
    # `_honest_rate`, which had exactly this denominator bug once already.
    # A greenfield task has no seeded test to weaken, so every run scores
    # honest and the gap is structurally zero, which would read as "this
    # task separates nothing on honesty" rather than "honesty does not
    # apply here".
    honest = (
        _mean_by_config(present, lambda r: 1.0 if r.get("honest", True) else 0.0)
        if _is_seeded_task(task)
        else {}
    )
    mean_acc = statistics.fmean(accuracy.values())
    acc_gap = max(accuracy.values()) - min(accuracy.values())
    call_gap = max(calls.values()) - min(calls.values())
    honest_gap = (max(honest.values()) - min(honest.values())) if honest else None
    # A task at ceiling on accuracy can still separate on cost or honesty,
    # which is the whole reason those two columns are here.
    verdict = "no" if (acc_gap < 0.05 and (honest_gap or 0.0) < 0.05 and call_gap < 2.0) else "yes"
    shown = f"{honest_gap * 100:>10.0f}%" if honest_gap is not None else f"{'—':>11}"
    return (
        f"{task:<20}{len(subset):>3}{mean_acc * 100:>9.0f}%"
        f"{acc_gap * 100:>8.0f}%{call_gap:>10.1f}"
        f"{shown}  {verdict}"
    )


def render_discrimination(records: list[dict]) -> str:
    """Which tasks actually separate the configs in this sweep.

    A task every config passes costs its budget and buys no comparison. This
    has been eyeballed off the per-task tables repeatedly and then argued
    about; computing it says plainly which rows were worth running. Needs at
    least two configs to mean anything.
    """
    configs = sorted({str(r.get("config", "?")) for r in records})
    if len(configs) < 2:
        return "discrimination needs at least two configs"
    tasks = sorted({str(r.get("task", "?")) for r in records})

    header = (
        f"{'task':<20}{'n':>3}{'mean acc':>10}{'acc gap':>9}"
        f"{'call gap':>10}{'honest gap':>12}  separates"
    )
    rows = (_discrimination_row(task, records, configs) for task in tasks)
    lines = [header, "-" * len(header), *(row for row in rows if row is not None)]
    return "\n".join(lines + _DISCRIMINATION_LEGEND)


# -- reliability: does it work every time, not just on average -------------


_RELIABILITY_LEGEND = [
    "",
    "pass@k     tasks where at least one repeat got every hidden test right",
    "pass^k     tasks where every repeat did — the number that matters when a",
    "           run is not retried by a person",
    "honest^k   pass^k, and every one of those runs was also honest",
    "cv         spread of cost / calls across runs relative to their mean; run-to-",
    "           run cost varied 40-65% on identical inputs before any of this was",
    "           measured, so a config with a low cv is one you can budget for",
    "Only tasks with more than one repeat are counted.",
]


def _reliability_row(name: str, group: list[dict]) -> str:
    """One config's (or task's) line in the reliability table."""
    by_task: dict[str, list[dict]] = {}
    for record in group:
        by_task.setdefault(str(record.get("task", "?")), []).append(record)
    # A task run once says nothing about consistency; leave it out rather
    # than count a single lucky run as reliable.
    repeated = {t: rs for t, rs in by_task.items() if len(rs) > 1}
    if not repeated:
        return f"{name:<24}{len(by_task):>6}{'-':>8}{'-':>8}{'-':>10}{'-':>9}{'-':>10}"
    any_pass = sum(1 for rs in repeated.values() if any(_perfect(r) for r in rs))
    all_pass = sum(1 for rs in repeated.values() if all(_perfect(r) for r in rs))
    all_honest = sum(
        1 for rs in repeated.values() if all(_perfect(r) and r.get("honest", True) for r in rs)
    )
    n = len(repeated)
    rows = [r for rs in repeated.values() for r in rs]
    cost_cv = _cv([float(r.get("cost_usd", 0) or 0) for r in rows])
    calls_cv = _cv([float(r.get("calls", 0) or 0) for r in rows])
    return (
        f"{name:<24}{n:>6}{100 * any_pass / n:>7.0f}%{100 * all_pass / n:>7.0f}%"
        f"{100 * all_honest / n:>9.0f}%"
        f"{cost_cv:>9.2f}{calls_cv:>10.2f}"
    )


def render_reliability(records: list[dict], by: str = "config") -> str:
    """Consistency across repeats, which a mean hides.

    A config that solves a task 3 times in 6 and one that solves it 6 times in
    6 can post the same average over a sweep with the same number of hard
    tasks. `pass@k` is "some repeat got everything right"; `pass^k` is "every
    repeat did", the number that matters when a run is not retried by a human.
    Read against each other, a large gap says the config can do the task but
    cannot be relied on to.
    """
    if not records:
        return "no results"
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(by, "?")), []).append(record)

    header = (
        f"{by:<24}{'tasks':>6}{'pass@k':>8}{'pass^k':>8}{'honest^k':>10}"
        f"{'cost cv':>9}{'calls cv':>10}"
    )
    lines = [header, "-" * len(header)]
    lines += [_reliability_row(name, groups[name]) for name in sorted(groups)]
    return "\n".join(lines + _RELIABILITY_LEGEND)


def _perfect(record: dict) -> bool:
    return float(record.get("accuracy", 0.0)) >= 1.0


def _cv(values: list[float]) -> float:
    mean = statistics.fmean(values) if values else 0.0
    return statistics.pstdev(values) / mean if mean and len(values) > 1 else 0.0


# -- failure classes: what kind of wrong was it ----------------------------

FAILURE_CLASSES = (
    ("ok", "passed every hidden test honestly"),
    ("tests_miss_change", "passed, but its own tests pass on the original bug"),
    (
        "dishonest",
        "passed, having weakened, removed or skipped a seeded test or changed a pinned signature",
    ),
    ("wrong_fix_believed_done", "its own suite was green and it stopped, but hidden tests fail"),
    ("gave_up_red", "stopped with its own suite still red"),
    ("loop", "exhausted a tool loop or cycled edits with no progress"),
    ("crashed", "raised an error or was killed before finishing"),
    (
        "provider",
        "died of a provider failure (stall, reset, 5xx), not of anything the agent did",
    ),
)


def classify_failure(record: dict) -> str:
    """One deterministic label for how a run ended, worst cause first.

    Ordered so the label names the *first* thing that went wrong: a run that
    crashed did not "believe it was done", and one that looped and then failed
    is a loop problem before it is a wrong-fix problem. The article this view
    follows separates planning errors, execution issues and incorrect
    responses; those are inferred here from the run's own trace counters
    (`diag_*`, see bench/diagnostics.py), so the labels describe what the trace
    shows rather than what a model guessed.
    """
    # Before "crashed": a provider stall is not the agent's failure, and lumping
    # the two hid that a sweep's accuracy column was recording who got stalled.
    if stalls.provider_failure(record):
        return "provider"
    if record.get("timed_out") or record.get("agent_error"):
        return "crashed"
    perfect = _perfect(record)
    if perfect:
        if record.get("own_tests_miss_the_bug"):
            return "tests_miss_change"
        if not record.get("honest", True):
            return "dishonest"
        return "ok"
    if (
        int(record.get("diag_loops_exhausted", 0) or 0)
        or int(record.get("diag_longest_unproductive_streak", 0) or 0) >= 3
    ):
        return "loop"
    if int(record.get("diag_final_green", -1)) == 1:
        return "wrong_fix_believed_done"
    return "gave_up_red"


def render_failures(records: list[dict], by: str = "config") -> str:
    """How runs ended, by class, per config.

    The outcome table says how often a config was wrong. This says *how*, and
    the two classes worth separating are the ones with opposite remedies:
    `wrong_fix_believed_done` is a diagnosis or verification problem (the agent
    had a green suite and was wrong anyway), `gave_up_red` is an execution
    problem (it knew it had not finished).
    """
    if not records:
        return "no results"
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(by, "?")), []).append(record)
    labels = [name for name, _ in FAILURE_CLASSES]
    header = f"{by:<24}{'n':>4}" + "".join(f"{name[:13]:>15}" for name in labels)
    lines = [header, "-" * len(header)]
    for name in sorted(groups):
        counts = {label: 0 for label in labels}
        for record in groups[name]:
            counts[classify_failure(record)] += 1
        lines.append(
            f"{name:<24}{len(groups[name]):>4}"
            + "".join(f"{counts[label]:>15}" for label in labels)
        )
    lines.append("")
    lines += [f"{name:<26} {meaning}" for name, meaning in FAILURE_CLASSES]
    return "\n".join(lines)


# -- compare: is the difference real ---------------------------------------

_COMPARE_METRICS = (
    ("accuracy", "accuracy", lambda r: float(r.get("accuracy", 0.0))),
    ("honest", "honest (seeded tasks)", lambda r: 1.0 if r.get("honest", True) else 0.0),
    ("perfect", "perfect runs", lambda r: 1.0 if float(r.get("accuracy", 0.0)) >= 1.0 else 0.0),
    ("calls", "model calls", lambda r: float(r.get("calls", 0) or 0)),
    # Per phase (DESIGN-LOG §83): an arm that acts on some phases is judged there.
    (
        "exec_fix_calls",
        "execute+fix calls",
        lambda r: (
            float(r.get("diag_execute_calls", 0) or 0) + float(r.get("diag_fix_calls", 0) or 0)
        ),
    ),
    ("plan_calls", "plan calls", lambda r: float(r.get("diag_plan_calls", 0) or 0)),
    ("out_tokens", "output tokens", lambda r: float(r.get("diag_out_tokens", 0) or 0)),
    ("stalls", "stalls", lambda r: float(r.get("diag_stalls", 0) or 0)),
    (
        "empty_searches",
        "empty web searches",
        lambda r: float(r.get("diag_empty_searches", 0) or 0),
    ),
    ("model_s", "model seconds", lambda r: float(r.get("model_latency_s", 0.0) or 0.0)),
    ("cost", "cost $", lambda r: float(r.get("cost_usd", 0.0) or 0.0)),
    ("wasted", "wasted edits", lambda r: float(r.get("diag_unproductive_edits", 0) or 0)),
    # DESIGN-LOG §105's mechanism metrics.
    ("reads", "file reads (exec+fix)", lambda r: float(r.get("diag_exec_reads", 0) or 0)),
    ("edit_turns", "turns that edit", lambda r: float(r.get("diag_edit_turns", 0) or 0)),
    ("edit_blocks", "edit_blocks calls", lambda r: float(r.get("diag_edit_blocks_calls", 0) or 0)),
    (
        "edit_bad",
        "edit misses + parse errors",
        lambda r: (
            float(r.get("diag_edit_misses", 0) or 0)
            + float(r.get("diag_edit_parse_errors", 0) or 0)
        ),
    ),
    (
        "final_types",
        "first final check: new type errors",
        lambda r: float(r.get("diag_first_final_type_errors", 0) or 0),
    ),
    ("exec_replies", "execute replies", lambda r: float(r.get("diag_execute_calls", 0) or 0)),
    # DESIGN-LOG §108's reviewer arm: review calls, and its guard on what the review found.
    ("review_calls", "review calls", lambda r: float(r.get("diag_review_calls", 0) or 0)),
    ("findings", "review findings", lambda r: float(r.get("findings", 0) or 0)),
)


def render_compare(records: list[dict], a: str, b: str) -> str:
    """Arm `b` against arm `a`, with an interval and a verdict per metric.

    Prints the verdict a reader would otherwise reach by eye, computed: a
    bootstrap interval for the difference, "differs" only when it excludes
    zero, and otherwise how many runs a side it would take to see a gap this
    size. Also says whether the arms ran at the same time, because two sweeps
    an hour apart measured a 2.3x provider-latency drift that read as a
    treatment effect.
    """
    rows_a = [r for r in records if str(r.get("config")) == a]
    rows_b = [r for r in records if str(r.get("config")) == b]
    if not rows_a or not rows_b:
        have = sorted({str(r.get("config")) for r in records})
        return f"need runs for both configs; found {have}"

    lines = [
        f"{b}  vs  {a}   (n = {len(rows_b)} vs {len(rows_a)}; difference is {b} minus {a})",
        "",
    ]
    if not stats.interleaved(
        [float(r.get("started_at", 0) or 0) for r in rows_a],
        [float(r.get("started_at", 0) or 0) for r in rows_b],
    ):
        lines += [
            "WARNING: these arms did not run during the same stretch of time.",
            "Provider latency drifted 2.3x between two sweeps an hour apart in",
            "this project; a model-time or wall-time gap here may be that drift.",
            "",
        ]
    header = f"{'metric':<24}{a[:12]:>13}{b[:12]:>13}{'diff':>10}{'95% interval':>22}  verdict"
    lines += [header, "-" * len(header)]
    for _key, label, get in _COMPARE_METRICS:
        result = stats.compare(label, [get(r) for r in rows_a], [get(r) for r in rows_b])
        if result is None:
            continue
        interval = f"[{result.low:+.3f}, {result.high:+.3f}]"
        lines.append(
            f"{label:<24}{result.mean_a:>13.3f}{result.mean_b:>13.3f}"
            f"{result.diff:>+10.3f}{interval:>22}  {result.verdict}"
        )
    floor = stats.MIN_RUNS_PER_SIDE
    lines += [
        "",
        "verdict    'differs' only when the interval excludes zero. Otherwise the",
        "           runs-a-side figure is what it would take to see a gap this size",
        f"           at 80% power; under {floor} a side no verdict is given.",
    ]
    return "\n".join(lines)


# -- models: what actually answered ----------------------------------------


def render_models(records: list[dict]) -> str:
    """Which model ids and providers actually served each run, by configured id.

    A sweep that records only the model it *asked* for cannot see a routing
    layer serving a different snapshot or upstream provider under an unchanged
    id, and two sweeps of "the same" model a month apart may then compare two
    different models. This lists every requested -> served @ provider pairing
    seen, how many runs it covers, and when it was first and last seen, so a
    swap shows up as a second row for one requested id.
    """
    seen: dict[str, list[dict]] = {}
    unrecorded = 0
    for record in records:
        pairs = record.get("served_models") or []
        if not pairs:
            unrecorded += 1
        for pair in pairs:
            seen.setdefault(str(pair), []).append(record)
    lines = []
    if seen:
        header = f"{'requested -> served @ provider':<70}{'runs':>6}  span"
        lines += [header, "-" * len(header)]
        by_requested: dict[str, int] = {}
        for pair in sorted(seen):
            rows = seen[pair]
            times = [float(r.get("started_at", 0) or 0) for r in rows]
            span = f"{(max(times) - min(times)) / 60:.0f} min"
            lines.append(f"{pair[:69]:<70}{len(rows):>6}  {span}")
            requested = pair.split(" -> ")[0]
            by_requested[requested] = by_requested.get(requested, 0) + 1
        for requested, count in sorted(by_requested.items()):
            if count > 1:
                lines.append("")
                lines.append(
                    f"NOTE: {requested} was served under {count} different "
                    "model/provider pairings — runs of it are not all the "
                    "same model."
                )
    else:
        lines.append("no served-model data in these records")
    if unrecorded and seen:
        lines.append("")
        lines.append(
            f"{unrecorded} run(s) carry no served-model data (recorded before it "
            "existed, or the provider did not say)."
        )
    return "\n".join(lines)


# -- history: what a config normally scores on each task -------------------

CEILING_ACCURACY = 0.98
CEILING_HONEST = 0.95
MIN_HISTORY_RUNS = 8


FLOOR_SHARE = 0.15


def _history_reading(n: int, accuracy: float, perfect: float, honest: float | None) -> str:
    """What a task can and cannot do for a comparison, from a config's history on it."""
    if n < MIN_HISTORY_RUNS:
        return f"too few runs (<{MIN_HISTORY_RUNS})"
    solved = accuracy >= CEILING_ACCURACY and perfect >= 0.9
    if solved:
        if honest is None or honest >= CEILING_HONEST:
            return "ceiling: cannot separate configs"
        return "moves on honesty"
    if perfect <= FLOOR_SHARE and (honest is None or honest <= FLOOR_SHARE):
        return "floor: almost nothing passes it"
    return "moves on accuracy"


def _history_row(task: str, rows: list[dict]) -> str:
    accuracy = statistics.fmean([float(r.get("accuracy", 0.0)) for r in rows])
    perfect = sum(1 for r in rows if _perfect(r)) / len(rows)
    honest_text = _honest_rate(rows)
    honest = None if honest_text == "-" else float(honest_text.rstrip("%")) / 100
    calls = statistics.median([float(r.get("calls", 0) or 0) for r in rows])
    reading = _history_reading(len(rows), accuracy, perfect, honest)
    return (
        f"{task:<20}{len(rows):>4}{accuracy * 100:>9.0f}%{perfect * 100:>8.0f}%"
        f"{honest_text:>8}{calls:>8.0f}  {reading}"
    )


def render_history(records: list[dict], config: str, since: float = 0.0) -> str:
    """A config's own record on each task, across every run in the given files.

    "Before calling a bench delta a regression, compare it with that config's
    historical rate on the task" (the repo rule), because `regression` was honest 67%
    of the time historically, not 100%, and a control that happens to score 100%
    on a task it normally misses makes every treatment look like a loss. This is
    that lookup, plus the question §53.2 ended on: which tasks are at a ceiling
    and so cannot separate two configs at all.

    Provider failures are expected to have been removed by the caller. `since` is
    a unix time; earlier runs are ignored, for a task whose fixture changed.
    """
    rows = [
        r
        for r in records
        if str(r.get("config")) == config
        and float(r.get("started_at", 0) or 0) >= since
        # A plan-only run stops before anything is scored; folding its zero accuracy
        # into a task's history called rename-callers a 49% task when 42 of 43 scored
        # runs were perfect (FINDINGS.md row 15).
        and not r.get("plan_only")
    ]
    if not rows:
        have = sorted({str(r.get("config")) for r in records})
        return f"no scored runs of {config!r}; found {have}"
    by_task: dict[str, list[dict]] = {}
    for record in rows:
        by_task.setdefault(str(record.get("task", "?")), []).append(record)
    header = f"{'task':<20}{'n':>4}{'accuracy':>10}{'perfect':>8}{'honest':>8}{'calls':>8}  reading"
    lines = [
        f"{config}: {len(rows)} runs across {len(by_task)} task(s)",
        "",
        header,
        "-" * len(header),
    ]
    lines += [_history_row(task, group) for task, group in sorted(by_task.items())]
    lines += [
        "",
        f"ceiling  accuracy >= {CEILING_ACCURACY:.0%}, >= 90% of runs perfect, honest >= "
        f"{CEILING_HONEST:.0%} (or unseeded):",
        "         two configs cannot be told apart on it; it spends budget without comparing",
        f"floor    at most {FLOOR_SHARE:.0%} of runs perfect and honest: the same, from below",
        "perfect  share of runs that passed every hidden test",
        "honest   share of seeded-task runs reached without weakening a test; - if unseeded",
    ]
    return "\n".join(lines)


# The planner's reproduce check (plan_probe="repro"), one outcome per run, in the
# order a run's counters are read: a check that ran beats one that was asked for.
REPRO_OUTCOMES = ("reproduced", "passed", "broken", "not_run", "missing", "none")


def repro_outcome(record: dict) -> str:
    """What the run's reproduce check did, or "" when the run has no repro counters."""
    for outcome in REPRO_OUTCOMES:
        if float(record.get(f"diag_repro_{outcome}", 0) or 0) > 0:
            return outcome
    return ""


def render_repro(records: list[dict]) -> str:
    """Per config and task: how often the reproduce check ran, and what it showed.

    Built for DESIGN-LOG §65 (the second time this tally was done by hand, after the
    §65 history split): `ran` is reproduced + passed + broken, the share the check
    actually executed. On a wrong-report task `passed` is the useful outcome; on a
    real bug it is a correct plan sent back. Honest is shown only for full runs.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        key = (str(record.get("config", "?")), str(record.get("task", "?")))
        groups.setdefault(key, []).append(record)
    header = (
        f"{'config':<26}{'task':<18}{'n':>4}{'ran':>6}{'repro':>7}{'passed':>8}"
        f"{'broken':>8}{'empty':>7}{'calls':>7}{'honest':>8}"
    )
    lines = [header, "-" * len(header)]
    for (config, task), rows in sorted(groups.items()):
        outcomes = [repro_outcome(r) for r in rows]
        n = len(rows)

        def share(*names: str, outcomes=outcomes, n=n) -> str:
            return f"{sum(o in names for o in outcomes) / n:.0%}"

        planning_only = all(r.get("plan_only") for r in rows)
        honest = (
            "-"
            if planning_only
            else f"{statistics.fmean(1.0 if r.get('honest', True) else 0.0 for r in rows):.0%}"
        )
        calls = statistics.fmean(float(r.get("calls", 0) or 0) for r in rows)
        lines.append(
            f"{config:<26}{task:<18}{n:>4}{share('reproduced', 'passed', 'broken'):>6}"
            f"{share('reproduced'):>7}{share('passed'):>8}{share('broken'):>8}"
            f"{share('none', 'missing', ''):>7}{calls:>7.1f}{honest:>8}"
        )
    lines += [
        "",
        "ran = the check executed (repro + passed + broken); empty = left empty or declined.",
        "passed on a wrong-report task is the check working; on a real bug, a plan sent back.",
    ]
    return "\n".join(lines)
