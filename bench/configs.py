"""Named agent configurations to compare.

A configuration is the whole agent, not just a model: which model plans, executes
and reviews, how hard each thinks, and which of the harness's accuracy and speed
mechanisms are switched on. They trade against each other, and the cheapest
configuration that is accurate enough is not obvious in advance — which is the
reason the benchmark exists.

Configurations come in four groups: the originals, ablations that price one
mechanism at a time, executor swaps against Mercury, and mixed-model hybrids.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Verified present on OpenRouter with tool-calling support, with $/M in/out.
MERCURY = "inception/mercury-2.5"  # 0.040 / 0.150   diffusion
MERCURY_3 = "inception/mercury-3"  # 0.100 / 0.350   diffusion (launch price; list 0.20 / 0.70)
SONNET = "anthropic/claude-sonnet-5"  # frontier autoregressive
HAIKU = "anthropic/claude-haiku-4.5"  # 1.000 / 5.000
QWEN_FLASH = "qwen/qwen3.7-flash"  # 0.030 / 0.130   cheapest tool-capable
GLM_FLASH = "z-ai/glm-5.3-flash"  # 0.070 / 0.233
DEEPSEEK_FLASH = "deepseek/deepseek-v4-flash"  # 0.089 / 0.177
GPT_NANO = "openai/gpt-5-nano"  # 0.050 / 0.400
GPT_OSS = "openai/gpt-oss-120b"  # 0.037 / 0.170
LAGUNA = "poolside/laguna-s-2.1"  # 0.090 / 0.180   coding-specialised
# §131.16: the models Inception compares Mercury 3 with, at its price class, and Sonnet 5.5.
HAIKU_55 = "anthropic/claude-haiku-5.5"  # 0.100 / 0.500
GPT6_LUNA = "openai/gpt-6-luna"  # 0.100 / 0.500
SONNET_55 = "anthropic/claude-sonnet-5.5"  # 2.000 / 10.000


@dataclass
class AgentConfig:
    name: str
    planner: str
    executor: str
    reviewer: str
    executor_reasoning: str = "high"
    planner_reasoning: str = "medium"
    # Optional phase 1b: a second model checks the plan against the goal before
    # execution starts, and can resubmit a corrected plan (see
    # agent/phases/plan_audit.py). None (the default) skips this entirely.
    plan_auditor: str | None = None
    plan_audit_reasoning: str = "low"
    # Which agent/relevance.py backend the tool-loop compactor uses, when it
    # fires. None (the default) is the harness's normal auto-preference
    # (Jev if a key is configured, else the executor model, else free
    # keyword overlap). Forcing it lets a sweep hold the task fixed and vary
    # only the judge backend -- see the `judge` group below.
    relevance_backend: str | None = None
    relevance_model: str | None = None
    # Execution tools to force into every step's kit regardless of the step
    # text's own optional-group triggers (see agent/tools/__init__.py). Empty
    # by default. Exists to measure a new tool's effect on a task whose
    # auto-generated step titles would not reliably say its trigger words --
    # see the `debug` group below.
    force_tools: tuple[str, ...] = ()
    # Where Mercury's calls go and at what temperature (Settings.mercury_provider,
    # Settings.mercury_temperature; DESIGN-LOG §85). Forwarded by name.
    mercury_provider: str = "openrouter"
    mercury_temperature: float | None = None
    executor_batch_last: bool = False
    # On since §106.2 (measured on Mercury; configs with another executor turn it off).
    executor_tail_reminder: bool = True
    executor_tail_items: tuple[str, ...] = ()
    planner_tail: bool = False
    plan_key_facts: bool = False
    plan_findings_last: bool = False
    plan_findings_required: bool = False
    plan_findings_hold: bool = False
    last_word_gates: str = "off"
    # Forwarded to Settings by name; the same defaults as Settings (§95.1).
    mercury_max_tokens: int | None = 16_000
    mercury_content_stall_s: float | None = 30.0
    # None (the default) leaves run_tool_loop's own budget math in charge.
    # The `judge` group sets this low on purpose, to force real compaction
    # (and therefore a real judge call) inside a short, cheap bench task
    # instead of relying on a fixture large enough to hit the default
    # quarter-context budget on its own. Named differently from
    # Budget.step_prompt_token_budget on purpose: Settings has no field of
    # that name (it lives on the nested Budget object bench/runner.py builds
    # separately), so a same-named AgentConfig field would auto-forward
    # nowhere and silently no-op -- bench/runner.py::_settings_for wires this
    # one into Budget explicitly.
    judge_step_budget: int | None = None
    baseline_digest: bool = True
    test_audit: bool = True
    fix_weak_tests: bool = False
    symptom_first_planning: bool = False
    strict_tool_schemas: bool = True
    callsite_check: bool = True
    plan_probe: str = "repro"
    repro_form: str = "snippet"
    research_first: bool = False
    numeric_check: bool = False
    check_after_edit: bool = True
    combined_fixes: bool = True
    edit_blocks: bool = False
    read_assist: bool = True
    types_after_edit: bool = True
    review_presupplied_checks: bool = True
    bugfix_evidence: bool = False
    premise_gate: bool = True
    test_change_justification: bool = False
    draft_style: str = "terse"
    tdd_gate: bool = False
    executor_order: str = "tdd_first"
    executor_sees_draft: bool = False
    executor_hypothesis: bool = False
    plan_task_kind: bool = False
    preload_neighbours: bool = False
    # Skip phase 3 entirely, to price what review is actually buying.
    review: bool = True
    # Skip the planner: one step for the whole goal (Settings.direct, DESIGN-LOG §129).
    direct: bool = False
    autolint: bool = True
    stream: bool = True
    # Only Anthropic models accept an explicit cache breakpoint, so this is
    # inert for every non-retired config here. Kept because the retired
    # Sonnet and Haiku configs are still runnable by name and it is real for
    # them; not worth an ablation while the active set is Mercury and GLM.
    cache_prompts: bool = True
    # Run each step's verify command and hand failures back.
    verify_steps: bool = True
    # Forwarded to Settings by name (§131.9; off by default since).
    expected_result_gate: bool = False
    # Forwarded to Settings by name (§131.12; on by default since).
    runaway_effort_step_down: bool = True
    # Retry a failed step once with reasoning turned up.
    escalate_on_failure: bool = True
    # The deterministic gate after review: re-run every verify command and the
    # test suite against the finished code, then fix what fails. Sweepable
    # because it is the one phase that costs wall time but no model calls
    # until something actually fails, so what it buys is worth pricing.
    final_check: bool = True
    # Overlap independent steps that touch disjoint files.
    step_concurrency: int = 1
    # Inject skills/*.md reference material into prompts.
    skills: bool = True
    skill_triggers: str = (
        "step"  # default since §117.1; mercury-skill-triggers-workspace is the control
    )
    # Let a green whole-suite run stand in for a step's own pytest-selection
    # verify command at the gate, instead of running both.
    skip_subsumed_verifies: bool = True
    # Type-check what a step changed against the rest of the project, so a
    # broken caller fails the step that broke it rather than the review.
    type_check_steps: bool = True
    # Narrow a step's whole-suite verify to the step's own files.
    scope_step_verify: bool = True
    # Skip a step's verify command when the executor already ran an
    # equivalent passing pytest invocation at the same workspace state.
    skip_verified_by_tests: bool = True
    # Replay a read-only check's output when the identical call is repeated
    # against a byte-for-byte unchanged workspace.
    replay_identical_checks: bool = True
    description: str = ""
    group: str = "core"
    # Registered and runnable by name, but kept out of `in_group` so a sweep
    # does not pull it in. Two reasons a config lands here, both measured
    # rather than assumed: it costs far more than it discriminates (the
    # Sonnet and Haiku configs -- `hybrid` scored 33% with real timeouts at
    # roughly 20x `mercury-only`'s cost across two sweeps), or it differs
    # from a config in an active group by one setting that would be better
    # swept as an axis than duplicated as a config (most of the `fix` group,
    # which restates core/executor/ablation entries at
    # planner_reasoning="medium").
    retired: bool = False

    @property
    def models(self) -> set[str]:
        models = {self.planner, self.executor, self.reviewer}
        if self.plan_auditor:
            models.add(self.plan_auditor)
        return models


CONFIGS: dict[str, AgentConfig] = {}


def register(config: AgentConfig) -> AgentConfig:
    CONFIGS[config.name] = config
    return config


# -- the original designs ------------------------------------------------

register(
    AgentConfig(
        name="hybrid",
        retired=True,
        planner=SONNET,
        executor=MERCURY,
        reviewer=SONNET,
        description="the harness's design: Claude plans and reviews, Mercury executes",
    )
)

# Deferred rather than run this session: `hybrid` at planner_reasoning="low"
# (review.py shares that same knob, so it would also drop the reviewer pass
# to "low") to see whether Sonnet's accuracy edge needs "medium" or holds
# cheaper. Not built yet because the 2026-09-19 sweep's `glm-plan-mercury`
# already prices "cheaper model in the same role"; a low-reasoning Sonnet
# variant prices the other lever (same model, less thinking) and is worth
# its own measurement once that sweep's results are in, not assumed now.

# Paired with plain `hybrid` (cache_prompts defaults True there) to measure
# the explicit cache breakpoint at all: it is a no-op on every active
# Mercury/GLM config (llm.CACHEABLE_PREFIXES is Anthropic-only), so testing
# it needs a config with an Anthropic model. Retired from routine sweeps --
# Sonnet costs ~20x Mercury for a comparable score -- and gettable by name
# for exactly this one measurement, against `hybrid` as the cache-on side.
register(
    AgentConfig(
        name="hybrid-nocache",
        retired=True,
        planner=SONNET,
        executor=MERCURY,
        reviewer=SONNET,
        cache_prompts=False,
        group="cache-test",
        description="hybrid with the explicit cache breakpoint off, same otherwise",
    )
)

register(
    AgentConfig(
        name="mercury-only",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        description="dLLM baseline: Mercury does all three phases",
    )
)

register(
    AgentConfig(
        name="mercury3-only",
        planner=MERCURY_3,
        executor=MERCURY_3,
        reviewer=MERCURY_3,
        planner_reasoning="high",
        group="executor",
        description="mercury-only with Mercury 3 in every phase",
    )
)

register(
    AgentConfig(
        name="sonnet-only",
        retired=True,
        planner=SONNET,
        executor=SONNET,
        reviewer=SONNET,
        description="autoregressive control: Claude does everything",
    )
)

register(
    AgentConfig(
        name="hybrid-cheap",
        retired=True,
        planner=HAIKU,
        executor=MERCURY,
        reviewer=HAIKU,
        description="a small autoregressive model plans and reviews",
    )
)

# -- ablations: price one mechanism at a time ----------------------------

register(
    AgentConfig(
        name="mercury-raw",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        verify_steps=False,
        escalate_on_failure=False,
        autolint=False,
        group="ablation",
        description="Mercury with every accuracy mechanism off — the floor",
    )
)

register(
    AgentConfig(
        name="mercury-noverify",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        verify_steps=False,
        escalate_on_failure=False,
        group="ablation",
        description="Mercury without the step verify gate — prices verification",
    )
)

register(
    AgentConfig(
        name="mercury-direct",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        review=False,
        direct=True,
        group="ablation",
        description="Mercury with no planner and no review: one step for the goal (§129)",
    )
)

register(
    AgentConfig(
        name="mercury-noreview",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        review=False,
        group="ablation",
        description="Mercury without phase 3 — prices review",
    )
)

register(
    AgentConfig(
        name="mercury-parallel",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        step_concurrency=3,
        group="ablation",
        description="Mercury overlapping independent steps — prices concurrency",
    )
)

# Skill triggers that see only the step itself (DESIGN-LOG §117), adopted in §117.1: this is
# now mercury-only, and mercury-skill-triggers-workspace is the control.
register(
    AgentConfig(
        name="mercury-skill-triggers-step",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        skill_triggers="step",
        group="ablation",
        description="step-only skill triggers, now the default (§117.1): see mercury-only",
    )
)

# The pre-§117.1 default, so a sweep can still price the trigger change.
register(
    AgentConfig(
        name="mercury-skill-triggers-workspace",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        skill_triggers="workspace",
        group="ablation",
        description="mercury-only with skill triggers matched to the workspace and the verify",
    )
)

register(
    AgentConfig(
        name="mercury-noskills",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        skills=False,
        group="ablation",
        description="Mercury without injected skill guidance — prices the skills",
    )
)

register(
    AgentConfig(
        name="mercury-nano",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="medium",
        executor_reasoning="none",
        group="ablation",
        description="the cheapest setting: Mercury with reasoning off",
    )
)

register(
    AgentConfig(
        name="mercury-high",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_reasoning="high",
        group="ablation",
        description="Mercury with reasoning turned up everywhere — now the default, see mercury-only",  # noqa: E501
    )
)

register(
    AgentConfig(
        name="mercury-nostream",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        stream=False,
        group="ablation",
        description="Mercury with streaming off — prices stream=True against a "
        "non-streamed call, per DESIGN-LOG's open question on whether "
        "streaming's early-TTFT signal is worth its own failure modes "
        "for a model whose decode is a fast burst, not progressive",
    )
)

register(
    AgentConfig(
        name="mercury-nogate",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        final_check=False,
        group="ablation",
        description="Mercury without the deterministic final gate — prices phase 4",
    )
)

# -- judge: which agent/relevance.py backend does the tool-loop compactor use? --
# Same base config, same tiny step_prompt_token_budget (forces real
# compaction inside a short bench task rather than relying on a fixture big
# enough to hit the default quarter-context budget on its own) -- the only
# axis that varies is relevance_backend, so a sweep across this group prices
# the judge backend itself, holding everything else constant. `judge-none`
# is what every run before agent/relevance.py existed did: no judge, plain
# drop-oldest FIFO compaction (see runner.py::_compact_fifo) -- the actual
# baseline the other three are upgrades over, not just another backend.
register(
    AgentConfig(
        name="judge-none",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        judge_step_budget=1_600,
        relevance_backend="heuristic",
        group="judge",
        description="no relevance judge — the pre-agent/relevance.py FIFO baseline "
        "(HeuristicJudge scores by keyword overlap only, same as the "
        "compactor's old drop-oldest behavior on realistic content)",
    )
)

register(
    AgentConfig(
        name="judge-mercury",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        judge_step_budget=1_600,
        relevance_backend="llm",
        relevance_model=MERCURY,
        group="judge",
        description="LLMJudge backed by Mercury 2.5 — no new API key or dependency",
    )
)

register(
    AgentConfig(
        name="judge-jev",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        judge_step_budget=1_600,
        relevance_backend="jev",
        group="judge",
        description="JevJudge (typesafe_sdk) — needs a TypeSafe key in Keychain; "
        "falls back to the auto-preference order if none is configured",
    )
)

# -- debug: does a one-shot execution tracer beat print-statement run_python? --
# A trace pass through 261 real loop_exhausted events (1,651 traces) found
# ~46% were a same-tool streak, run_python the largest single source: the
# model using it as an interactive debugger, adding one print() and
# re-running to see one more variable, sometimes literally re-running an
# identical snippet. trace_python (agent/tools/checks.py, pure sys.settrace,
# no dependency) answers that in one call instead of several. Same base
# config as mercury-only; force_tools is the only axis that varies, since
# expr/throttle/regression's auto-generated step titles won't reliably say
# "debug" on their own to trigger the tool's normal optional-group gating.
register(
    AgentConfig(
        name="mercury-trace-python",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        force_tools=("trace_python",),
        group="debug",
        description="mercury-only with trace_python forced into every step's kit",
    )
)

# Prices agent/failsig.py's baseline digest: the planner is told what each
# already-failing test reports, not only its name. Only meaningful on a task
# that starts red (the red-* fixtures); on every other task the two arms are
# byte-identical.
register(
    AgentConfig(
        name="mercury-nodigest",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        baseline_digest=False,
        group="digest",
        description="mercury-only with the baseline failure digest withheld from the planner",
    )
)

# Prices relaxing chain-of-draft: does letting the planner draft in full sentences
# and paragraphs (code sketches allowed in the draft only) change the plans it
# writes? The mechanism metrics are draft words, named tests per step and
# planning tool calls; see DESIGN-LOG §47 (plan W8).
register(
    AgentConfig(
        name="mercury-prose-draft",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        draft_style="prose",
        group="digest",
        description="mercury-only with a prose draft instead of chain-of-draft notes",
    )
)

# Prices Settings.strict_tool_schemas (DESIGN-LOG §63): does `strict: true` on
# every tool stop arguments arriving JSON-encoded or as the wrong scalar type?
# (A submit_plan-only version, mercury-strict-plan, was scouted first.)
register(
    AgentConfig(
        name="mercury-strict-tools",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        strict_tool_schemas=True,
        group="digest",
        retired=True,
        description="strict_tool_schemas=True, now the default: see mercury-only, and "
        "mercury-no-strict-tools for the old behaviour",
    )
)

# Prices Settings.callsite_check (DESIGN-LOG §66): does sending back a rename
# plan that leaves out a calling file prevent the §62.1 missed-callers failure?
register(
    AgentConfig(
        name="mercury-callsite-check",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        callsite_check=True,
        group="probe",
        retired=True,
        description="callsite_check=True, now the default: see mercury-only, and "
        "mercury-no-callsite-check for the old behaviour",
    )
)

# The pre-§66.1 default, so a sweep can still price the call-site check.
register(
    AgentConfig(
        name="mercury-no-callsite-check",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        callsite_check=False,
        group="probe",
        description="mercury-only without the call-site plan gate (the old default)",
    )
)

# The pre-§63.1 default, so a sweep can still price strict tool schemas.
register(
    AgentConfig(
        name="mercury-no-strict-tools",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        strict_tool_schemas=False,
        group="probe",
        description="mercury-only with tool schemas sent without strict (the old default)",
    )
)

# Reasoning-architecture arms (DESIGN-LOG §53). Mercury 2.5 reasons in hidden
# tokens — about 90% of its completion tokens at medium/high effort, returned
# as a count and never as text — so the first question is whether the
# executor's effort level is buying anything (§40 left it untested), and the
# second is whether reasoning the harness already pays for reaches the step
# that needs it. None of these adds a model call.
for _effort in ("none", "low", "medium"):
    register(
        AgentConfig(
            name=f"mercury-exec-{_effort}",
            planner=MERCURY,
            executor=MERCURY,
            reviewer=MERCURY,
            planner_reasoning="high",
            executor_reasoning=_effort,
            group="reasoning",
            description=f"mercury-only with executor reasoning effort {_effort} (default high)",
        )
    )

register(
    AgentConfig(
        name="mercury-draft-handoff",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_sees_draft=True,
        group="reasoning",
        description="mercury-only with the planner's draft notes passed to every executor step",
    )
)

register(
    AgentConfig(
        name="mercury-hypothesis",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_hypothesis=True,
        group="reasoning",
        description="mercury-only with a one-line `why` required on the call after a failure",
    )
)

register(
    AgentConfig(
        name="mercury-task-kind",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        plan_task_kind=True,
        group="reasoning",
        description="mercury-only with the planner classifying the goal (recorded, not routed)",
    )
)

register(
    AgentConfig(
        name="mercury-neighbours",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        preload_neighbours=True,
        group="reasoning",
        description="mercury-only with each step's importing tests and callers preloaded",
    )
)

# Prices the two ways of closing the gap between the executor prompt's
# write-a-failing-test-first rule and what steps do (13% of source-editing steps
# comply). One arm holds the first source edit until a test has run; the other
# only rewrites the prompt's ordering guidance. Separate configs, so each is
# swept against its own control and neither confounds the other.
register(
    AgentConfig(
        name="mercury-tdd-gate",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        tdd_gate=True,
        group="digest",
        description="mercury-only with the first source edit held until a test has run",
    )
)
register(
    AgentConfig(
        name="mercury-legacy-order",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_order="legacy",
        group="digest",
        description="mercury-only with the pre-§51 batch-everything order guidance in place of "
        "the ordered test-first procedure",
    )
)

# Prices agent/testcheck.py: the extra scratch-copy pytest runs a green run
# pays for after the gate. Report-only, so accuracy cannot move; what this
# measures is wall time and nothing else, which is the cost that decides
# whether the audit can stay on by default.
register(
    AgentConfig(
        name="mercury-notestaudit",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        test_audit=False,
        group="digest",
        description="mercury-only without the post-run test audit — prices its wall time",
    )
)

# Prices feeding agent/testcheck.py's findings back as one fix round: does telling
# the model its tests cannot see the change get it to write ones that can?
register(
    AgentConfig(
        name="mercury-fixtests",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        fix_weak_tests=True,
        group="digest",
        description="mercury-only plus one fix round on tests the audit says cannot see the change",
    )
)

# Prices prompts.py's SYMPTOM_FIRST_PLANNING against a goal that reports a
# symptom and names no defect — the shape an audit of 40 real runs found the
# planner diagnosing after a single read_files. Honesty, not accuracy, is the
# column to read: the failure it targets is a fix that passes tests aimed at
# the function the planner guessed about.
register(
    AgentConfig(
        name="mercury-symptom",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        symptom_first_planning=True,
        group="digest",
        description="mercury-only plus the planner's symptom-before-cause block — "
        "measured once at 83% vs 90% honest and 96.1% vs 98.3% accuracy "
        "against the default, so the block is off by default and this "
        "config is what would re-open the question",
    )
)

# Prices Settings.plan_probe: the planner can run code against a throwaway copy
# of the workspace before it names a cause. Sweep it against mercury-only on the
# `symptom` task set, where the goal reports a symptom and names no defect.
# Read honesty and plan_probes (did it use the tool at all?) before accuracy.
register(
    AgentConfig(
        name="mercury-plan-probe",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        plan_probe="offer",
        group="probe",
        description="mercury-only plus probe_python in the planner's kit",
    )
)

register(
    AgentConfig(
        name="mercury-plan-probe-required",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        plan_probe="require",
        group="probe",
        description="mercury-plan-probe, and the first submit_plan is held until a probe has run",
    )
)

register(
    AgentConfig(
        name="mercury-plan-repro",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        plan_probe="repro",
        group="probe",
        retired=True,
        description="plan_probe='repro', now the default: see mercury-only, and "
        "mercury-no-repro for the control",
    )
)

# Prices Settings.premise_gate against mercury-only (DESIGN-LOG §57).
register(
    AgentConfig(
        name="mercury-premise-gate",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        premise_gate=True,
        group="probe",
        retired=True,
        description="premise_gate=True, now the default: see mercury-only, and "
        "mercury-no-premise-gate for the control",
    )
)

# Prices Settings.test_change_justification against mercury-only (DESIGN-LOG §59): an
# edit that removes, weakens or rewrites a pre-existing test is held until it says
# why, and the reason is recorded and shown.
register(
    AgentConfig(
        name="mercury-test-justification",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        test_change_justification=True,
        group="digest",
        description="mercury-only plus a required justification for changing an existing test",
    )
)

# The pre-§57.1 default, so a sweep can still price premise_gate.
register(
    AgentConfig(
        name="mercury-no-premise-gate",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        premise_gate=False,
        group="probe",
        description="mercury-only without the premise holds (premise_gate=False)",
    )
)

# The pre-§54.2 default, so a sweep can still price plan_probe="repro".
register(
    AgentConfig(
        name="mercury-no-repro",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        plan_probe="off",
        group="probe",
        description="mercury-only without the planner's reproduce check (plan_probe='off')",
    )
)

# Settings.repro_form (agent/symptom.py, DESIGN-LOG §65): the reproduce check asked for
# before the steps with `none: why` instead of an empty string, and as a call plus its
# expected value that the harness turns into the assertion. Read with --plan-only first:
# the question is how often the check runs at all.
for _form in ("explicit", "symptom"):
    register(
        AgentConfig(
            name=f"mercury-repro-{_form}",
            planner=MERCURY,
            executor=MERCURY,
            reviewer=MERCURY,
            planner_reasoning="high",
            repro_form=_form,
            group="probe",
            description=f"mercury-only with repro_form={_form!r} (DESIGN-LOG §65)",
        )
    )

# -- executor swaps: who else can do the mechanical work? ----------------

register(
    AgentConfig(
        name="qwen-only",
        retired=True,
        planner=QWEN_FLASH,
        executor=QWEN_FLASH,
        reviewer=QWEN_FLASH,
        planner_reasoning="high",
        group="executor",
        description="Qwen3.7 Flash throughout — cheaper per token than Mercury",
    )
)

register(
    AgentConfig(
        name="glm-only",
        planner=GLM_FLASH,
        executor=GLM_FLASH,
        reviewer=GLM_FLASH,
        planner_reasoning="high",
        executor_tail_reminder=False,  # the reminder is measured on Mercury only (§106.2)
        review_presupplied_checks=False,  # measured with a Mercury reviewer only (§108.2)
        group="executor",
        description="GLM 5.3 Flash throughout",
    )
)

register(
    AgentConfig(
        name="deepseek-only",
        retired=True,
        planner=DEEPSEEK_FLASH,
        executor=DEEPSEEK_FLASH,
        reviewer=DEEPSEEK_FLASH,
        planner_reasoning="high",
        group="executor",
        description="DeepSeek V4 Flash throughout",
    )
)

register(
    AgentConfig(
        name="gptoss-only",
        retired=True,
        planner=GPT_OSS,
        executor=GPT_OSS,
        reviewer=GPT_OSS,
        planner_reasoning="high",
        group="executor",
        description="gpt-oss-120b throughout",
    )
)

register(
    AgentConfig(
        name="nano-only",
        retired=True,
        planner=GPT_NANO,
        executor=GPT_NANO,
        reviewer=GPT_NANO,
        planner_reasoning="high",
        group="executor",
        description="GPT-5 Nano throughout",
    )
)

register(
    AgentConfig(
        name="laguna-only",
        retired=True,
        planner=LAGUNA,
        executor=LAGUNA,
        reviewer=LAGUNA,
        planner_reasoning="high",
        group="executor",
        description="Laguna S 2.1 throughout — a coding-specialised model",
    )
)

# -- mixed: cheap thinking, fast hands ------------------------------------

register(
    AgentConfig(
        name="qwen-plan-mercury",
        retired=True,
        planner=QWEN_FLASH,
        executor=MERCURY,
        reviewer=QWEN_FLASH,
        planner_reasoning="high",
        group="mixed",
        description="a cheap autoregressive planner and reviewer with Mercury executing",
    )
)

register(
    AgentConfig(
        name="sonnet-plan-mercury",
        retired=True,
        planner=SONNET,
        executor=MERCURY,
        reviewer=MERCURY,
        group="mixed",
        description="Claude plans, Mercury executes and reviews — pays for planning only",
    )
)

register(
    AgentConfig(
        name="mercury-plan-sonnet-review",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=SONNET,
        planner_reasoning="high",
        group="mixed",
        description="Mercury plans and executes, Claude reviews — pays for the catch only",
    )
)

register(
    AgentConfig(
        name="mercury-turbo",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        step_concurrency=3,
        review=False,
        group="mixed",
        description="everything tuned for speed: parallel steps, no review",
    )
)

register(
    AgentConfig(
        name="hybrid-deepseek",
        retired=True,
        planner=DEEPSEEK_FLASH,
        executor=MERCURY,
        reviewer=DEEPSEEK_FLASH,
        planner_reasoning="high",
        group="mixed",
        description="DeepSeek V4 Flash plans and reviews, Mercury executes — a cheaper "
        "alternative to hybrid's Sonnet planner/reviewer",
    )
)


# The cheap tool-capable field, for pricing bug-fixing rather than greenfield
# work. Haiku appears only as a planner/reviewer over a Mercury executor: it is
# ~25x Mercury per token, but planning and review are a handful of calls while
# execution is most of them, so it buys better plans at a bounded cost.
register(
    AgentConfig(
        name="fix-mercury",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        description="Mercury throughout — the baseline",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-qwen",
        retired=True,
        planner=QWEN_FLASH,
        executor=QWEN_FLASH,
        reviewer=QWEN_FLASH,
        description="Qwen3.7 Flash throughout",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-deepseek",
        retired=True,
        planner=DEEPSEEK_FLASH,
        executor=DEEPSEEK_FLASH,
        reviewer=DEEPSEEK_FLASH,
        description="DeepSeek V4 Flash throughout",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-glm",
        retired=True,
        planner=GLM_FLASH,
        executor=GLM_FLASH,
        reviewer=GLM_FLASH,
        description="GLM 5.3 Flash throughout",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-gptoss",
        retired=True,
        planner=GPT_OSS,
        executor=GPT_OSS,
        reviewer=GPT_OSS,
        description="gpt-oss-120b throughout",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-haiku-plan",
        retired=True,
        planner=HAIKU,
        executor=MERCURY,
        reviewer=HAIKU,
        description="Haiku plans and reviews, Mercury executes",
        group="fix",
    )
)

# Dropped from the sweep after the first 27 runs rather than kept for
# completeness: gpt-oss was the slowest by model time (141s a run) *and* passed
# a task by deleting all four seeded tests, and DeepSeek was both slow (115s,
# 11.1s a call) and the least accurate, with the sweep's only timeout. Both are
# still defined above under the executor group if anyone wants to re-check them.

# Concurrency, which is worth measuring for the first time now that a step
# declares what it writes and is refused anything else (see tools/base.py).
# Until that existed, overlapping steps meant trusting a plan's file list.
register(
    AgentConfig(
        name="fix-mercury-parallel",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        step_concurrency=3,
        description="Mercury with independent steps overlapped",
        group="fix",
    )
)

# Mixed phases: a different model doing the thinking from the one doing the
# typing. This is the harness's founding bet, so it is worth pricing against
# more than one planner — GLM was the fastest model in the first sweep by model
# time and Qwen is the cheapest tool-capable one on the list.
register(
    AgentConfig(
        name="glm-plan-mercury",
        planner=GLM_FLASH,
        executor=MERCURY,
        reviewer=GLM_FLASH,
        planner_reasoning="high",
        group="mixed",
        review_presupplied_checks=False,  # measured with a Mercury reviewer only (§108.2)
        description="GLM plans and reviews, Mercury executes",
    )
)

# Prices the same lever `agent/audit.py`'s DEFAULT_REASONING_EFFORT="low"
# already paid off for GLM's audit pass (DESIGN-LOG §41.1: 4-6x faster,
# 6-11x cheaper, same findings) against `glm-plan-mercury` above: does a
# cheaper, faster GLM planning/review pass hold accuracy on real bench
# tasks, or was that win specific to the audit's narrower judgment call?
register(
    AgentConfig(
        name="glm-plan-mercury-low",
        planner=GLM_FLASH,
        executor=MERCURY,
        reviewer=GLM_FLASH,
        planner_reasoning="low",
        group="mixed",
        review_presupplied_checks=False,  # measured with a Mercury reviewer only (§108.2)
        description="GLM plans and reviews at low reasoning, Mercury executes — "
        "prices whether low reasoning holds accuracy for planning "
        "the way it did for the audit pass",
    )
)

# Mercury plans (fast, terser, more variable between repeats — see DESIGN-LOG's
# plan-comparison finding), GLM audits the plan against the goal before
# execution starts (agent/phases/plan_audit.py) and can patch it if it finds a
# real gap. The bet: a narrow alignment check is cheap relative to GLM writing
# a whole plan from scratch, since it skips workspace exploration entirely and
# reads only the goal and the plan already produced.
register(
    AgentConfig(
        name="mercury-plan-glm-audit",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        plan_auditor=GLM_FLASH,
        group="mixed",
        description="Mercury plans, GLM audits the plan against the goal and can patch it, "
        "Mercury executes",
    )
)
register(
    AgentConfig(
        name="fix-qwen-plan",
        retired=True,
        planner=QWEN_FLASH,
        executor=MERCURY,
        reviewer=QWEN_FLASH,
        description="Qwen plans and reviews, Mercury executes",
        group="fix",
    )
)

# The reverse pairing. Every mixed config above has Mercury executing, so
# "mixed-phase beats single-model" is confounded with "Mercury is a good
# executor" — this keeps GLM as the constant and swaps who executes, to see
# whether the effect survives moving to the other side of the plan/execute
# split.
register(
    AgentConfig(
        name="mercury-plan-glm",
        planner=MERCURY,
        executor=GLM_FLASH,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_tail_reminder=False,  # the reminder is measured on Mercury only (§106.2)
        group="mixed",
        description="Mercury plans and reviews, GLM executes",
    )
)

# What the last week of work actually bought. Same model throughout, with the
# newest gates switched off one at a time — and both at once, as the floor:
# fix-mercury-nogate's flailing (5 exhausted loops, 12,317 output tokens
# against fix-mercury's 0 and 8,979) could be losing the gate specifically or
# losing feedback generally, and nothing so far tells them apart.
register(
    AgentConfig(
        name="fix-mercury-noreview",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        review=False,
        description="Mercury without phase 3 — prices review",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-mercury-nogate",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        final_check=False,
        description="Mercury without the deterministic final gate",
        group="fix",
    )
)
register(
    AgentConfig(
        name="fix-mercury-raw",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        review=False,
        final_check=False,
        description="Mercury with review and the final gate both off — the floor",
        group="fix",
    )
)


def get(name: str) -> AgentConfig:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}. Available: {', '.join(sorted(CONFIGS))}")
    return CONFIGS[name]


def names() -> list[str]:
    return sorted(CONFIGS)


def in_group(group: str, include_retired: bool = False) -> list[str]:
    """The configs a sweep gets for `group`, retired ones excluded by default.

    A retired config is still `get`-able by name — retiring it says "not
    worth the budget in a routine sweep", not "never run this again".
    """
    return sorted(
        n for n, c in CONFIGS.items() if c.group == group and (include_retired or not c.retired)
    )


def groups() -> list[str]:
    return sorted({c.group for c in CONFIGS.values()})


def retired() -> list[str]:
    return sorted(n for n, c in CONFIGS.items() if c.retired)


# -- perf: what the 2026-09-16 trace's wall clock was actually spent on ----
#
# That run was 441 seconds, 73% of it pytest and 27% the model, so the
# mechanisms worth pricing here are about how often expensive commands run
# rather than how many tokens a prompt costs. Each config below turns one of
# them off against `perf-baseline`, which has both on (the shipped default) —
# the saving is the difference, and if there isn't one, these say so.
#
# Model held constant at Mercury on purpose: the question is what the gate
# and tool-loop changes cost, and varying the model would add spread to a
# comparison that is not about models. GLM is exercised separately.
register(
    AgentConfig(
        name="perf-nosubsume",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        skip_subsumed_verifies=False,
        group="perf",
        description="every step verify re-run at the gate, even one the suite covers",
    )
)

register(
    AgentConfig(
        name="perf-noreplay",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        replay_identical_checks=False,
        group="perf",
        description="identical read-only checks re-run rather than replayed",
    )
)

register(
    AgentConfig(
        name="perf-noverifyskip",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        skip_verified_by_tests=False,
        group="perf",
        description="every step verify re-run even when the step just ran it",
    )
)

register(
    AgentConfig(
        name="perf-noscope",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        scope_step_verify=False,
        group="perf",
        description="a step's whole-suite verify left as the planner wrote it",
    )
)

register(
    AgentConfig(
        name="perf-notypecheck",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        type_check_steps=False,
        group="perf",
        description="no per-step type check — prices catching a broken caller early",
    )
)

register(
    AgentConfig(
        name="perf-neither",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        skip_subsumed_verifies=False,
        replay_identical_checks=False,
        skip_verified_by_tests=False,
        scope_step_verify=False,
        type_check_steps=False,
        group="perf",
        description="both off — the pre-2026-09-16 behaviour, for the total",
    )
)

# Research before planning (agent/research.py, DESIGN-LOG §77): the installed libraries'
# API first, then skills, then the web, and a one-time hold on a plan that looked nothing
# up. On SciCode it holds in 6-11 of 15 runs (§77.2); on in-repo tasks the planner reads
# code first, which counts as a lookup, so it should rarely hold. This measures that.
register(
    AgentConfig(
        name="mercury-research-first",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        research_first=True,
        description="mercury-only with research-first planning",
    )
)

# Run the code on an independently known case before a step finishes (agent/numericcheck.py,
# DESIGN-LOG §81): 139 of 156 failing SciCode sub-steps were wrong numbers (§79.2).
register(
    AgentConfig(
        name="mercury-numeric-check",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        numeric_check=True,
        description="mercury-only with a numeric self-check before each step finishes",
    )
)


# The harness runs the step's tests after each turn that edits and appends the result
# (DESIGN-LOG §90 design C, §99): the "edit, then a turn that only checks" pair in one call.
register(
    AgentConfig(
        name="mercury-check-after-edit",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        check_after_edit=True,
        description="check_after_edit, now the default (§99.3): see mercury-only, and "
        "mercury-no-check-after-edit for the control",
    )
)

# The two runaway fixes adopted in §95.1, each switched off on its own, so a sweep
# can still price them.
register(
    AgentConfig(
        name="mercury-no-output-cap",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_max_tokens=None,
        description="mercury-only without the 16,000-token output cap (§95.1)",
    )
)
register(
    AgentConfig(
        name="mercury-no-stall-bound",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_content_stall_s=None,
        description="mercury-only without the 30 s cut on silence after the first delta (§95.1)",
    )
)

# The pre-§99.3 default, so a sweep can still price check_after_edit.
register(
    AgentConfig(
        name="mercury-no-check-after-edit",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        check_after_edit=False,
        description="mercury-only without the harness's test run after each edit turn",
    )
)


# One fixer loop for all of a round's findings (DESIGN-LOG §100).
register(
    AgentConfig(
        name="mercury-combined-fixes",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        combined_fixes=True,
        description="combined_fixes, now the default (§101): see mercury-only, and "
        "mercury-no-combined-fixes for the control",
    )
)

# The pre-§101 default, so a sweep can still price combined_fixes.
register(
    AgentConfig(
        name="mercury-no-combined-fixes",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        combined_fixes=False,
        description="mercury-only with one fixer loop per finding",
    )
)


# edit_blocks in place of the list-based batch edit tools (DESIGN-LOG §104, measured in §105).
register(
    AgentConfig(
        name="mercury-edit-blocks",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        edit_blocks=True,
        executor_tail_reminder=False,  # §106's arm C; mercury-edit-blocks-tail has both
        description="mercury-only with edit_blocks (SEARCH/REPLACE text) instead of edit_files, "
        "without the reminder",
    )
)

# The after-edit check also reports new type errors (DESIGN-LOG §104, measured in §105).
register(
    AgentConfig(
        name="mercury-types-after-edit",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        types_after_edit=True,
        description="types_after_edit, now the default (§105.2): see mercury-only, and "
        "mercury-no-types-after-edit for the control",
    )
)

# The pre-§105.2 default, so a sweep can still price types_after_edit.
register(
    AgentConfig(
        name="mercury-no-types-after-edit",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        types_after_edit=False,
        description="mercury-only without type errors in the after-edit check",
    )
)

# mercury-only's settings under a name of its own, for §108's stage 2: it runs from the §108
# commit while `mercury-only` runs from the commit before, so `bench report --view compare` can
# tell the two apart. Retired from the start: run by name only, never swept, and not a duplicate.
register(
    AgentConfig(
        name="mercury-s108",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        description="mercury-only's settings, labelled for DESIGN-LOG §108's stage 2",
    )
)

# The reviewer handed ty's report with the suite's, and no run_tests (DESIGN-LOG §108): §108.2's
# arm R, the default since. mercury-no-review-checks is the control.
register(
    AgentConfig(
        name="mercury-review-checks",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        review_presupplied_checks=True,
        description="review_presupplied_checks, now the default (§108.2): see mercury-only, and "
        "mercury-no-review-checks for the control",
    )
)

# The pre-§108.2 reviewer, so a sweep can still price the change.
register(
    AgentConfig(
        name="mercury-no-review-checks",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        review_presupplied_checks=False,
        description="mercury-only with the reviewer running the suite itself (before §108.2)",
    )
)

# A bug fix's own evidence must show the problem going away (DESIGN-LOG §120): the plan's
# reproduction re-run and the run's new tests run on the old code before a fix step finishes.
register(
    AgentConfig(
        name="mercury-bugfix-evidence",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        bugfix_evidence=True,
        description="mercury-only with Settings.bugfix_evidence (§120; dropped in §120.2)",
    )
)

# The defaults with the end-of-context reminder (DESIGN-LOG §106): does the reminder cut turns
# now that it names read_file sections and the harness runs the tests after an edit? Adopted
# in §106.2, so this is now mercury-only; mercury-no-tail is the control.
register(
    AgentConfig(
        name="mercury-tail",
        retired=True,
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_tail_reminder=True,
        description="the reminder, now the default (§106.2): see mercury-only, and "
        "mercury-no-tail for the control",
    )
)

# The pre-§106.2 default, so a sweep can still price the reminder.
register(
    AgentConfig(
        name="mercury-no-tail",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        executor_tail_reminder=False,
        description="mercury-only without the end-of-context batching reminder",
    )
)

# Edit blocks with the end-of-context reminder naming them (DESIGN-LOG §105's follow-up arm).
register(
    AgentConfig(
        name="mercury-edit-blocks-tail",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        edit_blocks=True,
        executor_tail_reminder=True,
        description="mercury-edit-blocks with the end-of-context batching reminder",
    )
)

# read_file without the §104 assists (no widening, no one-line re-read answer): the control.
register(
    AgentConfig(
        name="mercury-no-read-assist",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        read_assist=False,
        description="mercury-only with read_file as before §104",
    )
)


# Mercury through Inception's own API rather than OpenRouter (DESIGN-LOG §85):
# the same model at the same per-token price, paid from Inception's free credit.
# `mercury-inception` should behave as `mercury-only`; `-t05` is the one
# behaviour change, since every run so far has sampled at 1.0 whatever the
# harness asked for (Inception resets anything under 0.5 to 1.0).
register(
    AgentConfig(
        name="mercury-inception",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_provider="inception",
        executor_tail_reminder=False,  # §82-§83's control, before the reminder was a default
        group="provider",
        description="mercury-only with Mercury served by Inception's API instead of OpenRouter, "
        "without the reminder (§82-§83's control)",
    )
)
register(
    AgentConfig(
        name="mercury-inception-t05",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_provider="inception",
        executor_tail_reminder=False,  # one change against mercury-inception
        mercury_temperature=0.5,
        group="provider",
        description="mercury-inception at temperature 0.5, the lowest Inception honours",
    )
)


# Two executor prompt arms for Mercury's recency weighting (DESIGN-LOG §82),
# each one change against `mercury-inception`.
register(
    AgentConfig(
        name="mercury-inception-batchlast",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_provider="inception",
        executor_tail_reminder=False,  # one change against mercury-inception
        executor_batch_last=True,
        group="provider",
        description="mercury-inception with the batching rule last in the executor prompt, "
        "with worked examples",
    )
)
register(
    AgentConfig(
        name="mercury-inception-tail",
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_provider="inception",
        executor_tail_reminder=True,
        group="provider",
        description="mercury-inception with the step goal, batching rule and turns left "
        "appended after the last message on every executor call",
    )
)


# Round two of the tail (DESIGN-LOG §83): each arm is `mercury-inception-tail`
# plus one thing, so each is compared against that config; the output cap is
# compared against `mercury-inception`.
for _name, _items, _plan, _desc in (
    ("errors", ("errors",), False, "plus the last failing test run and its same-failure streak"),
    ("ledger", ("ledger",), False, "plus a running record of this step's edits and test runs"),
    ("plan", (), True, "plus the planner's own tail: goal, turns left, what a plan needs"),
):
    register(
        AgentConfig(
            name=f"mercury-inception-tail-{_name}",
            planner=MERCURY,
            executor=MERCURY,
            reviewer=MERCURY,
            planner_reasoning="high",
            mercury_provider="inception",
            executor_tail_reminder=True,
            executor_tail_items=_items,
            planner_tail=_plan,
            group="provider",
            description=f"mercury-inception-tail {_desc}",
        )
    )
register(
    AgentConfig(
        name="mercury-inception-cap16k",
        retired=True,  # the cap is the default since §95.1, so this equals mercury-inception
        planner=MERCURY,
        executor=MERCURY,
        reviewer=MERCURY,
        planner_reasoning="high",
        mercury_provider="inception",
        executor_tail_reminder=False,  # one change against mercury-inception
        mercury_max_tokens=16_000,
        group="provider",
        description="mercury-inception with Mercury's output capped at 16,000 tokens a call",
    )
)

# §131.6: mechanisms Mercury 2.5 was measured without, re-tried with Mercury 3. Each is its
# Mercury 2.5 namesake with the three models swapped.
for _base in (
    "mercury-direct",
    "mercury-noreview",
    "mercury-plan-probe-required",
    "mercury-exec-medium",  # §131.10: 14 SciCode calls ran to the 16k cap reasoning at high
):
    _cfg = get(_base)
    register(
        replace(
            _cfg,
            name=_base.replace("mercury-", "mercury3-", 1),
            planner=MERCURY_3,
            executor=MERCURY_3,
            reviewer=MERCURY_3,
            group="executor",
            description=f"{_base} with Mercury 3 in every phase (§131.6)",
        )
    )

# §131.9: §27's expected_result gate, off by default since the A/B; these price turning it back on.
for _base, _name in (("mercury-only", "mercury-er-gate"), ("mercury3-only", "mercury3-er-gate")):
    register(
        replace(
            get(_base),
            name=_name,
            expected_result_gate=True,
            group="executor",
            description=f"{_base} with §27's expected_result gate (a missing one retries the step)",
        )
    )

# §131.12: the runaway step-down is on by default since its A/B; this prices turning it off.
register(
    replace(
        get("mercury3-only"),
        name="mercury3-no-step-down",
        runaway_effort_step_down=False,
        group="executor",
        description="mercury3-only with a runaway call repeated, not stepped down (§131.12)",
    )
)

# §131.18: Mercury 3 at low effort in planning and execution, against mercury3-only (high).
register(
    replace(
        get("mercury3-only"),
        name="mercury3-low",
        planner_reasoning="low",
        executor_reasoning="low",
        group="executor",
        description="mercury3-only at low reasoning effort in planning and execution (§131.18)",
    )
)

# §131.16: other models in every phase, on the private fixtures (no model saw them in training).
# Sonnet 5.5 at low reasoning, as in §127.6; the others at mercury-only's settings.
for _name, _model, _low in (
    ("haiku55-only", HAIKU_55, False),
    ("luna-only", GPT6_LUNA, False),
    ("sonnet55-only", SONNET_55, True),
):
    register(
        replace(
            get("mercury-only"),
            name=_name,
            planner=_model,
            executor=_model,
            reviewer=_model,
            planner_reasoning="low" if _low else "high",
            executor_reasoning="low" if _low else "high",
            # OpenAI's strict mode wants every property listed as required; the harness's
            # optional parameters fail it (list_dir's `path`), so Luna runs without strict.
            strict_tool_schemas=_model != GPT6_LUNA,
            group="executor",
            description=f"mercury-only with {_model} in every phase (§131.16)",
        )
    )
