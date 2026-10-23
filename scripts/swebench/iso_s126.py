"""§126's django suite: the harness arms and the external agents, all isolated and sandboxed.

  python iso_s126.py run s126-dj --ids 9296,11790,11815,11848,11880,11885,11951,11964,11999,12708 \
      --arms h,h-nr,h-glm,mini,goose --repeats 3 --parallel 3 --max-spend 6

Harness arms (`h` = the default of 323fa64, `h-nr` = `--no-review`, `h-glm` = GLM 5.3 Flash in every
phase) run through iso.py's one-commit repositories and its sandbox profile; `mini` and `goose` run
through iso_agents.py, which does the same for them. Every arm is isolated the same way, so no arm
can read django's history, the answer key or the home folder.
"""

import sys

import iso
import iso_agents

SHA = "323fa64"
GLM = "z-ai/glm-5.3-flash"
GLM_FLAGS = ["--planner", GLM, "--executor", GLM, "--reviewer", GLM, "--provider-sort", "latency"]
iso.ARMS["h"] = (SHA, [])
iso.ARMS["h-nr"] = (SHA, ["--no-review"])
iso.ARMS["h-glm"] = (SHA, GLM_FLAGS)
iso.ARMS["h-idle"] = (SHA, ["--stop-after-idle-fix-round"])
# §131: Mercury 3 (released 2026-10-19) on `h`'s tree, so the model is the only difference from
# §126's numbers. --all-mercury stays and copies the executor into every phase.
M3 = "inception/mercury-3"
iso.ARMS["h-m3"] = (SHA, ["--executor", M3])
# §131.10: today's tree with §131.9-131.10's fixes (review off by default since §124.10), the
# preflight before the full run: Mercury 2.5, Mercury 3, and Mercury 3 with the gate off.
TODAY = "1ded9df"
iso.ARMS["t2"] = (TODAY, [])
iso.ARMS["t3"] = (TODAY, ["--executor", M3])
iso.ARMS["t3-nog"] = (TODAY, ["--executor", M3, "--no-expected-result-gate"])
# §131.12: the low-effort step-down A/B, on the tree that has it (gate off by default there).
STEP_TREE = "c39cea0"
iso.ARMS["n3"] = (STEP_TREE, ["--executor", M3])
iso.ARMS["n3-sd"] = (STEP_TREE, ["--executor", M3, "--runaway-step-down"])
# §131.13: the full run, on the tree with §131.9 and §131.12's defaults (gate off, step-down on).
FULL_TREE = "17bf856"
iso.ARMS["f2"] = (FULL_TREE, [])
iso.ARMS["f3"] = (FULL_TREE, ["--executor", M3])
# §131.16: Inception's comparison models, same tree, in every phase (--all-mercury copies the
# executor into every phase, whatever the model).
iso.ARMS["f-haiku55"] = (FULL_TREE, ["--executor", "anthropic/claude-haiku-5.5"])
iso.ARMS["f-luna"] = (FULL_TREE, ["--executor", "openai/gpt-6-luna"])
# §131.18: Mercury 3 at low effort in planning and execution, against f3 (high) on the same tree.
LOW = ["--executor-reasoning", "low", "--planner-reasoning", "low"]
iso.ARMS["f3-low"] = (FULL_TREE, ["--executor", M3, *LOW])
# Plan-only arms for §128: Mercury plans, nothing is executed or scored.
# Both arms run the same newer tree (84f3e19, which has --issue-symbols), so only the flag differs.
PLAN_TREE = "84f3e19"
iso.ARMS["p-def"] = (PLAN_TREE, ["--stop-after", "plan"])
iso.ARMS["p-sym"] = (PLAN_TREE, ["--stop-after", "plan", "--issue-symbols"])
# §129: default against --direct on one tree (8c845de), so the tool-schema change is in both arms.
DIRECT_TREE = "8c845de"
iso.ARMS["h-new"] = (DIRECT_TREE, [])
iso.ARMS["h-direct"] = (DIRECT_TREE, ["--direct"])
# §129.3: a direct step handed the files where the goal's names are defined; both on tree f6d1a77.
LOOKUP_TREE = "f6d1a77"
iso.ARMS["h-d2"] = (LOOKUP_TREE, ["--direct"])
iso.ARMS["h-dsym"] = (LOOKUP_TREE, ["--direct", "--issue-symbols"])
# §129.5 control: did the default Mercury score fall between trees? Review became off by default
# between 323fa64 and 8c845de (9a94bc3), so the older trees run with --no-review to match: h-nr
# (323fa64), h-tc (aa2b378, the toolchain timeout change only) and h-new (8c845de, adds the
# tool-schema diet), interleaved.
iso.ARMS["h-tc"] = ("aa2b378", ["--no-review"])
# §127.5: Sonnet 5.5 in the harness (B4 lets it run), on a tree with the refusal handling.
S55 = "anthropic/claude-sonnet-5-5"
S55_TREE = "226290d"
iso.ARMS["h-s55"] = (
    S55_TREE,
    [
        *("--planner", S55, "--executor", S55, "--reviewer", S55),
        *("--planner-reasoning", "low", "--executor-reasoning", "low"),
    ],
)
# §127.8: the day-one design, a strong model plans and Mercury executes, against GLM planning
# (§94's glm-plan) on one tree. Only the planner and its reasoning differ; review is off (the
# default of this tree), Mercury executes and would review, and both use §77's latency sort.
# On 5504418 Sonnet 5.5 refused every plan call (§127.9) and the run fell back to Mercury alone;
# 37e51e6 retries a refused plan call without the reasoning hand-off wording (§127.11).
PLAN_TREE_127 = "37e51e6"
MERCURY = "inception/mercury-2.5"
_PLANNED = ["--executor", MERCURY, "--reviewer", MERCURY]
# No latency sort for Sonnet (§127.10.2): it routes Sonnet 5.5 to Azure, which refuses the plan
# request; the harness has no provider order or ignore list, so it takes OpenRouter's default.
iso.ARMS["sonnet-plan"] = (
    PLAN_TREE_127,
    ["--planner", S55, "--planner-reasoning", "low", *_PLANNED],
)
# The author's fallback (§127.10.2): Sonnet 5.5 refused 9 of 12 plan runs on AWS too, so Sonnet 5
# plans instead, at reasoning off as in §127.7 (where it planned in 6 of 6), same routing.
iso.ARMS["sonnet5-plan"] = (
    PLAN_TREE_127,
    ["--planner", "anthropic/claude-sonnet-5", "--planner-reasoning", "none", *_PLANNED],
)
iso.ARMS["glm-plan"] = (
    PLAN_TREE_127,
    ["--planner", GLM, *_PLANNED, "--provider-sort", "latency"],
)
PLAN_ONLY_ARMS = {"p-def", "p-sym"}
_score = iso.score_state


def score_state(ws, d):
    """A plan-only run has no edit to score: skip the hidden tests."""
    if ws.parent.name.rsplit("-", 1)[-1] in PLAN_ONLY_ARMS:  # <prefix>-r<k>-<arm>/<task>
        return False, "plan-only"
    return _score(ws, d)


iso.score_state = score_state  # ty: ignore[invalid-assignment]
SONNET = "anthropic/claude-sonnet-5"  # through OpenRouter, billed there (§127.1)
# Reasoning off: with thinking on, Anthropic's API refuses the forced `tool_choice` the harness
# sends ("tool_choice: type tool and any are not supported for this model"); the first Sonnet run
# died on it.
SONNET_FLAGS = ["--planner", SONNET, "--executor", SONNET, "--reviewer", SONNET]
SONNET_FLAGS += ["--planner-reasoning", "none", "--executor-reasoning", "none"]
iso.ARMS["h-sonnet"] = (SHA, SONNET_FLAGS)
HAIKU = "anthropic/claude-haiku-4.5"  # the same reasoning-off setting as the Sonnet arm (§127.4)
iso.ARMS["h-haiku"] = (
    SHA,
    [
        "--planner",
        HAIKU,
        "--executor",
        HAIKU,
        "--reviewer",
        HAIKU,
        "--planner-reasoning",
        "none",
        "--executor-reasoning",
        "none",
    ],
)

_argv = iso.harness_argv


def harness_argv(tree, goal, ws, traces, flags):
    argv = _argv(tree, goal, ws, traces, flags)
    if "--planner" in flags:  # GLM in every phase: --all-mercury would override the models
        argv.remove("--all-mercury")
    return argv


iso.harness_argv = harness_argv  # ty: ignore[invalid-assignment]
iso_agents.install()

if __name__ == "__main__":
    sys.argv[0] = "iso_s126.py"
    iso.main()
