#!/bin/bash
# DESIGN-LOG §126: the full sandboxed benchmark, one stage at a time (smoke, django, scicode,
# fixtures). Serial on purpose: one provider window, one cost cap. Logs go to
# ~/.cache/mercury-harness/s126-<stage>.log. Run under caffeinate.
#   scripts/s126_sweep.sh smoke|django|scicode|fixtures [extra args]
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$HOME/.cache/mercury-harness"
TREE="$CACHE/iso-harness-323fa64"
GLM="z-ai/glm-5.3-flash"
GLM_FLAGS="--planner $GLM --executor $GLM --reviewer $GLM --provider-sort latency"
GLM_AGENT_MODEL="openrouter/$GLM"
SONNET_ID="anthropic/claude-sonnet-5"
HAIKU_ID="anthropic/claude-haiku-4.5"
IDS=9296,11790,11815,11848,11880,11885,11951,11964,11999,12708
stage="${1:-}"; shift || true

sci_round() {  # ROUND ARM-NAME PROBLEM-IDS...: one SciCode round for one arm
  local round="$1" arm="$2"; shift 2
  (cd "$REPO/scripts/scicode" && uv run python sci.py setup "$round" dev "$@" >/dev/null 2>&1)
  case "$arm" in
    h)      SCI_TREE=$TREE SCI_HARNESS=$REPO/scripts/scicode/sandboxed_harness.sh SCI_MODELS="--all-mercury" ;;
    h-nr)   SCI_TREE=$TREE SCI_HARNESS=$REPO/scripts/scicode/sandboxed_harness.sh SCI_MODELS="--all-mercury --no-review" ;;
    h-glm)  SCI_TREE=$TREE SCI_HARNESS=$REPO/scripts/scicode/sandboxed_harness.sh SCI_MODELS="$GLM_FLAGS" ;;
    h-haiku) SCI_TREE=$TREE SCI_HARNESS=$REPO/scripts/scicode/sandboxed_harness.sh SCI_MAX_USD=${SCI_MAX_USD:-0.5} SCI_MODELS="--planner $HAIKU_ID --executor $HAIKU_ID --reviewer $HAIKU_ID --planner-reasoning none --executor-reasoning none" ;;
    h-sonnet) SCI_TREE=$TREE SCI_HARNESS=$REPO/scripts/scicode/sandboxed_harness.sh SCI_MAX_USD=${SCI_MAX_USD:-0.5} SCI_MODELS="--planner $SONNET_ID --executor $SONNET_ID --reviewer $SONNET_ID --planner-reasoning none --executor-reasoning none" ;;
  esac
  export SCI_TREE SCI_HARNESS SCI_MODELS
  case "$arm" in
    mini|goose) (cd "$REPO/scripts/scicode" && AGENT=$arm uv run python agents_sci.py run "$round" "$@") ;;
    mini-glm|goose-glm|aider-glm) (cd "$REPO/scripts/scicode" && AGENT=${arm%-glm} AGENT_MODEL=$GLM_AGENT_MODEL uv run python agents_sci.py run "$round" "$@") ;;
    *)          (cd "$REPO/scripts/scicode" && uv run python sci.py run "$round" "$@") ;;
  esac
  (cd "$REPO/scripts/scicode" && uv run python sci.py score "$round" > "$CACHE/scicode-runs/$round.out" 2>&1)
  case "$arm" in  # the score's calls and cost columns read harness traces: an agent's own are here
    mini|goose) (cd "$REPO/scripts/scicode" && AGENT=$arm uv run python agents_sci.py stats "$round" >> "$CACHE/scicode-runs/$round.out" 2>&1) ;;
    mini-glm|goose-glm|aider-glm) (cd "$REPO/scripts/scicode" && AGENT=${arm%-glm} AGENT_MODEL=$GLM_AGENT_MODEL uv run python agents_sci.py stats "$round" >> "$CACHE/scicode-runs/$round.out" 2>&1) ;;
  esac
  echo "== $round ($arm)"; cat "$CACHE/scicode-runs/$round.out"
}

case "$stage" in
  smoke)
    echo "== django smoke (11880): h, h-glm, mini, goose"
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s126-smoke --ids 11880 \
      --arms h,h-glm,mini,goose --repeats 1 --parallel 4 --max-spend 0.8)
    echo "== scicode smoke (problem 1): h, h-glm, mini, goose"
    for arm in h h-glm mini goose; do sci_round "s126-smoke-sci-$arm" "$arm" 1; done
    echo "== fixtures smoke (expr): mercury-only"
    uv run python "$REPO/scripts/sandboxed_bench.py" "$TREE" "$CACHE/s126-smoke-fixtures" \
      --configs mercury-only --tasks expr --repeats 1 --timeout 600 -y
    ;;
  django)
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s126-dj --ids $IDS \
      --arms "${ARMS:-h,h-nr,h-glm,mini,goose}" --repeats 3 --parallel "${PAR:-3}" --max-spend "${CAP:-6}" "$@")
    ;;
  direct-django)
    # §129 stage 2: Mercury, default against --direct, all ten django tasks, 3 repeats a side.
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s129-dj --ids $IDS \
      --arms h-new,h-direct --repeats 3 --parallel "${PAR:-4}" --max-spend "${CAP:-1.5}")
    ;;
  direct-lookup)
    # §129.3: --direct against --direct --issue-symbols, all ten django tasks, 3 repeats a side.
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s129-dl --ids $IDS \
      --arms h-d2,h-dsym --repeats 3 --parallel "${PAR:-4}" --max-spend "${CAP:-1.5}")
    ;;
  mercury-regress)
    # §129.5: the default Mercury across three trees, django 10 x 3 each, interleaved.
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s129-rg --ids $IDS \
      --arms h-nr,h-tc,h-new --repeats 3 --parallel "${PAR:-4}" --max-spend "${CAP:-2.5}")
    ;;
  uncapped-haiku)
    # §127.5: the Haiku scout again with a $2 cap a run (the first had $0.60 and every run hit it).
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-2} uv run python iso_s126.py run s127u-haiku \
      --ids 11885,11848,11964,11790,12708,11951 --arms h-haiku --repeats 1 --parallel 3 \
      --max-spend "${CAP:-5}")
    ;;
  uncapped-sonnet5)
    # §127.5: this harness with Sonnet 5 (reasoning off), a $2 cap a run, the six hard django tasks once each.
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-2} uv run python iso_s126.py run s127u-s5 \
      --ids 11885,11848,11964,11790,12708,11951 --arms h-sonnet --repeats 1 --parallel 3 \
      --max-spend "${CAP:-8}")
    ;;
  harness-s55)
    # §127.5: this harness with Sonnet 5.5, a $2 cap a run, the six hard django tasks once each.
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-2} uv run python iso_s126.py run s127u-s55 \
      --ids 11885,11848,11964,11790,12708,11951 --arms h-s55 --repeats 1 --parallel 3 \
      --max-spend "${CAP:-6}")
    ;;
  plan-127)
    # §127.8: Sonnet 5.5 plans and Mercury executes, against GLM planning; six hard tasks, 2 repeats, $2 cap a run.
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-2} uv run python iso_s126.py run s127y-plan \
      --ids 11885,11848,11964,11790,12708,11951 --arms sonnet-plan,glm-plan --repeats 2 --parallel 3 \
      --max-spend "${CAP:-3}")
    ;;
  plan-127-s5)
    # §127.10.2 fallback: Sonnet 5 plans, Mercury executes; same six tasks, 2 repeats, against s127y-plan's glm-plan.
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-2} uv run python iso_s126.py run s127z-plan \
      --ids 11885,11848,11964,11790,12708,11951 --arms sonnet5-plan --repeats 2 --parallel 3 \
      --max-spend "${CAP:-2}")
    ;;
  scicode)
    for rep in 1 2 3; do for arm in h h-nr h-glm mini goose; do
      sci_round "s126-sci-$arm$rep" "$arm" ${PROBLEMS:-}
    done; done
    ;;
  glm-agents-smoke)
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s126g-smoke --ids 11880 \
      --arms mini-glm,goose-glm,aider-glm --repeats 1 --parallel 3 --max-spend 0.5)
    ;;
  glm-agents-django)
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s126g-dj --ids $IDS \
      --arms mini-glm,goose-glm,aider-glm --repeats 3 --parallel "${PAR:-6}" --max-spend "${CAP:-3}" "$@")
    ;;
  glm-agents-scicode-cheap)  # repeats 2-3 for the two cheap agents (Goose costs ~$0.9 a round: one round only)
    for rep in 2 3; do for arm in mini-glm aider-glm; do
      sci_round "s126g-sci-$arm$rep" "$arm" ${PROBLEMS:-}
    done; done
    ;;
  glm-agents-scicode)
    for rep in 1 2 3; do for arm in mini-glm goose-glm aider-glm; do
      sci_round "s126g-sci-$arm$rep" "$arm" ${PROBLEMS:-}
    done; done
    ;;
  harness-sonnet-scout)
    # This harness with Sonnet through OpenRouter (billed), one run on the §127 scout problems.
    for p in 7 29 51 49 70 10; do :; done
    sci_round "s127h-sci-sonnet" h-sonnet 7 29 51 49 70 10
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-0.6} uv run python iso_s126.py run s127h-dj \
      --ids 11885,11848,11964,11790,12708,11951 --arms h-sonnet --repeats 1 --parallel 3 \
      --max-spend "${CAP:-3}")
    ;;
  harness-haiku-scout)
    # This harness with Haiku 4.5 through OpenRouter (billed), one run on the §127 scout problems.
    sci_round "s127h-sci-haiku" h-haiku 7 29 51 49 70 10
    (cd "$REPO/scripts/swebench" && ISO_MAX_USD=${ISO_MAX_USD:-0.6} uv run python iso_s126.py run s127h-dj-haiku \
      --ids 11885,11848,11964,11790,12708,11951 --arms h-haiku --repeats 1 --parallel 3 \
      --max-spend "${CAP:-3}")
    ;;
  issue-symbols-plan)
    # §128 stage 1: Mercury plans only, default against --issue-symbols, 8 plans a side a task.
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s128-plan \
      --ids 11885,11848,11964,11790,12708,11951 --arms p-def,p-sym --repeats 8 \
      --parallel "${PAR:-6}" --max-spend "${CAP:-1.5}")
    ;;
  direct-scout)
    # §129: Mercury, default against --direct, six hard django tasks, 2 repeats a side.
    (cd "$REPO/scripts/swebench" && uv run python iso_s126.py run s129-direct \
      --ids 11885,11848,11964,11790,12708,11951 --arms h-new,h-direct --repeats 2 \
      --parallel "${PAR:-4}" --max-spend "${CAP:-1.5}")
    ;;
  fixtures-direct)
    # §129 guard: the 16 fixtures, Mercury with review off against --direct, 3 repeats a side.
    uv run python "$REPO/scripts/sandboxed_bench.py" "$CACHE/iso-harness-226290d" "$CACHE/s129-fixtures" \
      --configs mercury-noreview,mercury-direct --repeats 3 --parallel 3 --timeout 600 -y "$@"
    ;;
  fixtures)
    uv run python "$REPO/scripts/sandboxed_bench.py" "$TREE" "$CACHE/s126-fixtures" \
      --configs mercury-only,mercury-noreview,glm-only --repeats 3 --parallel 3 --timeout 600 -y "$@"
    ;;
  *) echo "usage: $0 smoke|django|scicode|fixtures"; exit 2 ;;
esac
