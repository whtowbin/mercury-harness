#!/bin/bash
# DESIGN-LOG §127: Claude Code as a benchmark agent on the user's subscription, one run a task.
#   scripts/s127_claude.sh scout haiku|sonnet|opus     the problems the other agents solve least reliably
#   scripts/s127_claude.sh full haiku|sonnet|opus      every django task and SciCode dev problem, once
# Serial, with the weekly-allowance guard (scripts/agents/claude.py: stops at CLAUDE_WEEK_CAP).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$HOME/.cache/mercury-harness"
stage="${1:-}"; model="${2:-haiku}"
# haiku|sonnet|opus are Claude Code's aliases; sonnet5 and sonnet5-nothink are Sonnet 5 itself (§127.3)
case "$model" in
  sonnet5)         cli_model="claude-sonnet-5" ;;
  sonnet5-nothink) cli_model="claude-sonnet-5:nothink" ;;
  *)               cli_model="$model" ;;
esac
case "$stage" in
  scout) DJ=11885,11848,11964,11790,12708,11951; SCI="7 29 51 49 70 10" ;;
  full)  DJ=9296,11790,11815,11848,11880,11885,11951,11964,11999,12708; SCI="" ;;
  *) echo "usage: $0 scout|full haiku|sonnet|opus"; exit 2 ;;
esac
tag="s127-$stage-$model"
djtag="s127-scout-$model"  # the full django run adds to the scout's (finished runs are kept)
echo "== django ($djtag): $DJ"
(cd "$REPO/scripts/swebench" && uv run python iso_s126.py run "$djtag-dj" --ids $DJ \
  --arms "claude-$model" --repeats 1 --parallel 1 --max-spend 1000)  # list price is not money here: the allowance guard stops it
echo "== scicode ($tag): ${SCI:-all 15}"
(cd "$REPO/scripts/scicode" && uv run python sci.py setup "$tag-sci" dev $SCI >/dev/null 2>&1 \
  && uv run python claude_sci.py run "$tag-sci" "$cli_model" $SCI \
  && uv run python sci.py score "$tag-sci" > "$CACHE/scicode-runs/$tag-sci.out" 2>&1 \
  && cat "$CACHE/scicode-runs/$tag-sci.out" \
  && uv run python claude_sci.py stats "$tag-sci")
echo "== allowance: $(cat "$CACHE/claude-usage.json")"
