#!/bin/bash
# Stop every benchmark sweep when the OpenRouter balance falls below a floor (default $1.00), so the
# account is never run dry mid-run. Checks every 2 minutes; logs to ~/.cache/mercury-harness/credit-guard.log.
#   scripts/credit_guard.sh [FLOOR_USD]
FLOOR="${1:-1.00}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$HOME/.cache/mercury-harness/credit-guard.log"
while pgrep -f "s126_sweep.sh|s127_claude.sh|iso_s126.py|agents_sci.py|sci.py run|sample.py run" >/dev/null; do
  b=$(cd "$REPO/scripts/swebench" && uv run python -c "import iso; print(iso.balance())" 2>/dev/null | tail -1)
  echo "$(date +%H:%M) balance $b" >> "$LOG"
  if [ -n "$b" ] && python3 -c "import sys; sys.exit(0 if float('$b') < float('$FLOOR') else 1)"; then
    echo "$(date +%H:%M) BALANCE $b BELOW $FLOOR: stopping the sweeps" >> "$LOG"
    pkill -f "s126_sweep.sh"; pkill -f "iso_s126.py"; pkill -f "agents_sci.py"; pkill -f "sci.py run"; pkill -f "sample.py run"
    exit 0
  fi
  sleep 120
done
