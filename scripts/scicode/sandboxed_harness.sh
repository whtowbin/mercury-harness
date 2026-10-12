#!/bin/sh
# SCI_HARNESS for sci.py (DESIGN-LOG §126): the harness of the frozen tree in $SCI_TREE, sandboxed.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
exec uv run --project "$REPO" python "$REPO/scripts/scicode/sandboxed_harness.py" "$@"
