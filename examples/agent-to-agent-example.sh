#!/bin/bash
# Example: Using one model to generate a plan, another to execute it
# This shows how to use the agent as a CLI for other models to call

set -euo pipefail

WORKSPACE="${1:-.}"

# Example 1: Use a saved plan file
echo "Example 1: Loading a plan from a file"
harness run \
  --workspace "$WORKSPACE" \
  --plan examples/agent-to-agent-plan.json \
  --executor anthropic/claude-opus-5 \
  --max-usd 1.0 \
  --stop-after plan  # Just show the plan, don't execute

# Example 2: Generate a plan with one model, execute with another
# (This requires another model's CLI to be available)
# echo "Example 2: Generated plan from stdin"
# your-model-cli generate-plan "add a sum function" | \
#   harness run \
#     --workspace "$WORKSPACE" \
#     --executor anthropic/claude-opus-5 \
#     --stdin-plan \
#     --max-usd 1.0

# Example 3: Chain multiple runs
# echo "Example 3: Chain runs with feedback"
# PLAN=$(your-model-cli generate-plan "implement quicksort")
#
# SUMMARY=$(echo "$PLAN" | harness run \
#   --workspace "$WORKSPACE" \
#   --executor anthropic/claude-opus-5 \
#   --stdin-plan \
#   --out summary.json)
#
# # Check if successful; if not, generate a new plan with feedback
# if ! echo "$SUMMARY" | jq -e '.ok' > /dev/null; then
#   ERROR=$(echo "$SUMMARY" | jq -r '.error')
#   echo "Run failed: $ERROR"
#   echo "Regenerating plan with feedback..."
#
#   NEW_PLAN=$(your-model-cli generate-plan \
#     "implement quicksort" \
#     --feedback "$ERROR")
#
#   echo "$NEW_PLAN" | harness run \
#     --workspace "$WORKSPACE" \
#     --executor anthropic/claude-opus-5 \
#     --stdin-plan
# fi
