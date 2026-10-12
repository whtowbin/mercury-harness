"""Execution-speed workload for `wordfreq`.

Sorting the whole vocabulary to take the top 10 is the common inefficiency here;
so is rebuilding the counter inside a loop. Both are correct and both are slow.
"""
import json
import random
import sys
import time

from wordfreq import top_n

random.seed(20260909)
VOCAB = [f"word{i}" for i in range(5_000)]
TEXT = " ".join(random.choice(VOCAB) for _ in range(400_000))

started = time.perf_counter()
result = top_n(TEXT, 10)
elapsed = time.perf_counter() - started

if len(result) != 10:
    print(json.dumps({"error": f"expected 10 pairs, got {len(result)}"}))
    sys.exit(1)
counts = [count for _, count in result]
if counts != sorted(counts, reverse=True):
    print(json.dumps({"error": "results are not ordered by descending count"}))
    sys.exit(1)
print(json.dumps({"seconds": elapsed, "workload": "top_n over 400k words"}))
