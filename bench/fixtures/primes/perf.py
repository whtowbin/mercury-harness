"""Execution-speed workload for `primes`.

count_primes(2_000_000) is the discriminator: a sieve finishes in well under a
second, trial division takes minutes. Both are "correct", which is exactly why
accuracy alone does not tell you which agent produced better code.
"""
import json
import sys
import time

from primes import count_primes

LIMIT = 2_000_000
EXPECTED = 148933  # primes below 2,000,000

started = time.perf_counter()
result = count_primes(LIMIT)
elapsed = time.perf_counter() - started

if result != EXPECTED:
    print(json.dumps({"error": f"wrong result: {result} != {EXPECTED}"}))
    sys.exit(1)
print(json.dumps({"seconds": elapsed, "workload": f"count_primes({LIMIT})"}))
