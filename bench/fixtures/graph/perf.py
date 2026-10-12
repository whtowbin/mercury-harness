"""Execution-speed workload for `graph`.

A grid graph makes BFS's O(V+E) trivial — the shortest path is found almost
immediately. A naive all-paths search (DFS enumerating every route, or BFS that
rebuilds the visited set incorrectly and revisits nodes) is exponential in path
length and will not finish this grid in any reasonable time, which is why the
timeout itself is part of the measurement: a correct-but-naive implementation
should be expected to time out here, not just run slow.
"""
import json
import sys
import time

from graph import Graph

N = 45  # a 45x45 grid: 2025 nodes, shortest path length 89

g = Graph()
for r in range(N):
    for c in range(N):
        if c + 1 < N:
            g.add_edge((r, c), (r, c + 1))
        if r + 1 < N:
            g.add_edge((r, c), (r + 1, c))

started = time.perf_counter()
path = g.shortest_path((0, 0), (N - 1, N - 1))
elapsed = time.perf_counter() - started

expected_len = 2 * (N - 1) + 1
if path is None or len(path) != expected_len:
    got = None if path is None else len(path)
    print(json.dumps({"error": f"wrong path length: {got} != {expected_len}"}))
    sys.exit(1)
print(json.dumps({"seconds": elapsed, "workload": f"shortest_path on a {N}x{N} grid"}))
