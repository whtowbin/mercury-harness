"""SciCode as it is officially prompted, one sub-step at a time, with optional candidate voting.

Usage: python3 sample.py run ROUND SPLIT [IDS]    then    python3 sci.py score ROUND

Environment:
  SCI_BACKGROUND=1   include each step's scientist-annotated background (Artificial Analysis's
                     setting; SciCode's "with background")
  SCI_SAMPLES=K      candidates per step (default 1: the bare model, one call a step)
  SCI_FIRST=N        with K > 1, draw N candidates first and stop if they all agree (default 3)
  SCI_EFFORT         reasoning effort (default high, the harness executor's)
  SCI_PARALLEL       problems at once (default 3); SCI_MAX_CALLS concurrent model calls (default 8)

The prompt is SciCode's own multistep template, built exactly as eval/inspect_ai/scicode.py
builds it: earlier steps' descriptions (plus background when on) with this run's own code for
them, then the next step's description and header. Code is extracted the same way.

With K > 1, candidates for a step are drawn in parallel from one prompt, and one more call asks
for a few probe inputs from the step's description and header only (never SciCode's test cases).
Every candidate runs on every probe in one subprocess; candidates whose outputs match are grouped,
and the largest group's first member is kept. If nothing runs, the first candidate is kept, which
is the bare answer. Selection never sees a target value.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sci

from agent.config import Settings
from agent.llm import OpenRouterClient

MODEL = os.environ.get("SCI_MODEL", "inception/mercury-2.5")
BACKGROUND = os.environ.get("SCI_BACKGROUND") == "1"
SAMPLES = int(os.environ.get("SCI_SAMPLES", "1"))
FIRST = int(os.environ.get("SCI_FIRST", "3"))
EFFORT = os.environ.get("SCI_EFFORT", "high")
MAX_TOKENS = 16_000  # the harness's Mercury cap (§95.1)
TEMPLATE = (sci.DATA / "repo" / "eval" / "data" / "multistep_template.txt").read_text()
CALLS = threading.Semaphore(int(os.environ.get("SCI_MAX_CALLS", "8")))
PROBE_TIMEOUT_S = 120

PROBE_PROMPT = """Write example inputs for testing the function below. Use only the description \
and header. Reply with one ```python``` block that defines PROBES, a list of 3 tuples of \
positional arguments: small, valid, varied inputs. You may use these imports:
{deps}

{description}

{header}"""

RUNNER = r"""
import json, sys, signal
import numpy as np
spec = json.load(open(sys.argv[1]))
def sig(value):
    try:
        a = np.asarray(value, dtype=float)
        return "A" + repr(np.round(a, 6).tolist())
    except Exception:
        return "R" + repr(value)[:2000]
out = []
for cand in spec["candidates"]:
    ns = {}
    try:
        signal.alarm(20)
        exec(spec["prefix"] + "\n" + spec["probes"] + "\n" + cand, ns)
        fn = ns[spec["name"]]
        out.append([sig(fn(*args)) for args in ns["PROBES"]])
    except BaseException as exc:
        out.append(None)
    finally:
        signal.alarm(0)
print(json.dumps(out))
"""


def extract(response: str) -> str:
    """SciCode's own extract_python_script: the first python block, imports stripped."""
    if "```" in response:
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        else:
            code = response.split("```")[1].split("```")[0]
    else:
        code = response
    return re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", code, flags=re.MULTILINE)


def describe(step: dict) -> str:
    text = step["step_description_prompt"]
    return text + "\n" + step["step_background"] if BACKGROUND else text


def prompt(problem: dict, n: int, previous: list[str]) -> str:
    """eval/inspect_ai/scicode.py's process_problem_steps + generate_prompt_with_steps."""
    lines = []
    for i in range(n):
        lines += [describe(problem["sub_steps"][i]), previous[i], "------"]
    step = problem["sub_steps"][n]
    header = f"{step['function_header']}\n\n{step['return_line']}"
    return TEMPLATE.format(
        problem_steps_str="\n\n".join(lines[:-1]),
        next_step_str="\n\n".join([describe(step), header]),
        dependencies=problem["required_dependencies"],
    )


def call(client: OpenRouterClient, text: str, log: list) -> str:
    with CALLS:
        reply = client.chat(
            MODEL,
            [{"role": "user", "content": text}],
            max_tokens=MAX_TOKENS,
            reasoning_effort=EFFORT,
        )
    log.append({"cost": reply.usage.cost_usd, "out": reply.usage.completion_tokens})
    if not (reply.content or "").strip():
        raise RuntimeError("empty answer from the model")  # fail loudly on an empty answer
    return reply.content


def run_candidates(prefix: str, probes: str, name: str, candidates: list[str], tmp: Path) -> list:
    spec = tmp / "spec.json"
    spec.write_text(
        json.dumps({"prefix": prefix, "probes": probes, "name": name, "candidates": candidates})
    )
    script = tmp / "runner.py"
    script.write_text(RUNNER)
    try:
        r = sci.sh([str(sci.SCORE_PY), str(script), str(spec)], cwd=tmp, timeout=PROBE_TIMEOUT_S)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        return [None] * len(candidates)


def choose(signatures: list) -> int:
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(signatures):
        if s is not None:
            groups.setdefault(json.dumps(s), []).append(i)
    if not groups:
        return 0
    return max(groups.values(), key=lambda g: (len(g), -g[0]))[0]


def agree(signatures: list) -> bool:
    return None not in signatures and len({json.dumps(s) for s in signatures}) == 1


def vote_step(client, problem, n, previous, text, tmp, log) -> tuple[str, dict]:
    step = problem["sub_steps"][n]
    name = sci_function_name(step["function_header"])
    probe_text = PROBE_PROMPT.format(
        deps=problem["required_dependencies"],
        description=describe(step),
        header=step["function_header"],
    )
    prefix = problem["required_dependencies"] + "\n" + "\n".join(previous)
    with ThreadPoolExecutor(max_workers=FIRST + 1) as pool:
        probe_f = pool.submit(call, client, probe_text, log)
        first = list(pool.map(lambda _: extract(call(client, text, log)), range(FIRST)))
        probes = extract(probe_f.result())
    candidates, sigs = first, run_candidates(prefix, probes, name, first, tmp)
    if not agree(sigs) and SAMPLES > FIRST:
        with ThreadPoolExecutor(max_workers=SAMPLES - FIRST) as pool:
            more = list(
                pool.map(lambda _: extract(call(client, text, log)), range(SAMPLES - FIRST))
            )
        candidates = first + more
        sigs = run_candidates(prefix, probes, name, candidates, tmp)
    pick = choose(sigs)
    groups = sorted({json.dumps(s) for s in sigs if s is not None})
    info = {
        "candidates": len(candidates),
        "ran": sum(s is not None for s in sigs),
        "groups": len(groups),
        "picked": pick,
        "unanimous": agree(sigs),
    }
    return candidates[pick], info


def sci_function_name(header: str) -> str:
    match = re.search(r"\b(?:def|class)\s+(\w+)", header)
    return match.group(1) if match else ""


def run_problem(client: OpenRouterClient, root: Path, problem: dict) -> None:
    pid = str(problem["problem_id"])
    ws = root / pid
    tmp = root / "votes" / pid
    ws.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    previous: list[str] = []
    calls: list[dict] = []
    steps = []
    started = time.time()
    for n in range(len(problem["sub_steps"])):
        text = prompt(problem, n, previous)
        try:
            if SAMPLES > 1:
                code, info = vote_step(client, problem, n, previous, text, tmp, calls)
            else:
                code, info = extract(call(client, text, calls)), {}
        except Exception as exc:  # a lost step is scored as a failure, like a crashed run
            code, info = "", {"error": f"{type(exc).__name__}: {exc}"[:300]}
        previous.append(code)
        steps.append({"step": problem["sub_steps"][n]["step_number"], **info})
    solution = problem["required_dependencies"] + "\n\n" + "\n\n".join(previous)
    (ws / "solution.py").write_text(solution)
    record = {
        "problem": pid,
        "samples": SAMPLES,
        "first": FIRST,
        "background": BACKGROUND,
        "effort": EFFORT,
        "model": MODEL,
        "calls": len(calls),
        "cost_usd": sum(c["cost"] or 0 for c in calls),
        "output_tokens": sum(c["out"] or 0 for c in calls),
        "wall_s": round(time.time() - started, 1),
        "steps": steps,
    }
    (root / f"{pid}.sample.json").write_text(json.dumps(record, indent=1))
    print(f"{pid}: {len(calls)} calls ${record['cost_usd']:.4f}", file=sys.stderr)


def main() -> None:
    cmd, round_name, split, *ids = sys.argv[1:]
    if cmd != "run":
        raise SystemExit(__doc__)
    root = sci.RUNS / round_name
    probs = sci.problems(split)
    chosen = ids or list(probs)
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(workspace=str(root))  # for the API key and the stall bound only
    client = OpenRouterClient(
        api_key=settings.api_key,
        mercury_max_tokens=MAX_TOKENS,
        content_stall_s=settings.mercury_content_stall_s,
    )
    for pid in chosen:  # sci.py score reads these
        (root).mkdir(parents=True, exist_ok=True)
        (root / f"{pid}.goal").write_text(f"sample.py SciCode step-wise run of {pid}\n")
        (root / f"{pid}.split").write_text(split)
    with ThreadPoolExecutor(max_workers=sci.PARALLEL) as pool:
        list(pool.map(lambda pid: run_problem(client, root, probs[pid]), chosen))


if __name__ == "__main__":
    main()
