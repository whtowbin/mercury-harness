"""SciCode spike runner: one harness run per main problem, scored per sub-step with
SciCode's own test cases and reference data (never shown to the agent).

Usage: python3 sci.py env | validate SPLIT | setup ROUND SPLIT [IDS] | run ROUND [IDS] | score ROUND
SPLIT is dev or test. Nothing here writes to the harness repository.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runlimits
from webdeny import GITHUB_MIRRORS, SEARCH_DENY

REPO = Path(__file__).resolve().parents[2]
# Problems, reference data (drive/test_data.h5, 1 GB, from the SciCode release) and the scoring
# env live outside git: the problems file holds the gold solutions.
DATA = Path(os.environ.get("SCI_DATA", REPO / "traces" / "scicode-data"))
# Each round's workspaces carry a .venv with numpy/scipy (~1 GB a round): kept out of the repo.
RUNS = Path(os.environ.get("SCI_RUNS", Path.home() / ".cache" / "mercury-harness" / "scicode-runs"))
H5 = DATA / "drive" / "test_data.h5"
SCORE_PY = DATA / "score-env" / "bin" / "python"
HARNESS = os.environ.get("SCI_HARNESS", str(REPO / ".venv" / "bin" / "harness"))
# SciCode's own sources (DESIGN-LOG §80): the answers were fetched from these.
WEB_DENY = ["scicode-bench", "SciCode1/SciCode"]
# Copies of the dataset Mercury 3 reached (§131.5): NeMo Gym's, Hugging Face mirrors, and a
# SciCode repository path anywhere. Not a bare "scicode": problem 51 is SciCode_Example.
WEB_DENY += [
    "nvidia-nemo/gym",
    "huggingface.co/datasets",
    "huggingface.co/api/datasets",
    "datasets-server.huggingface.co",
    "/scicode/",
    *GITHUB_MIRRORS,
    *SEARCH_DENY,
]
AGENT_DEPS = ["numpy", "scipy", "sympy", "matplotlib"]
# Each run's wall limit: about 1.5x the slowest SciCode run on record (15.6 min); §96.2.
RUN_TIMEOUT_S = runlimits.env_seconds("SCI_RUN_TIMEOUT_S", 1200)
PARALLEL = int(
    os.environ.get("SCI_PARALLEL", "3")
)  # Inception free tier: 2-3 concurrent Mercury runs
# Steps whose own gold solution fails SciCode's tests here (validated: 48/50 dev steps
# pass with gold code); scored as neither pass nor fail.
BROKEN_STEPS = {"70.8", "78.3"}
# Test-split steps with no stored target in test_data.h5 (and no gold code to check the scorer
# with: the test split publishes none). Scored as neither pass nor fail.
BROKEN_STEPS |= {"13.6", "76.3", "62.1"}
# Per-step scoring timeout. 600 s for every round so far; Artificial Analysis grades with 300 s
# (dataset v1.0.1), so a run comparable with their SciCode number sets
# SCI_SCORE_TIMEOUT_S=300 (§130).
SCORE_TIMEOUT_S = int(os.environ.get("SCI_SCORE_TIMEOUT_S", "600"))


def problems(split: str) -> dict[str, dict]:
    out = {}
    with open(DATA / f"problems_{split}.jsonl") as f:
        for line in f:
            p = json.loads(line)
            out[str(p["problem_id"])] = p
    return out


def sh(cmd, cwd=None, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def env() -> None:
    """One environment for scoring: the agent's packages plus h5py and scicode."""
    if not SCORE_PY.exists():
        sh(["uv", "venv", "-q", "-p", "3.11", str(DATA / "score-env")])
    e = {**os.environ, "VIRTUAL_ENV": str(DATA / "score-env")}
    r = sh(["uv", "pip", "install", "-q", *AGENT_DEPS, "h5py", "-e", str(DATA / "repo")], env=e)
    print("score env:", "ok" if r.returncode == 0 else r.stderr[-400:])


def step_script(code: str, step: dict) -> str:
    tests = step["test_cases"]
    lines = [
        code,
        "",
        "from scicode.parse.parse import process_hdf5_to_tuple",
        f"targets = process_hdf5_to_tuple({step['step_number']!r}, {len(tests)}, "
        f"h5py_file={str(H5)!r})",
    ]
    for i, test in enumerate(tests):
        lines += [f"target = targets[{i}]", test]
    return "\n".join(lines) + "\n"


def score_code(problem: dict, code: str, scratch: Path) -> list[bool]:
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for step in problem["sub_steps"]:
        if step["step_number"] in BROKEN_STEPS:
            continue
        path = scratch / f"step_{step['step_number']}.py"
        path.write_text(step_script(code, step))
        try:
            r = sh([str(SCORE_PY), str(path)], cwd=scratch, timeout=SCORE_TIMEOUT_S)
            results.append(r.returncode == 0)
        except subprocess.TimeoutExpired:
            results.append(False)
    return results


def gold_code(problem: dict) -> str:
    return (
        problem["required_dependencies"]
        + "\n\n"
        + "\n\n".join(s["ground_truth_code"] for s in problem["sub_steps"])
    )


def validate(split: str) -> None:
    total = passed = 0
    for pid, p in problems(split).items():
        res = score_code(p, gold_code(p), RUNS / "validate" / pid)
        total += len(res)
        passed += sum(res)
        if not all(res):
            print(
                pid,
                "gold fails steps",
                [s["step_number"] for s, ok in zip(p["sub_steps"], res, strict=False) if not ok],
            )
    print(f"gold solutions: {passed}/{total} steps pass")


def goal(p: dict) -> str:
    parts = [
        "Write solution.py, a single Python module implementing the scientific code below. "
        "Hidden tests will import nothing else: they run solution.py's functions with "
        "specific inputs and compare the results numerically, so each function must keep "
        "exactly the name and signature given.",
        "",
        p["problem_description_main"],
        "",
        p["problem_io"],
        "",
        "Use only these imports (put them at the top of solution.py):",
        p["required_dependencies"],
        "",
        "Implement these functions in order; later ones may call earlier ones.",
    ]
    # SCI_BACKGROUND=1 adds each step's scientist-annotated background, the setting Artificial
    # Analysis reports (and SciCode's "with background"); off by default, as every past round ran.
    background = os.environ.get("SCI_BACKGROUND") == "1"
    for i, s in enumerate(p["sub_steps"], 1):
        description = s["step_description_prompt"]
        if background and (s.get("step_background") or "").strip():
            description += "\n\n" + s["step_background"].strip()
        parts += [
            "",
            f"Step {i}: {description}",
            "```python",
            s["function_header"],
            "",
            s["return_line"] or "",
            "```",
        ]
    return "\n".join(parts)


def setup(root: Path, split: str, ids: list[str]) -> None:
    probs = problems(split)
    for pid in ids or list(probs):
        ws = root / pid
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "README.md").write_text(f"SciCode problem {pid}: {probs[pid]['problem_name']}\n")
        if not (ws / ".venv").exists():
            sh(["uv", "venv", "-q", "-p", "3.11", ".venv"], cwd=ws)
            e = {**os.environ, "VIRTUAL_ENV": str(ws / ".venv")}
            sh(["uv", "pip", "install", "-q", *AGENT_DEPS, "pytest"], cwd=ws, env=e)
        (root / f"{pid}.goal").write_text(goal(probs[pid]))
        (root / f"{pid}.split").write_text(split)


def run_one(root: Path, pid: str) -> None:
    ws = root / pid
    with open(root / f"{pid}.log", "w") as log:
        runlimits.run(
            [
                HARNESS,
                "run",
                (root / f"{pid}.goal").read_text(),
                "-w",
                str(ws),
                *os.environ.get("SCI_MODELS", "--all-mercury").split(),
                "-y",
                "--max-usd",
                os.environ.get("SCI_MAX_USD", "2.0"),  # a per-problem spend cap (§109)
                "--no-memory",
                "--no-clarify",
                "--no-ask",  # no one answers a SciCode run (§131.11: 5 ask_user calls in one round)
                "--no-animate",
                *[arg for pat in WEB_DENY for arg in ("--web-deny", pat)],
                "--trace-dir",
                str(root / "traces" / pid),
            ],
            RUN_TIMEOUT_S,
            log=log,
            cwd=ws,
        )


def run(root: Path, ids: list[str]) -> None:
    ids = ids or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "traces")
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        list(pool.map(lambda pid: run_one(root, pid), ids))


def _score_one(root: Path, pid: str) -> list[bool]:
    p = problems((root / f"{pid}.split").read_text())[pid]
    sol = root / pid / "solution.py"
    code = sol.read_text() if sol.exists() else ""
    if code:
        return score_code(p, code, root / "scoring" / pid)
    # No solution: every step fails, but broken steps stay unscored as in score_code (§131: a run
    # with no code for problem 70 was scored 0/8, not 0/7).
    return [False for s in p["sub_steps"] if s["step_number"] not in BROKEN_STEPS]


def score(root: Path) -> None:
    rows = []
    pids = [g.stem for g in sorted(root.glob("*.goal"))]
    # Problems are scored in parallel: each step still runs alone in its own process with the same
    # timeout, so no result changes, but one hung step (600 s) no longer holds up every problem
    # after it (§130: one hang was 10 of an arm's 18 minutes).
    with ThreadPoolExecutor(max_workers=4) as pool:
        scored = dict(zip(pids, pool.map(lambda pid: _score_one(root, pid), pids), strict=True))
    for pid in pids:
        p = problems((root / f"{pid}.split").read_text())[pid]
        res = scored[pid]
        events = [
            json.loads(line)
            for t in (root / "traces" / pid).glob("run-*.jsonl")
            for line in t.read_text().splitlines()
        ]
        calls = [e for e in events if e.get("kind") == "llm_call"]
        wall = (max(e["ts"] for e in events) - min(e["ts"] for e in events)) / 60 if events else 0
        cost = sum(e.get("cost_usd") or 0 for e in calls)
        rows.append((pid, all(res), sum(res), len(res), wall, len(calls), cost))
        print(
            f"{pid:>4} {p['problem_name'][:34]:<34} {'SOLVED' if all(res) else 'not   '} "
            f"steps {sum(res)}/{len(res)}  {wall:5.1f} min {len(calls):4d} calls ${cost:.3f}"
        )
    probs_solved = sum(r[1] for r in rows)
    steps = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(
        f"\nproblems {probs_solved}/{len(rows)}, steps {steps[0]}/{steps[1]}, "
        f"cost ${sum(r[6] for r in rows):.2f}"
    )


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "env":
        env()
    elif cmd == "validate":
        validate(args[0])
    elif cmd == "setup":
        setup(RUNS / args[0], args[1], args[2:])
    elif cmd == "run":
        run(RUNS / args[0], args[1:])
    elif cmd == "score":
        score(RUNS / args[0])
