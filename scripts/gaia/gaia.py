"""GAIA runner: one harness run per validation question, scored by GAIA's own quasi-exact match.

Usage: python3 gaia.py fetch | validate | setup ROUND [--level L] [--n N] [IDS] | run ROUND [IDS]
       | score ROUND | leaks ROUND | canary REFERENCE_ROUND ROUND
GAIA_SEARCH=brave puts `run` and `canary` on Brave's API instead of the local SearXNG (§115).
Text-only questions (no attached file) first. The metadata holds the answers, so it
lives outside git, and the benchmark's own pages are denied to the agent (DESIGN-LOG §80).
Nothing here writes to the harness repository.
"""

import argparse
import json
import os
import re
import string
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import runlimits  # noqa: E402  scripts/runlimits.py (DESIGN-LOG §96.2)

# Each run's wall limit: about 1.5x the slowest GAIA run on record (14.0 min); §96.2.
RUN_TIMEOUT_S = runlimits.env_seconds("GAIA_RUN_TIMEOUT_S", 1200)
DATA = Path(os.environ.get("GAIA_DATA", Path.home() / ".cache" / "mercury-harness" / "gaia"))
RUNS = Path(os.environ.get("GAIA_RUNS", Path.home() / ".cache" / "mercury-harness" / "gaia-runs"))
METADATA = DATA / "2023" / "validation" / "metadata.parquet"
HARNESS = os.environ.get("GAIA_HARNESS", str(REPO / ".venv" / "bin" / "harness"))
# The benchmark's home and every mirror of it. Named patterns missed two in scout g2
# (huggingface.co/datasets/<user>/gaia-validation-sample, github.com/harbor-framework/
# harbor-datasets/.../gaia/<task id>; DESIGN-LOG §93), so any URL or code naming GAIA is
# denied. No text-only validation question mentions it.
WEB_DENY = ["gaia", "metadata.jsonl", "metadata.parquet"]
LEAK_QUERY = r"(?i)\bgaia\b"
AGENT_DEPS = ["numpy", "pandas", "sympy", "requests", "pypdf"]
PARALLEL = int(os.environ.get("GAIA_PARALLEL", "4"))
ANSWER_FILE = "answer.txt"


# Derived from GAIA's scorer (scorer.py in the gaia-benchmark/leaderboard Space, Apache-2.0;
# see THIRD_PARTY_NOTICES.md), modified: annotations added, prints removed, helpers
# inlined. Kept to its behaviour: numbers compare as
# floats after dropping $ % and commas, lists element-wise, strings with whitespace and
# punctuation removed.
def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str).lower()
    if remove_punct:
        return no_spaces.translate(str.maketrans("", "", string.punctuation))
    return no_spaces


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def question_scorer(model_answer: str | None, ground_truth: str) -> bool:
    if model_answer is None:
        model_answer = "None"
    if _is_float(ground_truth):
        return normalize_number_str(model_answer) == float(ground_truth)
    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = re.split(r"[,;]", ground_truth)
        ma_elems = re.split(r"[,;]", model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        return all(
            normalize_number_str(ma) == float(gt)
            if _is_float(gt)
            else normalize_str(ma, remove_punct=False) == normalize_str(gt, remove_punct=False)
            for ma, gt in zip(ma_elems, gt_elems, strict=True)
        )
    return normalize_str(model_answer) == normalize_str(ground_truth)


def sh(cmd, cwd=None, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def fetch() -> None:
    """The validation metadata (gated: needs `hf auth login` and accepted terms)."""
    r = sh(
        [
            "uvx",
            "--from",
            "huggingface_hub",
            "hf",
            "download",
            "gaia-benchmark/GAIA",
            "--repo-type",
            "dataset",
            "--include",
            "2023/validation/metadata.parquet",
            "--local-dir",
            str(DATA),
        ]
    )
    print("metadata:", "ok" if METADATA.exists() else r.stderr[-400:])


def questions() -> dict[str, dict]:
    """Text-only validation questions by task id."""
    import duckdb

    rows = duckdb.sql(
        f"select task_id, Question, Level, \"Final answer\" from '{METADATA}' "
        "where coalesce(file_name, '') = '' order by task_id"
    ).fetchall()
    return {r[0]: {"question": r[1], "level": r[2], "answer": r[3]} for r in rows}


def validate() -> None:
    """The scorer against the gold answers, and against answers it must refuse."""
    qs = questions()
    gold = sum(question_scorer(q["answer"], q["answer"]) for q in qs.values())
    wrong = sum(question_scorer("zzz-not-an-answer", q["answer"]) for q in qs.values())
    print(f"gold answers: {gold}/{len(qs)} score; a wrong answer scores {wrong}/{len(qs)}")


def goal(q: dict) -> str:
    return (
        "Answer the question below. Find the facts you need (search the web, fetch pages, "
        "compute with Python), then write only the final answer to answer.txt: no "
        "explanation, no units unless the question asks for them. A number is written "
        "without thousands separators; a list is comma-separated; a string is as few words "
        "as possible, no articles or abbreviations. An empty file or an explanation scores "
        "zero: if the facts cannot be confirmed, write your best estimate.\n\n"
        f"Question: {q['question']}"
    )


def setup(root: Path, ids: list[str], level: str | None, n: int | None) -> None:
    qs = questions()
    chosen = ids or [t for t, q in qs.items() if level is None or q["level"] == level]
    for tid in chosen[:n] if n else chosen:
        ws = root / tid
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "README.md").write_text("A research question; the answer goes in answer.txt.\n")
        if not (ws / ".venv").exists():
            sh(["uv", "venv", "-q", "-p", "3.12", ".venv"], cwd=ws)
            e = {**os.environ, "VIRTUAL_ENV": str(ws / ".venv")}
            sh(["uv", "pip", "install", "-q", *AGENT_DEPS], cwd=ws, env=e)
        (root / f"{tid}.goal").write_text(goal(qs[tid]))


def run_one(root: Path, tid: str, env: dict[str, str]) -> None:
    ws = root / tid
    with open(root / f"{tid}.log", "w") as log:
        runlimits.run(
            [
                HARNESS,
                "run",
                (root / f"{tid}.goal").read_text(),
                "-w",
                str(ws),
                *os.environ.get("GAIA_MODELS", "--all-mercury").split(),
                "-y",
                "--max-usd",
                "1.0",
                "--no-memory",
                "--no-clarify",
                "--no-quick-answers",
                "--no-animate",
                # 8 of g5's 53 misses ended with no answer file (written elsewhere, or never):
                # the final gate gives a failed goal check its fix rounds (DESIGN-LOG §93.3).
                "--verify",
                f"test -s {ANSWER_FILE}",
                *[arg for pat in WEB_DENY for arg in ("--web-deny", pat)],
                "--trace-dir",
                str(root / "traces" / tid),
            ],
            RUN_TIMEOUT_S,
            log=log,
            cwd=ws,
            env=env,
        )


def _keychain(service: str) -> str:
    """A secret from the macOS Keychain (account `harness`), or "". Never printed."""
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-a", "harness", "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def run_env() -> dict[str, str]:
    """The environment every run gets. The harness loads `.env` from its own checkout, and a
    frozen worktree has none, so scout g1 ran with no search tool at all (DESIGN-LOG §93: 0
    web_search calls in 20 runs). The search backend is passed explicitly instead.

    GAIA_SEARCH=brave (DESIGN-LOG §115): every run uses Brave's API. SEARXNG_URL is dropped, so
    no harness, including an older one that predates SEARCH_BACKEND, can fall back to SearXNG,
    and the key (env, the repo's .env, or the Keychain item `-a harness -s BraveSearch`) goes
    to the runs in their environment only, never on a command line."""
    from dotenv import dotenv_values

    env = dict(os.environ)
    dotenv = dotenv_values(REPO / ".env")
    if env.get("GAIA_SEARCH", "").strip().lower() == "brave":
        env.pop("SEARXNG_URL", None)
        env["SEARCH_BACKEND"] = "brave"
        key = env.get("BRAVE_API_KEY") or dotenv.get("BRAVE_API_KEY") or _keychain("BraveSearch")
        if key:
            env["BRAVE_API_KEY"] = key
        return env
    if not env.get("SEARXNG_URL") and not env.get("BRAVE_API_KEY"):
        url = dotenv.get("SEARXNG_URL")
        if url:
            env["SEARXNG_URL"] = url
    return env


def search_live(env: dict[str, str]) -> tuple[bool, str]:
    """The canary search, by the harness's own interpreter with the runs' environment, from a
    directory with no `.env` above it (`python -c` looks for one from its working directory)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    r = sh(
        [
            str(Path(HARNESS).parent / "python"),
            "-c",
            "from agent.tools.web import search_health as s; ok, m = s(); print(ok, m)",
        ],
        cwd=RUNS,
        env=env,
    )
    out = (r.stdout or r.stderr).strip()
    return out.startswith("True"), out


def run(root: Path, ids: list[str]) -> None:
    env = run_env()
    live, why = search_live(env)
    if not live and not os.environ.get("GAIA_ALLOW_NO_SEARCH"):
        sys.exit(f"refusing to run: web search is not live for {HARNESS} ({why})")
    if env.get("SEARCH_BACKEND") == "brave" and "brave" not in why:
        sys.exit(f"refusing to run: GAIA_SEARCH=brave but {HARNESS} searched with: {why}")
    ids = ids or sorted(p.stem for p in root.glob("*.goal"))
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        list(pool.map(lambda tid: run_one(root, tid, env), ids))


def score(root: Path) -> None:
    qs = questions()
    rows = []
    for goal_file in sorted(root.glob("*.goal")):
        tid = goal_file.stem
        q = qs[tid]
        path = root / tid / ANSWER_FILE
        answer = path.read_text().strip() if path.exists() else None
        ok = answer is not None and question_scorer(answer, q["answer"])
        events = [
            json.loads(line)
            for t in (root / "traces" / tid).glob("run-*.jsonl")
            for line in t.read_text().splitlines()
            if line.strip()
        ]
        calls = [e for e in events if e.get("kind") == "llm_call"]
        cost = sum(e.get("cost_usd") or 0 for e in calls)
        wall = (max(e["ts"] for e in events) - min(e["ts"] for e in events)) / 60 if events else 0
        rows.append((tid, q["level"], ok, cost))
        shown = "(none)" if answer is None else answer[:40]
        print(
            f"{tid[:8]} L{q['level']} {'RIGHT' if ok else 'wrong'} {shown!r:<44} "
            f"gold {q['answer'][:30]!r:<32} {wall:5.1f} min {len(calls):3d} calls ${cost:.3f}"
        )
    for level in sorted({r[1] for r in rows}):
        sub = [r for r in rows if r[1] == level]
        print(f"level {level}: {sum(r[2] for r in sub)}/{len(sub)}")
    print(f"\nright {sum(r[2] for r in rows)}/{len(rows)}, cost ${sum(r[3] for r in rows):.2f}")


# Run by the harness's own interpreter, so the backend is chosen exactly as a run would choose
# it. Queries arrive on stdin; one JSON line comes back with the backend's name and, per query,
# the result text or an error. Nothing else is printed, so no secret can reach the output.
# The first version scored titles and whole snippets against a reference that had kept only
# 160 characters of each snippet, which favoured the live side (DESIGN-LOG §115.2).
_CANARY_CODE = r"""
import json, sys, time
from agent.tools import web
queries = json.load(sys.stdin)
backend = web.search_backend()
search = {"searxng": web._searxng, "brave": web._brave}.get(backend)
out = []
for q in queries:
    found = search(q, 7) if search else "error: no search backend"
    if not isinstance(found, str):
        # Scored as the trace records a hit (url and the first TRACE_SNIPPET_CHARS of the
        # snippet, no title), so the reference round's recorded hits compare like for like.
        n = web.TRACE_SNIPPET_CHARS
        found = " ".join(f"{u} {web._strip_tags(s)[:n]}" for t, u, s in found[0])
    out.append(found)
    time.sleep(0.3)
print(json.dumps({"backend": backend, "results": out}))
"""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", str(text).lower())


def _shows(gold: str, text: str) -> bool:
    """Whether `gold` appears in `text`, ignoring case, spaces and punctuation. Golds under two
    characters would match almost anything and never count."""
    g = _norm(gold)
    return len(g) >= 2 and g in _norm(text)


def canary_score(
    reference: Path, ids: list[str], gold: dict[str, str], env: dict[str, str], per: int = 2
) -> dict:
    """Search quality, not liveness (DESIGN-LOG §112.2, §115): replay the first `per` queries the
    `reference` round recorded for each question through the runs' backend, and count questions
    whose gold answer appears in what comes back. The reference's own recorded hits for the same
    queries give the number to compare against. Questions the reference never searched are
    left out."""
    queries: list[str] = []
    owner: list[str] = []
    then: dict[str, bool] = {}
    for tid in ids:
        traces = sorted((reference / "traces" / tid).glob("run-*.jsonl"))
        events = [json.loads(line) for t in traces[:1] for line in t.read_text().splitlines()]
        searches = [e for e in events if e.get("kind") == "web_search"][:per]
        if not searches:
            continue
        recorded = " ".join(
            f"{h.get('url', '')} {h.get('snippet', '')}"
            for e in searches
            for h in e.get("hits") or []
        )
        then[tid] = _shows(gold[tid], recorded)
        for e in searches:
            queries.append(e["query"])
            owner.append(tid)
    r = subprocess.run(
        [str(Path(HARNESS).parent / "python"), "-c", _CANARY_CODE],
        input=json.dumps(queries),
        capture_output=True,
        text=True,
        cwd=RUNS if RUNS.exists() else None,
        env=env,
        timeout=60 + 30 * len(queries),
    )
    if r.returncode != 0:
        raise RuntimeError(f"canary search failed (exit {r.returncode}): {r.stderr[-300:]}")
    data = json.loads(r.stdout.strip().splitlines()[-1])
    text: dict[str, str] = {}
    errors = 0
    for tid, found in zip(owner, data["results"], strict=True):
        errors += found.startswith("error:")
        text[tid] = text.get(tid, "") + " " + found
    return {
        "backend": data["backend"],
        "questions": len(then),
        "queries": len(queries),
        "errors": errors,
        "now": sum(_shows(gold[t], text.get(t, "")) for t in then),
        "reference": sum(then.values()),
    }


def canary(reference: Path, root: Path) -> None:
    ids = sorted(p.stem for p in root.glob("*.goal"))
    qs = questions()
    result = canary_score(reference, ids, {t: qs[t]["answer"] for t in ids}, run_env())
    print(
        f"{result['backend']}: gold shown for {result['now']} of {result['questions']} questions "
        f"({result['queries']} queries, {result['errors']} errors); {reference.name} recorded "
        f"{result['reference']} of {result['questions']} for the same queries"
    )


def leaks(root: Path) -> None:
    subprocess.run(
        [
            HARNESS,
            "trace",
            "leaks",
            str(root / "traces"),
            *[arg for pat in WEB_DENY for arg in ("--deny", pat)],
            "--query",
            LEAK_QUERY,
        ]
    )


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "fetch":
        fetch()
    elif cmd == "validate":
        validate()
    elif cmd == "setup":
        ap = argparse.ArgumentParser(prog="gaia.py setup")
        ap.add_argument("round")
        ap.add_argument("ids", nargs="*")
        ap.add_argument("--level")
        ap.add_argument("--n", type=int)
        a = ap.parse_args(args)
        setup(RUNS / a.round, a.ids, a.level, a.n)
    elif cmd == "run":
        run(RUNS / args[0], args[1:])
    elif cmd == "score":
        score(RUNS / args[0])
    elif cmd == "leaks":
        leaks(RUNS / args[0])
    elif cmd == "canary":
        canary(RUNS / args[0], RUNS / args[1])
