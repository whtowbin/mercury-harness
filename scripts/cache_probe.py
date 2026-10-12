"""Cache-affinity probe (DESIGN-LOG §121): does prompt_cache_key / user / the Responses API
raise Mercury's cached share on OpenRouter? One growing 5-turn conversation per arm x repeat,
each with a unique nonce at the top of the system prompt so arms cannot warm each other's cache.
Interleaved: repeat r runs the arms in a rotated order. Writes one JSON line per call."""

import json
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path.cwd()))
from agent.config import Settings
from agent.prompts import EXECUTOR_SYSTEM

MODEL = "inception/mercury-2.5"
BASE = "https://openrouter.ai/api/v1"
ARMS = ["a_plain", "b_key", "c_user", "d_resp", "e_resp_key"]
REPEATS = 4
TURNS = 5
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/s121-cache-probe.jsonl")

KEY = Settings(workspace=str(Path.cwd())).api_key
HDR = {"Authorization": f"Bearer {KEY}"}
SRC = [
    Path(p).read_text()[:9000]
    for p in (
        "agent/runner.py",
        "agent/llm.py",
        "agent/loop.py",
        "agent/providers.py",
        "agent/config.py",
    )
]


def conversation(nonce: str, turns: int) -> list[dict]:
    msgs = [
        {"role": "system", "content": f"Session {nonce}.\n\n{EXECUTOR_SYSTEM}"},
        {
            "role": "user",
            "content": "Step: review the harness modules below and say in one "
            "sentence what each does.",
        },
    ]
    for i in range(turns - 1):
        msgs.append({"role": "assistant", "content": f"Next file {i + 1}."})
        msgs.append(
            {
                "role": "user",
                "content": f"File {i + 1}:\n```python\n{SRC[i]}\n```\nOne sentence on it.",
            }
        )
    return msgs


def call(client: requests.Session, arm: str, msgs: list[dict], run_key: str) -> dict:
    t0 = time.monotonic()
    if arm.startswith(("d_", "e_")):
        body = {
            "model": MODEL,
            "input": msgs,
            "max_output_tokens": 300,
            "reasoning": {"effort": "low"},
        }
        if arm == "e_resp_key":
            body["prompt_cache_key"] = run_key
        r = client.post(f"{BASE}/responses", json=body, headers=HDR, timeout=120)
        data = r.json()
        u = data.get("usage") or {}
        prompt = u.get("input_tokens")
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens")
    else:
        body = {
            "model": MODEL,
            "messages": msgs,
            "max_tokens": 300,
            "reasoning": {"effort": "low"},
            "usage": {"include": True},
        }
        if arm == "b_key":
            body["prompt_cache_key"] = run_key
        if arm == "c_user":
            body["user"] = run_key
        r = client.post(f"{BASE}/chat/completions", json=body, headers=HDR, timeout=120)
        data = r.json()
        u = data.get("usage") or {}
        prompt = u.get("prompt_tokens")
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "status": r.status_code,
        "prompt": prompt,
        "cached": cached,
        "latency_s": round(time.monotonic() - t0, 3),
        "cost": u.get("cost"),
        "provider": data.get("provider"),
        "error": data.get("error"),
        "usage": u,
    }


def main() -> None:
    with (
        requests.Session() as client,
        OUT.open("a") as out,
    ):
        for rep in range(REPEATS):
            k = rep % len(ARMS)
            for arm in ARMS[k:] + ARMS[:k]:
                nonce = uuid.uuid4().hex[:12]
                run_key = f"probe-{nonce}"
                for turn in range(1, TURNS + 1):
                    row = call(client, arm, conversation(nonce, turn), run_key)
                    row.update(rep=rep, arm=arm, turn=turn, ts=time.time())
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(
                        arm,
                        rep,
                        turn,
                        row["status"],
                        row["prompt"],
                        row["cached"],
                        row["latency_s"],
                        row["error"] and str(row["error"])[:120],
                    )


if __name__ == "__main__":
    main()
