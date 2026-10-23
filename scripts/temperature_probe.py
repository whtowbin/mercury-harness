"""Does Mercury's temperature reach the model through OpenRouter? (DESIGN-LOG §125.1)

Three forced-choice prompts, N samples per arm each, run in parallel; arms are the temperature
sent (none = the harness default, 0.5, 0.0). One JSON line per call, then the modal answer's
share per arm and prompt: a temperature that arrives raises it. 0.0 is outside Inception's
[0.5, 1.0] and is the control that §85 found reset to 1.0.

  uv run python scripts/temperature_probe.py [OUT.jsonl] [N] [MODEL]
"""

import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path.cwd()))
from agent.config import Settings

MODEL = sys.argv[3] if len(sys.argv) > 3 else "inception/mercury-2.5"
URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPTS = {
    "word": "Choose one word at random from: apple, river, stone, cloud, ember, violin, harbor, "
    "lantern. Reply with just the word.",
    "number": "Name one random integer from 1 to 20. Reply with just the number.",
    "file": "A Django bug: a TextChoices model field returns the enum member from a freshly "
    "created instance, so str(instance.field) is 'MyChoice.FIRST' not 'first'. Which one file "
    "would you change? a) db/models/fields/__init__.py b) db/models/enums.py "
    "c) db/models/query.py d) forms/fields.py e) db/models/base.py. Reply with just the letter.",
}
ARMS: dict[str, float | None] = {"default": None, "t1.0": 1.0, "t0.5": 0.5, "t0.0": 0.0}
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/s125-temp-probe.jsonl")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
HDR = {"Authorization": f"Bearer {Settings(workspace=str(Path.cwd())).api_key}"}


def sample(arm: str, temp: float | None, name: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[name]}],
        "max_tokens": 300,
        "reasoning": {"effort": "none"},
    }
    if temp is not None:
        body["temperature"] = temp
    t0 = time.monotonic()
    r = requests.post(URL, json=body, headers=HDR, timeout=120)
    data = r.json()
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return {"prompt": name, "arm": arm, "temp": temp, "text": text, "status": r.status_code,
            "seconds": round(time.monotonic() - t0, 1),
            "warning": data.get("warning") or data.get("warnings")}  # fmt: skip


def main() -> None:
    jobs = [(arm, temp, name) for i in range(N) for name in PROMPTS for arm, temp in ARMS.items()]
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(lambda j: sample(*j), jobs))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("arm      modal share of the most common answer, per prompt (empty answers dropped)")
    for arm in ARMS:
        shares = []
        for name in PROMPTS:
            ans = [
                r["text"].strip().lower().strip(".")
                for r in rows
                if r["arm"] == arm and r["prompt"] == name and r["text"].strip()
            ]
            top = Counter(ans).most_common(1)
            shares.append(f"{name}={top[0][1] / len(ans):.2f} (n={len(ans)})" if top else "none")
        print(f"{arm:8s} " + "  ".join(shares))


if __name__ == "__main__":
    main()
