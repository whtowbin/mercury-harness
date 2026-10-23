"""One long, fully specified page of code from one call: how fast, how much reasoning, how correct
(DESIGN-LOG §131.17).

Each arm streams the same prompt (a sectioned calculator module, about 300 lines) with no tools,
one call at a time so that no run shares the provider with another. For every call it records the
time to the first byte, to the first reasoning delta and to the first content delta, the total
time, the tokens (reasoning and visible) and the cost; then it runs the returned module's own
self-test and a hidden test set under sandbox-exec: the benchmark agents' profile (home folder
unreadable), no network, and writes only to its own folder.

  uv run python scripts/long_output_probe.py OUT.jsonl [REPEATS] [ARM,ARM,...]
"""

import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path.cwd()))
from agent.providers import keychain_secret

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents import sandbox_profile

# Mercury 3 at high effort has reasoned past 16k tokens on one call (§131.11, §131.15).
MAX_TOKENS = 40_000
TIMEOUT_S = 600

OPENROUTER = ("https://openrouter.ai/api/v1/chat/completions", "OpenRouter")
INCEPTION = ("https://api.inceptionlabs.ai/v1/chat/completions", "InceptionLabs")

# arm: (route, model id at that route, effort). Mercury's harness default is high (agent/config.py).
ARMS = {
    "m3-or-high": (OPENROUTER, "inception/mercury-3", "high"),
    "m3-or-low": (OPENROUTER, "inception/mercury-3", "low"),
    "m3-inc-high": (INCEPTION, "mercury-3", "high"),
    "m3-inc-low": (INCEPTION, "mercury-3", "low"),
    "m25-or-high": (OPENROUTER, "inception/mercury-2.5", "high"),
    "m25-or-low": (OPENROUTER, "inception/mercury-2.5", "low"),
    "m25-inc-high": (INCEPTION, "mercury-2.5", "high"),
    "haiku55-or-high": (OPENROUTER, "anthropic/claude-haiku-5.5", "high"),
    "haiku55-or-low": (OPENROUTER, "anthropic/claude-haiku-5.5", "low"),
    "sonnet55-or-low": (OPENROUTER, "anthropic/claude-sonnet-5.5", "low"),
}
# Per million tokens in / out, for the direct route, which reports no cost (agent/providers.py).
DIRECT_PRICE = {"mercury-3": (0.10, 0.35), "mercury-2.5": (0.04, 0.15)}

PROMPT = """Write one complete, self-contained Python 3.12 module named calc.py, using the \
standard library only. Reply with the module as a single ```python code block and nothing else.

Organize it in six sections, each starting with a comment header such as `# 1. Tokens`.

0. Before the sections: `class CalcError(ValueError)`, the only exception the module raises on
   bad input.

1. Tokens. A dataclass `Token(kind: str, text: str, pos: int)` and
   `tokenize(src: str) -> list[Token]`.
   Kinds: NUM (integers, decimals such as 3.5 or .5, and exponents such as 1e-3 or 2.5E+2), NAME
   ([A-Za-z_][A-Za-z0-9_]*), OP (one of + - * / % ^ ( ) ,) and a final END token. Whitespace is
   skipped. Any other character raises CalcError naming its position. `pos` is the 0-based index of
   the token's first character.

2. Syntax tree. Dataclasses `Num(value: float)`, `Var(name: str)`, `Unary(op: str, operand)`,
   `Binary(op: str, left, right)` and `Call(name: str, args: list)`, compared by value.

3. Parser. `parse(src: str)` returns the tree. Precedence from lowest to highest:
   + and - (binary, left-associative); * / % (left-associative); unary + and -;
   ^ (right-associative,
   and binding tighter than a unary minus on its left, so -2^2 is -4, while 2^-1 is 0.5);
   primaries: a number, a name, a call name(arg, ...) with zero or more comma-separated arguments,
   or a parenthesized expression. A number parses to Num(float(text)). Any syntax error, including
   empty input and tokens left over after a complete expression, raises CalcError.

4. Evaluator. `evaluate(src: str, env: dict[str, float] | None = None) -> float`. Names come from
   env, plus the constants pi and e (env overrides them). Functions: sin, cos, sqrt, abs (one
   argument each), min and max (one or more), round(x) or round(x, n) using Python's round.
   % is Python's float modulo. An unknown name or function, the wrong number of arguments,
   division or modulo by zero, and sqrt of a negative number raise CalcError, never another
   exception type. Return a float.

5. Printer. `to_source(node) -> str` writes the expression back with the fewest parentheses such
   that parse(to_source(parse(s))) == parse(s) for every valid s.

6. Self-test. Under `if __name__ == "__main__":`, at least 25 assert statements covering every
   section, then print exactly: ALL TESTS PASSED

Give every public function a docstring."""

# Hidden checks, run against the returned module. Values compare with a 1e-9 tolerance.
HIDDEN = r"""
import math, sys
sys.path.insert(0, sys.argv[1])
import calc
ok = fail = 0
def check(name, fn):
    global ok, fail
    try:
        if fn():
            ok += 1
            return
    except Exception as exc:
        print("FAIL", name, type(exc).__name__, exc)
        fail += 1
        return
    print("FAIL", name)
    fail += 1
def val(src, want, env=None):
    check(src, lambda: math.isclose(calc.evaluate(src, env), want, abs_tol=1e-9))
def err(src, env=None):
    def f():
        try:
            calc.evaluate(src, env)
        except calc.CalcError:
            return True
        return False
    check("error " + repr(src), f)
for src, want in [("1+2*3", 7), ("(1+2)*3", 9), ("2^3^2", 512), ("-2^2", -4), ("2^-1", 0.5),
                  ("(-2)^2", 4), ("10 % 4", 2), ("-7 % 3", 2), ("7 - 2 - 1", 4), ("8/2/2", 2),
                  (".5 + 1e-3*1000", 1.5), ("2.5E+2", 250), ("max(1, 5, 3)", 5), ("min(4)", 4),
                  ("round(2.567, 2)", 2.57), ("abs(-3)", 3), ("sqrt(16)", 4), ("sin(0)", 0),
                  ("cos(pi)", -1), ("+-+3", -3), ("--2", 2), ("2*(3+4)^2", 98), ("e", math.e)]:
    val(src, want)
val("x*y+1", 7, {"x": 2, "y": 3})
val("pi", 3, {"pi": 3})
for src in ["1 +", "(1", "1 2", "", "foo(1)", "z", "1/0", "5%0", "sqrt(-1)", "sin(1,2)", "max()",
            "3 $ 4", "round(1,2,3)", "2^", "f(,)"]:
    err(src)
check("CalcError is ValueError", lambda: issubclass(calc.CalcError, ValueError))
KINDS = ["NAME", "OP", "NUM", "END"]
check("tokenize kinds", lambda: [t.kind for t in calc.tokenize("a1 + .5")] == KINDS)
check("tokenize pos", lambda: [t.pos for t in calc.tokenize("a1 + .5")][:3] == [0, 3, 5])
TREE = calc.Unary("-", calc.Binary("^", calc.Num(2.0), calc.Num(2.0)))
check("parse tree", lambda: calc.parse("-2^2") == TREE)
check("parse call", lambda: calc.parse("f()") == calc.Call("f", []))
for src in ["-2^2", "(-2)^2", "2^3^2", "(2^3)^2", "a-(b-c)", "a-b-c", "a/(b*c)", "-(a+b)",
            "f(a, -b)", "2^-1", "--x", "max(1,2)*-3", "(a+b)%c", "a^-b^c"]:
    rt = lambda s=src: calc.parse(calc.to_source(calc.parse(s))) == calc.parse(s)
    check("roundtrip " + src, rt)
def parens(src):
    return calc.to_source(calc.parse(src)).count("(")
check("minimal parens 1", lambda: parens("((1+2))*3") == 1)
check("minimal parens 2", lambda: parens("1+(2*3)") == 0)
check("minimal parens 3", lambda: parens("a-(b+c)") == 1)
check("minimal parens 4", lambda: parens("(a*b)+c") == 0)
print(f"RESULT {ok} {fail}")
"""


def sandbox_rules(d: Path) -> str:
    """The profile the benchmark agents run under (`agents.sandbox_profile`: home folder and the
    session scratch folders unreadable, only `d` writable there), narrowed further for graded
    code: no network, and no writes anywhere but `d` and /dev/null. The Python install stays
    readable. Later rules win, so the additions come last."""
    python = {Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).resolve().parents[1]}
    base = sandbox_profile(rw=[d], ro=sorted(python))
    return base + (
        # `security` and the keychain reach securityd and the Security Server over mach
        '(deny mach-lookup (global-name-prefix "com.apple.security")'
        ' (global-name-prefix "com.apple.SecurityServer"))\n'
        "(deny network*)\n"
        "(deny file-write*)\n"
        f'(allow file-write* (subpath "{d.resolve()}") (literal "/dev/null"))\n'
    )


def stream(arm: str) -> dict[str, object]:
    (url, key_service), model, effort = ARMS[arm]
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    if key_service == "OpenRouter":
        body |= {"reasoning": {"effort": effort}, "usage": {"include": True}}
    else:
        body |= {"reasoning_effort": effort, "stream_options": {"include_usage": True}}
    headers = {"Authorization": f"Bearer {keychain_secret(key_service)}"}
    t0 = time.monotonic()
    first_byte = first_reasoning = first_content = None
    content, usage, finish, provider = [], {}, None, None
    with requests.post(url, json=body, headers=headers, stream=True, timeout=TIMEOUT_S) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            now = time.monotonic() - t0
            first_byte = first_byte if first_byte is not None else now
            line = raw.decode()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            provider = chunk.get("provider") or provider
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if (
                    delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning_details")
                ):
                    first_reasoning = first_reasoning if first_reasoning is not None else now
                if delta.get("content"):
                    first_content = first_content if first_content is not None else now
                    content.append(delta["content"])
                finish = choice.get("finish_reason") or finish
    total = time.monotonic() - t0
    text = "".join(content)
    out_tok = int(usage.get("completion_tokens") or 0)
    reason_tok = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    cost = usage.get("cost")
    if cost is None and model in DIRECT_PRICE:
        p_in, p_out = DIRECT_PRICE[model]
        cost = (int(usage.get("prompt_tokens") or 0) * p_in + out_tok * p_out) / 1e6
    visible_s = total - first_content if first_content is not None else None
    return {
        "arm": arm,
        "model": model,
        "effort": effort,
        "route": key_service,
        "provider": provider,
        "finish": finish,
        "first_byte_s": round(first_byte or -1, 2),
        "first_reasoning_s": None if first_reasoning is None else round(first_reasoning, 2),
        "first_content_s": None if first_content is None else round(first_content, 2),
        "total_s": round(total, 2),
        "out_tok": out_tok,
        "reason_tok": reason_tok,
        "visible_tok": out_tok - reason_tok,
        "visible_tok_s": round((out_tok - reason_tok) / visible_s, 1) if visible_s else None,
        "e2e_tok_s": round(out_tok / total, 1) if total else None,
        "cost": cost,
        "chars": len(text),
        "text": text,
    }


def grade(text: str) -> dict:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[1] if m else text
    with tempfile.TemporaryDirectory() as d:
        Path(d, "calc.py").write_text(code)
        Path(d, "hidden.py").write_text(HIDDEN)
        Path(d, "box.sb").write_text(sandbox_rules(Path(d)))
        box = ["/usr/bin/sandbox-exec", "-f", f"{d}/box.sb", sys.executable, "-I"]

        def run(args: list[str]) -> subprocess.CompletedProcess:
            try:
                return subprocess.run(box + args, cwd=d, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                return subprocess.CompletedProcess(args, -9, "", "timeout")

        selftest = run(["calc.py"])
        hidden = run(["hidden.py", d])
    res = re.search(r"RESULT (\d+) (\d+)", hidden.stdout)
    ok, fail = (int(res[1]), int(res[2])) if res else (0, -1)
    return {
        "lines": code.count("\n") + 1,
        "selftest_pass": selftest.returncode == 0 and "ALL TESTS PASSED" in selftest.stdout,
        "hidden_ok": ok,
        "hidden_total": ok + fail if fail >= 0 else None,
        "hidden_fails": [ln for ln in hidden.stdout.splitlines() if ln.startswith("FAIL")][:8],
        "grader_error": None if res else (hidden.stderr or hidden.stdout)[-400:],
    }


def main() -> None:
    out = Path(sys.argv[1])
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ARMS)
    rng = random.Random(131)
    with out.open("a") as f:
        for rep in range(repeats):
            order = arms[:]
            rng.shuffle(order)
            for arm in order:
                try:
                    row = stream(arm)
                except (
                    Exception
                ) as exc:  # recorded, not fatal: one arm's outage must not end the probe
                    row: dict[str, object] = {"arm": arm, "error": repr(exc)[:400]}
                if row.get("text"):
                    row |= grade(str(row["text"]))
                elif "error" not in row:
                    row["error"] = (
                        "empty answer"  # fail loudly: a reasoning-only reply is not a result
                    )
                row["rep"] = rep
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    arm,
                    rep,
                    {
                        k: row.get(k)
                        for k in (
                            "total_s",
                            "first_content_s",
                            "out_tok",
                            "reason_tok",
                            "cost",
                            "hidden_ok",
                            "selftest_pass",
                            "error",
                        )
                    },
                    flush=True,
                )


if __name__ == "__main__":
    main()
