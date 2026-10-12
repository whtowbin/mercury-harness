"""Telling a provider failure from an agent failure, in a sweep's records.

A run that dies because the model provider stalled, reset a connection or
returned a 5xx says nothing about the configuration being measured, but it
scores zero and sits in the same table as a run that was wrong. Three sweeps
have now netted this out by hand ("honest below is on completed runs"): 27-31
of 320 runs in DESIGN-LOG §53.2, 32 of 216 in §54.3, and about two-thirds of
each arm in §51 before the stream watchdog was fixed. A table that mixes the two
mostly records who got stalled, and a stall that lands on one block of tasks
looks like a treatment effect.

The classification reads only what a record already carries (`agent_error`,
`timed_out`, `cost_usd`), so it works on every sweep on disk, not just future
ones. What counts as provider-side:

- an `LLMError` from the client's own retry exhaustion ("request failed after N
  attempts"): the agent never got a usable reply, whatever the cause;
- a bench timeout on a run that cost nothing: no model call ever completed, so the
  run was waiting on the provider, not misbehaving. A timeout on a run that spent
  money is a slow run and stays the agent's.

- an account refusal (HTTP 401/402/403): out of credits or a bad key, which fails
  every later run identically until the account is fixed;

What does not: a worker crash, a budget stop, or any error the agent raised
itself.
"""

from __future__ import annotations

from collections import Counter

# Substring -> kind, checked in order against the lower-cased error text. The
# first four are the shapes seen in real sweeps; the last is any other retry
# exhaustion, which is still the provider's for the reason given above.
_KINDS = (
    ("stream open past", "stream_stall"),
    ("connection aborted", "connection"),
    ("connection reset", "connection"),
    ("remotedisconnected", "connection"),
    ("connectionerror", "connection"),
    ("read timed out", "connection"),
    ("provider error", "server_error"),
    ("http 5", "server_error"),
    ("http 429", "rate_limit"),
    ("rate limit", "rate_limit"),
)

# A run that lost this share of its runs to the provider is not comparable with
# one that lost none: the runs that survive a stall are not a random sample.
LOST_SHARE_WARNING = 0.20


# An account failure is not a stall: the request was refused outright, so the client
# did not retry and the error carries no "request failed after". HTTP 402 (out of
# credits, or the in-flight budget exhausted), 401 and 403 (a bad or revoked key)
# say nothing about the configuration being measured, and every run after the first
# one fails the same way until someone fixes the account.
_ACCOUNT_CODES = ("http 401", "http 402", "http 403")


def is_account_failure(record: dict) -> bool:
    error = str(record.get("agent_error") or "")
    lowered = error.lower()
    return error.startswith("LLMError") and any(code in lowered for code in _ACCOUNT_CODES)


# A run whose wall time exceeds its monotonic time by more than this slept part of the
# way through. If it then errored or timed out, the sleep is the cause, not the agent;
# if it finished, its score stands and only its timing is off (DESIGN-LOG §61.2).
SLEPT_LIMIT_S = 30.0


def provider_failure(record: dict) -> str:
    """The kind of provider or environment failure this run died of, or "" if none."""
    error = str(record.get("agent_error") or "")
    lowered = error.lower()
    if float(record.get("slept_s") or 0.0) > SLEPT_LIMIT_S and (error or record.get("timed_out")):
        return "machine_slept"
    if is_account_failure(record):
        return "account"
    if error.startswith("LLMError") and "request failed after" in lowered:
        for needle, kind in _KINDS:
            if needle in lowered:
                return kind
        return "provider_other"
    if (record.get("timed_out") or error.startswith("BenchTimeout")) and _cost(record) == 0:
        return "zero_cost_timeout"
    return ""


def _cost(record: dict) -> float:
    try:
        return float(record.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """(completed, provider-failed), each in the original order."""
    completed: list[dict] = []
    stalled: list[dict] = []
    for record in records:
        (stalled if provider_failure(record) else completed).append(record)
    return completed, stalled


def note(records: list[dict], stalled: list[dict], by: str = "config") -> str:
    """One paragraph saying what was left out, and warning where it matters.

    Empty when nothing was excluded, so a clean sweep prints exactly what it
    printed before.
    """
    if not stalled:
        return ""
    kinds = Counter(provider_failure(r) for r in stalled)
    total = Counter(str(r.get(by, "?")) for r in records)
    lost = Counter(str(r.get(by, "?")) for r in stalled)
    kind_text = ", ".join(f"{kind} {count}" for kind, count in kinds.most_common())
    arm_text = ", ".join(f"{name} {lost[name]}/{total[name]}" for name in sorted(lost))
    lines = [
        f"{len(stalled)} of {len(records)} run(s) died of provider failures and are left out "
        f"of every figure below ({kind_text}).",
        f"Lost per {by}: {arm_text}. Pass --include-stalls to count them as ordinary runs.",
    ]
    if kinds.get("machine_slept"):
        lines.append(
            f"SLEEP: {kinds['machine_slept']} run(s) spent part of their time with the machine "
            "asleep (a closed lid on battery sleeps whatever caffeinate asks). Keep it awake "
            "and on power, and re-run those repeats."
        )
    if kinds.get("account"):
        lines.append(
            f"ACCOUNT: {kinds['account']} run(s) were refused with HTTP 401/402/403 (out of "
            "credits, or a bad key). Every run after the first such refusal is void; fix the "
            "account and re-run those repeats."
        )
    heavy = [
        name
        for name in sorted(lost)
        if total[name] and lost[name] / total[name] > LOST_SHARE_WARNING
    ]
    if heavy:
        lines.append(
            f"WARNING: {', '.join(heavy)} lost more than {LOST_SHARE_WARNING:.0%} of its runs; "
            "the survivors are not a random sample, so read its figures with care."
        )
    return "\n".join(lines) + "\n"
