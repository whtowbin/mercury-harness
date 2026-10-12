"""Where a model call goes: OpenRouter, or a model's own maker directly.

Every call used to go to OpenRouter. Mercury is also served by Inception's own
API (the same model at the same per-token price, plus a one-time 100M-token free
credit) and by Venice. They speak the same OpenAI wire format but not the same
dialect, measured against each API on 2026-09-29 (DESIGN-LOG §85):

- **Model ids.** The harness names Mercury `inception/mercury-2.5` everywhere
  (configs, traces, results). Inception calls it `mercury-2.5`, Venice
  `mercury-2-5`. The harness id stays the one recorded; only the request says
  the provider's.
- **Reasoning.** OpenRouter takes `reasoning: {"effort": ...}` with `none`
  meaning no reasoning. Inception ignores that object *silently* (the call runs
  at its default, medium) and wants top-level `reasoning_effort`, with
  `instant` for none.
- **Cost.** OpenRouter returns dollars in `usage.cost`; Inception returns only
  tokens, so cost comes from the price table below, or `max_usd` would never trip.
- **Usage when streaming.** Inception sends it only with
  `stream_options: {"include_usage": true}`.
- **OpenRouter-only fields** (`usage.include`, `provider`) are dropped.

A provider is data plus one translation function, so adding one is a table
entry, not a new client: `OpenRouterClient` stays the one transport, with its
retries, deadlines and stall detection.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Models whose temperature the provider restricts, as (low, high). Inception
# silently resets Mercury's temperature outside [0.5, 1.0] to the model default
# (1.0 for mercury-2.5) and says so only in a `warning` field that OpenRouter
# drops. Every temperature the harness asked for (0.0-0.3) was therefore
# running at 1.0 (DESIGN-LOG §85), so leaving an out-of-range value out of the
# request changes nothing but the warning.
TEMPERATURE_RANGES: dict[str, tuple[float, float]] = {"mercury": (0.5, 1.0)}


def is_mercury(model: str) -> bool:
    """The harness's Mercury ids (`inception/mercury-2.5`) and the providers' own
    (`mercury-2.5`, `mercury-2-5`)."""
    return model.startswith("inception/") or model.startswith("mercury")


def effective_temperature(
    model: str, requested: float | None, override: float | None
) -> float | None:
    """The temperature to send. For a model with a restricted range: `override`
    when one is set (Settings.mercury_temperature), otherwise the requested
    value only if the provider would honour it. None means "send none"."""
    family = "mercury" if is_mercury(model) else ""
    if family not in TEMPERATURE_RANGES:
        return requested
    low, high = TEMPERATURE_RANGES[family]
    chosen = override if override is not None else requested
    if chosen is None or not low <= chosen <= high:
        return None
    return chosen


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    cached_input: float
    output: float

    def cost(self, prompt: int, cached: int, completion: int) -> float:
        fresh = max(0, prompt - cached)
        return (fresh * self.input + cached * self.cached_input + completion * self.output) / 1e6


def _openrouter_body(body: dict) -> dict:
    return body


# Inception's reasoning_effort values. OpenRouter's `none` and `minimal` are
# its `instant` (0 reasoning tokens, measured); `xhigh`/`max` do not exist there.
_INCEPTION_EFFORT = {"none": "instant", "minimal": "instant", "xhigh": "high", "max": "high"}


def _inception_body(body: dict) -> dict:
    out = {k: v for k, v in body.items() if k not in ("usage", "provider", "reasoning")}
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        out["reasoning_effort"] = _INCEPTION_EFFORT.get(effort, effort)
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}
    return out


def _venice_body(body: dict) -> dict:
    # Untested: no Venice key yet. Venice documents the OpenAI format, so this is
    # Inception's translation with OpenAI's effort names (`none` passes as is).
    out = {k: v for k, v in body.items() if k not in ("usage", "provider", "reasoning")}
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        out["reasoning_effort"] = effort
    if out.get("stream"):
        out["stream_options"] = {"include_usage": True}
    return out


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    # Keychain item (service, account `harness`) and environment variable the key
    # is read from, in that order.
    key_service: str
    key_env: str
    # harness model id -> the provider's own id. Missing: sent unchanged.
    models: dict[str, str] = field(default_factory=dict)
    translate: Callable[[dict], dict] = _openrouter_body
    # By the provider's own model id. Used only when the response has no cost.
    prices: dict[str, Price] = field(default_factory=dict)
    # Sent only to OpenRouter, which uses them for attribution.
    headers: dict[str, str] = field(default_factory=dict)

    def request(self, body: dict) -> dict:
        """`body` as this provider wants it: its model id, its dialect."""
        out = self.translate(body)
        model = out.get("model", "")
        return {**out, "model": self.models.get(model, model)}

    def cost(self, model: str, prompt: int, cached: int, completion: int) -> float:
        price = self.prices.get(self.models.get(model, model))
        return price.cost(prompt, cached, completion) if price else 0.0

    def api_key(self) -> str:
        return keychain_secret(self.key_service) or os.getenv(self.key_env, "")


def keychain_secret(service: str, account: str = "harness") -> str:
    """A key from this machine's secret store (agent/keystore.py), or ""."""
    from .keystore import read

    return read(service)


OPENROUTER = Provider(
    name="openrouter",
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    key_service="OpenRouter",
    key_env="OPENROUTER_API_KEY",
    headers={
        "HTTP-Referer": "https://localhost/mercury-harness",
        "X-Title": "Mercury dLLM coding harness",
    },
)

INCEPTION = Provider(
    name="inception",
    base_url="https://api.inceptionlabs.ai/v1",
    key_service="InceptionLabs",
    key_env="INCEPTION_API_KEY",
    models={"inception/mercury-2.5": "mercury-2.5", "inception/mercury-2": "mercury-2"},
    translate=_inception_body,
    # docs.inceptionlabs.ai/get-started/models, 2026-09-29 (mercury-2.5 at its
    # listed 80% discount; OpenRouter charges the same).
    prices={
        "mercury-2.5": Price(input=0.04, cached_input=0.004, output=0.15),
        "mercury-2": Price(input=0.25, cached_input=0.025, output=0.75),
    },
)

VENICE = Provider(
    name="venice",
    base_url="https://api.venice.ai/api/v1",
    key_service="Venice",
    key_env="VENICE_API_KEY",
    models={"inception/mercury-2.5": "mercury-2-5"},
    translate=_venice_body,
    # venice.ai/models/mercury-2-5, 2026-09-29: cached input is free.
    prices={"mercury-2-5": Price(input=0.05, cached_input=0.0, output=0.19)},
)

PROVIDERS = {p.name: p for p in (OPENROUTER, INCEPTION, VENICE)}


class RoutingClient:
    """Mercury to `mercury_provider`, every other model to OpenRouter.

    Each provider's client is built on first use, so a run that never calls a
    non-Mercury model never needs an OpenRouter key.
    """

    def __init__(self, mercury_provider: str, build: Callable[[Provider], Any]):
        self.mercury_provider = PROVIDERS[mercury_provider]
        self._build = build
        self._clients: dict[str, Any] = {}

    def client_for(self, model: str) -> Any:
        provider = self.mercury_provider if is_mercury(model) else OPENROUTER
        if provider.name not in self._clients:
            self._clients[provider.name] = self._build(provider)
        return self._clients[provider.name]

    def chat(self, model: str, messages: list[dict], **kwargs: Any) -> Any:
        return self.client_for(model).chat(model, messages, **kwargs)

    def close(self) -> None:
        for client in self._clients.values():
            with contextlib.suppress(Exception):
                client.close()


def build_client(settings: Any, **kwargs: Any) -> Any:
    """The run's model client: plain OpenRouter unless Settings.mercury_provider
    names another provider for Mercury. `kwargs` go to every OpenRouterClient."""
    from .llm import OpenRouterClient

    choice = getattr(settings, "mercury_provider", "openrouter") or "openrouter"
    temperature = getattr(settings, "mercury_temperature", None)
    max_tokens = getattr(settings, "mercury_max_tokens", None)
    content_stall = getattr(settings, "mercury_content_stall_s", None)

    def build(provider: Provider) -> OpenRouterClient:
        key = settings.api_key if provider is OPENROUTER else provider.api_key()
        if not key:
            raise RuntimeError(
                f"no API key for {provider.name}. Run `harness setup --provider "
                f"{provider.name}`, or set {provider.key_env}."
            )
        return OpenRouterClient(
            key,
            provider=provider,
            mercury_temperature=temperature,
            mercury_max_tokens=max_tokens,
            content_stall_s=content_stall,
            **kwargs,
        )

    if choice == "openrouter":
        return build(OPENROUTER)
    if choice not in PROVIDERS:
        raise ValueError(f"unknown mercury provider {choice!r}; one of {sorted(PROVIDERS)}")
    return RoutingClient(choice, build)
