"""Settings' two Keychain-backed API key properties.

The gap that motivated this file: `api_key` silently returned None instead of
the fetched key (a dropped `return key` at the end of the property) for an
entire session's worth of edits, and nothing caught it until a live bench run
hit a 401. Both properties cache into a `_*_cache` field and both have a
"key found" path that must actually return the key, not just cache it.
"""

from __future__ import annotations

import pytest

from agent.config import Settings


def settings_for(**overrides) -> Settings:
    import dataclasses

    return dataclasses.replace(Settings(workspace=".", goal="g"), **overrides)


def test_api_key_returns_the_key_found_in_the_keychain(monkeypatch):
    monkeypatch.setattr(
        "agent.config._get_keychain_secret", lambda service, account: "sk-or-v1-real-key"
    )
    assert settings_for().api_key == "sk-or-v1-real-key"


def test_api_key_falls_back_to_the_env_var(monkeypatch):
    monkeypatch.setattr("agent.config._get_keychain_secret", lambda service, account: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env-key")
    assert settings_for().api_key == "sk-or-v1-env-key"


def test_api_key_raises_a_clear_error_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr("agent.config._get_keychain_secret", lambda service, account: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _ = settings_for().api_key


def test_jev_api_key_returns_the_key_found_in_the_keychain(monkeypatch):
    monkeypatch.setattr(
        "agent.config._get_keychain_secret", lambda service, account: "fake-jev-key"
    )
    assert settings_for().jev_api_key == "fake-jev-key"


def test_jev_api_key_is_empty_string_not_an_exception_when_unconfigured(monkeypatch):
    monkeypatch.setattr("agent.config._get_keychain_secret", lambda service, account: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert settings_for().jev_api_key == ""
