"""`harness setup`: ask for an OpenRouter or Inception API key, check it, and store it.

Where a key goes: this machine's secret store when it has one (agent/keystore.py: the macOS
Keychain, or the Linux Secret Service or Windows Credential Manager through the `keyring`
extra; item "OpenRouter" or "InceptionLabs", account "harness", which is where agent/config.py
and agent/providers.py look first), else `credentials.env` in the per-user config folder,
readable only by its owner, which agent/config.py loads at startup. Choosing Inception also
persists `mercury_provider = inception`, so Mercury calls go there without a flag on every run.

The key is checked before it is stored: OpenRouter's key endpoint costs nothing; Inception has
no such endpoint (its model list answers without a key, and it validates a request's body
before its key), so the check is one chat request capped at one output token, well under
$0.00001. 401/403 means rejected; any other failure is "could not confirm", and the key can
still be saved.

The logic is here and the prompting in cli.py, so tests drive both against a local HTTP
server, a temporary keychain and a temporary folder.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

import requests

from . import keystore, userconfig


@dataclass(frozen=True)
class KeyProvider:
    name: str
    label: str
    keychain_service: str
    env_var: str
    signup_url: str
    check_url: str
    key_hint: str = ""

    def check_request(self, key: str) -> dict:
        headers = {"Authorization": f"Bearer {key}"}
        if self.name == "openrouter":
            return {"method": "GET", "url": self.check_url, "headers": headers}
        body = {"model": "mercury-2.5", "messages": [{"role": "user", "content": "ok"}]}
        return {
            "method": "POST",
            "url": self.check_url,
            "headers": headers,
            "json": {**body, "max_tokens": 1},
        }


KEY_PROVIDERS = {
    "openrouter": KeyProvider(
        name="openrouter",
        label="OpenRouter: one key for Mercury and any other model (planner, reviewer)",
        keychain_service="OpenRouter",
        env_var="OPENROUTER_API_KEY",
        signup_url="https://openrouter.ai/keys",
        check_url="https://openrouter.ai/api/v1/key",
        key_hint="sk-or-",
    ),
    "inception": KeyProvider(
        name="inception",
        label="Inception: Inception's own API, Mercury models only",
        keychain_service="InceptionLabs",
        env_var="INCEPTION_API_KEY",
        signup_url="https://platform.inceptionlabs.ai",
        check_url="https://api.inceptionlabs.ai/v1/chat/completions",
    ),
}
KEY_SHAPE = keystore.KEY_SHAPE
CHECK_TIMEOUT_S = 15


@dataclass(frozen=True)
class CheckResult:
    ok: bool | None  # True accepted, False rejected, None could not confirm
    detail: str


def check_key(provider: KeyProvider, key: str, timeout: float = CHECK_TIMEOUT_S) -> CheckResult:
    try:
        response = requests.request(**provider.check_request(key), timeout=timeout)
    except requests.RequestException as exc:
        return CheckResult(None, f"could not reach {provider.check_url}: {exc}")
    if response.status_code in (401, 403):
        return CheckResult(False, f"{provider.name} rejected the key (HTTP {response.status_code})")
    if response.ok:
        return CheckResult(True, f"{provider.name} accepted the key")
    return CheckResult(
        None, f"{provider.name} answered HTTP {response.status_code}: {response.text[:200]}"
    )


def store_in_file(path: Path, env_var: str, key: str) -> None:
    """Set `env_var=key` in an env file, keeping its other lines, readable by its owner only."""
    if not KEY_SHAPE.fullmatch(key):
        raise ValueError("the key has characters a stored key cannot hold")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    lines = [line for line in lines if not line.startswith(f"{env_var}=")]
    lines.append(f"{env_var}={key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    path.chmod(0o600)  # an existing file keeps its old mode through os.open


def store_key(provider: KeyProvider, key: str, where: str) -> str:
    """Store the key in the secret store ("secure") or the owner-only file ("file"); returns
    where it went, in words."""
    if where == "secure":
        return keystore.store(provider.keychain_service, key)
    path = userconfig.credentials_path()
    store_in_file(path, provider.env_var, key)
    return f"{path} (readable only by you)"


def keyring_hint() -> str:
    """What to install for a secret store on a machine that has none, or ""."""
    if keystore.secret_store() or platform.system() == "Darwin":
        return ""
    try:
        import keyring  # noqa: F401
    except ImportError:
        return (
            "Install the keyring extra to keep keys in your desktop's secret store (GNOME "
            "Keyring, KWallet) instead of a file: uv tool install "
            "'mercury-harness[keyring] @ git+https://github.com/whtowbin/mercury-harness'"
        )
    return "keyring is installed but found no running secret store (common on servers)."


def existing_key(provider: KeyProvider) -> str:
    """Where a key for this provider is already set, or ""."""
    if keystore.read(provider.keychain_service):
        return keystore.secret_store()
    if os.getenv(provider.env_var):
        sources = f"the environment, a .env file or {userconfig.credentials_path()}"
        return f"{provider.env_var} ({sources})"
    return ""


def use_provider_for_mercury(name: str) -> Path:
    """Persist where Mercury calls go, so no run needs --mercury-provider."""
    prefs = userconfig.load()
    prefs.values["mercury_provider"] = name
    return userconfig.save(prefs)
