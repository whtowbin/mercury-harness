"""The OS secret store for API keys: the macOS Keychain, or what `keyring` finds elsewhere.

Every key is an item with service <name> (OpenRouter, InceptionLabs, BraveSearch, TypeSafe) and
account "harness", the names the Keychain items have always had.

- **macOS:** the Keychain, through `/usr/bin/security`. Not through `keyring` even when it is
  installed: a Keychain item remembers the program that created it, and items made by
  `security` (every one this project has ever written) make macOS ask "allow python to
  access?" when another program reads them, which would hang a run nobody is watching.
- **Elsewhere:** `keyring` when the `keyring` extra is installed and a store answers: the
  Secret Service on Linux (GNOME Keyring, KWallet), Credential Manager on Windows. A headless
  Linux machine usually has none; `keyring` then reports its fail backend and there is no store.

With no store, `harness setup` writes an owner-only `credentials.env` instead (agent/setup.py),
which agent/config.py loads into the environment, so the environment variable is always the
second place a key is read from (agent/config.py, agent/providers.py).
"""

from __future__ import annotations

import functools
import platform
import re
import shutil
import subprocess
from typing import Any

ACCOUNT = "harness"
# Stored through `security -i` and in an env file, so no quotes, spaces or newlines. Every
# provider's keys fit.
KEY_SHAPE = re.compile(r"[A-Za-z0-9._~+/=-]{16,}")


def _macos_keychain() -> bool:
    return platform.system() == "Darwin" and shutil.which("security") is not None


@functools.cache
def _keyring() -> Any:
    """The keyring module when it is installed and a real backend answers, else None."""
    try:
        import keyring
    except ImportError:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:  # a broken keyring config reads as no store, never as a crash
        return None
    return keyring if getattr(backend, "priority", 0) > 0 else None


def secret_store() -> str:
    """The name of the store keys go to on this machine, or "" when there is none."""
    if _macos_keychain():
        return "the macOS Keychain"
    keyring = _keyring()
    if keyring is not None:
        return f"the system keyring ({keyring.get_keyring().name})"
    return ""


def read(service: str) -> str:
    """The stored key for `service`, or "" (none stored, no store, or a locked store)."""
    if _macos_keychain():
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-a", ACCOUNT, "-s", service, "-w"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""
    keyring = _keyring()
    if keyring is None:
        return ""
    try:
        return (keyring.get_password(service, ACCOUNT) or "").strip()
    except Exception:  # a locked or unavailable store reads as no key
        return ""


def store_in_macos_keychain(service: str, key: str, keychain: str | None = None) -> None:
    """`security -i` reads the command from stdin, so the key never appears in a process
    listing, which `security add-generic-password -w KEY` would show. `keychain` names a
    keychain file other than the login one (tests use a temporary one)."""
    if not KEY_SHAPE.fullmatch(key):
        raise ValueError("the key has characters a stored key cannot hold")
    command = f'add-generic-password -U -a {ACCOUNT} -s {service} -w "{key}"'
    if keychain:
        command += f' "{keychain}"'
    result = subprocess.run(
        ["security", "-i"], input=command + "\n", capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0 or "error" in (result.stderr + result.stdout).lower():
        raise RuntimeError(f"security failed: {(result.stderr or result.stdout).strip()}")


def store(service: str, key: str) -> str:
    """Put the key in this machine's secret store; returns where, in words. Raises
    RuntimeError when there is no store (the caller falls back to a file)."""
    if not KEY_SHAPE.fullmatch(key):
        raise ValueError("the key has characters a stored key cannot hold")
    if _macos_keychain():
        store_in_macos_keychain(service, key)
        return f'the macOS Keychain (item "{service}", account "{ACCOUNT}")'
    keyring = _keyring()
    if keyring is None:
        raise RuntimeError("no secret store on this machine")
    keyring.set_password(service, ACCOUNT, key)
    return f'{secret_store()} (service "{service}", user "{ACCOUNT}")'
