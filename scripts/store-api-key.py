#!/usr/bin/env python3
"""Helper to securely store your OpenRouter API key in macOS Keychain."""

from __future__ import annotations

import getpass
import subprocess
import sys


def store_key() -> int:
    print("Store your OpenRouter API key in macOS Keychain")
    print("=" * 50)
    key = getpass.getpass("Paste your API key (sk-or-v1-...): ").strip()
    if not key:
        print("error: no key provided", file=sys.stderr)
        return 1
    if not key.startswith("sk-or-v1-"):
        print(
            "warning: key doesn't look like an OpenRouter key (should start with 'sk-or-v1-')",
            file=sys.stderr,
        )
        confirm = input("Continue anyway? [y/n]: ").strip().lower()
        if confirm != "y":
            return 1

    try:
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-a",
                "harness",
                "-s",
                "OpenRouter",
                "-w",
                key,
            ],
            check=True,
        )
        print("✓ API key stored in Keychain")
        print("You can now run: uv run harness ...")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(store_key())
