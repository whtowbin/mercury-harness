"""`harness setup` (agent/setup.py): the key check against a real local HTTP server, storage in a
real temporary keychain and a real file, and the command end to end with the check stubbed."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from agent import keystore, setup, userconfig
from agent.cli import app

REPO = Path(__file__).resolve().parents[1]
GOOD = "sk-or-v1-good0000000000000000"
runner = CliRunner()


class _Handler(BaseHTTPRequestHandler):
    bodies: ClassVar[list[bytes]] = []

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.bodies.append(self.rfile.read(length) if length else b"")
        auth = self.headers.get("Authorization", "")
        status = {f"Bearer {GOOD}": 200, "Bearer sk-or-v1-bad00000000000000": 401}.get(auth, 500)
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b'{"data": {}}' if status == 200 else b'{"error": "nope"}')

    do_GET = do_POST = _answer

    def log_message(self, format: str, *args: object) -> None:  # keep test output quiet
        pass


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/check"
    httpd.shutdown()


@pytest.mark.parametrize("name", ["openrouter", "inception"])
def test_check_key_accepts_rejects_and_cannot_confirm(server, name):
    provider = dataclasses.replace(setup.KEY_PROVIDERS[name], check_url=server)
    assert setup.check_key(provider, GOOD).ok is True
    assert setup.check_key(provider, "sk-or-v1-bad00000000000000").ok is False
    assert setup.check_key(provider, "sk-or-v1-other000000000000").ok is None  # HTTP 500


def test_the_inception_check_is_one_capped_request(server):
    _Handler.bodies.clear()
    provider = dataclasses.replace(setup.KEY_PROVIDERS["inception"], check_url=server)
    setup.check_key(provider, GOOD)
    body = json.loads(_Handler.bodies[-1])
    assert body["max_tokens"] == 1 and len(body["messages"]) == 1


def test_an_unreachable_endpoint_cannot_confirm():
    provider = dataclasses.replace(
        setup.KEY_PROVIDERS["openrouter"], check_url="http://127.0.0.1:9/nothing"
    )
    result = setup.check_key(provider, GOOD, timeout=2)
    assert result.ok is None and "could not reach" in result.detail


def test_store_in_file_is_owner_only_and_replaces_its_own_line(tmp_path):
    path = tmp_path / "config" / "credentials.env"
    path.parent.mkdir()
    path.write_text("OTHER=1\nOPENROUTER_API_KEY=old\n")
    path.chmod(0o644)
    setup.store_in_file(path, "OPENROUTER_API_KEY", GOOD)
    assert path.read_text() == f"OTHER=1\nOPENROUTER_API_KEY={GOOD}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("bad", ['sk-or-v1-"; rm -rf ~', "short", "has space in it 000000"])
def test_keys_that_could_break_storage_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        setup.store_in_file(tmp_path / "c.env", "X", bad)
    with pytest.raises(ValueError):
        keystore.store_in_macos_keychain("HarnessSetupTest", bad)


@pytest.mark.skipif(not keystore._macos_keychain(), reason="needs the macOS security tool")
def test_store_in_a_real_temporary_keychain(tmp_path):
    keychain = str(tmp_path / "test.keychain")
    subprocess.run(["security", "create-keychain", "-p", "pw", keychain], check=True)
    try:
        keystore.store_in_macos_keychain("HarnessSetupTest", GOOD, keychain)
        keystore.store_in_macos_keychain("HarnessSetupTest", GOOD + "x", keychain)  # -U replaces
        find = ["security", "find-generic-password", "-a", "harness", "-s", "HarnessSetupTest"]
        read = subprocess.run(
            [*find, "-w", keychain],
            capture_output=True,
            text=True,
        )
        assert read.stdout.strip() == GOOD + "x"
    finally:
        subprocess.run(["security", "delete-keychain", keychain], check=False)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No real Keychain, environment key or config folder is read or written."""
    monkeypatch.setattr(userconfig, "config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr(keystore, "secret_store", lambda: "")
    monkeypatch.setattr(keystore, "read", lambda service: "")
    for provider in setup.KEY_PROVIDERS.values():
        monkeypatch.delenv(provider.env_var, raising=False)
    return tmp_path / "config"


def test_setup_inception_stores_the_key_and_routes_mercury_there(isolated, monkeypatch):
    monkeypatch.setattr(setup, "check_key", lambda p, k: setup.CheckResult(True, "accepted"))
    result = runner.invoke(app, ["setup", "--provider", "inception"], input=GOOD + "\n")
    assert result.exit_code == 0, result.output
    assert (isolated / "credentials.env").read_text() == f"INCEPTION_API_KEY={GOOD}\n"
    assert json.loads((isolated / "config.json").read_text()) == {"mercury_provider": "inception"}


def test_setup_asks_which_provider(isolated, monkeypatch):
    monkeypatch.setattr(setup, "check_key", lambda p, k: setup.CheckResult(True, "accepted"))
    result = runner.invoke(app, ["setup"], input=f"1\n{GOOD}\n")
    assert result.exit_code == 0, result.output
    assert "OPENROUTER_API_KEY=" in (isolated / "credentials.env").read_text()
    assert not (isolated / "config.json").exists()  # OpenRouter is already the default route


def test_a_rejected_key_is_never_stored(isolated, monkeypatch):
    monkeypatch.setattr(setup, "check_key", lambda p, k: setup.CheckResult(False, "rejected"))
    result = runner.invoke(app, ["setup", "--provider", "openrouter"], input=f"{GOOD}\n" * 3)
    assert result.exit_code == 1
    assert not (isolated / "credentials.env").exists()


def test_config_loads_the_credentials_file_in_a_real_process(tmp_path):
    """A key stored in a file must reach the harness from any folder, as installed: config.py
    loads it at import. HOME moves the per-user config folder on macOS and Linux alike."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env |= {"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / ".config")}
    code = (
        "from agent import userconfig; p = userconfig.credentials_path();"
        "p.parent.mkdir(parents=True, exist_ok=True);"
        "p.write_text('INCEPTION_API_KEY=from-file\\n');"
        "import agent.config, os; print(os.getenv('INCEPTION_API_KEY'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert out.stdout.strip() == "from-file", out.stderr


def test_mercury_provider_is_a_persistable_preference(isolated):
    prefs = userconfig.set_value("mercury_provider", "inception")
    assert prefs.get("mercury_provider") == "inception"


def test_secure_storage_without_a_store_says_how_to_get_one(isolated, monkeypatch):
    """A Linux machine with no keyring installed: --store secure cannot work, and the error
    names the extra that would make it work."""
    import builtins
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    real_import = builtins.__import__

    def no_keyring(name, *a, **k):
        if name == "keyring":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_keyring)
    result = runner.invoke(app, ["setup", "--provider", "openrouter", "--store", "secure"])
    assert result.exit_code == 2
    assert "mercury-harness[keyring]" in " ".join(result.output.split())


def test_setup_stores_in_the_secret_store_when_there_is_one(isolated, monkeypatch):
    stored = {}
    monkeypatch.setattr(keystore, "secret_store", lambda: "the system keyring (Test)")
    monkeypatch.setattr(
        keystore, "store", lambda service, key: stored.update({service: key}) or "the keyring"
    )
    monkeypatch.setattr(setup, "check_key", lambda p, k: setup.CheckResult(True, "accepted"))
    result = runner.invoke(app, ["setup", "--provider", "openrouter"], input=GOOD + "\n")
    assert result.exit_code == 0, result.output
    assert stored == {"OpenRouter": GOOD}
    assert not (isolated / "credentials.env").exists()
