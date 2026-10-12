"""agent/keystore.py: the keyring path through keyring's own API with an in-memory backend (the
OS store behind it is keyring's to test), the no-store cases, and the macOS path, whose store
test against a real temporary keychain is in tests/test_setup.py."""

from __future__ import annotations

import keyring
import keyring.backend
import keyring.backends.fail
import keyring.errors
import pytest

from agent import config, keystore, providers

KEY = "sk-or-v1-abcdefghijklmnop0123"


class MemoryKeyring(keyring.backend.KeyringBackend):
    priority = 1  # a real backend: anything above 0

    def __init__(self):
        super().__init__()
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.items.get((service, username))

    def set_password(self, service, username, password):
        self.items[(service, username)] = password

    def delete_password(self, service, username):
        self.items.pop((service, username), None)


class BrokenKeyring(MemoryKeyring):
    def get_password(self, service, username):
        raise keyring.errors.KeyringLocked("locked")


@pytest.fixture
def not_macos(monkeypatch):
    """The Linux/Windows path, whatever this machine is; keyring's chosen backend restored."""
    monkeypatch.setattr(keystore, "_macos_keychain", lambda: False)
    original = keyring.get_keyring()
    keystore._keyring.cache_clear()
    yield
    keyring.set_keyring(original)
    keystore._keyring.cache_clear()


def use(backend):
    keyring.set_keyring(backend)
    keystore._keyring.cache_clear()
    return backend


def test_store_and_read_through_the_system_keyring(not_macos):
    backend = use(MemoryKeyring())
    where = keystore.store("OpenRouter", KEY)
    assert backend.items == {("OpenRouter", "harness"): KEY}
    assert "system keyring" in where and keystore.secret_store().startswith("the system keyring")
    assert keystore.read("OpenRouter") == KEY
    assert keystore.read("InceptionLabs") == ""


def test_the_harness_reads_keys_from_the_keyring(not_macos, monkeypatch):
    use(MemoryKeyring()).set_password("InceptionLabs", "harness", KEY)
    assert providers.keychain_secret("InceptionLabs") == KEY
    assert config._get_keychain_secret("InceptionLabs", "harness") == KEY


def test_no_store_when_keyring_has_only_its_fail_backend(not_macos):
    """A headless Linux machine: keyring is installed but finds no Secret Service."""
    use(keyring.backends.fail.Keyring())
    assert keystore.secret_store() == ""
    assert keystore.read("OpenRouter") == ""
    with pytest.raises(RuntimeError, match="no secret store"):
        keystore.store("OpenRouter", KEY)


def test_a_locked_store_reads_as_no_key(not_macos):
    use(BrokenKeyring())
    assert keystore.read("OpenRouter") == ""


def test_no_store_without_keyring_installed(not_macos, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "keyring", None)  # import keyring fails
    keystore._keyring.cache_clear()
    assert keystore.secret_store() == ""
    assert keystore.read("OpenRouter") == ""


def test_keys_that_could_break_storage_are_refused(not_macos):
    use(MemoryKeyring())
    with pytest.raises(ValueError):
        keystore.store("OpenRouter", 'bad "key" with spaces')
