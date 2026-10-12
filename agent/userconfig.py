"""Persisted user preferences: default models, reasoning, and a user skills directory.

Lives outside the installed package (in the OS's standard per-user config
location), so a preference survives an upgrade or reinstall and so a skill you
add is yours, not a patch to something pip owns. Deliberately small: the knobs
worth not retyping every invocation, not every Settings field — see
PERSISTABLE_FIELDS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs

APP_NAME = "mercury-harness"

# What a user is allowed to persist. Anything else is a per-run choice, not a
# standing preference, and stays a CLI flag only.
PERSISTABLE_FIELDS = {
    "executor_model": str,
    "planner_model": str,
    "reviewer_model": str,
    "executor_reasoning": str,
    "planner_reasoning": str,
    "auto_approve": bool,
    "auto_approve_python": bool,
    # The named trust tier, persisted in preference to the two booleans above
    # — it is the setting people actually think in, and leaving it out meant
    # re-typing --approval-mode every invocation. It is resolved into those
    # booleans at startup, and an explicit flag still overrides it, so there
    # is one source of truth at any moment rather than two competing ones.
    "approval_mode": str,
    "clarify": bool,
    "review_plan": bool,
    "final_check": bool,
    "fix_existing_failures": bool,
    "require_tests": bool,
    "check_coverage": bool,
    "memory": bool,
    "step_concurrency": int,
    "skills_enabled": bool,
    "ask_enabled": bool,
    "cache_prompts": bool,
    "stream": bool,
    "autolint": bool,
    "failure_log_dir": str,
    "narrate": bool,
    "quick_answers": bool,
    "show_stats": bool,
    "animate": bool,
    "spinner_style": str,
    # Where Mercury calls go (agent/providers.py): set by `harness setup` for a user whose
    # only key is Inception's, so no run needs --mercury-provider.
    "mercury_provider": str,
}


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME))


def config_path() -> Path:
    return config_dir() / "config.json"


def credentials_path() -> Path:
    """API keys stored by `harness setup` where there is no Keychain; agent/config.py loads it."""
    return config_dir() / "credentials.env"


def user_skills_dir() -> Path:
    return config_dir() / "skills"


@dataclass
class UserPreferences:
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.values


def load() -> UserPreferences:
    """Read persisted preferences. Any problem reading them means "none set" —
    a corrupt or hand-edited config file should not stop the tool from running."""
    path = config_path()
    if not path.is_file():
        return UserPreferences()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return UserPreferences()
    if not isinstance(data, dict):
        return UserPreferences()
    return UserPreferences(values={k: v for k, v in data.items() if k in PERSISTABLE_FIELDS})


def save(prefs: UserPreferences) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prefs.values, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_value(key: str, raw: str) -> Any:
    """Turn a CLI string into the type this field is stored as."""
    if key not in PERSISTABLE_FIELDS:
        raise ValueError(f"unknown setting {key!r}. Known: {', '.join(sorted(PERSISTABLE_FIELDS))}")
    kind = PERSISTABLE_FIELDS[key]
    if kind is bool:
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{key} is a yes/no setting; got {raw!r}")
    if kind is int:
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{key} must be a whole number; got {raw!r}") from None
    return raw


def set_value(key: str, raw: str) -> UserPreferences:
    prefs = load()
    prefs.values[key] = parse_value(key, raw)
    save(prefs)
    return prefs


def unset_value(key: str) -> UserPreferences:
    prefs = load()
    prefs.values.pop(key, None)
    save(prefs)
    return prefs


def reset() -> None:
    path = config_path()
    if path.is_file():
        path.unlink()
