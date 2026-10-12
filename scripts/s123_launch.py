"""Run the harness CLI with Settings fields that have no flag (DESIGN-LOG §123).

`preload_neighbours`, `executor_hypothesis` and `force_tools` exist only as bench
`AgentConfig` fields, but `sci.py` and `iso.py` start the harness through its CLI. This
launcher sets them without touching `agent/`:

    S123_SET="preload_neighbours=1,executor_hypothesis=1,force_tools=trace_python+inspect_module" \
        python s123_launch.py run GOAL -w WS ...

`--s123-set SPEC` anywhere in argv does the same and is removed before the CLI sees it.
Values are coerced to the type of the field's default (bool, int, a tuple split on `+`,
or str). An unknown field name stops the run, so a typo cannot run as the control.
Run it with the frozen harness tree's python, so `agent` is that tree's.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any


def parse_overrides(spec: str, settings_cls: Any) -> dict[str, object]:
    fields = {f.name: f for f in dataclasses.fields(settings_cls)}
    out: dict[str, object] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, raw = item.partition("=")
        if name not in fields:
            raise SystemExit(f"s123_launch: unknown Settings field {name!r}")
        default = fields[name].default
        if isinstance(default, bool):
            out[name] = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            out[name] = int(raw)
        elif isinstance(default, tuple):
            out[name] = tuple(filter(None, raw.split("+")))
        else:
            out[name] = raw
    return out


def main() -> None:
    import agent.cli as cli
    import agent.config as config

    spec = os.environ.get("S123_SET", "")
    if "--s123-set" in sys.argv:  # in argv, for runs whose environment is allowlisted
        i = sys.argv.index("--s123-set")
        spec = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    overrides = parse_overrides(spec, config.Settings)

    class OverriddenSettings(config.Settings):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for name, value in overrides.items():
                setattr(self, name, value)

    cli.Settings = OverriddenSettings  # ty: ignore[invalid-assignment]
    print(f"s123_launch: Settings overrides {overrides}", file=sys.stderr)
    cli.main()


if __name__ == "__main__":
    main()
