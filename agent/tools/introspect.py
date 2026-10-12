"""Library knowledge: introspection of installed packages, and PyPI lookup.

The most expensive kind of turn is the one spent guessing an API. `inspect_module`
answers "what does this take and what does it return" from the actual installed
package, in one tool call, without the model having to write and run a script.

Introspection happens in a subprocess: importing arbitrary modules runs their
top-level code, and that must not be able to crash or pollute the harness.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from .. import toolchain
from .base import Tool, ToolContext, obj

INSPECT_TIMEOUT = 30
PYPI_TIMEOUT = 10
MAX_MEMBERS = 60

# Runs inside the workspace's interpreter so it sees the workspace's dependencies.
_PROBE = r"""
import importlib, inspect, json, sys, pkgutil

target, member = sys.argv[1], sys.argv[2]
out = {}
try:
    module = importlib.import_module(target)
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(0)

obj = module
path = [target]
if member:
    for part in member.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            available = [n for n in dir(module) if not n.startswith("_")][:40]
            print(json.dumps({"error": f"{'.'.join(path)} has no attribute {part!r}",
                              "available": available}))
            raise SystemExit(0)
        path.append(part)

out["name"] = ".".join(path)
out["kind"] = type(obj).__name__
out["version"] = getattr(module, "__version__", "") or ""
try:
    out["signature"] = str(inspect.signature(obj))
except (TypeError, ValueError):
    out["signature"] = ""
doc = inspect.getdoc(obj) or ""
out["doc"] = doc[:2500]
try:
    out["file"] = inspect.getfile(obj)
except (TypeError, OSError):
    out["file"] = ""

if inspect.ismodule(obj):
    names = [n for n in dir(obj) if not n.startswith("_")]
    out["members"] = names[:200]
    if hasattr(obj, "__path__"):
        out["submodules"] = sorted(m.name for m in pkgutil.iter_modules(obj.__path__))[:80]
elif inspect.isclass(obj):
    methods = []
    for n, v in inspect.getmembers(obj):
        if n.startswith("_") and n != "__init__":
            continue
        try:
            methods.append(f"{n}{inspect.signature(v)}")
        except (TypeError, ValueError):
            methods.append(n)
    out["members"] = methods[:200]

print(json.dumps(out))
"""


def inspect_module(ctx: ToolContext, module: str, member: str = "") -> str:
    """Import a module in a subprocess and report its real signature and docs."""
    # Prefer the workspace's own interpreter so its dependencies are importable.
    argv = [sys.executable] if not shutil.which("uv") else ["uv", "run", "--quiet", "python"]
    try:
        proc = subprocess.run(
            [*argv, "-c", _PROBE, module, member],
            cwd=str(ctx.workspace.root),
            capture_output=True,
            text=True,
            timeout=INSPECT_TIMEOUT,
            env=toolchain.restricted_env(ctx.workspace.root),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"error: importing {module} timed out after {INSPECT_TIMEOUT}s"
    except OSError as exc:
        return f"error: could not run the interpreter: {exc}"

    raw = (proc.stdout or "").strip().splitlines()
    if not raw:
        return f"error: no output from the probe. stderr:\n{(proc.stderr or '')[:600]}"
    try:
        data = json.loads(raw[-1])
    except json.JSONDecodeError:
        return f"error: could not parse the probe output:\n{raw[-1][:400]}"

    ctx.trace.event("inspect_module", module=module, member=member, ok="error" not in data)
    if "error" in data:
        body = f"error: {data['error']}"
        if data.get("available"):
            body += "\nAvailable names: " + ", ".join(data["available"])
        return body

    lines = [f"{data['name']}  ({data['kind']})"]
    if data.get("version"):
        lines.append(f"version: {data['version']}")
    if data.get("signature"):
        lines.append(f"signature: {data['name']}{data['signature']}")
    if data.get("file"):
        lines.append(f"file: {data['file']}")
    if data.get("doc"):
        lines.append(f"\n{data['doc']}")
    if data.get("submodules"):
        lines.append("\nsubmodules: " + ", ".join(data["submodules"]))
    members = data.get("members") or []
    if members:
        shown = members[:MAX_MEMBERS]
        lines.append(f"\nmembers ({len(members)}): " + ", ".join(shown))
        overflow = members[MAX_MEMBERS:]
        if overflow:
            # Never let a name disappear entirely: the model would conclude it does
            # not exist. Drop the signatures for the tail, keep every name.
            names = [m.split("(", 1)[0] for m in overflow]
            lines.append(f"... {len(overflow)} more, names only: " + ", ".join(names))
    return "\n".join(lines)


def pypi_info(ctx: ToolContext, package: str) -> str:
    """Look a package up on PyPI: current version, summary, requires-python, links."""
    name = package.strip().split("[")[0]
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "mercury-harness"})
    try:
        with urllib.request.urlopen(request, timeout=PYPI_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return (
                f"'{name}' is not on PyPI. Check the spelling — the import name often "
                "differs from the package name (e.g. import yaml, package PyYAML)."
            )
        return f"error: PyPI returned HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return f"error: could not reach PyPI ({exc})"

    info = data.get("info") or {}
    ctx.trace.event("pypi_info", package=name, version=info.get("version"))
    lines = [
        f"{info.get('name', name)} {info.get('version', '?')}",
        f"summary: {info.get('summary') or '(none)'}",
        f"requires-python: {info.get('requires_python') or 'unspecified'}",
    ]
    requires = info.get("requires_dist") or []
    if requires:
        hard = [r for r in requires if "extra ==" not in r][:12]
        if hard:
            lines.append("dependencies: " + ", ".join(hard))
    home = info.get("home_page") or (info.get("project_urls") or {}).get("Homepage") or ""
    if home:
        lines.append(f"home: {home}")
    lines.append(f"install with: uv add {name}")
    return "\n".join(lines)


INTROSPECT_TOOLS = [
    Tool(
        "inspect_module",
        "Introspect an installed Python module, class, or function: real signature, docstring, "
        "version, and members. Use this instead of guessing an API or writing a scratch script. "
        "Example: inspect_module('pathlib', 'Path.write_text').",
        obj(
            {
                "module": {"type": "string", "description": "Importable module, e.g. 'json'."},
                "member": {
                    "type": "string",
                    "description": "Dotted attribute inside it, e.g. 'Client.request'.",
                },
            },
            ["module"],
        ),
        inspect_module,
    ),
    Tool(
        "pypi_info",
        "Look up a package on PyPI: latest version, summary, required Python, dependencies. "
        "Use it to confirm a package exists and is maintained before adding it.",
        obj(
            {
                "package": {
                    "type": "string",
                    "description": "Distribution name as it appears on PyPI, e.g. 'pydantic'.",
                }
            },
            ["package"],
        ),
        pypi_info,
    ),
]
