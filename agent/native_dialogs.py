"""Native OS folder-picker dialogs, for a surface (the TUI) that wants a
familiar system picker instead of a typed path.

Deliberately kept separate from the TUI itself: a subprocess call to a GUI
tool is exactly the kind of thing that needs mocking in tests rather than
exercised for real (see tests/test_native_dialogs.py), and keeping it in its
own module with no Textual import means the TUI's own tests never risk a
real dialog popping up during a test run.

No timeout on the subprocess call: the whole point is waiting on a human to
click something, and a picker that can be open for as long as they need is
not a hang. Cancelling the dialog and no native picker being available on
this platform are treated identically — both return None — since neither is
an error the caller needs to distinguish from the other; the TUI's own
manual-path Input field is the fallback for both.
"""

from __future__ import annotations

import platform

from . import toolchain


def choose_folder(prompt: str = "Select a folder") -> str | None:
    """Open the OS's native folder picker and return the chosen absolute
    path, or None if the user cancelled, the dialog failed, or this
    platform has no native picker this module knows how to drive.
    """
    system = platform.system()
    if system == "Darwin":
        return _choose_folder_macos(prompt)
    if system == "Linux":
        return _choose_folder_linux(prompt)
    return None


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _choose_folder_macos(prompt: str) -> str | None:
    """`choose folder` is AppleScript's own native folder picker — the exact
    dialog Finder and every Mac app use, not a custom recreation of one.
    Cancelling it is AppleScript error -128, which osascript reports as a
    non-zero exit; that is not distinguished from any other failure here,
    per the module docstring."""
    script = f'POSIX path of (choose folder with prompt "{_escape_applescript(prompt)}")'
    result = toolchain.run_capturing(["osascript", "-e", script])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _choose_folder_linux(prompt: str) -> str | None:
    """Best-effort: zenity is common on GNOME-based desktops but is not
    guaranteed to be installed, and this is not the harness's primary
    platform (nothing else here supports Windows either — see
    agent/toolchain.py's Unix-shaped PATH handling). A missing binary
    reports the same 127 `run_capturing` already returns for that case,
    read the same way as a cancel: not a native picker on this machine."""
    result = toolchain.run_capturing(
        ["zenity", "--file-selection", "--directory", f"--title={prompt}"]
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
