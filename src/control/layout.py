"""Switch the active keyboard layout while injecting keys.

``ydotool`` sends **physical keycodes**, and the compositor turns those into
characters using the *active layout*. That is why ``key = "g"`` does nothing
useful on a Russian layout: physical ``g`` (code 34) resolves to ``п``.

To make Latin bindings reliable regardless of the current layout, we briefly
select the layout that maps to Latin (usually ``us``) around each key
emission and restore the previous one afterwards.

The implementation talks to KDE over D-Bus (``org.kde.keyboard``), which is
the desktop in use here. On a desktop without that service the switcher
simply disables itself and everything keeps working as before.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from contextlib import contextmanager

# qdbus --literal prints the a(sss) layout list like:
#   [Argument: a(sss) {[Argument: (sss) "us", "", "English (US)"], ...}]
_LAYOUT_ROW = re.compile(r'\(sss\)\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"')

_SERVICE = "org.kde.keyboard"
_PATH = "/Layouts"
_IFACE = "org.kde.KeyboardLayouts"

# Values that mean "leave the layout alone".
_OFF = {"", "0", "off", "none", "false", "no", "disable", "disabled"}
# Values that mean "use the default Latin layout".
_AUTO = {"1", "on", "true", "yes", "auto", "enable", "enabled"}


def parse_layouts(text: str) -> list[tuple[str, str, str]]:
    """Parse ``qdbus --literal getLayoutsList`` into ``(code, variant, name)``.

    Returns an empty list on anything unparseable, so callers can degrade
    gracefully instead of crashing on a different D-Bus output format.
    """
    return _LAYOUT_ROW.findall(text or "")


class LayoutSwitcher:
    """Momentarily selects a Latin layout around key injection (KDE/Wayland).

    Args:
        target: the layout to switch *to* while sending keys, e.g. ``"us"``.
            Matched by layout code first, then by its human-readable name.
        binary: the qdbus client to use (``qdbus6`` on modern KDE).
    """

    def __init__(self, target: str = "us", binary: str = "qdbus6") -> None:
        self._binary = binary
        self._target = (target or "us").lower()
        self._index: int | None = None
        self._enabled = False
        self._probe()

    # -- discovery --------------------------------------------------------- #
    def _call(self, method: str, *args: str, literal: bool = False) -> str | None:
        if shutil.which(self._binary) is None:
            return None
        cmd = [self._binary]
        if literal:
            cmd.append("--literal")
        cmd += [_SERVICE, _PATH, f"{_IFACE}.{method}", *args]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout

    def _probe(self) -> None:
        out = self._call("getLayoutsList", literal=True)
        if not out:
            return
        rows = parse_layouts(out)
        for i, (code, _variant, name) in enumerate(rows):
            if code.lower() == self._target or self._target in name.lower():
                self._index = i
                self._enabled = True
                return

    # -- state ------------------------------------------------------------- #
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def index(self) -> int | None:
        return self._index

    def active(self) -> int | None:
        out = self._call("getLayout")
        if not out:
            return None
        digits = re.search(r"\d+", out)
        return int(digits.group()) if digits else None

    def set(self, index: int) -> bool:
        out = self._call("setLayout", str(index))
        return bool(out) and "true" in out

    # -- switching --------------------------------------------------------- #
    def activate(self) -> int | None:
        """Switch to the Latin layout and return the previous index.

        Returns ``None`` when nothing was switched (disabled, already Latin,
        or the current layout is unreadable) -- pass that back to
        :meth:`restore` and it does nothing.
        """
        if not self._enabled or self._index is None:
            return None
        saved = self.active()
        if saved is None or saved == self._index:
            return None
        self.set(self._index)
        return saved

    def restore(self, saved: int | None) -> None:
        """Undo a previous :meth:`activate` (no-op when ``saved`` is None)."""
        if saved is not None:
            self.set(saved)

    @contextmanager
    def latin(self):
        """Temporarily switch to the Latin layout, then restore it."""
        saved = self.activate()
        try:
            yield
        finally:
            self.restore(saved)
