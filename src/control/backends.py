"""Layer E: input backends.

A backend knows *how* to emit input; actions know *what* to emit. Choosing a
backend is a runtime/config decision, so the rest of the code never imports
``pynput`` or ``ydotool`` directly.

Given this machine runs **Wayland (KDE)**, the backends are not equally
capable:

* ``NullBackend``   -- logs only. Safe default. Works everywhere.
* ``PynputBackend`` -- X11 (via Xwayland on Wayland). May be ignored by
                       native Wayland windows, so it is *not* reliable here.
* ``YdotoolBackend``-- injects via ``/dev/uinput`` at kernel level, so the
                       compositor cannot tell it from a real device. This is
                       the Wayland-capable option; requires the ``ydotool``
                       daemon to be running (as *your user*, not root).

Backends must never be imported for their side effects at module import
time; ``pynput``/``ydotool`` are imported lazily inside their classes.
"""

from __future__ import annotations

import os
import shutil
import subprocess


class NullBackend:
    """Logs actions instead of performing them (dry-run)."""

    def __init__(self, logger=print) -> None:
        self._log = logger

    def tap_key(self, key: str) -> None:
        self._log(f"[dry-run] tap key {key!r}")

    def key_down(self, key: str) -> None:
        self._log(f"[dry-run] key down {key!r}")

    def key_up(self, key: str) -> None:
        self._log(f"[dry-run] key up {key!r}")

    def move_mouse(self, x: float, y: float) -> None:
        self._log(f"[dry-run] move mouse -> ({x:.0f}, {y:.0f})")

    def move_mouse_relative(self, dx: float, dy: float) -> None:
        self._log(f"[dry-run] move mouse by ({dx:+.0f}, {dy:+.0f})")

    def press_mouse(self, button: str) -> None:
        self._log(f"[dry-run] press mouse {button!r}")

    def release_mouse(self, button: str) -> None:
        self._log(f"[dry-run] release mouse {button!r}")

    def scroll(self, amount: int) -> None:
        self._log(f"[dry-run] scroll {amount:+d}")


_BUTTONS = {"left": "left", "right": "right", "middle": "middle"}


class PynputBackend:
    """Real input via ``pynput`` (X11 / Xwayland)."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller as KeyboardController
        from pynput.mouse import Button, Controller as MouseController

        self._keyboard = KeyboardController()
        self._mouse = MouseController()
        self._Button = Button

    def _button(self, name: str):
        if name not in _BUTTONS:
            raise ValueError(f"unknown mouse button {name!r}")
        return getattr(self._Button, _BUTTONS[name])

    def tap_key(self, key: str) -> None:
        self._keyboard.press(key)
        self._keyboard.release(key)

    def key_down(self, key: str) -> None:
        self._keyboard.press(key)

    def key_up(self, key: str) -> None:
        self._keyboard.release(key)

    def move_mouse(self, x: float, y: float) -> None:
        self._mouse.position = (int(x), int(y))

    def move_mouse_relative(self, dx: float, dy: float) -> None:
        self._mouse.move(int(dx), int(dy))

    def press_mouse(self, button: str) -> None:
        self._mouse.press(self._button(button))

    def release_mouse(self, button: str) -> None:
        self._mouse.release(self._button(button))

    def scroll(self, amount: int) -> None:
        self._mouse.scroll(0, amount)


# Linux evdev keycodes (see /usr/include/linux/input-event-codes.h).
# ydotool 1.x takes raw keycodes, not key names, so we map the common ones.
_YDOTOOL_KEYCODES: dict[str, int] = {
    "esc": 1, "backspace": 14, "tab": 15, "enter": 28,
    "leftctrl": 29, "leftshift": 42, "leftalt": 56, "space": 57,
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34,
    "h": 35, "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49,
    "o": 24, "p": 25, "q": 16, "r": 19, "s": 31, "t": 20, "u": 22,
    "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
}

# Mouse button base numbers; `| 0x40` = down, `| 0x80` = up.
_YDOTOOL_BUTTONS = {"left": 0x00, "right": 0x01, "middle": 0x02}


def _keycode(key: str) -> int:
    """Resolve a key name (or a literal keycode) to an evdev keycode."""
    name = key.strip().lower()
    if name in _YDOTOOL_KEYCODES:
        return _YDOTOOL_KEYCODES[name]
    if name.isdigit():
        return int(name)
    raise ValueError(
        f"unknown key {key!r} for the ydotool backend; use a single letter, "
        f"digit, or one of {sorted(_YDOTOOL_KEYCODES)}"
    )


class YdotoolBackend:
    """Real input via ``ydotool`` 1.x (kernel ``uinput``; works on Wayland).

    This backend speaks ydotool 1.x's syntax:

    * keys:   ``ydotool key <evdev_code>:1`` (down) / ``<code>:0`` (up)
    * buttons:``ydotool click 0x40`` (left down) / ``0x80`` (left up)
    * move:   ``ydotool mousemove -x <dx> -y <dy>`` (RELATIVE only)
    * wheel:  ``ydotool mousemove -w -x 0 -y <n>``

    Absolute positioning is *not* supported: ydotool's virtual device
    exposes only relative axes, so a cursor position cannot be set directly.
    Motion must be expressed as deltas (like a trackpad).

    The daemon must run as your user so the client can find the socket at
    ``$XDG_RUNTIME_DIR/.ydotool_socket``. Run ``ydotoold`` via the user
    systemd unit, not with sudo.
    """

    def __init__(self, binary: str = "ydotool", socket: str | None = None,
                 probe: bool = True) -> None:
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"{binary!r} not found on PATH. Install ydotool and start its "
                "daemon, or use the NullBackend (dry-run)."
            )
        self._bin = binary
        self._env = dict(os.environ)
        if socket:
            self._env["YDOTOOL_SOCKET"] = socket
        if probe:
            self._warn_if_unreachable()

    # -- process plumbing -------------------------------------------------- #
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self._bin, *args], capture_output=True, text=True, env=self._env
        )

    def _warn_if_unreachable(self) -> None:
        # `key 0:1` is harmless (non-interpretable -> only a delay) but still
        # needs the socket, so it makes a good connectivity probe.
        result = self._run("key", "0:1")
        if result.returncode != 0 or "failed to connect" in (result.stderr or ""):
            print(
                "[ydotool] WARNING: cannot reach the daemon.\n"
                "         Start it for your user, not root:\n"
                "           sudo pkill ydotoold\n"
                "           systemctl --user enable --now ydotool.service"
            )

    # -- keyboard ---------------------------------------------------------- #
    def tap_key(self, key: str) -> None:
        code = _keycode(key)
        self._run("key", f"{code}:1", f"{code}:0")

    def key_down(self, key: str) -> None:
        self._run("key", f"{_keycode(key)}:1")

    def key_up(self, key: str) -> None:
        self._run("key", f"{_keycode(key)}:0")

    # -- mouse ------------------------------------------------------------- #
    def move_mouse(self, x: float, y: float) -> None:
        # NOTE: the ydotool virtual device exposes only REL axes, so absolute
        # positioning cannot work on Wayland. Use move_mouse_relative instead.
        raise NotImplementedError(
            "ydotool cannot do absolute mouse positioning (its virtual device "
            "has no absolute axes); use move_mouse_relative"
        )

    def move_mouse_relative(self, dx: float, dy: float) -> None:
        self._run("mousemove", "-x", str(int(round(dx))), "-y", str(int(round(dy))))

    def _click_code(self, button: str, action: str) -> str:
        if button not in _YDOTOOL_BUTTONS:
            raise ValueError(f"unknown mouse button {button!r}")
        base = _YDOTOOL_BUTTONS[button]
        mask = {"down": 0x40, "up": 0x80, "click": 0xC0}[action]
        return hex(base | mask)

    def press_mouse(self, button: str) -> None:
        self._run("click", self._click_code(button, "down"))

    def release_mouse(self, button: str) -> None:
        self._run("click", self._click_code(button, "up"))

    def scroll(self, amount: int) -> None:
        # Positive amount scrolls up (REL_WHEEL convention).
        self._run("mousemove", "-w", "-x", "0", "-y", str(int(amount)))


_BACKENDS = {
    "null": NullBackend,
    "pynput": PynputBackend,
    "ydotool": YdotoolBackend,
}


def build_backend(name: str):
    """Instantiate a backend by name (``null`` / ``pynput`` / ``ydotool``)."""
    name = (name or "null").lower()
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(_BACKENDS)}")
    return _BACKENDS[name]()

