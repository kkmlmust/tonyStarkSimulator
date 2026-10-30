"""The "send a random line from a CSV" macro.

Flow (as requested):

    1. open the chat            -> Shift+Enter (or Enter, configurable)
    2. pick a random line       -> from a CSV file
    3. put it on the clipboard  -> avoid the US-keymap problem
    4. paste it                 -> Ctrl+V
    5. send it                  -> Enter

The action itself only knows *how to sequence the steps*; the actual key
and clipboard emission is delegated to the backend, so it stays testable
with the NullBackend.
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .actions import Action, ActionContext


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path | str) -> Path:
    """Resolve a possibly-relative path against CWD, then the project root."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def load_lines(path: Path | str, column: int | None = None) -> list[str]:
    """Read phrases from a CSV/text file.

    By default each non-empty **line is used whole** -- this matches the
    intuitive "a list of phrases, one per line" and keeps commas inside a
    phrase intact (e.g. ``Привет, мир``).

    If ``column`` is given (0-based), the file is parsed as real CSV and
    that column is extracted instead, for structured data.
    """
    path = resolve_path(path)
    text = path.read_text(encoding="utf-8-sig")

    if column is None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        lines = []
        for row in csv.reader(text.splitlines()):
            if len(row) > column and row[column].strip():
                lines.append(row[column].strip())

    if not lines:
        raise ValueError(f"no usable lines found in {path}")
    return lines


@dataclass
class SendRandomLineAction(Action):
    """Open chat, paste a random CSV line, then send it.

    Args:
        csv_path: the CSV to read phrases from.
        open_chat: how to open the chat ("shift_enter" or "enter").
        send: how to send after pasting ("enter" or "shift_enter").
        pre_delay: seconds to wait for the chat box to open.
        post_delay: seconds to wait after pasting before sending.
    """

    csv_path: str = ""
    open_chat: str = "shift_enter"
    send: str = "enter"
    column: int | None = None
    pre_delay: float = 0.15
    post_delay: float = 0.1
    _lines: list[str] | None = field(default=None, init=False, repr=False)

    def _get_lines(self) -> list[str]:
        if self._lines is None:
            self._lines = load_lines(self.csv_path, self.column)
        return self._lines

    def _combo_for(self, which: str) -> list[str]:
        which = (which or "enter").lower()
        if which == "shift_enter":
            return ["leftshift", "enter"]
        if which == "enter":
            return ["enter"]
        if which == "ctrl_v":
            return ["leftctrl", "v"]
        raise ValueError(
            f"unknown key {which!r} (use 'enter', 'shift_enter' or 'ctrl_v')"
        )

    def _combo(self, backend, which: str) -> None:
        combo = self._combo_for(which)
        tap_combo = getattr(backend, "tap_combo", None)
        if tap_combo is not None:
            tap_combo(combo)
            return
        # Fallback for backends without combo support (e.g. NullBackend).
        for key in combo:
            backend.tap_key(key)

    def perform(self, ctx: ActionContext, backend) -> None:
        phrase = random.choice(self._get_lines())

        # 2. open the chat
        self._combo(backend, self.open_chat)
        if self.pre_delay:
            time.sleep(self.pre_delay)

        # 3-4. put the phrase on the clipboard and paste it
        backend.set_clipboard(phrase)
        self._combo(backend, "ctrl_v")
        if self.post_delay:
            time.sleep(self.post_delay)

        # 5. send
        self._combo(backend, self.send)
