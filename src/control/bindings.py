"""Layer D: bind gesture events to actions.

Bindings come from a TOML file so they can be edited without touching code.
Each binding says: *when hand X makes gesture G, do action A*.

Two modes, matching the two kinds of behaviour a gesture can have:

``trigger``  -- fire the action only on ACTIVATED (a single discrete
               action, e.g. "fist -> press D once"). HELD frames are
               ignored, so holding the gesture does not repeat the action.

``hold``     -- the action is active for the whole duration: it runs on
               ACTIVATED and on every HELD frame, and the action's
               ``release`` is called on DEACTIVATED (e.g. "palm -> move
               mouse", "fist -> hold left mouse button").

``interval_ms`` adds optional throttling to ``hold`` bindings (an
auto-repeat), which is useful for things like repeated key presses.
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .actions import Action, ActionContext, build_action
from .state import Event, Phase


class Mode(Enum):
    TRIGGER = "trigger"
    HOLD = "hold"


@dataclass
class Binding:
    gesture: str
    mode: Mode
    action: Action
    hand: str | None = None  # "Left"/"Right"/""/None = any hand
    interval_ms: int = 0
    _last_fired: float = field(default=0.0, init=False, repr=False)

    def matches(self, event: Event) -> bool:
        if event.gesture != self.gesture:
            return False
        if self.hand and event.hand != self.hand:
            return False
        return True

    def should_fire(self, event: Event, now: float) -> bool:
        if self.mode is Mode.TRIGGER:
            return event.phase is Phase.ACTIVATED
        if event.phase in (Phase.ACTIVATED, Phase.HELD):
            if self.interval_ms <= 0:
                return True
            if (now - self._last_fired) * 1000.0 >= self.interval_ms:
                self._last_fired = now
                return True
            return False
        return False

    def should_release(self, event: Event) -> bool:
        return self.mode is Mode.HOLD and event.phase is Phase.DEACTIVATED


def parse_binding(raw: dict) -> Binding:
    try:
        gesture = raw["gesture"]
        mode = Mode(raw.get("mode", "trigger").lower())
    except KeyError as exc:
        raise ValueError(f"binding missing required field: {exc}") from exc

    action = build_action(raw.get("action", {}), mode=mode)
    hand = raw.get("hand") or None
    return Binding(
        gesture=gesture,
        mode=mode,
        action=action,
        hand=hand,
        interval_ms=int(raw.get("interval_ms", 0)),
    )


def load_bindings(path: Path | str) -> list[Binding]:
    """Load a list of bindings from a TOML file with ``[[binding]]`` blocks."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return [parse_binding(raw) for raw in data.get("binding", [])]


@dataclass
class ControlConfig:
    bindings: list[Binding]
    backend: str = "null"
    dry_run: bool = True
    layout: str | None = None


def load_config(path: Path | str) -> ControlConfig:
    """Load bindings plus backend settings from a TOML config file."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    return ControlConfig(
        bindings=[parse_binding(raw) for raw in data.get("binding", [])],
        backend=str(data.get("backend", "null")),
        dry_run=bool(data.get("dry_run", True)),
        layout=data.get("layout"),
    )


@dataclass
class Dispatcher:
    """Turns per-hand events into backend calls via the configured bindings."""

    bindings: list[Binding]
    backend: object
    clock=time.monotonic  # injectable for tests

    def handle(self, events, reading, frame_size) -> None:
        """Process one hand's events for a frame.
            frame_size: ``(width, height)`` of the frame.
        """
        now = self.clock()
        for event in events:
            for binding in self.bindings:
                if not binding.matches(event):
                    continue
                ctx = ActionContext(
                    landmarks=reading.landmarks,
                    frame_size=frame_size,
                    gesture=event.gesture,
                    hand=reading.hand,
                    slot=reading.slot,
                )
                if binding.should_release(event):
                    binding.action.release(ctx, self.backend)
                elif binding.should_fire(event, now):
                    binding.action.perform(ctx, self.backend)