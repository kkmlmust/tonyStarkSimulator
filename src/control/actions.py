"""Layer E: actions bound to gestures.

An :class:`Action` is a *description* of something to do; the actual input
injection is delegated to a backend (see ``backends.py``). This keeps the
action layer pure and testable with :class:`~control.backends.NullBackend`.

A key design point: a gesture is an on/off *modifier*, while the hand's
position is the actual signal. So actions receive a full
:class:`ActionContext` (landmarks + frame size), not just the gesture name.
Moving the mouse uses the index fingertip's coordinates; the gesture merely
decides whether the mouse moves at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# MediaPipe landmark index of the index fingertip -- used for pointing.
INDEX_FINGERTIP = 8


@dataclass
class ActionContext:
    """Everything an action might need for the current frame."""

    landmarks: object  # 21 MediaPipe landmarks
    frame_size: tuple[int, int]  # (width, height)
    gesture: str
    hand: str
    slot: int


@dataclass
class Action:
    """Base class. Subclasses implement :meth:`perform` (and optionally
    :meth:`release` for `hold` bindings)."""

    def perform(self, ctx: ActionContext, backend) -> None:
        raise NotImplementedError

    def release(self, ctx: ActionContext, backend) -> None:
        """Called when a `hold` binding's gesture ends. Default: nothing."""


@dataclass
class KeyAction(Action):
    """Press a key.

    In ``trigger`` mode the key is tapped once (press + release). In
    ``hold`` mode it is pressed down on activation and held until the
    gesture ends, when it is released -- so holding a gesture does not
    repeat the keypress every frame.
    """

    key: str = ""
    hold: bool = False
    _down: bool = field(default=False, init=False, repr=False)

    def perform(self, ctx, backend) -> None:
        if not self.hold:
            backend.tap_key(self.key)
            return
        if not self._down:
            backend.key_down(self.key)
            self._down = True

    def release(self, ctx, backend) -> None:
        if self.hold and self._down:
            backend.key_up(self.key)
            self._down = False


@dataclass
class MouseMoveAction(Action):
    """Move the cursor like a trackpad: the hand's motion becomes deltas.

    Absolute positioning is not possible on Wayland (see the ydotool
    backend), so we track how far the reference point moved since the last
    frame and forward that as a relative movement, multiplied by
    ``sensitivity``.

    A small ``deadzone`` (in normalized units) ignores hand jitter so the
    cursor does not drift while the hand is held still.
    """

    sensitivity: float = 1200.0  # normalized hand motion -> pixels
    deadzone: float = 0.004  # ignore movements smaller than this
    smoothing: float = 0.4  # 0 = raw, ->1 = smoother (laggier)
    reference: int = INDEX_FINGERTIP
    _last: tuple[float, float] | None = None
    _acc: tuple[float, float] = (0.0, 0.0)

    def perform(self, ctx, backend) -> None:
        point = ctx.landmarks[self.reference]
        current = (float(point.x), float(point.y))

        if self._last is None:
            self._last = current
            return

        dx = current[0] - self._last[0]
        dy = current[1] - self._last[1]
        self._last = current

        if abs(dx) < self.deadzone and abs(dy) < self.deadzone:
            return

        # Exponential smoothing on the *velocity* to damp shake.
        a = self.smoothing
        self._acc = (
            a * self._acc[0] + (1 - a) * dx,
            a * self._acc[1] + (1 - a) * dy,
        )

        move = getattr(backend, "move_mouse_relative", None)
        if move is None:
            # Backend without relative support: fall back to absolute.
            backend.move_mouse(current[0] * ctx.frame_size[0], current[1] * ctx.frame_size[1])
            return
        move(self._acc[0] * self.sensitivity, self._acc[1] * self.sensitivity)

    def release(self, ctx, backend) -> None:
        # Reset so the next activation does not jump by the accumulated delta.
        self._last = None
        self._acc = (0.0, 0.0)


@dataclass
class MouseButtonAction(Action):
    """Press a mouse button while held, release it on DEACTIVATED."""

    button: str = "left"

    def perform(self, ctx, backend) -> None:
        backend.press_mouse(self.button)

    def release(self, ctx, backend) -> None:
        backend.release_mouse(self.button)


@dataclass
class ScrollAction(Action):
    """Scroll the mouse wheel.

    ``amount`` is one scroll step per fire; ``ticks`` (hold mode) lets a
    held gesture repeat the scroll every frame or on an interval. Positive
    scrolls up.
    """

    amount: int = 1
    ticks: int = 1

    def perform(self, ctx, backend) -> None:
        backend.scroll(self.amount * self.ticks)


def build_action(spec: dict, mode: "Mode | None" = None) -> Action:
    """Construct an Action from a TOML binding's action spec.

    ``mode`` (the binding's mode, e.g. ``Mode.HOLD``) lets a key action know
    whether to hold the key down rather than tap it.

    Expected shapes::

        { "type": "key",          "key": "d" }
        { "type": "mouse_move",   "smoothing": 0.5 }
        { "type": "mouse_button", "button": "left" }
        { "type": "scroll",       "amount": 1 }
    """
    action_type = spec.get("type", "").lower()
    if action_type == "key":
        # A `hold` binding holds the key down for its whole duration.
        hold = bool(spec.get("hold", False)) or getattr(mode, "value", mode) == "hold"
        return KeyAction(key=str(spec.get("key", "")), hold=hold)
    if action_type == "mouse_move":
        return MouseMoveAction(
            sensitivity=float(spec.get("sensitivity", 1200.0)),
            deadzone=float(spec.get("deadzone", 0.004)),
            smoothing=float(spec.get("smoothing", 0.4)),
        )
    if action_type == "mouse_button":
        return MouseButtonAction(button=str(spec.get("button", "left")))
    if action_type == "scroll":
        return ScrollAction(
            amount=int(spec.get("amount", 1)),
            ticks=int(spec.get("ticks", 1)),
        )
    raise ValueError(f"unknown action type: {action_type!r}")
