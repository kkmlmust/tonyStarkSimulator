"""Layers C and D: per-hand temporal state and gesture events.

One :class:`HandState` is kept **per hand slot**. It turns a stream of raw
per-frame probabilities into a stream of discrete *events*:

    ACTIVATED    the stable gesture just appeared
    HELD         the stable gesture is still active (every frame)
    DEACTIVATED  the stable gesture just went away

Two mechanisms keep this stable:

* **Smoothing** -- a majority vote over the last ``smooth_window`` frames,
  with a confidence gate, so a single noisy frame cannot flip the label.
* **Hysteresis** -- a new gesture must persist ``confirm_frames`` frames to
  be accepted, and must be absent for ``release_frames`` frames to be
  dropped. Different thresholds for entering and leaving a state is what
  stops flicker on the boundaries.

This module imports no cv2/torch/MediaPipe, so the whole state machine can be
tested with plain label sequences (see ``tests/test_state.py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    ACTIVATED = "activated"
    HELD = "held"
    DEACTIVATED = "deactivated"


@dataclass(frozen=True)
class Event:
    """A single transition of one hand's stable gesture."""

    slot: int
    hand: str  # handedness hint from the detector ("Left"/"Right"/"")
    gesture: str  # the gesture this event is about
    phase: Phase


@dataclass
class HandState:
    """Smoothed, hysteretic gesture state for a single hand slot."""

    slot: int
    labels: list[str]
    smooth_window: int = 10
    min_confidence: float = 0.6
    confirm_frames: int = 4
    release_frames: int = 5

    current: str | None = None
    _votes: deque = field(default_factory=deque, repr=False)
    _candidate: str | None = None
    _candidate_frames: int = 0
    _absent_frames: int = 0

    def __post_init__(self) -> None:
        self._votes = deque(maxlen=self.smooth_window)

    @property
    def stable(self) -> str | None:
        """The currently accepted gesture, or ``None``."""
        return self.current

    def reset(self) -> None:
        self.current = None
        self._votes.clear()
        self._candidate = None
        self._candidate_frames = 0
        self._absent_frames = 0

    def mark_lost(self, hand: str = "") -> list[Event]:
        """Called when the hand leaves the frame entirely.

        Without this, a `hold` binding's ``release`` would never run (e.g. a
        mouse button could stay pressed forever after the hand disappears),
        because :meth:`update` is simply not called for a hand that is gone.
        """
        if self.current is None:
            return []
        event = Event(self.slot, hand, self.current, Phase.DEACTIVATED)
        self.reset()
        return [event]

    def update(self, probs, hand: str = "") -> list[Event]:
        """Feed one frame's probabilities; return any events produced.

        Args:
            probs: probability vector from the classifier.
            hand: handedness hint, forwarded to the emitted events.
        """
        smoothed = self._smooth(probs)
        return self._advance(smoothed, hand)

    # ------------------------------------------------------------------ #
    # Smoothing
    # ------------------------------------------------------------------ #
    def _smooth(self, probs) -> str | None:
        index = int(probs.argmax())
        self._votes.append(index)

        counts: dict[int, int] = {}
        for vote in self._votes:
            counts[vote] = counts.get(vote, 0) + 1

        best_index = max(counts, key=counts.get)
        confidence = counts[best_index] / len(self._votes)

        if confidence < self.min_confidence:
            return None
        return self.labels[best_index]

    # ------------------------------------------------------------------ #
    # Hysteresis / event generation
    # ------------------------------------------------------------------ #
    def _advance(self, smoothed: str | None, hand: str) -> list[Event]:
        events: list[Event] = []

        # Same as what we already accepted: nothing to confirm.
        if smoothed == self.current:
            self._candidate = None
            self._candidate_frames = 0
            self._absent_frames = 0
            if self.current is not None:
                events.append(Event(self.slot, hand, self.current, Phase.HELD))
            return events

        # No confident gesture this frame: count towards release.
        if smoothed is None:
            self._candidate = None
            self._candidate_frames = 0
            if self.current is not None:
                self._absent_frames += 1
                if self._absent_frames >= self.release_frames:
                    events.append(Event(self.slot, hand, self.current, Phase.DEACTIVATED))
                    self.current = None
                    self._absent_frames = 0
            return events

        # A different gesture candidate: count towards confirmation.
        self._absent_frames = 0
        if smoothed == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = smoothed
            self._candidate_frames = 1

        if self._candidate_frames >= self.confirm_frames:
            if self.current is not None:
                events.append(Event(self.slot, hand, self.current, Phase.DEACTIVATED))
            self.current = smoothed
            self._candidate = None
            self._candidate_frames = 0
            events.append(Event(self.slot, hand, self.current, Phase.ACTIVATED))

        return events
