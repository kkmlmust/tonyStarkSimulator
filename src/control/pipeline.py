"""Tie layers A (identity) and C/D (state) together.

:class:`GesturePipeline` is the single entry point the camera loop calls
each frame. Given the detector's raw result it returns a list of
``HandReading`` objects -- one per tracked hand -- each carrying its stable
slot, the current smoothed gesture, and the events produced this frame.

Keeping this in its own class means the camera loop deals with one call
instead of juggling slot dictionaries and vote buffers itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .identity import HandIdentity
from .state import Event, HandState


@dataclass
class HandReading:
    slot: int
    hand: str  # "Left"/"Right"/""
    landmarks: object  # the raw MediaPipe landmarks for drawing
    gesture: str | None  # the stable gesture, or None
    events: list[Event] = field(default_factory=list)


class GesturePipeline:
    def __init__(
        self,
        labels: list[str],
        smooth_window: int = 10,
        min_confidence: float = 0.6,
        confirm_frames: int = 4,
        release_frames: int = 5,
    ) -> None:
        self.labels = labels
        self.identity = HandIdentity()
        self._settings = dict(
            labels=labels,
            smooth_window=smooth_window,
            min_confidence=min_confidence,
            confirm_frames=confirm_frames,
            release_frames=release_frames,
        )
        self._states: dict[int, HandState] = {}

    def _state_for(self, slot: int) -> HandState:
        if slot not in self._states:
            self._states[slot] = HandState(slot=slot, **self._settings)
        return self._states[slot]

    def update(self, result, classifier) -> list[HandReading]:
        """Process one detector result.

        Args:
            result: a MediaPipe ``HandLandmarkerResult`` (or ``None``).
            classifier: a :class:`GestureClassifier`, or ``None`` to skip
                classification (labels will be ``None``).

        Returns:
            One :class:`HandReading` per tracked hand.
        """
        if result is None or not result.hand_landmarks:
            # No hands this frame. Age out slots and release any held state so
            # "hold" bindings do not stay engaged when the hand disappears.
            self.identity.update([], None)
            return self._release_missing(present_slots=set())

        assignments = self.identity.update(result.hand_landmarks, result.handedness)

        readings: list[HandReading] = []
        present_slots = {slot for slot, _, _ in assignments}
        for slot, landmarks, hand in assignments:
            state = self._state_for(slot)
            events: list[Event] = []
            if classifier is not None:
                probs = classifier.predict(landmarks)
                events = state.update(probs, hand)
            readings.append(
                HandReading(
                    slot=slot,
                    hand=hand,
                    landmarks=landmarks,
                    gesture=state.stable,
                    events=events,
                )
            )

        # Emit releases for hands that vanished, so actions do not stay stuck.
        readings.extend(self._release_missing(present_slots))
        return readings

    def _release_missing(self, present_slots: set[int]) -> list[HandReading]:
        """Produce reading-less DEACTIVATED events for slots that disappeared."""
        released: list[HandReading] = []
        for slot, state in self._states.items():
            if slot in present_slots or state.stable is None:
                continue
            events = state.mark_lost()
            if events:
                released.append(
                    HandReading(
                        slot=slot,
                        hand="",
                        landmarks=None,
                        gesture=None,
                        events=events,
                    )
                )
        return released
