"""Layer A: assign stable slots to detected hands across frames.

MediaPipe returns a *list* of hands, but the order of that list is not
guaranteed to be stable between frames. If we naively keyed per-hand state
by list index, the two hands would swap slots whenever MediaPipe reorders
them, mixing their histories and firing bindings on the wrong hand.

We instead match each detected hand to the closest previously-seen hand by
wrist position (hands do not teleport in ~33 ms), and inherit that hand's
slot id. The MediaPipe ``handedness`` label is only used as a hint, never as
ground truth: it is defined relative to a mirrored (selfie) view, and we
flip the frame, so its meaning is easy to get backwards.

This module is pure numpy -- no cv2, no torch, no MediaPipe imports -- so it
can be unit-tested with plain coordinate arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_DISTANCE = 0.25  # normalized units; wrist jump larger than this = new hand
DEFAULT_GRACE_FRAMES = 15  # frames a missing hand keeps its slot before freeing it


def wrist_xy(hand_landmarks) -> np.ndarray:
    """Return the wrist (landmark 0) as ``(x, y)`` in normalized coords."""
    lm = hand_landmarks[0]
    return np.array([lm.x, lm.y], dtype=np.float32)


@dataclass
class _Track:
    slot: int
    position: np.ndarray
    handedness: str
    missing: int = 0


@dataclass
class HandIdentity:
    """Tracks at most two hands and yields stable slot ids.

    Usage::

        identity = HandIdentity()
        slots = identity.update(result.hand_landmarks, result.handedness)
        # slots: list of (slot_id, hand_landmarks, handedness_str)
    """

    max_distance: float = DEFAULT_MAX_DISTANCE
    grace_frames: int = DEFAULT_GRACE_FRAMES
    max_hands: int = 2

    _tracks: dict[int, _Track] = field(default_factory=dict)
    _next_slot: int = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_slot = 0

    def _handedness_of(self, handedness, index: int) -> str:
        try:
            categories = handedness[index]
            if categories:
                return categories[0].category_name
        except (IndexError, TypeError):
            pass
        return ""

    def update(self, hand_landmarks, handedness=None):
        """Match this frame's hands to slots.

        Returns a list of ``(slot_id, hand_landmarks, handedness_str)``,
        sorted by slot id.
        """
        if not hand_landmarks:
            for track in self._tracks.values():
                track.missing += 1
            self._expire()
            return []

        positions = [wrist_xy(h) for h in hand_landmarks]
        labels = [
            self._handedness_of(handedness, i) for i in range(len(hand_landmarks))
        ]

        available = list(hand_landmarks)
        available_pos = positions
        available_lab = labels

        assignments: list[tuple[int, object, str]] = []
        used_slots: set[int] = set()

        # Pass 1: greedily match the closest existing track to each hand.
        # We take the globally-closest pair first so a single confident match
        # is not stolen by a worse one.
        while available:
            best = None  # (distance, hand_idx, slot)
            for hi, pos in enumerate(available_pos):
                for slot, track in self._tracks.items():
                    if slot in used_slots:
                        continue
                    distance = float(np.linalg.norm(pos - track.position))
                    # Prefer matches that also agree on handedness.
                    if available_lab[hi] and track.handedness:
                        if available_lab[hi] != track.handedness:
                            distance += 0.05  # small tie-break penalty
                    if distance > self.max_distance:
                        continue
                    if best is None or distance < best[0]:
                        best = (distance, hi, slot)

            if best is None:
                break

            _, hi, slot = best
            used_slots.add(slot)
            assignments.append((slot, available[hi], available_lab[hi]))
            self._tracks[slot].position = available_pos[hi]
            self._tracks[slot].handedness = available_lab[hi]
            self._tracks[slot].missing = 0
            del available[hi]
            del available_pos[hi]
            del available_lab[hi]

        # Pass 2: any unmatched hand starts a new slot.
        for hand, pos, lab in zip(available, available_pos, available_lab):
            slot = self._next_slot
            self._next_slot += 1
            self._tracks[slot] = _Track(slot=slot, position=pos, handedness=lab)
            assignments.append((slot, hand, lab))

        # Age tracks that were not matched this frame.
        matched_slots = {slot for slot, _, _ in assignments}
        for slot, track in self._tracks.items():
            if slot not in matched_slots:
                track.missing += 1
        self._expire()

        assignments.sort(key=lambda item: item[0])
        return assignments

    def _expire(self) -> None:
        expired = [
            slot
            for slot, track in self._tracks.items()
            if track.missing > self.grace_frames
        ]
        for slot in expired:
            del self._tracks[slot]
