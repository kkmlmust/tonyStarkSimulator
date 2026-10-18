"""Collect hand-landmark samples into a labelled gesture dataset.

See ``README.md`` in this directory for the full specification. Summary:

* The collector is independent of the webcam, keyboard and MediaPipe setup.
* It is the single consumer of the detector's result queue.
* Each recording burst gets a ``clip_id`` so the dataset can later be split
  by whole clips (avoiding train/validation leakage between near-identical
  consecutive frames).
"""

from __future__ import annotations

import csv
from collections import deque
from enum import Enum
from pathlib import Path
from queue import Empty

from .preprocess import extract_features

# Number of feature columns: 21 landmarks * 3 coordinates.
NUM_LANDMARKS = 21
FEATURE_COLUMNS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]
META_COLUMNS = ["label", "clip_id", "session", "handedness", "detection_score"]


class Gesture(Enum):
    """The gestures this project is built to recognise.

    The value is the human-readable label written into the dataset, so it
    stays stable even if the enum members are reordered later.
    """

    F_SIGN = "f_sign"
    V_SIGN = "v_sign"
    PEN_HOLD = "pen_hold"
    PALM_HAND = "palm_hand"
    FIST = "fist"


class State(Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"


class DataCollector:
    """Records landmark samples for one gesture at a time.

    Typical use::

        collector = DataCollector(session_name="day1")
        collector.start_recording(Gesture.FIST)
        ...
        collector.collect(landmarker.results_queue)   # call once per frame
        ...
        collector.stop_recording()
    """

    def __init__(
        self,
        session_name: str,
        base_dir: Path | str | None = None,
        sample_interval_ms: int = 150,
        trim_head: int = 5,
        trim_tail: int = 5,
    ) -> None:
        """Create a collector.

        Args:
            session_name: name of the session folder to write into.
            base_dir: where sessions are stored. Defaults to ``./sessions``
                next to this file.
            sample_interval_ms: minimum delay between saved samples for a
                single hand. Throttling reduces near-duplicate frames.
            trim_head: number of leading valid samples to drop per clip
                (the hand is still moving into the pose).
            trim_tail: number of trailing valid samples to hold back and
                drop per clip (the hand is moving out of the pose).
        """
        if sample_interval_ms < 0:
            raise ValueError("sample_interval_ms must be >= 0")
        if trim_head < 0 or trim_tail < 0:
            raise ValueError("trim_head and trim_tail must be >= 0")

        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent / "sessions"
        self.session_name = session_name
        self.sample_interval_ms = sample_interval_ms
        self.trim_head = trim_head
        self.trim_tail = trim_tail

        self.state = State.IDLE
        self.gesture: Gesture | None = None
        self.samples_written = 0

        self._clip_id = 0
        self._file = None
        self._writer: csv.writer | None = None
        self._warmup_left = 0
        self._last_timestamp = -(10**12)
        self._tail: deque = deque(maxlen=max(trim_tail + 1, 1))

    # ------------------------------------------------------------------ #
    # Recording control
    # ------------------------------------------------------------------ #
    def start_recording(self, gesture: Gesture) -> None:
        """Begin a new recording burst for ``gesture``.

        Safe to call only while IDLE. Creates this session's folder and
        appends to ``<gesture>.csv`` (writing a header the first time),
        so previously recorded data is never overwritten.
        """
        if self.state is State.RECORDING:
            raise RuntimeError("recording already in progress; call stop_recording() first")
        if not isinstance(gesture, Gesture):
            raise TypeError("gesture must be a Gesture member")

        session_dir = self.base_dir / self.session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        self._path = session_dir / f"{gesture.value}.csv"
        is_new = not self._path.exists()
        self._file = self._path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(FEATURE_COLUMNS + META_COLUMNS)
            self._file.flush()

        self.gesture = gesture
        self.state = State.RECORDING
        self._clip_id += 1
        self._warmup_left = self.trim_head
        self._last_timestamp = -(10**12)
        self._tail.clear()
        self._clip_samples = 0

    def stop_recording(self) -> None:
        """End the current burst, flush and close the file.

        Safe to call while IDLE (it does nothing).
        """
        if self.state is State.IDLE:
            return

        # Whatever is still in the tail buffer is the "leaving the pose"
        # part of the clip, so it is intentionally dropped.
        self._tail.clear()

        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None
        self.state = State.IDLE
        self.gesture = None

    # ------------------------------------------------------------------ #
    # Result consumption
    # ------------------------------------------------------------------ #
    def collect(self, queue):
        """Consume every pending result; save valid samples while recording.

        This must be the *only* place the result queue is read from. It
        returns the newest ``FrameResult`` (or ``None``) so the caller can
        still draw the current frame.
        """
        latest = None
        while True:
            try:
                frame = queue.get_nowait()
            except Empty:
                break
            latest = frame
            self._handle(frame)
        return latest

    def _handle(self, frame) -> None:
        if self.state is not State.RECORDING:
            return

        result = frame.result
        if not result.hand_landmarks:
            return

        hand = result.hand_landmarks[0]
        if len(hand) != NUM_LANDMARKS:
            return

        # Throttle: skip samples that arrive too soon after the last one.
        if frame.timestamp_ms - self._last_timestamp < self.sample_interval_ms:
            return

        # Leading trim: the hand may still be moving into the pose.
        if self._warmup_left > 0:
            self._warmup_left -= 1
            return

        handedness, score = "", ""
        if result.handedness and result.handedness[0]:
            category = result.handedness[0][0]
            handedness = category.category_name
            score = round(float(category.score), 4)

        row = list(extract_features(hand)) + [
            self.gesture.value,
            self._clip_id,
            self.session_name,
            handedness,
            score,
        ]

        # Trailing trim: hold back `trim_tail` samples, writing older ones
        # progressively so we never buffer the whole clip in memory.
        if self.trim_tail == 0:
            self._write(row)
        else:
            self._tail.append(row)
            if len(self._tail) > self.trim_tail:
                self._write(self._tail.popleft())

        self._last_timestamp = frame.timestamp_ms

    def _write(self, row) -> None:
        self._writer.writerow(row)
        self._file.flush()
        self.samples_written += 1
        self._clip_samples += 1

    # ------------------------------------------------------------------ #
    # Resource management
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Release the open file, if any. Safe to call multiple times."""
        if self.state is State.RECORDING:
            self.stop_recording()

    def __enter__(self) -> "DataCollector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
