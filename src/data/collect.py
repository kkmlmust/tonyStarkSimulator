"""Record a labelled hand-gesture dataset with a webcam.

Run from the repository root:

    uv run python src/data/collect.py

Controls
--------
    1  f_sign      Start recording this gesture (with a short countdown)
    2  v_sign
    3  pen_hold
    4  palm_hand
    5  fist
    space          Stop the current recording early
    q              Quit

Each recording is a short *burst* (``RECORD_SECONDS``) and gets its own
``clip_id``, which is what lets the dataset be split by whole clips later
instead of by individual (near-duplicate) frames.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make both the ``src`` root (for hand_detection / draw) and this package
# importable regardless of where the script is launched from.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2  # noqa: E402

from hand_detection import Hand_tracker  # noqa: E402
from draw import draw_result, draw_fps, draw_status  # noqa: E402
from data.data_collector import DataCollector, Gesture, State  # noqa: E402

# Number keys 1-5 map to the five gestures, in README order.
GESTURE_KEYS = {
    ord("1"): Gesture.F_SIGN,
    ord("2"): Gesture.V_SIGN,
    ord("3"): Gesture.PEN_HOLD,
    ord("4"): Gesture.PALM_HAND,
    ord("5"): Gesture.FIST,
}

COUNTDOWN_SECONDS = 3.0
RECORD_SECONDS = 3.0
SESSION_NAME = time.strftime("session_%Y%m%d_%H%M%S")


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open the webcam")

    landmarker = Hand_tracker()
    collector = DataCollector(session_name=SESSION_NAME)

    pending_gesture: Gesture | None = None
    countdown_start = 0.0
    stop_at = 0.0

    prev = time.time()
    fps = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Не удалось получить кадр с камеры.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            landmarker.procces_frame(rgb_frame, timestamp_ms)

            # The collector is the single consumer of the result queue.
            latest = collector.collect(landmarker.results_queue)
            if latest is not None:
                draw_result(frame, latest.result)

            now = time.time()

            # --- recording state machine ------------------------------- #
            if collector.state is State.RECORDING and now >= stop_at:
                collector.stop_recording()

            if pending_gesture is not None:
                remaining = COUNTDOWN_SECONDS - (now - countdown_start)
                if remaining <= 0:
                    collector.start_recording(pending_gesture)
                    stop_at = now + RECORD_SECONDS
                    pending_gesture = None
                else:
                    cv2.putText(
                        frame,
                        f"{int(remaining) + 1}...",
                        (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 0, 255),
                        4,
                    )

            # --- HUD ---------------------------------------------------- #
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            draw_fps(frame, fps)

            if collector.state is State.RECORDING:
                status = f"REC {collector.gesture.value}  samples={collector.samples_written}"
            else:
                status = f"IDLE  session={SESSION_NAME}  samples={collector.samples_written}"
            draw_status(frame, status)

            cv2.imshow("Gesture collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                collector.stop_recording()
                pending_gesture = None
            elif key in GESTURE_KEYS and collector.state is State.IDLE:
                pending_gesture = GESTURE_KEYS[key]
                countdown_start = time.time()
    finally:
        collector.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Готово. Сессия сохранена в {collector.base_dir / SESSION_NAME}")


if __name__ == "__main__":
    main()
