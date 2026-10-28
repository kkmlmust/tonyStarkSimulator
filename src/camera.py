"""Real-time gesture recognition.

Run from the repository root:

    uv run python src/camera.py

Shows the detected hand skeleton plus the predicted gesture. Requires a
trained checkpoint (``uv run python src/mlp/train.py`` first); if none
exists the script still runs, just without a prediction.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2  # noqa: E402

from hand_detection import Hand_tracker  # noqa: E402
from draw import draw_result, draw_fps, draw_hand_label, draw_status  # noqa: E402
from data.preprocess import drain_latest  # noqa: E402
from control.pipeline import GesturePipeline  # noqa: E402
from control.bindings import Dispatcher, load_config  # noqa: E402
from control.backends import build_backend  # noqa: E402

CONFIG_PATH = SRC_DIR / "control" / "config.toml"


def load_classifier():
    """Load the gesture classifier, or return None if not trained yet."""
    try:
        from mlp.classifier import GestureClassifier

        return GestureClassifier()
    except FileNotFoundError as exc:
        print(f"{exc}\n--> running without predictions for now.")
        return None

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open the webcam")

    landmarker = Hand_tracker()
    classifier = load_classifier()
    pipeline = None

    # Control layer (layer E). dry_run guarantees nothing is typed/clicked
    # until you explicitly arm it.
    config = load_config(CONFIG_PATH)
    backend = build_backend("null" if config.dry_run else config.backend)
    dispatcher = Dispatcher(bindings=config.bindings, backend=backend)
    armed = False  # press `a` to arm/disarm real actions

    prev = time.time()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Не удалось получить кадр с камеры.")
                break

            frame = cv2.flip(frame, 1)
            frame_size = (frame.shape[1], frame.shape[0])
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            landmarker.procces_frame(rgb_frame, timestamp_ms)

            # Single consumer: take the newest result, drop stale ones.
            latest = drain_latest(landmarker.results_queue)
            result = latest.result if latest is not None else None

            if classifier is None:
                # No model yet: still draw the skeleton.
                draw_result(frame, result)
            else:
                if pipeline is None:
                    pipeline = GesturePipeline(labels=classifier.labels)
                for reading in pipeline.update(result, classifier):
                    if reading.landmarks is not None:
                        draw_hand_label(
                            frame,
                            reading.landmarks,
                            reading.slot,
                            reading.gesture,
                            reading.hand,
                        )
                    if armed and reading.events:
                        dispatcher.handle(reading.events, reading, frame_size)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            draw_fps(frame, fps)
            mode = "ARMED" if armed else "disarmed"
            draw_status(
                frame,
                f"{mode}  backend={config.backend}"
                + (" (dry-run)" if config.dry_run else ""),
            )

            cv2.imshow("Hand Control Pipeline", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                armed = not armed
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
