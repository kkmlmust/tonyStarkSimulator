from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"model file not found: {MODEL_PATH}")

MODEL_PATH = str(MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions


class Hand_tracker:

    def __init__(
        self,
        path: str = MODEL_PATH,
        running_mode=vision.RunningMode.LIVE_STREAM,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=running_mode,
            num_hands=2,
            result_callback=self._on_result,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.last_result: HandLandmarkerResult | None = None
        self.last_timestamp: int | None = None

    def _on_result(
        self,
        result: HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        self.last_result = result
        self.last_timestamp = timestamp_ms

    def procces_frame(self, frame: np.ndarray, timestamp: int) -> None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(image, timestamp)

    def close(self) -> None:
        self.landmarker.close()