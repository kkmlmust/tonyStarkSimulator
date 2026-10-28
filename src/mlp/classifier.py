"""Load a trained checkpoint and classify hand landmarks.

The classifier is intentionally **stateless**. It knows nothing about
hands, slots, or smoothing: given one hand's landmarks it returns the raw
probability vector. All temporal reasoning (smoothing, hysteresis, events)
lives in ``control.state``, which keeps one tracker per hand.

This separation matters: with two hands in the frame, a single shared
vote buffer would mix their votes together.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.preprocess import extract_features  # noqa: E402
from mlp.model import MLP  # noqa: E402

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "models" / "gesture_mlp.pt"


class GestureClassifier:
    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint_path}. Train it first with "
                "`uv run python src/mlp/train.py`"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.labels: list[str] = list(checkpoint["labels"])
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["std"], dtype=np.float32)

        self.model = MLP(input_dim=checkpoint["input_dim"], output_dim=len(self.labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, hand_landmarks) -> np.ndarray:
        """Return the class probabilities for one hand's landmarks.

        Args:
            hand_landmarks: sequence of 21 MediaPipe landmarks.

        Returns:
            A ``float32`` array of length ``len(self.labels)`` that sums to
            1. Use :meth:`label` to turn it into a name.
        """
        features = extract_features(hand_landmarks)
        x = (features - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None, :].astype(np.float32)))
            probs = torch.softmax(logits, dim=1).numpy()[0]

        return probs.astype(np.float32)

    def label(self, probs: np.ndarray) -> tuple[str, float]:
        """Pick the most likely label from a probability vector."""
        index = int(np.argmax(probs))
        return self.labels[index], float(probs[index])
