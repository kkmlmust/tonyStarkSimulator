"""Shared feature extraction and queue helpers.

The single most important rule in this file: `extract_features` must be
called in exactly the same way when collecting the dataset and when
running live inference. If the two paths preprocess differently, the
model will look great in training and fail in real life.
"""

from __future__ import annotations

from queue import Empty

import numpy as np


def extract_features(hand_landmarks) -> np.ndarray:
    """Turn 21 hand landmarks into a 63-dim feature vector.

    Steps (see ``context.md`` -> preprocessing):
      1. Stack the (x, y, z) of each landmark into an array of shape (21, 3).
      2. Translate so the wrist (landmark 0) becomes the origin. This
         removes dependence on where the hand is on the screen.
      3. Scale by the wrist -> middle-finger-MCP distance (landmark 9).
         This removes dependence on how far the hand is from the camera.

    Args:
        hand_landmarks: sequence of 21 MediaPipe landmarks, each with
            ``.x``, ``.y`` and ``.z``.

    Returns:
        A flat ``float32`` array of length 63.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    pts = pts - pts[0]
    scale = float(np.linalg.norm(pts[9]))
    return (pts / max(scale, 1e-6)).flatten()


def drain_latest(queue):
    """Return the newest item from ``queue``, discarding older ones.

    MediaPipe runs in ``LIVE_STREAM`` mode asynchronously, and the camera
    can enqueue frames faster than we consume them. Draining to the newest
    result keeps the queue from growing without bound and guarantees we
    always act on the freshest frame. Returns ``None`` when empty.
    """
    latest = None
    while True:
        try:
            latest = queue.get_nowait()
        except Empty:
            break
    return latest
