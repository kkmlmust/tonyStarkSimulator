import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerResult, HandLandmarksConnections



HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIPS = [4, 8, 12, 16, 20]
print(type(HAND_CONNECTIONS))
print(type(next(iter(HAND_CONNECTIONS))))
print(next(iter(HAND_CONNECTIONS)))

def draw_landmarks(frame, hand_landmarks):
    """Рисует 21 точку руки. Кончики пальцев — другим цветом."""
    h, w, _ = frame.shape
    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)

        # кончики пальцев — красные, остальное — зелёное
        if idx in FINGERTIPS:
            color = (0, 0, 255)      # BGR: красный
            radius = 7
        else:
            color = (0, 255, 0)      # BGR: зелёный
            radius = 5

        cv2.circle(frame, (cx, cy), radius, color, -1)


def draw_connections(frame, hand_landmarks):
    """Рисует линии между точками по HAND_CONNECTIONS."""
    h, w, _ = frame.shape
    for connection in HAND_CONNECTIONS:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


def draw_handedness(frame, hand_landmarks, handedness):
    """Пишет 'Left'/'Right' рядом с запястьем (точка 0)."""
    h, w, _ = frame.shape
    wrist = hand_landmarks[0]
    cx, cy = int(wrist.x * w), int(wrist.y * h)

    label = handedness.category_name   # "Left" / "Right"
    score = handedness.score           # 0.0 ... 1.0
    text = f"{label} {score:.2f}"

    cv2.putText(frame, text, (cx + 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_fps(frame, fps):
    """Пишет FPS в левом верхнем углу."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_result(frame, result):
    """Главная функция: рисует всё, что есть в результате.

    Вызывай её из camera.py после получения result.
    Если рук нет — просто ничего не делает.
    """
    if not result or not result.hand_landmarks:
        return

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        draw_connections(frame, hand_landmarks)

        draw_landmarks(frame, hand_landmarks)

        if i < len(result.handedness) and result.handedness[i]:
            draw_handedness(frame, hand_landmarks, result.handedness[i][0])