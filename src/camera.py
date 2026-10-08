import cv2
import time
from hand_detection import Hand_tracker
import numpy as np
from draw import draw_result



cap = cv2.VideoCapture(0)

# Настройка разрешения камеры на уровне железа (необязательно, но полезно)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
flag = True

landmarker = Hand_tracker() 
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получsить кадр с камеры.")
        break
    
    # 1. Отзеркаливаем для удобства
    frame = cv2.flip(frame, 1)
       # 2. Подготавливаем кадр для ML-пайплайна
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)
    
    landmarker.procces_frame(rgb_frame, timestamp_ms)
    
    draw_result(frame, landmarker.last_result)
    # 4. Отображаем результат
    cv2.imshow("Hand Control Pipeline", frame)

    # 5. Выход по кнопке Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()