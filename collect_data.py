# collect_data.py - Captures hand gesture images and stores them in dataset/

import cv2
import os
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

LABEL = "Hello"  # Change this to the sign you are collecting
data_dir = f"dataset/{LABEL}"
os.makedirs(data_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 100:  # Capture 100 images
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        cv2.imwrite(f"{data_dir}/{count}.jpg", frame)
        count += 1
    
    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()