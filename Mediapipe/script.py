import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
import time
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_BUFFER = 3
CONFIRM_THRESHOLD = 5
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_model.pkl")
MODEL_CONF_THRESHOLD = 0.6
COMMAND_DELAY = 0.2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONF,
    min_tracking_confidence=TRACKING_CONF
)

gesture_buffer = deque(maxlen=MAX_BUFFER)
last_command_time = 0

if not os.path.exists(MODEL_PATH):
    print(f"[✗] No trained model found at '{MODEL_PATH}'.")
    print("    Run 'python train_model.py' after collecting data.")
    sys.exit(1)

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    gesture_model = model_data["model"]
    model_classes = model_data["classes"]
    print(f"Loaded gesture model — classes: {model_classes}")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)


def extract_landmarks_live(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return np.array(features).reshape(1, -1)


def classify_gesture(hand_landmarks):
    features = extract_landmarks_live(hand_landmarks)
    proba = gesture_model.predict_proba(features)[0]
    max_idx = np.argmax(proba)
    predicted_class = model_classes[max_idx]
    confidence = proba[max_idx]

    if confidence < MODEL_CONF_THRESHOLD:
        return "UNKNOWN", confidence

    return predicted_class, confidence


def execute_command(gesture, confidence=0.0):
    global last_command_time
    now = time.time()
    if now - last_command_time < COMMAND_DELAY:
        return
    print(f"{gesture} ({confidence:.2f})")
    last_command_time = now


def get_stable_gesture():
    if len(gesture_buffer) < CONFIRM_THRESHOLD:
        return None
    most_common = max(set(gesture_buffer), key=gesture_buffer.count)
    count = gesture_buffer.count(most_common)
    if count >= CONFIRM_THRESHOLD and most_common != "UNKNOWN":
        return most_common
    return None


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "NONE"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture, confidence = classify_gesture(hand_landmarks)
                gesture_buffer.append(gesture)
        else:
            gesture_buffer.clear()

        stable_gesture = get_stable_gesture()
        if stable_gesture:
            execute_command(stable_gesture, confidence)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow("Drone Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
