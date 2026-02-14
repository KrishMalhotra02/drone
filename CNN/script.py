import cv2
import json
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import transforms
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_model.pth")
CLASSES_PATH = os.path.join(SCRIPT_DIR, "class_names.json")
IMG_SIZE = 128
CONF_THRESHOLD = 0.6
MAX_BUFFER = 7
CONFIRM_THRESHOLD = 5
COMMAND_DELAY = 0.2


class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
    print(f"[✗] Model files not found. Need both '{MODEL_PATH}' and '{CLASSES_PATH}'.")
    print("    Run 'python train_model.py' after collecting data.")
    sys.exit(1)

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

device = get_device()
model = GestureCNN(len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

print(f"[✓] Loaded CNN model on {device} — classes: {class_names}")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

gesture_buffer = deque(maxlen=MAX_BUFFER)
last_command_time = 0


def classify_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        proba = torch.softmax(outputs, dim=1)[0]
        max_idx = proba.argmax().item()
        confidence = proba[max_idx].item()

    if confidence < CONF_THRESHOLD:
        return "UNKNOWN", confidence

    return class_names[max_idx].upper(), confidence


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

    print("\n Drone Control Started [CNN]")
    print("Show gestures to control drone\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)

        gesture, confidence = classify_frame(frame)
        gesture_buffer.append(gesture)

        stable_gesture = get_stable_gesture()
        if stable_gesture:
            execute_command(stable_gesture, confidence)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Stable: {stable_gesture}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow("Drone Control [CNN]", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
