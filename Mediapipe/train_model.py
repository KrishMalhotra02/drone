import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_model.pkl")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    wrist = hand.landmark[0]
    features = []
    for lm in hand.landmark:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    return features


def load_dataset():
    X = []
    y = []
    skipped = 0

    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' directory not found. Run collect_data.py first.")
        return None, None

    gesture_classes = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not gesture_classes:
        print("Error: No gesture folders found in dataset/")
        return None, None

    print(f"Found gesture classes: {gesture_classes}")
    print()

    for gesture in gesture_classes:
        gesture_dir = os.path.join(DATA_DIR, gesture)
        images = [f for f in os.listdir(gesture_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  Processing '{gesture}': {len(images)} images...", end=" ")
        count = 0

        for img_file in images:
            img_path = os.path.join(gesture_dir, img_file)
            features = extract_landmarks(img_path)

            if features is not None:
                X.append(features)
                y.append(gesture.upper())
                count += 1
            else:
                skipped += 1

        print(f"{count} landmarks extracted")

    print(f"\nTotal samples: {len(X)} | Skipped (no hand detected): {skipped}")
    return np.array(X), np.array(y)


def train():
    print("Gesture Model Training")
    print()

    X, y = load_dataset()

    if X is None or len(X) == 0:
        print("No data to train on. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples")
    print()

    classifier = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=1, n_jobs=-1)

    print("Training Random Forest...")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.2%}")
    print()
    print(classification_report(y_test, y_pred))

    model_data = {"model": classifier, "classes": list(classifier.classes_)}

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to '{MODEL_PATH}'")
    print(f"Classes: {model_data['classes']}")


if __name__ == "__main__":
    train()
