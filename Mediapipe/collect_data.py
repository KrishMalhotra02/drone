import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")

def create_directory(gesture_name):
    path = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def collect_images():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    gesture_name = input("Enter the name of the gesture you want to collect data for: ").strip()

    save_path = create_directory(gesture_name)

    existing_files = len(os.listdir(save_path))
    count = existing_files

    print(f"\nCollecting data for '{gesture_name}'")
    print(f"Saving to: {save_path}")
    print("---------------------------------")
    print("Press 's' to save a single image")
    print("Press 'hold s' to rapid-fire save")
    print("Press 'q' to quit")
    print("---------------------------------")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Data Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            img_name = f"{gesture_name}_{count}.jpg"
            full_path = os.path.join(save_path, img_name)
            cv2.imwrite(full_path, frame)
            print(f"Saved: {img_name}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished. Total {count - existing_files} new images saved for '{gesture_name}'.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    while True:
        collect_images()
        cont = input("\nDo you want to collect for another gesture? (y/n): ").lower()
        if cont != 'y':
            break
