import cv2
import mediapipe as mp
import time

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

DEAD_ZONE = 0.08
Kp = 1.5
SIZE_TARGET = 0.25
SIZE_DEAD_ZONE = 0.05
COMMAND_DELAY = 0.15

last_command_time = 0


def get_command(error_x, error_y, size_error):
    commands = []

    if abs(error_x) > DEAD_ZONE:
        if error_x > 0:
            commands.append(("RIGHT", abs(error_x)))
        else:
            commands.append(("LEFT", abs(error_x)))

    if abs(error_y) > DEAD_ZONE:
        if error_y > 0:
            commands.append(("DOWN", abs(error_y)))
        else:
            commands.append(("UP", abs(error_y)))

    if abs(size_error) > SIZE_DEAD_ZONE:
        if size_error > 0:
            commands.append(("BACKWARD", abs(size_error)))
        else:
            commands.append(("FORWARD", abs(size_error)))

    return commands


def execute_commands(commands):
    global last_command_time
    now = time.time()
    if now - last_command_time < COMMAND_DELAY:
        return
    for cmd, strength in commands:
        speed = min(int(strength * Kp * 100), 100)
        print(f"{cmd} (speed: {speed}%)")
    last_command_time = now


def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    tracking_lost_time = None

    print("\n Face Tracking Started")
    print("Drone will follow the detected face")
    print("Press ESC to quit\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        status = "NO FACE"
        commands = []

        if results.detections:
            tracking_lost_time = None
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            face_x = bbox.xmin + bbox.width / 2
            face_y = bbox.ymin + bbox.height / 2
            face_size = bbox.width * bbox.height

            error_x = face_x - 0.5
            error_y = face_y - 0.5
            size_error = face_size - SIZE_TARGET

            commands = get_command(error_x, error_y, size_error)

            if commands:
                execute_commands(commands)
                status = " | ".join([f"{c} {s:.0%}" for c, s in commands])
            else:
                status = "CENTERED"

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            color = (0, 255, 0) if status == "CENTERED" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            center_px = (int(face_x * w), int(face_y * h))
            cv2.circle(frame, center_px, 5, (0, 0, 255), -1)

            cv2.line(frame, (w // 2, h // 2), center_px, (255, 255, 0), 1)

            confidence = detection.score[0]
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            if tracking_lost_time is None:
                tracking_lost_time = time.time()
            elif time.time() - tracking_lost_time > 3.0:
                status = "FACE LOST — HOVERING"

        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 1)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
