
import cv2
import mediapipe as mp
import csv
import os

MODEL_PATH = "hand_landmarker.task"
DATA_FILE = "gesture_data.csv"

labels = {
    "1": "HELP",
    "2": "YES",
    "3": "NO",
    "4": "WATER",
    "5": "THANK YOU",
    "6": "I",
    "7": "YOU",
    "8": "PLEASE",
    "9": "HELLO",
    "a": "GOODBYE",
    "b": "SORRY",
    "c": "FOOD",
    "d": "GOOD",
    "e": "DRINK",
    "f": "NEED",
    "g": "HOME",
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

header = ["label", "hands_detected"]

for hand_no in range(1, 3):
    for point_no in range(21):
        for axis in ("x", "y", "z"):
            header.append(
                f"hand{hand_no}_point{point_no}_{axis}"
            )

file_exists = os.path.exists(DATA_FILE)

current_label = None
frame_number = 0

with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    if not file_exists or os.path.getsize(DATA_FILE) == 0:
        writer.writerow(header)

    with HandLandmarker.create_from_options(options) as landmarker:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Camera could not be opened.")
            raise SystemExit

        print("\n========== PSL DATA COLLECTION ==========")
        print("1 HELP | 2 YES | 3 NO | 4 WATER")
        print("5 THANK YOU | 6 I | 7 YOU | 8 PLEASE")
        print("9 HELLO | A GOODBYE | B SORRY | C FOOD")
        print("D GOOD | E DRINK | F NEED | G HOME")
        print("------------------------------------------")
        print("Select a word, then press SPACE to save.")
        print("Press Q to quit.")
        print("==========================================")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read from camera.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            frame_number += 1
            timestamp_ms = frame_number * 33

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            detected_hands = result.hand_landmarks
            hand_count = len(detected_hands)

            for hand in detected_hands:
                for point in hand:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                f"Hands: {hand_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            label_text = current_label or "Not selected"

            cv2.putText(
                frame,
                f"Word: {label_text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SPACE = Save | Q = Quit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("PSL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == 32:
                if current_label and hand_count > 0:
                    row = [current_label, hand_count]

                    for hand_index in range(2):

                        if hand_index < hand_count:
                            hand = detected_hands[hand_index]
                            wrist = hand[0]

                            coords = []

                            for point in hand:
                                coords.extend([
                                    point.x - wrist.x,
                                    point.y - wrist.y,
                                    point.z - wrist.z
                                ])

                            scale = max(
                                max(
                                    abs(coords[i])
                                    for i in range(0, 63, 3)
                                ),
                                max(
                                    abs(coords[i])
                                    for i in range(1, 63, 3)
                                ),
                                0.0001
                            )

                            coords = [
                                value / scale
                                for value in coords
                            ]

                            row.extend(coords)

                        else:
                            row.extend([0.0] * 63)

                    writer.writerow(row)
                    file.flush()

                    print(
                        f"Saved: {current_label} | "
                        f"Hands: {hand_count}"
                    )

                else:
                    print(
                        "Select a word and show at least one hand."
                    )

            else:
                pressed = chr(key).lower() if key < 128 else ""

                if pressed in labels:
                    current_label = labels[pressed]
                    print("Selected:", current_label)

        camera.release()
        cv2.destroyAllWindows()

print("Data collection closed.")
print("Saved data file:", DATA_FILE)
