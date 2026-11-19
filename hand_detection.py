
import cv2
import mediapipe as mp
import time

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

connections = mp.tasks.vision.HandLandmarker

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

previous_time = time.time()

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert OpenCV image to MediaPipe format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # Draw the 21 landmarks and their connections
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                height, width, _ = frame.shape

                points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Connect the points using standard hand connections
                for start, end in [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),
                    (0,17)
                ]:
                    cv2.line(
                        frame, points[start], points[end],
                        (255, 0, 0), 2
                    )

            cv2.putText(
                frame, "Hand Detected!",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, "Show your hand",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

        cv2.imshow("PSL - Hand Detection", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
