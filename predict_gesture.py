
import cv2
import pickle
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue
from collections import deque

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Voice queue
speech_queue = queue.Queue()

# Voice function: initialize engine for every word
def speak_word(word):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)

        print("SPEAKING:", word)

        engine.say(word)
        engine.runAndWait()

        engine.stop()

        print("VOICE COMPLETED:", word)

    except Exception as error:
        print("VOICE ERROR:", error)

# Voice worker: process words one by one
def voice_worker():
    print("VOICE WORKER READY")

    while True:
        word = speech_queue.get()

        if word is None:
            speech_queue.task_done()
            break

        try:
            speak_word(word)

        finally:
            speech_queue.task_done()

# Start voice thread
voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)
voice_thread.start()

# MediaPipe settings
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
)

# Webcam settings
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_number = 0

CONFIDENCE_LIMIT = 0.55
stable_predictions = deque(maxlen=4)

last_spoken_word = ""

# Start detector
with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_number += 1

        result = detector.detect_for_video(
            mp_image,
            frame_number * 33
        )

        display_word = "NO HAND DETECTED"
        confidence_text = ""

        if result.hand_landmarks:

            landmarks = []

            for hand in result.hand_landmarks:

                wrist = hand[0]
                coords = []

                for point in hand:
                    coords.extend([
                        point.x - wrist.x,
                        point.y - wrist.y,
                        point.z - wrist.z
                    ])

                scale = max(
                    max(abs(coords[i]) for i in range(0, 63, 3)),
                    max(abs(coords[i]) for i in range(1, 63, 3)),
                    0.0001
                )

                coords = [value / scale for value in coords]
                landmarks.extend(coords)

            while len(landmarks) < 126:
                landmarks.extend([0.0] * 63)

            features = np.array(
                landmarks[:126]
            ).reshape(1, -1)

            probabilities = model.predict_proba(features)[0]

            best_index = np.argmax(probabilities)
            confidence = probabilities[best_index]
            prediction = model.classes_[best_index]

            confidence_text = f"Confidence: {confidence:.0%}"

            if confidence >= CONFIDENCE_LIMIT:

                display_word = prediction
                stable_predictions.append(prediction)

                if len(stable_predictions) == 4:

                    if len(set(stable_predictions)) == 1:

                        if prediction != last_spoken_word:

                            print("GESTURE CONFIRMED:", prediction)

                            speech_queue.put(prediction)
                            last_spoken_word = prediction

                            print("WORD SENT TO VOICE QUEUE")

            else:

                display_word = "SIGN NOT RECOGNIZED"
                stable_predictions.clear()
                last_spoken_word = ""

        else:

            stable_predictions.clear()
            last_spoken_word = ""

        # Display word
        cv2.putText(
            frame,
            "WORD: " + display_word,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Show one sign at a time",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        cv2.imshow(
            "Pakistani Sign Language Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

# Finish pending voice requests
speech_queue.join()

# Stop voice worker
speech_queue.put(None)
voice_thread.join()

print("PROGRAM CLOSED")