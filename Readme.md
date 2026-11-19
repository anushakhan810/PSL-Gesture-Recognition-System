
# AI-Based Pakistani Sign Language Gesture Recognition and Communication System

An AI-based project that recognizes selected Pakistani Sign Language (PSL) hand gestures using computer vision and machine learning. The system displays the recognized English word and provides spoken audio output.

## Project Overview

This project aims to make basic communication easier by recognizing selected hand gestures through a webcam.

The system uses MediaPipe to detect hand landmarks, a machine learning model to classify gestures, and text-to-speech technology to speak the recognized words.

This is a prototype for selected gestures, not a complete Pakistani Sign Language translator.

## Features

- Real-time hand detection using a webcam.
- Supports detection of up to two hands.
- Recognizes 16 selected gestures.
- Displays the predicted English word.
- Shows prediction confidence.
- Uses stable predictions to reduce incorrect recognition.
- Provides spoken output using text-to-speech.
- Allows users to collect additional gesture samples and retrain the model.

## Supported Gestures

1. HELP
2. YES
3. NO
4. WATER
5. THANK YOU
6. I
7. YOU
8. PLEASE
9. HELLO
10. GOODBYE
11. SORRY
12. FOOD
13. GOOD
14. DRINK
15. NEED
16. HOME

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Pickle
- Pyttsx3

## Project Files

```text
PSL_Gesture_Recognition_System/
│
├── collect_data.py
├── train_model.py
├── predict_gesture.py
├── hand_landmarker.task
├── gesture_data.csv
├── gesture_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Download the Project

Download the ZIP file from the GitHub repository and extract it.

### 2. Install Python

Install a compatible Python version.

### 3. Open the Project Folder

Open the extracted folder in VS Code.

### 4. Create a Virtual Environment

```bash
python -m venv .venv
```

### 5. Activate the Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 6. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7. Run the Recognition System

```bash
python predict_gesture.py
```

Allow webcam access when requested.

Press **Q** to close the recognition window.

## How the System Works

1. The webcam captures live video.
2. OpenCV reads the video frames.
3. MediaPipe detects hand landmarks.
4. The system extracts and normalizes landmark coordinates.
5. The trained machine learning model predicts the gesture.
6. A confidence threshold and prediction stability check help confirm the result.
7. The recognized English word appears on the screen.
8. Pyttsx3 generates spoken audio output.

## Training Your Own Model

To collect additional gesture samples:

```bash
python collect_data.py
```

After collecting samples, train the model:

```bash
python train_model.py
```

This updates the trained model file:

```text
gesture_model.pkl
```

Then run the recognition system again:

```bash
python predict_gesture.py
```

## Important Notes

- A working webcam is required.
- The project is designed for Windows setup instructions.
- Install the dependencies listed in `requirements.txt`.
- Keep `hand_landmarker.task`, `gesture_data.csv`, and `gesture_model.pkl` in the project directory.
- Recognition performance may vary with lighting, camera quality, hand position, and individual signing styles.
- The model recognizes only the selected gestures included in its training dataset.

## Project Scope and Limitations

This project recognizes a limited vocabulary of selected static hand gestures. It does not translate complete sign language sentences, facial expressions, or continuous signing.

The recognition model may require additional training samples to improve performance for certain gestures.

## Future Improvements

- Expand the gesture vocabulary.
- Improve recognition accuracy.
- Add support for dynamic gestures.
- Develop a graphical user interface.
- Explore sentence-level sign language recognition.

## Project Purpose

This project was developed as an academic AI and computer vision project to explore how machine learning can support accessible communication.

---

Project: AI-Based Pakistani Sign Language Gesture Recognition and Communication System
