
import csv
import os
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"

EXPECTED_WORDS = [
    "HELP", "YES", "NO", "WATER",
    "THANK YOU", "I", "YOU", "PLEASE",
    "HELLO", "GOODBYE", "SORRY", "FOOD",
    "GOOD", "DRINK", "NEED", "HOME"
]

if not os.path.exists(DATA_FILE):
    print("Error: gesture_data.csv was not found.")
    raise SystemExit

features = []
labels = []

with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader, None)

    for row in reader:
        if len(row) != 128:
            print("Skipping a row with an unexpected number of columns.")
            continue

        label = row[0].strip().upper()

        try:
            values = [float(value) for value in row[2:]]
        except ValueError:
            print("Skipping a row containing invalid data.")
            continue

        features.append(values)
        labels.append(label)

if len(features) < 5:
    print("Please collect more samples before training.")
    raise SystemExit

counts = Counter(labels)

print("\n----- SAMPLE COUNT CHECK -----")
for word in EXPECTED_WORDS:
    print(f"{word}: {counts.get(word, 0)} samples")

extra_labels = set(counts) - set(EXPECTED_WORDS)
missing_words = set(EXPECTED_WORDS) - set(counts)

if extra_labels:
    print("\nUnexpected labels:", sorted(extra_labels))

if missing_words:
    print("\nMissing labels:", sorted(missing_words))

if missing_words or extra_labels:
    print("\nPlease check your collected data before training.")
    raise SystemExit

if any(counts[word] < 2 for word in EXPECTED_WORDS):
    print("\nSome gestures have too few samples.")
    raise SystemExit

print("\nTotal samples loaded:", len(features))
print("Total gesture classes:", len(counts))

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True
    )
)

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

print("\nTraining complete!")
print("Model saved as:", MODEL_FILE)
