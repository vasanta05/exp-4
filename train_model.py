
import csv
import json
import random
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    random.seed(42)
    data = []

    for _ in range(500):
        cgpa = round(random.uniform(4, 10), 2)
        aptitude = random.randint(20, 100)
        communication = random.randint(20, 100)
        internships = random.randint(0, 3)

        score = (
            0.4 * cgpa / 10
            + 0.3 * aptitude / 100
            + 0.2 * communication / 100
            + 0.1 * internships / 3
        )

        placed = int(score >= 0.65)

        data.append([
            cgpa, aptitude, communication,
            internships, placed
        ])

    return data


def train_model():
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    data = create_dataset()

    columns = [
        "cgpa",
        "aptitude_score",
        "communication_score",
        "internships"
    ]

    with open(output_dir / "student_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns + ["placed"])
        writer.writerows(data)

    X = [row[:4] for row in data]
    y = [row[4] for row in data]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, output_dir / "placement_model.joblib")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Student Placement Model Trained Successfully!")
    print(f"Model Accuracy: {accuracy:.2%}")
    print("Model, metrics, and dataset saved.")

    return model, metrics


if __name__ == "__main__":
    train_model()
