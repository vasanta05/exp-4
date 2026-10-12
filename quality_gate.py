
import json
import sys
from pathlib import Path

MINIMUM_ACCURACY = 0.85
METRICS_FILE = Path("artifacts/metrics.json")

print("Reading model evaluation metrics...")

if not METRICS_FILE.exists():
    print("QUALITY GATE FAILED")
    print("Error: Model metrics file not found.")
    sys.exit(1)

with open(METRICS_FILE, "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model Accuracy: {accuracy:.2%}")
print(f"Required Accuracy: {MINIMUM_ACCURACY:.2%}")

if accuracy >= MINIMUM_ACCURACY:
    print("QUALITY GATE PASSED")
    print("Model performance satisfies the threshold.")
else:
    print("QUALITY GATE FAILED")
    print("Model accuracy is below the required threshold.")
    sys.exit(1)
