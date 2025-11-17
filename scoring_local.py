import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd

# ------------------------------
# PATHS
# ------------------------------
reference_dir = Path("./ref")      # Contains dev_track_a.jsonl
prediction_dir = Path("./res")     # Contains track_a.jsonl predictions
score_dir = Path("./outputs")      # Output folder for scores.json

# Ensure score directory exists
score_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------
# READ FILES
# ------------------------------
print("Reading prediction")
pred_path = prediction_dir / "track_a.jsonl"
pred = pd.read_json(pred_path, lines=True)

print("Reading gold (dev set)")
gold_path = reference_dir / "dev_track_a.jsonl"   # <-- UPDATED HERE
gold = pd.read_json(gold_path, lines=True)

# ------------------------------
# ACCURACY CHECK
# ------------------------------
print("Checking Accuracy")
accuracy = accuracy_score(gold["text_a_is_closer"], pred["text_a_is_closer"])

scores = {
    "accuracy": accuracy,
}

print(scores)

# ------------------------------
# SAVE SCORE FILE
# ------------------------------
scores_path = score_dir / "scores.json"

with open(scores_path, "w") as score_file:
    score_file.write(json.dumps(scores))

print(f"Saved scores to {scores_path}")
