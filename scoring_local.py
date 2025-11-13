import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd

reference_dir = Path("./ref") 
prediction_dir = Path("./res")
score_dir = "./output/"

print("Reading prediction")
pred = pd.read_json(prediction_dir / "track_a.jsonl", lines=True)
print("Reading gold")
gold = pd.read_json(reference_dir / "track_a.jsonl", lines=True)

print("Checking Accuracy")
accuracy = accuracy_score(gold["text_a_is_closer"], pred["text_a_is_closer"])
scores = {
    "accuracy": accuracy,
}
print(scores)

with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
    score_file.write(json.dumps(scores))