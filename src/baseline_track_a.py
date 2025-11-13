"""
Modified baseline system for Track A (local version).

- Compatible with official SemEval structure
- Uses your fine-tuned SentenceTransformer model instead of the OpenAI API
- Outputs valid 'track_a.jsonl' for CodaBench submission
"""

import random
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ------------------------------
# CONFIGURATION
# ------------------------------
baseline = "sbert"  # or "random"

input_file = "../ref/track_a.jsonl"   # Input dataset of triples
output_file = "../outputs/track_a.jsonl"  # Output predictions for submission

# ------------------------------
# MODEL SETUP
# ------------------------------
if baseline == "sbert":
    # Load fine-tuned model from the training step
    model = SentenceTransformer("../models/fine_tuned_tracka")
    print("✅ Using fine-tuned model: ../models/fine_tuned_tracka")
else:
    model = None
    print("⚠️ Using random baseline (for testing only)")

# ------------------------------
# PREDICTIONS
# ------------------------------
predictions = []

with open(input_file, "r") as f:
    for line in tqdm(f, desc="Processing Track A examples"):
        example = json.loads(line)
        anchor = example["anchor_text"]
        text_a = example["text_a"]
        text_b = example["text_b"]

        if baseline == "sbert":
            # Compute embeddings
            emb_anchor = model.encode(anchor)
            emb_a = model.encode(text_a)
            emb_b = model.encode(text_b)

            # Compute cosine similarities
            sim_a = util.cos_sim(emb_anchor, emb_a)
            sim_b = util.cos_sim(emb_anchor, emb_b)

            # Determine which is narratively closer
            text_a_is_closer = bool(sim_a > sim_b)

        elif baseline == "random":
            # Randomly assign for testing
            text_a_is_closer = random.choice([True, False])

        predictions.append({"text_a_is_closer": text_a_is_closer})

# ------------------------------
# SAVE OUTPUT FILE
# ------------------------------
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")

print(f"✅ Done! Predictions saved to {output_file}")