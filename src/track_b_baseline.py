import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import json
import os


def evaluate(labeled_data_path, embedding_lookup):
    """Evaluate embeddings on Track A labels using cosine similarity."""
    df = pd.read_json(labeled_data_path, lines=True)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Compute cosine similarities
    df["sim_a"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["a_embedding"]), axis=1
    )
    df["sim_b"] = df.apply(
        lambda row: cos_sim(row["anchor_embedding"], row["b_embedding"]), axis=1
    )

    # Predict and calculate accuracy
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


# ------------------------------
# CONFIGURATION
# ------------------------------
baseline = "sbert"  # or "random"

# Paths adjusted for your setup
data_path_b = "../ref/sample_track_b.jsonl"   # Track B input (stories only)
data_path_a = "../ref/track_a.jsonl"          # Track A labels for evaluation
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# MODEL AND EMBEDDINGS
# ------------------------------
if baseline == "sbert":
    # Use the stronger model from Track A
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("Encoding Track B stories with all-mpnet-base-v2...")
    data = pd.read_json(data_path_b, lines=True)
    embeddings = model.encode(data["text"], show_progress_bar=True)

    # Normalize embeddings for better cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

elif baseline == "random":
    data = pd.read_json(data_path_b, lines=True)
    embeddings = torch.rand((len(data), 512))
else:
    sys.exit("Invalid baseline method. Choose 'sbert' or 'random'.")

# Map story text → embedding
embedding_lookup = dict(zip(data["text"], embeddings))

# ------------------------------
# EVALUATION (optional local check)
# ------------------------------
print("Evaluating on Track A data...")
accuracy = evaluate(data_path_a, embedding_lookup)
print(f"✅ Local evaluation accuracy: {accuracy:.3f}")

# ------------------------------
# SAVE EMBEDDINGS
# ------------------------------
np.save(f"{output_dir}/track_b.npy", embeddings)

# Also save JSONL version for easier inspection
with open(f"{output_dir}/track_b.jsonl", "w") as f:
    for emb in embeddings:
        f.write(json.dumps({"embedding": emb.tolist()}) + "\n")

print(f"✅ Embeddings saved to:\n- {output_dir}/track_b.npy\n- {output_dir}/track_b.jsonl")
