from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os
import pandas as pd

# ------------------------------
# CONFIG
# ------------------------------

synthetic_file = "../synthetic/synthetic_data_for_classification.jsonl"
dev_file = "../ref/dev_track_a.jsonl"
save_dir = "../models/fine_tuned_synthetic_tracka"

batch_size = 16
epochs = 3
lr = 2e-5

# ------------------------------
# LOAD BASE MODEL
# ------------------------------

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Loaded MPNet base model.")

# ------------------------------
# LOAD SYNTHETIC DATA AND CREATE TRIPLETS
# ------------------------------

train_examples = []

with open(synthetic_file, "r") as f:
    for line in tqdm(f, desc="Reading synthetic data"):
        ex = json.loads(line)

        anchor = ex["anchor_text"]
        a = ex["text_a"]
        b = ex["text_b"]

        if ex["text_a_is_closer"]:
            pos, neg = a, b
        else:
            pos, neg = b, a

        train_examples.append(InputExample(texts=[anchor, pos, neg]))

print(f"Total training triplets: {len(train_examples)}")

# ------------------------------
# DATALOADER
# ------------------------------

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.TripletLoss(model=model, triplet_margin=0.3)

# ------------------------------
# TRAIN
# ------------------------------

model._model_card_data = None

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=100,
    optimizer_params={"lr": lr},
    show_progress_bar=True
)

# ------------------------------
# SAVE MODEL
# ------------------------------

os.makedirs(save_dir, exist_ok=True)
model.save(save_dir)
print(f"Saved fine-tuned model to: {save_dir}")

# ------------------------------
# OPTIONAL: QUICK EVAL ON DEV (200 items)
# ------------------------------

from sentence_transformers import util

dev_df = pd.read_json(dev_file, lines=True)
preds = []

for _, row in dev_df.iterrows():
    anchor = model.encode(row["anchor_text"])
    a = model.encode(row["text_a"])
    b = model.encode(row["text_b"])

    sim_a = util.cos_sim(anchor, a)
    sim_b = util.cos_sim(anchor, b)

    preds.append(bool(sim_a > sim_b))

accuracy = (preds == dev_df["text_a_is_closer"]).mean()
print(f"Dev Accuracy after synthetic training: {accuracy:.4f}")
