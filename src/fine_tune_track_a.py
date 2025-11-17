from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os

# 1. Load a stronger base model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Loaded base model: all-mpnet-base-v2")

# 2. Prepare training data
train_examples = []
input_file = "../ref/track_a.jsonl"

with open(input_file, "r") as f:
    for line in tqdm(f, desc="Reading triplets"):
        ex = json.loads(line)
        anchor = ex["anchor_text"]

        if ex["text_a_is_closer"]:
            pos, neg = ex["text_a"], ex["text_b"]
        else:
            pos, neg = ex["text_b"], ex["text_a"]

        # For MultiNegativeLoss we add only anchor + positive
        train_examples.append(InputExample(texts=[anchor, pos]))

print("Total training pairs:", len(train_examples))

# 3. Use a multi-negative approach for better generalization
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

train_loss = losses.MultipleNegativesRankingLoss(model)

# 4. Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,                         # more epochs for tiny dataset 
    warmup_steps=20,                   # small warmup
    optimizer_params={"lr": 3e-5},     # better LR
    checkpoint_path="../models/checkpoints",
    checkpoint_save_steps=100,
    output_path="../models/fine_tuned_tracka_v2"
)

print("Training complete! Model saved to ../models/fine_tuned_tracka_v2")
