from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import os

# 1. Load base model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("✅ Loaded base model: all-mpnet-base-v2")

# 2. Prepare training examples (anchor, positive, negative)
train_examples = []
input_file = "../ref/track_a.jsonl"

with open(input_file, "r") as f:
    for line in tqdm(f, desc="Preparing training triplets"):
        ex = json.loads(line)
        anchor = ex["anchor_text"]
        if ex["text_a_is_closer"]:
            pos, neg = ex["text_a"], ex["text_b"]
        else:
            pos, neg = ex["text_b"], ex["text_a"]
        train_examples.append(InputExample(texts=[anchor, pos, neg]))

print(f"✅ Total triplets prepared: {len(train_examples)}")

# 3. DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# 4. Define Triplet Loss
train_loss = losses.TripletLoss(model=model, triplet_margin=0.3)

# 5. Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    optimizer_params={'lr': 1e-5}
)

# 6. Save fine-tuned model
os.makedirs("../models", exist_ok=True)
model.save("../models/fine_tuned_tracka")
print("✅ Fine-tuned model saved to ../models/fine_tuned_tracka")
