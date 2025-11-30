"""
Track A baseline system.

We use a naive prompt for chatGPT.
"""

import random
from enum import Enum

from openai import OpenAI
import pandas as pd
from pydantic import BaseModel


class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum


def predict(row):
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert narrative analyst. "
                    "Your job is to determine whether Story A or Story B is more "
                    "narratively similar to the Anchor story.\n\n"
                    "Narrative similarity must be judged using these criteria:\n"
                    "1. Theme similarity (core ideas, abstract conflict)\n"
                    "2. Event similarity (sequence of major plot actions)\n"
                    "3. Outcome similarity (how the story resolves)\n\n"
                    "You must return a JSON object with:\n"
                    "- explanation: a brief explanation of your reasoning\n"
                    "- closer: 'A' or 'B' indicating which is more similar\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Anchor story:\n{anchor}\n\n"
                    f"Story A:\n{text_a}\n\n"
                    f"Story B:\n{text_b}\n\n"
                    "Compare A and B to the Anchor using theme, events, and outcome. "
                    "Pick ONLY the most similar one."
                ),
            },
        ],
        response_format=SimilarityPrediction,
        temperature=0.0,
    )

    # Return True if A is closer, False otherwise
    return completion.choices[0].message.parsed.closer == ResponseEnum.A



baseline = "random"  # or "openai"
df = pd.read_json("data/dev_track_a.jsonl", lines=True)

if baseline == "openai":
    client = OpenAI()
    df["predicted_text_a_is_closer"] = df.apply(predict, axis=1)
elif baseline == "random":
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )
accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
print(f"Accuracy: {accuracy:.3f}")


df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

open("output/track_a.jsonl", "w").write(df.to_json(orient='records', lines=True))
