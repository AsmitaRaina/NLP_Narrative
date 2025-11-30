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
                    "You are a classification system. "
                    "Given an anchor story and two comparison stories, "
                    "decide which comparison story (A or B) is narratively closer "
                    "to the anchor.\n\n"
                    "Return a JSON object with the field 'closer' only, "
                    "set to either 'A' or 'B'. No explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Anchor: {anchor}\n\n"
                    f"Story A: {text_a}\n\n"
                    f"Story B: {text_b}\n\n"
                    "Which is closer to the anchor? Return only JSON."
                ),
            },
        ],
        response_format=SimilarityPrediction,
        temperature=0.0,
    )

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
