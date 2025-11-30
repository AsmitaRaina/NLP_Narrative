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
    """
    Uses the OpenAI API with an explicitly defined narrative similarity criteria.

    Returns:
        bool: True if Story A is predicted to be more similar to the anchor than Story B; False otherwise.
    """
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    
    # Define the precise rules based on the task guidelines
    system_prompt = (
        "You are an expert on narrative similarity. Your task is to select which story, A or B, is "
        "**more narratively similar** to the Anchor story. Narrative similarity must be judged "
        "primarily on the structure and content of the plot, not just general semantics or style. "
        "Your decision must prioritize alignment in:\n"
        "1. **Course of Action:** The sequence of central events, turning points, or core plot structure.\n"
        "2. **Abstract Theme:** The underlying ideas, motives, or moral message (e.g., revenge, coming of age, tragic love).\n"
        "3. **Outcomes:** The final results or consequences of the story's events.\n"
        "Generate a brief explanation that explicitly references these elements before stating your final, single-letter choice."
    )
    
    user_prompt = f"Anchor story: {anchor}\n\nStory A: {text_a}\n\nStory B: {text_b}"

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=SimilarityPrediction,
    )
    
    # Return True if the parsed response is 'A', False otherwise.
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
