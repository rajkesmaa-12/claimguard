"""Natural Language Inference: does an evidence sentence entail or contradict the claim?"""
from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"


class Verifier(Protocol):
    def __call__(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return probabilities for 'entailment', 'contradiction', 'neutral'."""


def _normalise(label: str) -> str:
    label = label.lower()
    if label.startswith("entail"):
        return "entailment"
    if label.startswith("contra"):
        return "contradiction"
    return "neutral"


class TransformersNLI:
    """Wraps any Hugging Face NLI model (e.g. DeBERTa, RoBERTa-MNLI)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int | str = -1):
        from transformers import pipeline

        self.pipe = pipeline("text-classification", model=model_name, device=device, top_k=None)

    def __call__(self, premise: str, hypothesis: str) -> dict[str, float]:
        out = self.pipe({"text": premise, "text_pair": hypothesis})
        rows = out[0] if out and isinstance(out[0], list) else out
        probs = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for row in rows:
            probs[_normalise(row["label"])] += float(row["score"])
        return probs
