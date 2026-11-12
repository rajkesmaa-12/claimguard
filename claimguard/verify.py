"""Claim -> evidence -> NLI -> verdict."""
from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import Evidence, rank
from .nli import Verifier

SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NOT ENOUGH INFO"


@dataclass
class ScoredEvidence:
    evidence: Evidence
    entailment: float
    contradiction: float
    neutral: float


@dataclass
class Verdict:
    claim: str
    label: str
    confidence: float
    evidence: list[ScoredEvidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "evidence": [
                {"text": e.evidence.text, "source": e.evidence.source,
                 "entailment": round(e.entailment, 3), "contradiction": round(e.contradiction, 3)}
                for e in self.evidence
            ],
        }


def decide(scored: list[ScoredEvidence], threshold: float = 0.6) -> tuple[str, float]:
    """Strongest single piece of evidence wins; ties or weak evidence -> NOT ENOUGH INFO."""
    if not scored:
        return NEI, 0.0
    best_e = max(s.entailment for s in scored)
    best_c = max(s.contradiction for s in scored)
    if best_e >= threshold and best_e > best_c:
        return SUPPORTED, best_e
    if best_c >= threshold and best_c > best_e:
        return REFUTED, best_c
    return NEI, 1 - max(best_e, best_c)


class ClaimGuard:
    def __init__(self, source, verifier: Verifier, top_k: int = 5, threshold: float = 0.6):
        self.source = source
        self.verifier = verifier
        self.top_k = top_k
        self.threshold = threshold

    def check(self, claim: str) -> Verdict:
        evidence = rank(claim, self.source.candidates(claim), k=self.top_k)
        scored = []
        for ev in evidence:
            p = self.verifier(ev.text, claim)
            scored.append(ScoredEvidence(ev, p["entailment"], p["contradiction"], p["neutral"]))
        scored.sort(key=lambda s: max(s.entailment, s.contradiction), reverse=True)
        label, conf = decide(scored, self.threshold)
        return Verdict(claim, label, conf, scored)
