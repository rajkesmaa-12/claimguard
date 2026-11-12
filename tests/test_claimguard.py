"""Pipeline tests with a tiny rule-based stand-in for the NLI model (no downloads needed)."""
from pathlib import Path

from claimguard.evidence import Evidence, LocalSource, rank, split_sentences
from claimguard.nli import _normalise
from claimguard.verify import NEI, REFUTED, SUPPORTED, ClaimGuard, ScoredEvidence, decide

CORPUS = Path(__file__).parent.parent / "examples"


def fake_nli(premise: str, hypothesis: str) -> dict:
    p, h = premise.lower(), hypothesis.lower()
    if "eiffel" in p and "eiffel" in h:
        return ({"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03} if "paris" in h
                else {"entailment": 0.02, "contradiction": 0.94, "neutral": 0.04})
    return {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9}


def guard():
    return ClaimGuard(LocalSource(CORPUS), fake_nli, top_k=3)


def test_split_sentences():
    s = split_sentences("This is the first sentence here. And this is the second one. Short.")
    assert s == ["This is the first sentence here.", "And this is the second one."]


def test_rank_puts_relevant_sentence_first():
    cands = LocalSource(CORPUS).candidates("")
    top = rank("When was the University of Passau founded?", cands, k=1)
    assert "Passau" in top[0].text


def test_supported_claim():
    v = guard().check("The Eiffel Tower is in Paris.")
    assert v.label == SUPPORTED and v.confidence > 0.9
    assert "Eiffel" in v.evidence[0].evidence.text


def test_refuted_claim():
    assert guard().check("The Eiffel Tower is in Berlin.").label == REFUTED


def test_not_enough_info():
    assert guard().check("Bananas are the best fruit for programmers.").label == NEI


def test_decide_thresholds():
    ev = Evidence("x", "s")
    assert decide([]) == (NEI, 0.0)
    assert decide([ScoredEvidence(ev, 0.55, 0.1, 0.35)])[0] == NEI
    assert decide([ScoredEvidence(ev, 0.7, 0.75, 0.0)])[0] == REFUTED


def test_label_normalisation():
    assert _normalise("ENTAILMENT") == "entailment"
    assert _normalise("contradiction") == "contradiction"
    assert _normalise("LABEL_1") == "neutral"


def test_verdict_serialises():
    d = guard().check("The Eiffel Tower is in Paris.").to_dict()
    assert d["label"] == SUPPORTED and d["evidence"][0]["source"] == "corpus.md"
