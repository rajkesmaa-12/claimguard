"""Find evidence sentences for a claim.

Sources:
* WikipediaSource – live search via the public MediaWiki API
* LocalSource     – your own .txt/.md files (offline, reproducible)

Candidate sentences are ranked against the claim with TF-IDF cosine similarity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ClaimGuard/0.1 (student research project)"}


@dataclass
class Evidence:
    text: str
    source: str
    score: float = 0.0


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", re.sub(r"\s+", " ", text))
    return [s.strip() for s in sents if 15 <= len(s.strip()) <= 500]


class LocalSource:
    def __init__(self, folder: str | Path):
        self.docs = {p.name: p.read_text(encoding="utf-8", errors="ignore")
                     for p in sorted(Path(folder).rglob("*")) if p.suffix in {".txt", ".md"}}
        if not self.docs:
            raise FileNotFoundError(f"No .txt/.md files in {folder}")

    def candidates(self, claim: str) -> list[Evidence]:
        return [Evidence(s, name) for name, text in self.docs.items() for s in split_sentences(text)]


class WikipediaSource:
    def __init__(self, n_pages: int = 3, timeout: float = 10.0):
        self.n_pages = n_pages
        self.timeout = timeout

    def _get(self, **params) -> dict:
        params |= {"format": "json", "formatversion": 2}
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def candidates(self, claim: str) -> list[Evidence]:
        hits = self._get(action="query", list="search", srsearch=claim, srlimit=self.n_pages)
        titles = [h["title"] for h in hits["query"]["search"]]
        if not titles:
            return []
        pages = self._get(action="query", prop="extracts", explaintext=1, exintro=1, exlimit="max", titles="|".join(titles))
        out = []
        for page in pages["query"]["pages"]:
            url = "https://en.wikipedia.org/wiki/" + page["title"].replace(" ", "_")
            out += [Evidence(s, url) for s in split_sentences(page.get("extract", ""))]
        return out


def rank(claim: str, candidates: list[Evidence], k: int = 5) -> list[Evidence]:
    if not candidates:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform([claim] + [c.text for c in candidates])
    scores = (m[1:] @ m[0].T).toarray().ravel()
    order = np.argsort(-scores)[:k]
    return [Evidence(candidates[i].text, candidates[i].source, float(scores[i])) for i in order if scores[i] > 0]
