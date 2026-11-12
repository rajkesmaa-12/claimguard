# 🛡️ ClaimGuard — Evidence-Based Claim Verification

[![tests](https://github.com/rajkesmaa-12/claimguard/actions/workflows/tests.yml/badge.svg)](https://github.com/rajkesmaa-12/claimguard/actions/workflows/tests.yml) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

ClaimGuard checks a factual claim against real evidence and returns **SUPPORTED**, **REFUTED** or **NOT ENOUGH INFO**, together with the sentences and sources behind the decision.

Output format:

```
$ python -m claimguard "The Eiffel Tower is located in Berlin."

REFUTED  (confidence <score>)

  • The Eiffel Tower is a wrought-iron lattice tower ... in Paris, France.
    entail=<p> contra=<p>  https://en.wikipedia.org/wiki/Eiffel_Tower
```

## Pipeline

```
claim ─► 1. Evidence retrieval   Wikipedia API (live) or your own .txt/.md corpus
         2. Sentence ranking     TF-IDF cosine similarity → top-k sentences
         3. NLI scoring          DeBERTa-v3 cross-encoder: entailment / contradiction / neutral
         4. Verdict              strongest evidence above threshold wins, else NOT ENOUGH INFO
         5. Explanation (opt.)   Claude summarises why, using only the retrieved evidence
```

This is the same retrieve-then-verify setup used in fact-checking research (e.g. the FEVER shared task), built from small, testable parts.

## Quick start

```bash
pip install -r requirements.txt

python -m claimguard "Mount Everest is in the Alps."           # checks against Wikipedia
python -m claimguard "Passau was founded in 1978." --local examples   # offline corpus
python -m claimguard "Water boils at 100 °C at sea level." --json

streamlit run app.py                                          # web demo
```

The first run downloads the NLI model `cross-encoder/nli-deberta-v3-small` (~180 MB). Any Hugging Face NLI model works via `--model`, e.g. `roberta-large-mnli`.
Set `ANTHROPIC_API_KEY` to get a short plain-language explanation of each verdict.

## Project structure

```
claimguard/
├── evidence.py   # Wikipedia + local sources, sentence splitting, TF-IDF ranking
├── nli.py        # Hugging Face NLI wrapper (label normalisation across models)
├── verify.py     # ClaimGuard pipeline + verdict rule
├── explain.py    # optional Claude explanation
└── __main__.py   # CLI
app.py            # Streamlit demo
tests/            # pytest (uses a stub NLI model, runs offline in ~1 s)
```

The NLI model is passed into `ClaimGuard` as a plain function, so the pipeline can be tested without downloading a model, and models can be swapped easily.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Roadmap
- [ ] Evaluate on a FEVER dev subset (label accuracy, evidence recall)
- [ ] Split multi-claim sentences into atomic claims with an LLM
- [ ] Dense retrieval + cross-encoder re-ranking
- [ ] Fine-tune a verdict classifier on aggregated evidence

## Tech
Python · Hugging Face Transformers (DeBERTa NLI) · scikit-learn · MediaWiki API · Anthropic API · Streamlit · pytest
