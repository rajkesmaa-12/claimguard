"""python -m claimguard "The Eiffel Tower is in Berlin." [--local examples]"""
import argparse
import json

from .evidence import LocalSource, WikipediaSource
from .explain import explain
from .nli import DEFAULT_MODEL, TransformersNLI
from .verify import ClaimGuard


def main() -> None:
    p = argparse.ArgumentParser(prog="claimguard")
    p.add_argument("claim")
    p.add_argument("--local", help="folder of .txt/.md files to use instead of Wikipedia")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    source = LocalSource(a.local) if a.local else WikipediaSource()
    verdict = ClaimGuard(source, TransformersNLI(a.model), top_k=a.k).check(a.claim)
    if a.json:
        print(json.dumps(verdict.to_dict(), indent=2))
        return
    print(f"\n{verdict.label}  (confidence {verdict.confidence:.2f})\n")
    for e in verdict.evidence[:3]:
        print(f"  • {e.evidence.text}\n    entail={e.entailment:.2f} contra={e.contradiction:.2f}  {e.evidence.source}")
    if (why := explain(verdict)):
        print(f"\nWhy: {why}")


if __name__ == "__main__":
    main()
