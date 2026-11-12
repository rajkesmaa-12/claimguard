"""Optional: a short natural-language explanation of the verdict from Claude."""
from __future__ import annotations

import os

from .verify import Verdict


def explain(verdict: Verdict) -> str | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    evidence = "\n".join(f"- {e.evidence.text} ({e.evidence.source})" for e in verdict.evidence[:3])
    prompt = (
        f"Claim: {verdict.claim}\nVerdict from an NLI model: {verdict.label}\nEvidence:\n{evidence}\n\n"
        "In 2-3 sentences, explain to a non-expert why the evidence supports this verdict. "
        "Only use the evidence above."
    )
    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("CLAIMGUARD_CLAUDE_MODEL", "claude-sonnet-4-5"),
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
