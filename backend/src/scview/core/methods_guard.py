"""Keep a written methods section to the steps the recipe records.

The methods writer (``assistant.write_methods``) is told to use only the recorded
recipe, but a model fills in the steps a typical workflow has: a methods text
once said batch correction was performed on a dataset that had none. This check
reads the text for the steps that are commonly assumed and drops each sentence
that claims one the recipe does not name. A sentence that says a step was NOT
run ("No batch correction was applied") is kept: it is true.
"""

from __future__ import annotations

import re

# Steps a model tends to add, as (name, pattern). A step is recorded when the
# recipe digest itself matches the pattern (its step names have "_" for " ").
STEPS: tuple[tuple[str, re.Pattern], ...] = tuple((name, re.compile(rx, re.I)) for name, rx in (
    ("batch correction", r"batch[ -]?(?:effect )?correct\w*|batch[ -]?integrat\w*|\bharmony\b|"
                         r"\bscanorama\b|\bbbknn\b|\bcombat\b|\bintegrat(?:ed|ion)\b"),
    ("regression of covariates", r"\bregress\w*\b"),
    ("ambient RNA correction", r"\bambient\b|\bsoupx\b|\bcellbender\b|\bdecontx\b"),
    ("cell cycle scoring", r"\bcell[ -]cycle\b"),
    ("doublet detection", r"\bdoublets?\b|\bscrublet\b|\bscdblfinder\b"),
    ("trajectory inference", r"\bpseudotime\b|\btrajector\w+\b|\bpaga\b|\bdiffusion map\b"),
    ("cell type annotation", r"\bcelltypist\b|\bannotat\w+\b"),
    ("t-SNE", r"\bt-?sne\b"),
    ("gene set enrichment", r"\benrich\w*\b|\bgsea\b|over-?representation"),
))
_NEGATED = re.compile(r"\b(?:no|not|without|neither|nor|none)\b|n't\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ABBREV = re.compile(r"\b(?:e\.g|i\.e|et al|vs|approx|cf)\.", re.I)


def unrecorded_steps(text: str, recipe: str) -> list[str]:
    """The commonly assumed steps ``text`` mentions in an affirmative sentence
    that the ``recipe`` digest does not record."""
    plain = (recipe or "").replace("_", " ")
    out: list[str] = []
    for sentence in _sentences(text):
        if _NEGATED.search(sentence):
            continue
        for name, rx in STEPS:
            if rx.search(sentence) and not rx.search(plain) and name not in out:
                out.append(name)
    return out


def _sentences(text: str) -> list[str]:
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x01"), text or "")
    return [s.replace("\x01", ".") for s in _SENTENCE_END.split(protected)]


def keep_recorded(text: str, recipe: str) -> tuple[str, list[str]]:
    """``text`` without the sentences that claim a step the recipe does not record,
    and the names of the steps dropped. Paragraph breaks are kept."""
    dropped: list[str] = []
    paragraphs = []
    for para in re.split(r"\n\s*\n", text or ""):
        kept = []
        for sentence in _sentences(para):
            bad = [] if _NEGATED.search(sentence) else unrecorded_steps(sentence, recipe)
            if bad:
                dropped += [b for b in bad if b not in dropped]
                continue
            kept.append(sentence)
        if kept:
            paragraphs.append(" ".join(kept))
    return "\n\n".join(paragraphs), dropped
