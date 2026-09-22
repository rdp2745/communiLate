"""Targeted checks that corpus metrics cannot see.

Aggregate scores understate what these experiments change. A model that
correctly switches ``vosotros`` to ``ustedes`` moves chrF++ by a fraction of a
point while getting the *entire point of the experiment* right. These checks
measure the specific behaviour directly, and they are what belongs in the
write-up's results table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..data.regional_rules import ES_419, ES_ES, detect_region

# Discourse markers, hedges and fillers that separate spoken-style Spanish from
# written/formal Spanish. Presence rate is a crude but honest proxy for whether
# a register adapter is actually shifting register.
CONVERSATIONAL_MARKERS_ES = [
    "o sea", "pues", "bueno", "vale", "venga", "oye", "mira", "digo",
    "la verdad", "en plan", "ya ves", "qué va", "claro", "a ver",
    "este", "pos", "órale", "ándale", "che", "bah", "eh",
]

CONVERSATIONAL_MARKERS_EN = [
    "you know", "i mean", "like", "well", "yeah", "okay", "so",
    "kind of", "sort of", "gonna", "wanna", "gotta", "right",
]


@dataclass
class ContrastiveResult:
    name: str
    score: float
    n_scored: int
    n_skipped: int = 0
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}={self.score:.3f} (n={self.n_scored}, skipped={self.n_skipped})"


def regional_accuracy(
    hypotheses: Sequence[str],
    expected_region: str,
) -> ContrastiveResult:
    """Fraction of outputs carrying the expected plural-you marking.

    Given English prompts that force plural "you", does the Spain adapter
    produce ``vosotros`` forms and the Latin American adapter ``ustedes`` forms?

    **The measurement is asymmetric, and the write-up has to say so.**
    Peninsular morphology is unique -- ``podeis`` can only be ``vosotros`` -- so
    es-ES is detected reliably. Latin American plural-you is syncretic with the
    third person plural: once the pronoun is dropped, ``pueden`` is identical
    whether it means "you all can" or "they can". So an es-419 output is often
    detectable only when the explicit pronoun survives.

    In practice this means the trustworthy number is the **es-ES rate**: how
    often each adapter produces Peninsular morphology. A Spain adapter scoring
    high and a Latin American adapter scoring near zero on that same check is
    the real result. Reading a low es-419 score as failure would be a mistake --
    it mostly reflects pronoun dropping, not a wrong dialect.

    Outputs with no plural-you marking are *skipped*, not counted wrong: a
    sentence can be a fine translation without marking plural you, and scoring
    those as failures would conflate "picked the wrong region" with "did not
    exercise the distinction". A high skip count on the es-ES check means the
    test prompts are not forcing the distinction hard enough; a high skip count
    on the es-419 check is expected for the reason above.
    """
    if expected_region not in (ES_ES, ES_419):
        raise ValueError(f"expected_region must be {ES_ES!r} or {ES_419!r}")

    correct = 0
    scored = 0
    skipped = 0
    wrong_examples: list[str] = []

    for hypothesis in hypotheses:
        detected = detect_region(hypothesis)
        if detected is None:
            skipped += 1
            continue
        scored += 1
        if detected == expected_region:
            correct += 1
        elif len(wrong_examples) < 10:
            wrong_examples.append(hypothesis)

    return ContrastiveResult(
        name=f"regional_accuracy[{expected_region}]",
        score=correct / scored if scored else float("nan"),
        n_scored=scored,
        n_skipped=skipped,
        detail={"correct": correct, "wrong_examples": wrong_examples},
    )


def marker_rate(hypotheses: Sequence[str], markers: Sequence[str]) -> ContrastiveResult:
    """Fraction of outputs containing at least one conversational marker.

    Interpret as a *direction* indicator, not a quality score: a higher rate
    after adaptation means the model is reaching for spoken-style discourse
    markers more often, which is the intended effect. It says nothing about
    whether it used them appropriately.
    """
    patterns = [re.compile(rf"\b{re.escape(m)}\b", re.IGNORECASE) for m in markers]
    hits = sum(any(p.search(h) for p in patterns) for h in hypotheses)
    return ContrastiveResult(
        name="conversational_marker_rate",
        score=hits / len(hypotheses) if hypotheses else float("nan"),
        n_scored=len(hypotheses),
        detail={"n_with_marker": hits, "n_markers_tested": len(markers)},
    )


def diff_rate(baseline: Sequence[str], adapted: Sequence[str]) -> ContrastiveResult:
    """Fraction of inputs where adaptation changed the output at all.

    The first number to look at after any training run. A near-zero rate means
    the adapter is not doing anything and no downstream metric will be
    meaningful; a rate near 1.0 on a small LoRA is worth being suspicious of.
    """
    if len(baseline) != len(adapted):
        raise ValueError("baseline and adapted must be the same length")
    changed = sum(b.strip() != a.strip() for b, a in zip(baseline, adapted))
    return ContrastiveResult(
        name="diff_rate",
        score=changed / len(baseline) if baseline else float("nan"),
        n_scored=len(baseline),
        detail={"n_changed": changed},
    )


def side_by_side(
    sources: Sequence[str],
    baseline: Sequence[str],
    adapted: Sequence[str],
    references: Sequence[str] | None = None,
    limit: int = 25,
    only_changed: bool = True,
) -> list[dict]:
    """Collect frozen-vs-adapted examples for manual inspection.

    These examples carry more weight in a write-up than any aggregate score,
    because they let a reader judge the change directly.
    """
    rows: list[dict] = []
    for i, (source, base, adapt) in enumerate(zip(sources, baseline, adapted)):
        if only_changed and base.strip() == adapt.strip():
            continue
        row = {"source": source, "baseline": base, "adapted": adapt}
        if references is not None and i < len(references):
            row["reference"] = references[i]
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows
