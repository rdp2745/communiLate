"""Confidence gating: choose between translation memory, adapter and baseline.

Three routes, in priority order:

1. **Translation memory.** If the nearest stored source clears
   ``gating.tm_threshold``, return the human translation already paired with it.
   A near-exact match against curated data beats anything the model generates.
2. **Adapter.** Otherwise use the adapted model's output, provided its
   length-normalised confidence clears ``gating.confidence_threshold``.
3. **Frozen baseline.** If the adapter is not confident, fall back to the
   unadapted model.

Route 3 is the part that earns the architecture its keep: a LoRA trained on a
few thousand in-domain pairs will confidently produce nonsense on out-of-domain
input, and the frozen baseline is the safer answer there. Whether that is
actually true for your data is an empirical question -- which is why
:func:`route_batch` records the chosen route on every item, so the ablation can
report how often each path fires and what it cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .config import Config
from .retrieval import Retrieved, TranslationMemory
from .translate import Translation


class Route(str, Enum):
    TRANSLATION_MEMORY = "translation_memory"
    ADAPTER = "adapter"
    BASELINE = "baseline"


@dataclass
class GatedTranslation:
    """A final translation plus the evidence for how it was chosen."""

    source: str
    hypothesis: str
    route: Route
    reference: str | None = None
    adapter_hypothesis: str | None = None
    baseline_hypothesis: str | None = None
    tm_hit: Retrieved | None = None
    confidence: float | None = None


def route_one(
    source: str,
    adapter: Translation,
    baseline: Translation | None,
    tm_hit: Retrieved | None,
    tm_threshold: float,
    confidence_threshold: float | None,
) -> GatedTranslation:
    """Apply the three-way decision to a single item."""
    if tm_hit is not None and tm_hit.score >= tm_threshold:
        chosen, route = tm_hit.target, Route.TRANSLATION_MEMORY
    elif (
        confidence_threshold is not None
        and adapter.confidence is not None
        and adapter.confidence < confidence_threshold
        and baseline is not None
    ):
        chosen, route = baseline.hypothesis, Route.BASELINE
    else:
        chosen, route = adapter.hypothesis, Route.ADAPTER

    return GatedTranslation(
        source=source,
        hypothesis=chosen,
        route=route,
        reference=adapter.reference,
        adapter_hypothesis=adapter.hypothesis,
        baseline_hypothesis=baseline.hypothesis if baseline else None,
        tm_hit=tm_hit,
        confidence=adapter.confidence,
    )


def route_batch(
    cfg: Config,
    sources: Sequence[str],
    adapter_outputs: Sequence[Translation],
    baseline_outputs: Sequence[Translation] | None = None,
    memory: TranslationMemory | None = None,
) -> list[GatedTranslation]:
    """Route a whole batch, returning one decision per input."""
    tm_threshold = float(cfg.get("gating.tm_threshold", 0.95))
    confidence_threshold = cfg.get("gating.confidence_threshold")
    confidence_threshold = None if confidence_threshold is None else float(confidence_threshold)

    hits: list[Retrieved | None]
    if memory is not None and cfg.get("retrieval.enabled", False):
        hits = [group[0] if group else None for group in memory.query(sources, top_k=1)]
    else:
        hits = [None] * len(sources)

    baselines = baseline_outputs or [None] * len(sources)
    return [
        route_one(src, adapter, baseline, hit, tm_threshold, confidence_threshold)
        for src, adapter, baseline, hit in zip(sources, adapter_outputs, baselines, hits)
    ]


def route_counts(decisions: Sequence[GatedTranslation]) -> dict[str, int]:
    """Tally routes -- the headline number for the gating ablation."""
    counts = {route.value: 0 for route in Route}
    for decision in decisions:
        counts[decision.route.value] += 1
    return counts
