"""Corpus metrics.

chrF++ is the headline number rather than BLEU. BLEU is close to insensitive to
the thing these experiments manipulate: register and dialect shifts are lexical
and morphological, and a correct ``vosotros -> ustedes`` rewrite barely moves
n-gram overlap. chrF++ is character-based and handles Spanish morphology far
better. BLEU is still reported because reviewers expect it, not because it is
informative here.

COMET is optional -- it is a large neural metric and slow on CPU -- but it is
the best proxy for human judgement available, so it is worth running on the
final test sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class MetricResult:
    name: str
    score: float
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}={self.score:.2f}"


def chrf(hypotheses: Sequence[str], references: Sequence[str], word_order: int = 2) -> MetricResult:
    """chrF++ (``word_order=2`` is what makes it chrF++ rather than chrF)."""
    import sacrebleu

    metric = sacrebleu.CHRF(word_order=word_order)
    score = metric.corpus_score(list(hypotheses), [list(references)])
    return MetricResult("chrf++", score.score, {"word_order": word_order})


def bleu(hypotheses: Sequence[str], references: Sequence[str], tokenize: str = "13a") -> MetricResult:
    import sacrebleu

    metric = sacrebleu.BLEU(tokenize=tokenize)
    score = metric.corpus_score(list(hypotheses), [list(references)])
    return MetricResult("bleu", score.score, {"signature": str(metric.get_signature())})


def ter(hypotheses: Sequence[str], references: Sequence[str]) -> MetricResult:
    import sacrebleu

    score = sacrebleu.TER().corpus_score(list(hypotheses), [list(references)])
    return MetricResult("ter", score.score)


def comet(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 16,
) -> MetricResult:
    """Learned metric. Downloads a sizeable checkpoint on first use."""
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(model_name))
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=batch_size, progress_bar=False)
    return MetricResult("comet", float(output.system_score) * 100, {"model": model_name})


def compute_all(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str],
    which: Sequence[str] = ("chrf", "bleu"),
) -> list[MetricResult]:
    """Run the metrics named in ``eval.metrics``.

    A failing metric is reported rather than raised: COMET in particular can
    fail for environment reasons (no network, no disk for the checkpoint), and
    losing the cheap metrics because the expensive one could not load would be
    a bad trade mid-experiment.
    """
    results: list[MetricResult] = []
    for name in which:
        try:
            if name == "chrf":
                results.append(chrf(hypotheses, references))
            elif name == "bleu":
                results.append(bleu(hypotheses, references))
            elif name == "ter":
                results.append(ter(hypotheses, references))
            elif name == "comet":
                results.append(comet(sources, hypotheses, references))
            else:
                raise ValueError(f"unknown metric {name!r}")
        except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
            results.append(MetricResult(name, float("nan"), {"error": str(exc)}))
    return results
