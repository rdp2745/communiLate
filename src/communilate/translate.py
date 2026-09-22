"""Inference: frozen baseline vs. adapted model, through one code path.

The comparison at the centre of every experiment is "same input, adapter on vs.
adapter off". Keeping that in a single function -- with the adapter as an
argument rather than a separate script -- is what keeps the two sides honestly
comparable: identical generation settings, identical preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from .config import Config
from .model import load_for_inference, target_lang_id
from .schema import Pair


@dataclass
class Translation:
    """One generated translation plus the signal the gate needs."""

    source: str
    hypothesis: str
    reference: str | None = None
    src_lang: str = ""
    tgt_lang: str = ""
    confidence: float | None = None
    adapter: str | None = None
    meta: dict = field(default_factory=dict)


def _batched(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Translator:
    """Wraps a loaded model so a caller can translate many batches cheaply."""

    def __init__(self, cfg: Config, adapter: str | None = None):
        self.cfg = cfg
        self.model, self.tokenizer, self.adapter_path = load_for_inference(cfg, adapter)
        self.adapter_name = str(self.adapter_path) if self.adapter_path else None

    def translate(
        self,
        texts: Sequence[str],
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        with_confidence: bool | None = None,
    ) -> list[Translation]:
        """Translate a batch of strings."""
        import torch

        src_lang = src_lang or self.cfg["model.src_lang"]
        tgt_lang = tgt_lang or self.cfg["model.tgt_lang"]
        if with_confidence is None:
            with_confidence = bool(self.cfg.get("gating.enabled", False))

        self.tokenizer.src_lang = src_lang
        batch_size = int(self.cfg.get("inference.batch_size", 16))
        device = next(self.model.parameters()).device

        results: list[Translation] = []
        for chunk in _batched(list(texts), batch_size):
            encoded = self.tokenizer(
                list(chunk),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.cfg.get("model.max_source_length", 128)),
            ).to(device)

            gen_kwargs = {
                "forced_bos_token_id": target_lang_id(self.tokenizer, tgt_lang),
                "num_beams": int(self.cfg.get("generation.num_beams", 4)),
                "max_new_tokens": int(self.cfg.get("generation.max_new_tokens", 128)),
                "length_penalty": float(self.cfg.get("generation.length_penalty", 1.0)),
                "no_repeat_ngram_size": int(self.cfg.get("generation.no_repeat_ngram_size", 0)),
            }
            if with_confidence:
                gen_kwargs.update(output_scores=True, return_dict_in_generate=True)

            with torch.no_grad():
                output = self.model.generate(**encoded, **gen_kwargs)

            if with_confidence:
                sequences = output.sequences
                scores = _sequence_confidence(self.model, output, gen_kwargs["num_beams"])
            else:
                sequences = output
                scores = [None] * len(chunk)

            decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            for source, hypothesis, score in zip(chunk, decoded, scores):
                results.append(
                    Translation(
                        source=source,
                        hypothesis=hypothesis,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        confidence=score,
                        adapter=self.adapter_name,
                    )
                )
        return results

    def translate_pairs(self, pairs: Iterable[Pair]) -> list[Translation]:
        """Translate schema records, carrying the reference through for scoring."""
        pairs = list(pairs)
        if not pairs:
            return []
        # Records are grouped by direction so a both-direction dataset does not
        # silently get translated with one language pair's settings.
        out: list[Translation] = []
        by_direction: dict[tuple[str, str], list[Pair]] = {}
        for pair in pairs:
            by_direction.setdefault((pair.src_lang, pair.tgt_lang), []).append(pair)

        for (src_lang, tgt_lang), group in by_direction.items():
            translations = self.translate(
                [p.src for p in group], src_lang=src_lang, tgt_lang=tgt_lang
            )
            for pair, translation in zip(group, translations):
                translation.reference = pair.tgt
                translation.meta = {"id": pair.id, "region": pair.region, "register": pair.register}
                out.append(translation)
        return out


def _sequence_confidence(model, output, num_beams: int) -> list[float]:
    """Length-normalised mean token log-probability for each generated sequence.

    This is the signal the gate thresholds on. Length normalisation matters:
    raw sequence log-prob falls monotonically with length, so without it the
    gate would simply prefer short outputs.
    """
    import torch

    if getattr(output, "sequences_scores", None) is not None:
        # Beam search already reports a length-normalised score per sequence.
        return [float(s) for s in output.sequences_scores]

    try:
        transition = model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
    except (AttributeError, RuntimeError):
        return [None] * len(output.sequences)  # type: ignore[list-item]

    mask = torch.isfinite(transition)
    counts = mask.sum(dim=1).clamp(min=1)
    totals = torch.where(mask, transition, torch.zeros_like(transition)).sum(dim=1)
    return [float(v) for v in (totals / counts)]
