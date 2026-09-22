"""Evaluation orchestration: frozen baseline vs. adapted model on one test set."""

from __future__ import annotations

from pathlib import Path

from ..config import REPO_ROOT, Config
from ..data import load_split
from ..translate import Translator
from . import contrastive as contrastive_checks
from .metrics import compute_all
from .report import build_report, save_report


def evaluate(cfg: Config, adapter: str | None = None, split: str = "test") -> dict:
    """Run both systems over ``split`` and write a report.

    Always evaluates the frozen baseline alongside the adapter: a score with
    nothing to compare against cannot answer the question any of these
    experiments ask.
    """
    pairs = list(load_split(cfg, split))
    if not pairs:
        raise RuntimeError(f"no records in split {split!r} for data.path={cfg.get('data.path')!r}")

    sources = [p.src for p in pairs]
    references = [p.tgt for p in pairs]
    metric_names = list(cfg.get("eval.metrics", ["chrf", "bleu"]))

    baseline_translator = Translator(cfg, adapter="none")
    baseline = baseline_translator.translate_pairs(pairs)
    baseline_hyps = [t.hypothesis for t in baseline]

    metrics = {"baseline": compute_all(sources, baseline_hyps, references, metric_names)}
    checks = []
    adapted_hyps = baseline_hyps

    adapter_path = adapter or cfg.get("inference.adapter_path")
    if adapter_path and adapter != "none":
        adapted = Translator(cfg, adapter=adapter_path).translate_pairs(pairs)
        adapted_hyps = [t.hypothesis for t in adapted]
        metrics["adapted"] = compute_all(sources, adapted_hyps, references, metric_names)
        checks.append(contrastive_checks.diff_rate(baseline_hyps, adapted_hyps))
    else:
        print("no adapter configured -- reporting the frozen baseline only")

    expected_region = cfg.get("eval.expected_region")
    if expected_region:
        checks.append(contrastive_checks.regional_accuracy(adapted_hyps, expected_region))
        checks.append(
            contrastive_checks.regional_accuracy(baseline_hyps, expected_region)
        )

    if cfg.get("eval.marker_rate", False):
        markers = (
            contrastive_checks.CONVERSATIONAL_MARKERS_ES
            if cfg["model.tgt_lang"].startswith("spa")
            else contrastive_checks.CONVERSATIONAL_MARKERS_EN
        )
        checks.append(contrastive_checks.marker_rate(adapted_hyps, markers))

    examples = contrastive_checks.side_by_side(
        sources,
        baseline_hyps,
        adapted_hyps,
        references,
        limit=int(cfg.get("eval.n_examples", 25)),
    )

    report = build_report(cfg, metrics, checks, examples)
    out_dir = Path(cfg.get("eval.output_dir", f"runs/{cfg.get('experiment.name','eval')}/eval"))
    json_path, md_path = save_report(report, out_dir)
    print(f"report written to {md_path}")
    for system, results in metrics.items():
        print(f"  {system}: " + "  ".join(str(r) for r in results))
    for check in checks:
        print(f"  {check}")
    return report
