"""Evaluation: corpus metrics, targeted checks, and report assembly."""

from .contrastive import (
    ContrastiveResult,
    diff_rate,
    marker_rate,
    regional_accuracy,
    side_by_side,
)
from .metrics import MetricResult, bleu, chrf, comet, compute_all, ter
from .report import build_report, save_report, to_markdown
from .runner import evaluate

__all__ = [
    "evaluate",
    "compute_all",
    "chrf",
    "bleu",
    "ter",
    "comet",
    "MetricResult",
    "ContrastiveResult",
    "regional_accuracy",
    "marker_rate",
    "diff_rate",
    "side_by_side",
    "build_report",
    "save_report",
    "to_markdown",
]
