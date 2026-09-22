"""Data loaders.

Importing this package registers every loader, so ``data.loader`` in a config
resolves without the caller importing the specific module.
"""

from . import jsonl_dir, opensubtitles  # noqa: F401  (import for side effects)
from .base import apply_direction, load_split, stable_split
from .filters import FilterReport, apply_filters, normalise_text

__all__ = [
    "load_split",
    "apply_direction",
    "stable_split",
    "apply_filters",
    "normalise_text",
    "FilterReport",
]
