"""The default loader: read pre-built JSONL from a directory.

This is the one that makes corpora swappable. Anything that can be written into
the :class:`~communilate.schema.Pair` schema -- OpenSubtitles today, a better
conversational corpus later, hand-transcribed Gujarati recordings at the end --
is consumed by pointing ``data.path`` at a different folder::

    data:
      loader: jsonl_dir
      path: data/processed/opensubtitles_es_en

No other config or code has to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..registry import LOADERS
from ..schema import Pair, read_jsonl
from .base import resolve_path, stable_split

DEFAULT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


@LOADERS.register("jsonl_dir")
def load(data_cfg: dict, split: str) -> Iterator[Pair]:
    """Yield records for ``split`` from the directory at ``data_cfg['path']``.

    Two layouts are supported. If a per-split file exists it is used directly.
    Otherwise the loader falls back to ``all.jsonl`` (or any single ``*.jsonl``
    in the folder) and derives splits by hashing record ids, so a corpus you
    have not split yet still works without a preprocessing step.
    """
    root = Path(resolve_path(data_cfg["path"]))
    if not root.exists():
        raise FileNotFoundError(
            f"data.path {root} does not exist -- point it at a folder of JSONL "
            f"files matching the Pair schema"
        )

    files = {**DEFAULT_FILES, **(data_cfg.get("files") or {})}
    split_file = root / files.get(split, f"{split}.jsonl")

    if split_file.exists():
        yield from read_jsonl(split_file)
        return

    pooled = _find_pooled(root)
    ratios = data_cfg.get("split_ratios") or {"train": 0.9, "dev": 0.05, "test": 0.05}
    for pair in read_jsonl(pooled):
        if stable_split(pair.id, ratios) == split:
            pair.split = split
            yield pair


def _find_pooled(root: Path) -> Path:
    candidate = root / "all.jsonl"
    if candidate.exists():
        return candidate
    found = sorted(root.glob("*.jsonl"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"no per-split file and no unambiguous pooled file in {root}; "
        f"expected all.jsonl or exactly one *.jsonl, found {[f.name for f in found]}"
    )
