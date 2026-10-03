"""Loader dispatch and direction handling.

A loader is any callable ``(data_cfg, split) -> Iterable[Pair]`` registered under
a name. The pipeline only ever calls :func:`load_split`, so every consumer --
training, evaluation, retrieval index building -- sees the same records
regardless of which corpus the config points at.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Iterator

from ..config import REPO_ROOT, Config
from ..registry import LOADERS
from ..schema import Pair

DIRECTIONS = ("as_is", "swap", "both")


def resolve_path(raw: str) -> "object":
    """Resolve a config path relative to the repo root unless it is absolute."""
    from pathlib import Path

    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def _as_plain(cfg) -> dict:
    return cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)


def stable_split(record_id: str, ratios: dict[str, float]) -> str:
    """Assign a split by hashing the record id.

    Hash-based rather than random so the assignment is reproducible across runs
    and stable when the corpus grows: adding records never reshuffles the ones
    already placed, which would otherwise leak train data into a test set you
    had already reported numbers on.
    """
    digest = hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest()
    position = int.from_bytes(digest, "big") / float(1 << 64)
    cumulative = 0.0
    for name in ("train", "dev", "test"):
        cumulative += ratios.get(name, 0.0)
        if position < cumulative:
            return name
    return "train"


def apply_direction(pairs: Iterable[Pair], direction: str) -> Iterator[Pair]:
    """Emit pairs in the configured translation direction(s).

    ``both`` yields each pair twice, forward then reversed. Training on both
    directions from one corpus is why the JSONL stores a single canonical
    ordering -- the reverse is derived, never duplicated on disk.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    for pair in pairs:
        # Derive the reversed pair before yielding anything. Generators
        # interleave with their consumer, so computing it after the forward yield
        # would read whatever state the consumer left the record in -- defence in
        # depth alongside the consumers themselves not mutating.
        reversed_pair = pair.swapped() if direction in ("swap", "both") else None
        if direction in ("as_is", "both"):
            yield pair
        if reversed_pair is not None:
            yield reversed_pair


def load_split(cfg: Config, split: str) -> Iterator[Pair]:
    """Load one split of the dataset named by ``data.loader`` in the config."""
    data_cfg = cfg["data"]
    loader_name = data_cfg["loader"]
    loader = LOADERS.get(loader_name)

    pairs = loader(_as_plain(data_cfg), split)

    direction = data_cfg.get("direction", "as_is")
    pairs = apply_direction(pairs, direction)

    limit = data_cfg.get("max_records")
    if limit:
        pairs = _take(pairs, int(limit))
    return pairs


def _take(it: Iterable[Pair], n: int) -> Iterator[Pair]:
    for i, item in enumerate(it):
        if i >= n:
            return
        yield item
