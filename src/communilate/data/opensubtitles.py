"""OpenSubtitles (OPUS) raw reader.

Reads the Moses-format download -- two parallel plain-text files, one segment
per line, plus an optional ``.ids`` file -- and yields schema records. Filtering
is heavy enough that it is meant to run once via ``prepare``, writing clean
JSONL that training then reads through :mod:`.jsonl_dir`.

Caveats worth carrying into any write-up that uses this corpus: these are film
and TV subtitles, so the dialogue is *written to be spoken* rather than
spontaneous speech. Disfluencies, self-repairs and backchannels -- the strongest
markers of conversational register -- are largely absent, and subtitle
reading-speed limits mean many targets are condensed rather than faithful. It is
a dialogue-register corpus, not a spontaneous-speech corpus.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterator

from ..registry import LOADERS
from ..schema import Pair
from .base import resolve_path, stable_split

# OPUS id lines look like:
#   en/2009/1000001/3941878.xml.gz <TAB> es/2009/1000001/3941879.xml.gz <TAB> 1 <TAB> 1
# The third path component is the IMDb id, which is the join key for genre
# metadata -- the highest-leverage filter available for this corpus.
_IMDB = re.compile(r"/(\d{5,9})/")


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _find(root: Path, lang: str) -> Path:
    """Locate the side of the corpus for ``lang`` inside a Moses download."""
    patterns = [f"*.{lang}", f"*.{lang}.gz", f"*.{lang}.txt"]
    for pattern in patterns:
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(
        f"no '{lang}' side found in {root} (looked for {patterns}); expected a "
        f"Moses-format OPUS download"
    )


def _load_genre_map(path: Path | None) -> dict[str, list[str]]:
    """Optional ``{imdb_id: [genre, ...]}`` JSON, used by the genre filters."""
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8") as fh:
        return {str(k): list(v) for k, v in json.load(fh).items()}


@LOADERS.register("opensubtitles")
def load(data_cfg: dict, split: str) -> Iterator[Pair]:
    """Yield raw (unfiltered) pairs from an OPUS Moses download.

    Splits are derived by hashing the record id, so the same segment always
    lands in the same split even as the corpus is re-downloaded or extended.
    """
    root = Path(resolve_path(data_cfg["path"]))
    src_lang = data_cfg.get("raw_src_lang", "en")
    tgt_lang = data_cfg.get("raw_tgt_lang", "es")

    src_file = _find(root, src_lang)
    tgt_file = _find(root, tgt_lang)
    ids_file = next(iter(sorted(root.glob("*.ids*"))), None)
    genre_map = _load_genre_map(data_cfg.get("genre_map"))

    ratios = data_cfg.get("split_ratios") or {"train": 0.98, "dev": 0.01, "test": 0.01}
    corpus_name = data_cfg.get("source_corpus", "opensubtitles")
    register = data_cfg.get("register", "dialogue")
    region = data_cfg.get("region")

    src_fh, tgt_fh = _open(src_file), _open(tgt_file)
    ids_fh = _open(ids_file) if ids_file else None
    try:
        for lineno, (src_line, tgt_line) in enumerate(zip(src_fh, tgt_fh), start=1):
            src_text, tgt_text = src_line.strip(), tgt_line.strip()
            meta: dict = {}

            if ids_fh is not None:
                id_line = ids_fh.readline()
                imdb = _IMDB.search(id_line)
                if imdb:
                    meta["imdb_id"] = imdb.group(1)
                    if imdb.group(1) in genre_map:
                        meta["genres"] = genre_map[imdb.group(1)]

            if not src_text or not tgt_text:
                continue

            record_id = f"{corpus_name}-{lineno:09d}"
            assigned = stable_split(record_id, ratios)
            if assigned != split:
                continue

            yield Pair(
                id=record_id,
                src=src_text,
                tgt=tgt_text,
                src_lang=data_cfg.get("src_lang", "eng_Latn"),
                tgt_lang=data_cfg.get("tgt_lang", "spa_Latn"),
                register=register,
                region=region,
                source_corpus=corpus_name,
                split=assigned,
                meta=meta,
            )
    finally:
        src_fh.close()
        tgt_fh.close()
        if ids_fh is not None:
            ids_fh.close()
