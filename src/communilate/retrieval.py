"""Translation memory over the training corpus.

NLLB-200 is an encoder-decoder MT model, not an instruction-following LLM, so
retrieval here cannot mean "put similar examples in the prompt" -- there is no
prompt to put them in. What retrieval *can* do for this architecture is
translation-memory lookup: embed the source, find the nearest source in the
corpus, and if it is close enough, return the human translation that was already
paired with it.

That matters most for the Gujarati phase, where the family corpus is small,
idiosyncratic, and contains exactly the fixed expressions a general model has
never seen -- a near-exact match there is more trustworthy than anything the
model will generate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import REPO_ROOT, Config
from .schema import Pair


@dataclass
class Retrieved:
    """A translation-memory hit."""

    source: str
    target: str
    score: float
    record_id: str


class TranslationMemory:
    """Embedding-based nearest-neighbour lookup over stored pairs.

    Uses exact inner-product search: these corpora are small (a family corpus is
    hundreds to low thousands of pairs), so an approximate index would add
    tuning surface for no measurable speedup.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.get(
            "retrieval.encoder", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._encoder = None
        self._index = None
        self.pairs: list[Pair] = []

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def build(self, pairs: Iterable[Pair]) -> "TranslationMemory":
        import numpy as np

        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("cannot build a translation memory from zero pairs")

        embeddings = self.encoder.encode(
            [p.src for p in self.pairs],
            batch_size=int(self.cfg.get("retrieval.encode_batch_size", 64)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=bool(self.cfg.get("retrieval.progress", True)),
        )
        self._index = np.asarray(embeddings, dtype="float32")
        return self

    def query(self, texts: Sequence[str], top_k: int = 1) -> list[list[Retrieved]]:
        """Return the ``top_k`` nearest stored pairs for each input."""
        import numpy as np

        if self._index is None:
            raise RuntimeError("translation memory has not been built or loaded")

        queries = self.encoder.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        # Vectors are L2-normalised, so the dot product is cosine similarity.
        similarities = queries @ self._index.T

        results: list[list[Retrieved]] = []
        for row in similarities:
            best = np.argsort(-row)[:top_k]
            results.append(
                [
                    Retrieved(
                        source=self.pairs[i].src,
                        target=self.pairs[i].tgt,
                        score=float(row[i]),
                        record_id=self.pairs[i].id,
                    )
                    for i in best
                ]
            )
        return results

    def save(self, path: str | Path) -> Path:
        import numpy as np

        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self._index)
        with (path / "pairs.jsonl").open("w", encoding="utf-8") as fh:
            for pair in self.pairs:
                fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        with (path / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump({"encoder": self.model_name, "n_pairs": len(self.pairs)}, fh, indent=2)
        return path

    def load(self, path: str | Path) -> "TranslationMemory":
        import numpy as np

        from .schema import read_jsonl

        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        self._index = np.load(path / "embeddings.npy")
        self.pairs = list(read_jsonl(path / "pairs.jsonl"))
        return self
