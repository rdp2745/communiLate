"""The one record contract.

Every corpus -- OpenSubtitles, a rule-transformed regional set, hand-built
Gujarati family recordings -- is normalised into :class:`Pair` and written as
JSONL. Keeping the schema identical across all three experiments is what lets
the training and eval code stay corpus-agnostic: swapping data sources is a
config change, not a rewrite.

One record per line::

    {"id": "...", "src_lang": "eng_Latn", "tgt_lang": "spa_Latn",
     "src": "...", "tgt": "...", "register": "dialogue",
     "region": "es-ES", "source_corpus": "opensubtitles",
     "split": "train", "meta": {}}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# FLORES-200 codes. NLLB will not accept bare ISO codes ("es", "en"), and a
# wrong code fails at generation time rather than load time, so validate early.
ENGLISH = "eng_Latn"
SPANISH = "spa_Latn"
GUJARATI = "guj_Gujr"

KNOWN_LANGS = frozenset({ENGLISH, SPANISH, GUJARATI})

# Free-form by design, but these are the values the shipped configs use.
REGISTERS = frozenset({"formal", "dialogue", "conversational", "mixed", "unknown"})
SPLITS = frozenset({"train", "dev", "test"})


class SchemaError(ValueError):
    """Raised when a record does not satisfy the contract."""


@dataclass(slots=True)
class Pair:
    """A single aligned translation pair."""

    id: str
    src: str
    tgt: str
    src_lang: str
    tgt_lang: str
    register: str = "unknown"
    region: str | None = None
    source_corpus: str = "unknown"
    split: str = "train"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.id:
            raise SchemaError("record has an empty id")
        for name in ("src", "tgt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"record {self.id!r} has an empty {name}")
        for name in ("src_lang", "tgt_lang"):
            value = getattr(self, name)
            if value not in KNOWN_LANGS:
                raise SchemaError(
                    f"record {self.id!r} has {name}={value!r}; expected a FLORES-200 "
                    f"code from {sorted(KNOWN_LANGS)}"
                )
        if self.src_lang == self.tgt_lang:
            raise SchemaError(f"record {self.id!r} has src_lang == tgt_lang")
        if self.split not in SPLITS:
            raise SchemaError(
                f"record {self.id!r} has split={self.split!r}; expected one of {sorted(SPLITS)}"
            )
        if not isinstance(self.meta, dict):
            raise SchemaError(f"record {self.id!r} has a non-dict meta")

    def swapped(self) -> "Pair":
        """Return the same pair with the translation direction reversed.

        This is how both-direction training works without storing the corpus
        twice: the JSONL holds one canonical direction and the loader emits the
        reverse on demand.
        """
        return Pair(
            id=f"{self.id}::rev",
            src=self.tgt,
            tgt=self.src,
            src_lang=self.tgt_lang,
            tgt_lang=self.src_lang,
            register=self.register,
            region=self.region,
            source_corpus=self.source_corpus,
            split=self.split,
            meta={**self.meta, "reversed": True},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pair":
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise SchemaError(f"record {data.get('id')!r} has unknown fields: {sorted(unknown)}")
        return cls(**data)


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[Pair]:
    """Stream ``Pair`` records from a JSONL file.

    Errors name the line number -- with corpora this size, "bad record" without
    a line number is not an actionable message.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Pair.from_dict(json.loads(line))
            except (json.JSONDecodeError, SchemaError, TypeError) as exc:
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc
            if limit is not None and lineno >= limit:
                return


def write_jsonl(pairs: Iterable[Pair], path: str | Path) -> int:
    """Write records to JSONL, creating parent directories. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            pair.validate()
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
