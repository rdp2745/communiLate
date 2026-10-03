"""Corpus filters.

Sized for OpenSubtitles, where the raw dump is tens of millions of noisy pairs
and the useful yield is a small fraction of that. Each filter is a predicate
over a :class:`~communilate.schema.Pair` so they compose in any order, and the
chain reports how many records each one dropped -- without that breakdown it is
impossible to tell an over-aggressive threshold from a genuinely dirty corpus.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from ..schema import Pair

Predicate = Callable[[Pair], bool]

_WS = re.compile(r"\s+")
# Subtitle cruft: speaker dashes, italics/font tags, [sound effects], (asides),
# and the "ripped by" / OpenSubtitles.org advertising lines that ride along in
# a surprising share of community-uploaded files.
_TAG = re.compile(r"<[^>]{1,40}>")
_BRACKETED = re.compile(r"[\[\(][^\]\)]{0,60}[\]\)]")
_LEAD_DASH = re.compile(r"^\s*[-–—]\s*")
_AD = re.compile(
    r"(opensubtitles|subtitles? by|sync(hroniz|hronis)ed by|ripped by|www\.|http|"
    r"subtitulos por|subtitulado por|traducido por|resync)",
    re.IGNORECASE,
)


def normalise_text(text: str) -> str:
    """Strip subtitle markup and collapse whitespace. Cheap, applied to all."""
    text = unicodedata.normalize("NFC", text)
    text = _TAG.sub(" ", text)
    text = _BRACKETED.sub(" ", text)
    text = _LEAD_DASH.sub("", text)
    return _WS.sub(" ", text).strip()


def min_tokens(n: int = 4) -> Predicate:
    """Drop fragments.

    Short segments are overwhelmingly interjections and stock phrases; they are
    also the most duplicated lines in the corpus, so a LoRA trained on unfiltered
    data spends much of its capacity on '¿Qué?'.
    """

    def _f(pair: Pair) -> bool:
        return len(pair.src.split()) >= n and len(pair.tgt.split()) >= n

    return _f


def max_tokens(n: int = 100) -> Predicate:
    """Drop over-long segments -- usually merged cues or alignment failures."""

    def _f(pair: Pair) -> bool:
        return len(pair.src.split()) <= n and len(pair.tgt.split()) <= n

    return _f


def length_ratio(max_ratio: float = 2.5) -> Predicate:
    """Drop pairs whose sides differ wildly in length.

    The single highest-yield signal for timing-misaligned subtitle pairs: a
    one-word line matched against a full sentence is almost always an off-by-one
    shift between two independently authored subtitle files.
    """

    def _f(pair: Pair) -> bool:
        a, b = len(pair.src.split()), len(pair.tgt.split())
        if a == 0 or b == 0:
            return False
        return max(a, b) / min(a, b) <= max_ratio

    return _f


def min_alignment_score(threshold: float) -> Predicate:
    """Keep pairs whose OPUS alignment confidence clears ``threshold``.

    Records without a score are kept -- corpora other than OpenSubtitles do not
    carry one, and silently dropping all of them would be a nasty surprise.
    """

    def _f(pair: Pair) -> bool:
        score = pair.meta.get("alignment_score")
        return True if score is None else float(score) >= threshold

    return _f


def no_advertising() -> Predicate:
    """Drop the uploader-credit and site-advertising lines."""

    def _f(pair: Pair) -> bool:
        return not (_AD.search(pair.src) or _AD.search(pair.tgt))

    return _f


def not_mostly_punctuation(max_ratio: float = 0.5) -> Predicate:
    def _f(pair: Pair) -> bool:
        for text in (pair.src, pair.tgt):
            letters = sum(ch.isalpha() for ch in text)
            if not text or letters / max(len(text), 1) < (1 - max_ratio):
                return False
        return True

    return _f


def exclude_genres(genres: Iterable[str]) -> Predicate:
    """Drop titles whose IMDb genres intersect ``genres``.

    The highest-leverage filter for this project. OpenSubtitles skews hard
    toward action and crime, whose dialogue is threats and shouting; domestic
    conversation lives in comedy and family drama. Only applies to records that
    carry genre metadata.
    """
    blocked = {g.strip().lower() for g in genres}

    def _f(pair: Pair) -> bool:
        found = pair.meta.get("genres")
        if not found:
            return True
        return not blocked.intersection({str(g).lower() for g in found})

    return _f


def include_genres(genres: Iterable[str]) -> Predicate:
    allowed = {g.strip().lower() for g in genres}

    def _f(pair: Pair) -> bool:
        found = pair.meta.get("genres")
        if not found:
            return True
        return bool(allowed.intersection({str(g).lower() for g in found}))

    return _f


def no_untranslated_copy() -> Predicate:
    """Drop pairs whose two sides are the same string.

    Common in subtitles: untranslated lines, proper nouns standing alone, and
    songs left in the original. They teach the model to copy rather than
    translate, which is a failure mode worth avoiding outright.
    """

    def _f(pair: Pair) -> bool:
        return pair.src.strip().casefold() != pair.tgt.strip().casefold()

    return _f


def language_match() -> Predicate:
    """Drop pairs where either side is confidently the wrong language.

    Abstains on short segments rather than guessing -- see
    :mod:`communilate.data.langid`.
    """
    from .langid import matches_expected

    def _f(pair: Pair) -> bool:
        return matches_expected(pair.src, pair.src_lang) and matches_expected(
            pair.tgt, pair.tgt_lang
        )

    return _f


def no_all_caps(min_length: int = 8) -> Predicate:
    """Drop shouted lines.

    All-caps subtitle text is usually on-screen signage, titles, or a stylistic
    choice of one uploader, none of which is the conversational register this
    project is after.
    """

    def _f(pair: Pair) -> bool:
        for text in (pair.src, pair.tgt):
            letters = [c for c in text if c.isalpha()]
            if len(letters) >= min_length and all(c.isupper() for c in letters):
                return False
        return True

    return _f


def max_repeated_chars(limit: int = 4) -> Predicate:
    """Drop lines with long character runs (``noooooo``, ``!!!!!!!``)."""
    pattern = re.compile(rf"(.)\1{{{limit},}}")

    def _f(pair: Pair) -> bool:
        return not (pattern.search(pair.src) or pattern.search(pair.tgt))

    return _f


def _fingerprint(pair: Pair) -> str:
    key = f"{pair.src.casefold()}|||{pair.tgt.casefold()}"
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class FilterReport:
    """Per-filter drop counts, so a low yield can be attributed."""

    seen: int = 0
    kept: int = 0
    dropped: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dropped is None:
            self.dropped = Counter()

    def as_text(self) -> str:
        lines = [f"seen={self.seen}  kept={self.kept}  ({self.yield_pct:.2f}% yield)"]
        for name, count in self.dropped.most_common():
            lines.append(f"  dropped by {name}: {count}")
        return "\n".join(lines)

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.kept / self.seen if self.seen else 0.0


def build_chain(spec: dict) -> list[tuple[str, Predicate]]:
    """Build a named filter chain from the ``data.filters`` config block.

    Order matters for speed, not correctness: cheap predicates run first so the
    expensive ones see fewer records.
    """
    chain: list[tuple[str, Predicate]] = []
    if spec.get("min_tokens"):
        chain.append((f"min_tokens({spec['min_tokens']})", min_tokens(spec["min_tokens"])))
    if spec.get("max_tokens"):
        chain.append((f"max_tokens({spec['max_tokens']})", max_tokens(spec["max_tokens"])))
    if spec.get("max_length_ratio"):
        chain.append(
            (f"length_ratio({spec['max_length_ratio']})", length_ratio(spec["max_length_ratio"]))
        )
    if spec.get("drop_advertising", True):
        chain.append(("advertising", no_advertising()))
    if spec.get("drop_punctuation_only", True):
        chain.append(("punctuation_only", not_mostly_punctuation()))
    if spec.get("min_alignment_score") is not None:
        chain.append(
            (
                f"alignment<{spec['min_alignment_score']}",
                min_alignment_score(spec["min_alignment_score"]),
            )
        )
    if spec.get("drop_untranslated", True):
        chain.append(("untranslated_copy", no_untranslated_copy()))
    if spec.get("drop_all_caps", True):
        chain.append(("all_caps", no_all_caps()))
    if spec.get("max_repeated_chars"):
        chain.append(
            (f"repeated_chars>{spec['max_repeated_chars']}", max_repeated_chars(spec["max_repeated_chars"]))
        )
    # Language ID is the most expensive predicate, so it runs last -- by then the
    # cheap filters have already removed most of what it would have scored.
    if spec.get("check_language", False):
        chain.append(("language_mismatch", language_match()))
    if spec.get("include_genres"):
        chain.append(("genre_not_allowed", include_genres(spec["include_genres"])))
    if spec.get("exclude_genres"):
        chain.append(("genre_excluded", exclude_genres(spec["exclude_genres"])))
    return chain


def apply_filters(
    pairs: Iterable[Pair],
    spec: dict,
    report: FilterReport | None = None,
) -> Iterator[Pair]:
    """Run the configured chain, optionally de-duplicating, streaming throughout."""
    chain = build_chain(spec)
    # dedupe: false | true ("pair") | "pair" | "src" | "tgt" | "all"
    #
    # "src" matters more than it looks for subtitles: the same English line is
    # often aligned to several different Spanish lines across films, and keeping
    # all of them teaches the model that one input has many unrelated outputs.
    dedupe = spec.get("dedupe", True)
    dedupe_mode = "pair" if dedupe is True else (dedupe or "")
    seen_fps: set[str] = set()
    seen_src: set[str] = set()
    seen_tgt: set[str] = set()
    report = report if report is not None else FilterReport()

    for pair in pairs:
        report.seen += 1
        if spec.get("normalise", True):
            pair.src = normalise_text(pair.src)
            pair.tgt = normalise_text(pair.tgt)
            if not pair.src or not pair.tgt:
                report.dropped["empty_after_normalise"] += 1
                continue

        for name, predicate in chain:
            if not predicate(pair):
                report.dropped[name] += 1
                break
        else:
            if dedupe_mode in ("pair", "all"):
                fp = _fingerprint(pair)
                if fp in seen_fps:
                    report.dropped["duplicate_pair"] += 1
                    continue
                seen_fps.add(fp)
            if dedupe_mode in ("src", "all"):
                key = pair.src.casefold()
                if key in seen_src:
                    report.dropped["duplicate_src"] += 1
                    continue
                seen_src.add(key)
            if dedupe_mode in ("tgt", "all"):
                key = pair.tgt.casefold()
                if key in seen_tgt:
                    report.dropped["duplicate_tgt"] += 1
                    continue
                seen_tgt.add(key)
            report.kept += 1
            yield pair
