"""Corpus profiling: look at the data before training on it.

Answers the questions that decide whether a corpus is usable and how hard to
filter it, rather than guessing at thresholds:

* How long are segments, and how skewed is the length ratio? (misalignment)
* How much of it is duplicated, or untranslated copies? (stock-phrase overfit)
* How much is markup, advertising, or the wrong language entirely?
* **How much plural-you marking is there at all?** Experiment 2 is only viable
  if the corpus actually contains Peninsular forms in usable quantity -- this is
  the number to check before committing to that experiment.
* How much survives the configured filter chain?

Sampling is hash-based rather than head-of-file: the first N lines of an OPUS
dump are one alphabetically-early film, so ``head`` tells you about that film and
nothing about the corpus. Hashing on record id gives a reproducible random
sample that streams.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .data.filters import FilterReport, apply_filters
from .data.langid import guess_language, matches_expected
from .data.regional_rules import ES_419, ES_ES, detect_region
from .eval.contrastive import CONVERSATIONAL_MARKERS_EN, CONVERSATIONAL_MARKERS_ES
from .schema import Pair

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_MARKUP = re.compile(r"(<[^>]{1,40}>|\[[^\]]{0,60}\]|^\s*[-–—]\s)")
_AD = re.compile(
    r"(opensubtitles|subtitles? by|sync(hroniz|hronis)ed by|ripped by|www\.|http|"
    r"subtitulos por|traducido por|resync)",
    re.IGNORECASE,
)
_DIGIT_HEAVY = re.compile(r"^\W*[\d\W]{3,}\W*$")

# Distinctive voseo forms. -ir verbs are deliberately excluded: `venís`, `decís`
# and `vivís` are identical for vos and vosotros, so counting them would inflate
# both buckets.
_VOS_MARKERS = re.compile(
    r"\b(vos|sos|tenés|querés|podés|sabés|hacés|comés|hablás|estás\s+vos|"
    r"andá|mirá|vení|dale|tomá|esperá|fijate|bancá)\b",
    re.IGNORECASE,
)
_TU_MARKERS = re.compile(
    r"\b(tú|tienes|quieres|puedes|sabes|haces|eres|estás|comes|hablas|"
    r"contigo|tuyo|tuya)\b",
    re.IGNORECASE,
)
_USTEDES = re.compile(r"\bustedes\b", re.IGNORECASE)


def hash_sample(record_id: str, rate: float) -> bool:
    """Deterministic inclusion test for a ``rate`` fraction of records."""
    if rate >= 1.0:
        return True
    digest = hashlib.blake2b(record_id.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "big") / float(1 << 64)) < rate


def percentiles(values: Sequence[float], points=(1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in points:
        index = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = float(ordered[index])
    out["mean"] = float(statistics.fmean(ordered))
    return out


def text_histogram(values: Sequence[float], bins: int = 10, width: int = 40) -> list[str]:
    """A terminal histogram -- enough to see a distribution's shape."""
    if not values:
        return ["(no data)"]
    low, high = min(values), max(values)
    if low == high:
        return [f"{low:g}: {'#' * width} ({len(values)})"]

    step = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / step))
        counts[index] += 1

    peak = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        edge_lo, edge_hi = low + i * step, low + (i + 1) * step
        bar = "#" * int(round(width * count / peak))
        lines.append(f"{edge_lo:7.1f}-{edge_hi:<7.1f} {bar:<{width}} {count}")
    return lines


@dataclass
class CorpusProfile:
    """Everything the profiler measured, ready to serialise."""

    n_seen: int = 0
    n_sampled: int = 0
    sample_rate: float = 1.0

    src_lang: str = ""
    tgt_lang: str = ""

    src_tokens: dict = field(default_factory=dict)
    tgt_tokens: dict = field(default_factory=dict)
    src_chars: dict = field(default_factory=dict)
    length_ratio: dict = field(default_factory=dict)

    rates: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    dialect: dict[str, int] = field(default_factory=dict)
    markers: dict[str, float] = field(default_factory=dict)

    vocab_src: dict = field(default_factory=dict)
    vocab_tgt: dict = field(default_factory=dict)

    top_src_segments: list = field(default_factory=list)
    top_tgt_segments: list = field(default_factory=list)
    genres: dict[str, int] = field(default_factory=dict)

    filter_yield: dict = field(default_factory=dict)
    _ratio_values: list = field(default_factory=list, repr=False)
    _src_token_values: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


def profile_corpus(
    pairs: Iterable[Pair],
    sample_rate: float = 1.0,
    top_n: int = 15,
    filter_spec: dict | None = None,
) -> CorpusProfile:
    """Walk a corpus once and collect everything worth knowing about it."""
    profile = CorpusProfile(sample_rate=sample_rate)

    src_token_counts: list[int] = []
    tgt_token_counts: list[int] = []
    src_char_counts: list[int] = []
    ratios: list[float] = []

    src_segments: Counter[str] = Counter()
    tgt_segments: Counter[str] = Counter()
    src_vocab: Counter[str] = Counter()
    tgt_vocab: Counter[str] = Counter()
    genres: Counter[str] = Counter()

    seen_pair_fp: set[str] = set()
    seen_src: set[str] = set()
    seen_tgt: set[str] = set()

    tally = Counter()

    for pair in pairs:
        profile.n_seen += 1
        if not hash_sample(pair.id, sample_rate):
            continue
        profile.n_sampled += 1

        if not profile.src_lang:
            profile.src_lang, profile.tgt_lang = pair.src_lang, pair.tgt_lang

        src, tgt = pair.src, pair.tgt
        src_tokens = _TOKEN.findall(src)
        tgt_tokens = _TOKEN.findall(tgt)
        n_src, n_tgt = len(src_tokens), len(tgt_tokens)

        src_token_counts.append(n_src)
        tgt_token_counts.append(n_tgt)
        src_char_counts.append(len(src))
        if n_src and n_tgt:
            ratios.append(max(n_src, n_tgt) / min(n_src, n_tgt))

        src_segments[src] += 1
        tgt_segments[tgt] += 1
        src_vocab.update(t.lower() for t in src_tokens)
        tgt_vocab.update(t.lower() for t in tgt_tokens)

        # --- quality signals ---
        fingerprint = hashlib.blake2b(
            f"{src.casefold()}|||{tgt.casefold()}".encode(), digest_size=16
        ).hexdigest()
        if fingerprint in seen_pair_fp:
            tally["duplicate_pair"] += 1
        else:
            seen_pair_fp.add(fingerprint)

        if src.casefold() in seen_src:
            tally["duplicate_src"] += 1
        else:
            seen_src.add(src.casefold())
        if tgt.casefold() in seen_tgt:
            tally["duplicate_tgt"] += 1
        else:
            seen_tgt.add(tgt.casefold())

        if src.strip().casefold() == tgt.strip().casefold():
            tally["untranslated_copy"] += 1
        if _MARKUP.search(src) or _MARKUP.search(tgt):
            tally["has_markup"] += 1
        if _AD.search(src) or _AD.search(tgt):
            tally["advertising"] += 1
        if _DIGIT_HEAVY.match(src) or _DIGIT_HEAVY.match(tgt):
            tally["digit_or_punct_only"] += 1
        if n_src < 4 or n_tgt < 4:
            tally["under_4_tokens"] += 1
        if n_src and n_tgt and max(n_src, n_tgt) / min(n_src, n_tgt) > 2.5:
            tally["length_ratio_over_2.5"] += 1
        if src.isupper() and len(src) > 8:
            tally["all_caps"] += 1

        # Language sanity. matches_expected handles the ISO-639-1 vs FLORES-200
        # mapping and abstains on short or unrecognised text.
        if not matches_expected(src, pair.src_lang):
            tally["src_language_mismatch"] += 1
        if not matches_expected(tgt, pair.tgt_lang):
            tally["tgt_language_mismatch"] += 1

        # --- dialect markers on the Spanish side ---
        spanish = tgt if pair.tgt_lang.startswith("spa") else (
            src if pair.src_lang.startswith("spa") else None
        )
        if spanish is not None:
            region = detect_region(spanish)
            if region == ES_ES:
                tally["dialect_peninsular"] += 1
            elif region == ES_419:
                tally["dialect_ustedes"] += 1
            else:
                tally["dialect_unmarked"] += 1
            if _VOS_MARKERS.search(spanish):
                tally["dialect_voseo"] += 1
            if _TU_MARKERS.search(spanish):
                tally["dialect_tu"] += 1
            if _USTEDES.search(spanish):
                tally["dialect_ustedes_explicit"] += 1

        # --- conversational markers ---
        if any(m in src.lower() for m in CONVERSATIONAL_MARKERS_EN):
            tally["marker_en"] += 1
        if any(m in tgt.lower() for m in CONVERSATIONAL_MARKERS_ES):
            tally["marker_es"] += 1

        for genre in pair.meta.get("genres") or []:
            genres[str(genre).title()] += 1
        if pair.meta.get("imdb_id"):
            tally["has_imdb_id"] += 1

    n = profile.n_sampled or 1
    profile.src_tokens = percentiles(src_token_counts)
    profile.tgt_tokens = percentiles(tgt_token_counts)
    profile.src_chars = percentiles(src_char_counts)
    profile.length_ratio = percentiles(ratios)

    profile.counts = dict(sorted(tally.items()))
    profile.rates = {k: v / n for k, v in sorted(tally.items())}

    profile.dialect = {k[8:]: v for k, v in sorted(tally.items()) if k.startswith("dialect_")}
    profile.markers = {
        "english_marker_rate": tally["marker_en"] / n,
        "spanish_marker_rate": tally["marker_es"] / n,
    }

    profile.vocab_src = _vocab_stats(src_vocab)
    profile.vocab_tgt = _vocab_stats(tgt_vocab)
    profile.top_src_segments = [
        {"text": text, "count": count, "pct": 100 * count / n}
        for text, count in src_segments.most_common(top_n)
    ]
    profile.top_tgt_segments = [
        {"text": text, "count": count, "pct": 100 * count / n}
        for text, count in tgt_segments.most_common(top_n)
    ]
    profile.genres = dict(genres.most_common(25))
    profile._ratio_values = ratios
    profile._src_token_values = src_token_counts

    return profile


def _vocab_stats(counter: Counter) -> dict:
    total = sum(counter.values())
    types = len(counter)
    hapax = sum(1 for c in counter.values() if c == 1)
    return {
        "tokens": total,
        "types": types,
        "type_token_ratio": types / total if total else 0.0,
        "hapax_rate": hapax / types if types else 0.0,
        "top": [{"token": t, "count": c} for t, c in counter.most_common(20)],
    }


def estimate_filter_yield(pairs: Iterable[Pair], filter_spec: dict) -> FilterReport:
    """Run the configured filter chain purely to measure what it would keep."""
    report = FilterReport()
    for _ in apply_filters(pairs, filter_spec, report):
        pass
    return report


# --- reporting ---------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def profile_to_markdown(profile: CorpusProfile, title: str = "Corpus profile") -> str:
    lines = [f"# {title}", ""]
    lines.append(
        f"{profile.n_seen:,} records seen, {profile.n_sampled:,} sampled "
        f"(rate {profile.sample_rate:g}), {profile.src_lang} -> {profile.tgt_lang}"
    )

    lines += ["", "## Length", "", "| measure | p1 | p25 | p50 | p75 | p95 | p99 | mean |", "|---|---|---|---|---|---|---|---|"]
    for name, data in (
        ("source tokens", profile.src_tokens),
        ("target tokens", profile.tgt_tokens),
        ("source chars", profile.src_chars),
        ("length ratio", profile.length_ratio),
    ):
        if data:
            cells = " | ".join(
                f"{data.get(k, float('nan')):.2f}" for k in ("p1", "p25", "p50", "p75", "p95", "p99", "mean")
            )
            lines.append(f"| {name} | {cells} |")

    if profile._src_token_values:
        lines += ["", "Source token-count distribution:", "", "```"]
        lines += text_histogram(profile._src_token_values)
        lines.append("```")

    lines += ["", "## Quality signals", "", "| signal | count | rate |", "|---|---|---|"]
    for key, count in profile.counts.items():
        if key.startswith("dialect_") or key.startswith("marker_"):
            continue
        lines.append(f"| {key} | {count:,} | {_pct(profile.rates[key])} |")

    if profile.dialect:
        total = profile.n_sampled or 1
        lines += [
            "",
            "## Spanish dialect markers",
            "",
            "Decides whether Experiment 2 is viable: it needs Peninsular forms in",
            "usable quantity. `unmarked` segments carry no plural-you marking at all",
            "and are unusable for that experiment.",
            "",
            "| marker | count | rate |",
            "|---|---|---|",
        ]
        for key, count in profile.dialect.items():
            lines.append(f"| {key} | {count:,} | {_pct(count / total)} |")

    lines += [
        "",
        "## Conversational markers",
        "",
        f"- English side: {_pct(profile.markers.get('english_marker_rate', 0))}",
        f"- Spanish side: {_pct(profile.markers.get('spanish_marker_rate', 0))}",
    ]

    lines += ["", "## Vocabulary", "", "| side | tokens | types | type/token | hapax |", "|---|---|---|---|---|"]
    for name, stats in (("source", profile.vocab_src), ("target", profile.vocab_tgt)):
        if stats:
            lines.append(
                f"| {name} | {stats['tokens']:,} | {stats['types']:,} | "
                f"{stats['type_token_ratio']:.4f} | {_pct(stats['hapax_rate'])} |"
            )

    lines += [
        "",
        "## Most repeated segments",
        "",
        "A high share here is the stock-phrase problem: without de-duplication a",
        "LoRA spends much of its capacity on these.",
        "",
        "| count | share | source segment |",
        "|---|---|---|",
    ]
    for entry in profile.top_src_segments:
        text = entry["text"][:70].replace("|", "\\|")
        lines.append(f"| {entry['count']:,} | {entry['pct']:.2f}% | {text} |")

    if profile.genres:
        lines += ["", "## Genres", "", "| genre | segments |", "|---|---|"]
        for genre, count in profile.genres.items():
            lines.append(f"| {genre} | {count:,} |")

    if profile.filter_yield:
        lines += ["", "## Configured filter chain", "", "```", profile.filter_yield["text"], "```"]

    return "\n".join(lines)


def save_profile(profile: CorpusProfile, out_dir: Path | str, title: str = "Corpus profile"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "profile.json"
    md_path = out_dir / "profile.md"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(profile.to_dict(), fh, indent=2, ensure_ascii=False)
    md_path.write_text(profile_to_markdown(profile, title), encoding="utf-8")
    return json_path, md_path
