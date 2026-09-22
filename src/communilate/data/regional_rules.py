"""Rule-based regional transformation of Spanish text (Experiment 2).

Scope is deliberately narrow: the *plural you* axis, ``vosotros`` (Peninsular)
vs. ``ustedes`` (general Latin American). That is the one regional distinction
that is genuinely binary, near-exceptionless, and checkable by rule -- which is
what makes it a usable evaluation target.

Three things this module does not claim:

* **It is not a voseo transformer.** ``vos`` is not a Latin-America-wide
  feature: it dominates in Rioplatense and much of Central America, while
  Mexico, most of Colombia, Peru and the Caribbean are ``tu``-using. Voseo needs
  country-level labels and a third bucket, not a Spain/LatAm binary.
* **The transformation is one-way.** ``vosotros -> ustedes`` preserves meaning;
  the reverse does not, because ``ustedes`` is *also* the formal plural in
  Spain. Given ``ustedes`` you cannot recover whether Peninsular Spanish would
  have said ``vosotros``, so no ``to_peninsular`` exists.
* **It is not a morphological analyser.** Conversion is lexicon-first with a
  suffix-rule fallback for regular verbs. Anything it cannot handle confidently
  is reported rather than guessed at, so those records can be dropped from the
  training set instead of quietly corrupting it.

The lexicon lives in ``lexicons/vosotros_ustedes.json`` and is meant to be
extended there, not here.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ES_ES = "es-ES"
ES_419 = "es-419"

LEXICON_PATH = Path(__file__).parent / "lexicons" / "vosotros_ustedes.json"


@functools.lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> tuple[dict[str, str], frozenset[str]]:
    """Load and flatten the explicit-form lexicon.

    Returns ``(forms, exceptions)``. Cached because conversion is applied across
    hundreds of thousands of records.
    """
    target = Path(path) if path else LEXICON_PATH
    with target.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    forms: dict[str, str] = {}
    for key, value in raw.items():
        if key.startswith("_") or key == "exceptions":
            continue
        if isinstance(value, dict):
            forms.update({k.lower(): v for k, v in value.items()})
    exceptions = frozenset(w.lower() for w in raw.get("exceptions", []))
    return forms, exceptions


PRONOUNS: dict[str, str] = {"vosotros": "ustedes", "vosotras": "ustedes"}
POSSESSIVES: dict[str, str] = {
    "vuestro": "su",
    "vuestra": "su",
    "vuestros": "sus",
    "vuestras": "sus",
}

# Suffix fallback, applied longest-first. Each rule is tagged by risk:
#
#   safe  -- the conjugation is regular across the board for this tense, and the
#            handful of irregulars (ibais, erais, haréis, podréis, ...) are all
#            covered by the lexicon, which is consulted first.
#   risky -- present-tense endings, where stem-changing verbs live. The vosotros
#            form is the one present form that does NOT stem-change, so applying
#            the suffix rule to a stem-changer missing from the lexicon produces
#            a wrong form (venís -> "venen" instead of "vienen"). In strict mode
#            these are reported instead of converted.
#
# -isteis is treated as risky for the same reason: -ir preterites stem-change
# (dormisteis -> durmieron), and the long tail of those is not fully lexicalised.
SUFFIX_RULES: list[tuple[str, str, str]] = [
    ("aríais", "arían", "safe"),
    ("eríais", "erían", "safe"),
    ("iríais", "irían", "safe"),
    ("ierais", "ieran", "safe"),
    ("asteis", "aron", "safe"),
    ("isteis", "ieron", "risky"),
    ("abais", "aban", "safe"),
    ("arais", "aran", "safe"),
    ("aréis", "arán", "safe"),
    ("eréis", "erán", "safe"),
    ("iréis", "irán", "safe"),
    ("íais", "ían", "safe"),
    ("áis", "an", "risky"),
    ("éis", "en", "risky"),
    ("ís", "en", "risky"),
]

# A token ending in one of these *looks* like a vosotros form. In strict mode,
# any such token that the lexicon does not cover is reported as unhandled
# rather than run through the suffix fallback.
_VOSOTROS_SHAPED = re.compile(
    r"(áis|éis|ís|abais|íais|arais|ierais|asteis|isteis|aríais|eríais|iríais|aréis|eréis|iréis)$",
    re.IGNORECASE,
)

_PENINSULAR_MARKERS = re.compile(
    r"\b(vosotros|vosotras|vuestr[oa]s?|"
    r"\w+(?:áis|éis|ís|abais|íais|asteis|isteis|aríais|eríais|iríais|aréis|eréis|iréis))\b",
    re.IGNORECASE,
)
_LATAM_MARKERS = re.compile(r"\bustedes\b", re.IGNORECASE)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _match_case(source: str, replacement: str) -> str:
    """Carry the source token's casing onto its replacement."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


@dataclass
class ConversionResult:
    """Outcome of one conversion.

    ``ambiguous`` and ``unhandled`` are the important fields: a record with
    either populated should be excluded from the regional training set rather
    than trusted.
    """

    text: str
    changed: bool = False
    n_changes: int = 0
    ambiguous: list[str] = field(default_factory=list)
    unhandled: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.ambiguous and not self.unhandled


def to_latin_american(
    text: str,
    strict: bool = True,
    lexicon_path: str | None = None,
) -> ConversionResult:
    """Rewrite Peninsular plural-you forms as general Latin American ones.

    ``strict=True`` (the default) converts lexicon entries and regular tenses,
    but reports present-tense and -ir-preterite forms it does not recognise
    rather than guessing -- those are where stem changes live. ``strict=False``
    applies the suffix fallback everywhere, which keeps more of the corpus but
    will mangle any stem-changing verb missing from the lexicon.

    Check the unhandled rate on your corpus before choosing. A high rate means
    the lexicon needs extending, not that strict mode is wrong.

    The clitic ``os`` is never rewritten: its correct replacement is ``los`` for
    direct objects and ``les`` for indirect ones, and that call needs parsing
    this module deliberately does not do. It is reported in ``ambiguous``.
    """
    forms, exceptions = load_lexicon(lexicon_path)
    result = ConversionResult(text=text)

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        lower = token.lower()

        if lower == "os":
            result.ambiguous.append(token)
            return token
        if lower in exceptions:
            return token
        if lower in PRONOUNS:
            result.n_changes += 1
            return _match_case(token, PRONOUNS[lower])
        if lower in POSSESSIVES:
            result.n_changes += 1
            return _match_case(token, POSSESSIVES[lower])
        if lower in forms:
            result.n_changes += 1
            return _match_case(token, forms[lower])

        for suffix, replacement, risk in SUFFIX_RULES:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
                if strict and risk == "risky":
                    result.unhandled.append(token)
                    return token
                result.n_changes += 1
                return _match_case(token, lower[: -len(suffix)] + replacement)

        if strict and _VOSOTROS_SHAPED.search(lower):
            result.unhandled.append(token)
        return token

    result.text = _WORD.sub(_sub, text)
    result.changed = result.n_changes > 0
    return result


def detect_region(text: str) -> str | None:
    """Classify a Spanish string on the plural-you axis.

    Returns ``es-ES``, ``es-419``, or ``None`` when the sentence carries no
    plural-you marking at all -- which is most sentences, and the reason the
    Experiment 2 test set must be built from prompts that *force* plural you.
    """
    _, exceptions = load_lexicon()
    for match in _PENINSULAR_MARKERS.finditer(text):
        if match.group(0).lower() not in exceptions:
            return ES_ES
    if _LATAM_MARKERS.search(text):
        return ES_419
    return None


def has_plural_you(text: str) -> bool:
    """True when a sentence is informative for regional evaluation."""
    return detect_region(text) is not None
