"""Dependency-free language identification.

A stopword-overlap scorer, not a real LID model. It exists to catch the one
failure mode that actually occurs in community-uploaded subtitle files -- an
English line sitting in the Spanish column, usually from a mis-tagged upload --
without pulling in fasttext or langdetect for a single boolean.

It abstains (``"unknown"``) rather than guessing on short segments, because a
three-word line frequently contains no stopword at all and a confident wrong
answer there would make the mismatch rate meaningless.
"""

from __future__ import annotations

import re

STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "the", "and", "you", "that", "was", "for", "are", "with", "his", "they",
        "this", "have", "from", "not", "but", "what", "all", "were", "when",
        "your", "can", "said", "there", "been", "has", "would", "she", "him",
        "about", "could", "them", "then", "just", "know", "like", "well",
    }),
    "es": frozenset({
        "que", "de", "no", "la", "el", "es", "en", "lo", "un", "por", "qué",
        "me", "una", "te", "los", "se", "con", "para", "mi", "está", "si",
        "bien", "pero", "yo", "eso", "las", "sí", "su", "tu", "del", "al",
        "como", "más", "muy", "esto", "nada", "ser", "hay", "ya",
    }),
}

GUJARATI = re.compile(r"[઀-૿]")
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Below this many tokens the scorer abstains rather than guessing.
MIN_TOKENS_FOR_GUESS = 4


def guess_language(text: str, min_tokens: int = MIN_TOKENS_FOR_GUESS) -> str:
    """Return ``'en'``, ``'es'``, ``'gu'`` or ``'unknown'``."""
    if GUJARATI.search(text):
        return "gu"

    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if len(tokens) < min_tokens:
        return "unknown"

    unique = set(tokens)
    scores = {lang: len(unique & words) for lang, words in STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    # Require both a hit and a clear margin: "no" and "me" are stopwords in
    # both lists, so a tie is not evidence either way.
    if best_score == 0 or best_score == runner_up:
        return "unknown"
    return best


# The scorer speaks ISO-639-1 while the schema speaks FLORES-200, and the two do
# not share a prefix: "spa_Latn".startswith("es") is False. Mapping explicitly
# rather than comparing prefixes -- a naive prefix check silently rejects every
# record, which looks like an empty corpus rather than a bug.
ISO_TO_FLORES_PREFIX: dict[str, str] = {"en": "eng", "es": "spa", "gu": "guj"}


def matches_expected(text: str, flores_code: str) -> bool:
    """True when the text is not *confidently* the wrong language.

    Deliberately lenient: abstentions pass, and so does any language the scorer
    does not know about. A filter built on this should remove only clear
    mismatches, not everything it could not read.
    """
    guess = guess_language(text)
    if guess == "unknown":
        return True
    expected_prefix = ISO_TO_FLORES_PREFIX.get(guess)
    if expected_prefix is None:
        return True
    code = flores_code.lower()
    # Only contradict the declared language when the scorer recognises it as one
    # of the languages it can actually distinguish.
    known = tuple(ISO_TO_FLORES_PREFIX.values())
    if not code.startswith(known):
        return True
    return code.startswith(expected_prefix)
