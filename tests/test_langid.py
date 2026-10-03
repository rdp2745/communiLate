import pytest

from communilate.data.langid import (
    ISO_TO_FLORES_PREFIX,
    STOPWORDS,
    guess_language,
    matches_expected,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("the quick brown fox and the dog", "en"),
        ("¿qué es lo que no te gusta de eso?", "es"),
        ("કેમ છો તમે આજે સવારે", "gu"),
        ("Hi.", "unknown"),            # too short to score
        ("Madrid 1987", "unknown"),    # no stopwords at all
    ],
)
def test_guess_language(text, expected):
    assert guess_language(text) == expected


@pytest.mark.parametrize(
    "text,code,expected",
    [
        # Regression: guess_language speaks ISO-639-1 and the schema speaks
        # FLORES-200. A prefix comparison ("spa_Latn".startswith("es")) is always
        # False, which made the language filter reject every single record.
        ("que de no la es en", "spa_Latn", True),
        ("the and you that was for", "eng_Latn", True),
        ("the and you that was for", "spa_Latn", False),
        ("que de no la es en", "eng_Latn", False),
        ("કેમ છો તમે આજે સવારે", "guj_Gujr", True),
        ("કેમ છો તમે આજે સવારે", "spa_Latn", False),
        # Abstentions and unknown languages pass rather than being rejected.
        ("Hi.", "spa_Latn", True),
        ("que de no la es en", "fra_Latn", True),
    ],
)
def test_matches_expected(text, code, expected):
    assert matches_expected(text, code) is expected


def test_flores_mapping_covers_every_language_the_scorer_returns():
    assert set(ISO_TO_FLORES_PREFIX) == {"en", "es", "gu"}


def test_stopword_lists_are_disjoint():
    # The scorer's margin rule assumes a hit for one language is not also a hit
    # for the other.
    assert STOPWORDS["en"] & STOPWORDS["es"] == frozenset()


def test_ambiguous_tie_abstains():
    # Equal evidence for both languages (code-switched or mixed text) is not
    # evidence for either, so the scorer abstains rather than picking one.
    assert guess_language("the and que de") == "unknown"
