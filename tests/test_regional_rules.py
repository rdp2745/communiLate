import pytest

from communilate.data.regional_rules import (
    ES_419,
    ES_ES,
    detect_region,
    has_plural_you,
    to_latin_american,
)


@pytest.mark.parametrize(
    "source,expected",
    [
        # Stem-changing verbs: the vosotros form is the one present form that
        # does NOT stem-change, so conversion must re-apply it. A blind suffix
        # rule gives "venen" / "poden" / "queren" / "durmen" / "jugan".
        ("venís", "vienen"),
        ("podéis", "pueden"),
        ("queréis", "quieren"),
        ("dormís", "duermen"),
        ("jugáis", "juegan"),
        ("pedís", "piden"),
        ("construís", "construyen"),
        # Irregulars.
        ("sois", "son"),
        ("estáis", "están"),
        ("habéis", "han"),
        ("vais", "van"),
        # Regular tenses handled by the safe suffix rules.
        ("hablabais", "hablaban"),
        ("hablasteis", "hablaron"),
        ("hablaríais", "hablarían"),
        ("hablaréis", "hablarán"),
    ],
)
def test_verb_forms_convert_correctly(source, expected):
    assert to_latin_american(source).text == expected


def test_pronouns_and_possessives():
    result = to_latin_american("Vosotros y vuestra madre")
    assert result.text == "Ustedes y su madre"
    assert to_latin_american("vuestros libros").text == "sus libros"


def test_casing_is_preserved():
    assert to_latin_american("Vosotros").text == "Ustedes"
    assert to_latin_american("VOSOTROS").text == "USTEDES"


def test_clitic_os_is_flagged_not_guessed():
    # "os" maps to "los" for direct objects and "les" for indirect ones; that
    # call needs parsing this module deliberately does not do.
    result = to_latin_american("no os entiendo")
    assert "os" in result.ambiguous
    assert result.text == "no os entiendo"


@pytest.mark.parametrize("word", ["país", "además", "jamás", "anís", "compás"])
def test_non_verbs_ending_in_vosotros_suffixes_are_untouched(word):
    # Without the exception list the -ís rule turns "país" into "paen".
    assert to_latin_american(word).text == word
    assert detect_region(word) is None


def test_strict_mode_reports_unknown_present_tense_instead_of_guessing():
    result = to_latin_american("cantáis", strict=True)
    assert result.unhandled == ["cantáis"]
    assert result.text == "cantáis"
    assert not result.is_clean


def test_lenient_mode_applies_the_suffix_fallback():
    assert to_latin_american("cantáis", strict=False).text == "cantan"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("¿Vosotros venís?", ES_ES),
        ("¿Vuestra madre?", ES_ES),
        ("Hablabais rápido", ES_ES),
        ("Ustedes ya comieron", ES_419),
        ("Hace frío hoy", None),
    ],
)
def test_detect_region(text, expected):
    assert detect_region(text) == expected


def test_has_plural_you():
    assert has_plural_you("¿Vosotros venís?")
    assert not has_plural_you("Hace frío hoy")


def test_full_sentence_conversion():
    result = to_latin_american("¿Vosotros podéis venir a vuestra casa?")
    assert result.text == "¿Ustedes pueden venir a su casa?"
    assert result.is_clean
    assert detect_region(result.text) == ES_419
