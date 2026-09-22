import pytest

from communilate.schema import Pair, SchemaError, read_jsonl, write_jsonl


def make(**kw):
    base = dict(
        id="t1", src="Are you coming?", tgt="¿Vienen?",
        src_lang="eng_Latn", tgt_lang="spa_Latn",
    )
    base.update(kw)
    return Pair(**base)


def test_valid_pair_round_trips(tmp_path):
    path = tmp_path / "x.jsonl"
    written = write_jsonl([make(), make(id="t2")], path)
    assert written == 2
    assert [p.id for p in read_jsonl(path)] == ["t1", "t2"]


def test_swapped_reverses_direction_and_marks_meta():
    swapped = make().swapped()
    assert (swapped.src, swapped.src_lang) == ("¿Vienen?", "spa_Latn")
    assert (swapped.tgt, swapped.tgt_lang) == ("Are you coming?", "eng_Latn")
    assert swapped.meta["reversed"] is True


def test_bare_iso_code_is_rejected():
    # NLLB needs FLORES-200 codes; a bare "es" fails at generation time
    # otherwise, which is far harder to debug than failing here.
    with pytest.raises(SchemaError, match="FLORES-200"):
        make(tgt_lang="es")


@pytest.mark.parametrize("field,value", [("src", "  "), ("tgt", ""), ("id", "")])
def test_empty_fields_are_rejected(field, value):
    with pytest.raises(SchemaError):
        make(**{field: value})


def test_same_source_and_target_language_is_rejected():
    with pytest.raises(SchemaError):
        make(tgt_lang="eng_Latn")


def test_unknown_field_is_rejected():
    with pytest.raises(SchemaError, match="unknown fields"):
        Pair.from_dict({
            "id": "t", "src": "a", "tgt": "b",
            "src_lang": "eng_Latn", "tgt_lang": "spa_Latn",
            "typo_field": 1,
        })


def test_read_error_names_the_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"ok","src":"a","tgt":"b","src_lang":"eng_Latn","tgt_lang":"spa_Latn"}\n'
                    "not json\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r":2:"):
        list(read_jsonl(path))
