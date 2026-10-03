import json

import pytest

from communilate.config import Config, load_config
from communilate.data import load_split
from communilate.data.base import apply_direction, stable_split
from communilate.data.filters import FilterReport, apply_filters, normalise_text
from communilate.schema import Pair, write_jsonl


def make(i=1, src="Are you all coming over later today?", tgt="¿Vosotros venís luego a casa?", **kw):
    base = dict(
        id=f"t{i}", src=src, tgt=tgt,
        src_lang="eng_Latn", tgt_lang="spa_Latn",
    )
    base.update(kw)
    return Pair(**base)


# -- normalisation ------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<i>Hola</i> mundo", "Hola mundo"),
        ("- Vamos ya", "Vamos ya"),
        ("[puerta se cierra] Vamos", "Vamos"),
        ("Hola    mundo", "Hola mundo"),
    ],
)
def test_normalise_strips_subtitle_markup(raw, expected):
    assert normalise_text(raw) == expected


# -- filters ------------------------------------------------------------

def test_min_tokens_drops_fragments():
    pairs = [make(1), make(2, src="What?", tgt="¿Qué?")]
    kept = list(apply_filters(pairs, {"min_tokens": 4}))
    assert [p.id for p in kept] == ["t1"]


def test_length_ratio_drops_misaligned_pairs():
    skewed = make(2, src="Yes.", tgt="Sí, claro, por supuesto, no hay ningún problema con eso.")
    kept = list(apply_filters([make(1), skewed], {"max_length_ratio": 2.0, "min_tokens": 1}))
    assert [p.id for p in kept] == ["t1"]


def test_advertising_lines_are_dropped():
    ad = make(2, src="Subtitles by OpenSubtitles.org team", tgt="Subtitulos por el equipo")
    kept = list(apply_filters([make(1), ad], {"min_tokens": 1}))
    assert [p.id for p in kept] == ["t1"]


def test_dedupe_removes_repeats_and_is_reported():
    report = FilterReport()
    kept = list(apply_filters([make(1), make(2)], {"min_tokens": 1, "dedupe": True}, report))
    assert len(kept) == 1
    assert report.dropped["duplicate_pair"] == 1
    assert report.seen == 2 and report.kept == 1


def test_dedupe_src_mode_catches_one_source_many_targets():
    # The same English line aligned to different Spanish lines across films is
    # common in subtitles, and pair-level dedupe does not catch it.
    same_src = [
        make(1, tgt="¿Vosotros venís luego a casa?"),
        make(2, tgt="¿Vosotros venís más tarde aquí?"),
    ]
    assert len(list(apply_filters(same_src, {"min_tokens": 1, "dedupe": "pair"}))) == 2
    assert len(list(apply_filters(same_src, {"min_tokens": 1, "dedupe": "src"}))) == 1


def test_dedupe_can_be_disabled():
    assert len(list(apply_filters([make(1), make(2)], {"min_tokens": 1, "dedupe": False}))) == 2


def test_untranslated_copies_are_dropped():
    copy = make(2, src="Madrid 1987", tgt="Madrid 1987")
    kept = list(apply_filters([make(1), copy], {"min_tokens": 1, "dedupe": False}))
    assert [p.id for p in kept] == ["t1"]


def test_all_caps_is_dropped():
    shouted = make(2, src="STOP THE CAR NOW", tgt="PARA EL COCHE AHORA")
    kept = list(apply_filters([make(1), shouted], {"min_tokens": 1, "dedupe": False}))
    assert [p.id for p in kept] == ["t1"]


def test_repeated_character_runs_are_dropped():
    spam = make(2, src="Noooooooo do not", tgt="Noooooooo no lo hagas")
    kept = list(apply_filters(
        [make(1), spam], {"min_tokens": 1, "dedupe": False, "max_repeated_chars": 4}
    ))
    assert [p.id for p in kept] == ["t1"]


def test_language_check_drops_wrong_language_but_abstains_on_short_text():
    wrong = make(2, src="the and you that was for", tgt="the and you that was for now")
    short = make(3, src="Hola amigo", tgt="Hey friend")
    kept = list(apply_filters(
        [make(1), wrong, short],
        {"min_tokens": 1, "dedupe": False, "check_language": True, "drop_untranslated": False},
    ))
    # The confidently-wrong pair goes; the too-short one is kept rather than guessed at.
    assert [p.id for p in kept] == ["t1", "t3"]


def test_genre_filters_only_apply_to_records_that_have_metadata():
    action = make(2, meta={"genres": ["Action"]})
    comedy = make(3, meta={"genres": ["Comedy"]})
    no_meta = make(4)
    kept = list(apply_filters(
        [action, comedy, no_meta],
        {"min_tokens": 1, "dedupe": False, "exclude_genres": ["action"]},
    ))
    # Records without genre metadata survive rather than being silently lost.
    assert [p.id for p in kept] == ["t3", "t4"]


def test_report_attributes_every_drop():
    report = FilterReport()
    pairs = [make(1), make(2, src="No.", tgt="No."), make(3)]
    list(apply_filters(pairs, {"min_tokens": 4, "dedupe": True}, report))
    assert report.seen == 3
    assert sum(report.dropped.values()) == report.seen - report.kept


# -- direction ----------------------------------------------------------

def test_direction_as_is_swap_and_both():
    pair = make(1)
    assert len(list(apply_direction([pair], "as_is"))) == 1
    assert list(apply_direction([pair], "swap"))[0].src_lang == "spa_Latn"

    both = list(apply_direction([pair], "both"))
    assert [p.src_lang for p in both] == ["eng_Latn", "spa_Latn"]


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError):
        list(apply_direction([make(1)], "sideways"))


# -- splitting ----------------------------------------------------------

def test_stable_split_is_deterministic_and_roughly_proportional():
    ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    assignments = [stable_split(f"id-{i}", ratios) for i in range(2000)]
    assert stable_split("id-7", ratios) == assignments[7]  # deterministic
    train_share = assignments.count("train") / len(assignments)
    assert 0.75 < train_share < 0.85


def test_stable_split_does_not_reshuffle_when_corpus_grows():
    # Hash-based assignment means adding records never moves existing ones --
    # otherwise train data could leak into a test set already reported on.
    ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    before = {f"id-{i}": stable_split(f"id-{i}", ratios) for i in range(100)}
    after = {f"id-{i}": stable_split(f"id-{i}", ratios) for i in range(500)}
    assert all(after[k] == v for k, v in before.items())


# -- jsonl_dir loader ---------------------------------------------------

def _cfg(tmp_path, **data_overrides):
    data = {"loader": "jsonl_dir", "path": str(tmp_path), "direction": "as_is"}
    data.update(data_overrides)
    return Config({"data": data})


def test_jsonl_dir_reads_per_split_files(tmp_path):
    write_jsonl([make(1), make(2)], tmp_path / "train.jsonl")
    write_jsonl([make(3)], tmp_path / "test.jsonl")
    assert len(list(load_split(_cfg(tmp_path), "train"))) == 2
    assert len(list(load_split(_cfg(tmp_path), "test"))) == 1


def test_jsonl_dir_falls_back_to_pooled_file_and_derives_splits(tmp_path):
    write_jsonl([make(i) for i in range(200)], tmp_path / "all.jsonl")
    cfg = _cfg(tmp_path, split_ratios={"train": 0.8, "dev": 0.1, "test": 0.1})
    counts = {s: len(list(load_split(cfg, s))) for s in ("train", "dev", "test")}
    assert sum(counts.values()) == 200
    assert counts["train"] > counts["dev"]


def test_jsonl_dir_reports_a_missing_folder_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.path"):
        list(load_split(_cfg(tmp_path / "nope"), "train"))


def test_max_records_caps_the_stream(tmp_path):
    write_jsonl([make(i) for i in range(50)], tmp_path / "train.jsonl")
    cfg = _cfg(tmp_path, max_records=5)
    assert len(list(load_split(cfg, "train"))) == 5


def test_swapping_data_path_is_the_only_change_needed(tmp_path):
    # The core design claim: pointing data.path at a different folder is all it
    # takes to train on a different corpus.
    corpus_a, corpus_b = tmp_path / "a", tmp_path / "b"
    write_jsonl([make(1)], corpus_a / "train.jsonl")
    write_jsonl([make(i) for i in range(4)], corpus_b / "train.jsonl")

    cfg = load_config("exp1_register", [f"data.path={corpus_a}", "data.direction=as_is"])
    assert len(list(load_split(cfg, "train"))) == 1

    cfg = load_config("exp1_register", [f"data.path={corpus_b}", "data.direction=as_is"])
    assert len(list(load_split(cfg, "train"))) == 4


# -- regression: generator/consumer interaction ------------------------

def test_filters_do_not_mutate_the_records_they_are_given():
    # apply_direction derives the reversed pair after the forward one has been
    # consumed, so a consumer mutating the record in place corrupted a record the
    # producer was about to read.
    original = make(1, src="<i>Are you all coming later?</i>", tgt="- ¿Vosotros venís luego?")
    list(apply_filters([original], {"min_tokens": 1}))
    assert original.src == "<i>Are you all coming later?</i>"
    assert original.tgt == "- ¿Vosotros venís luego?"


def test_pure_markup_lines_drop_cleanly_in_both_directions():
    # Repro for the crash on real OpenSubtitles: "[gunshot]" normalises to an
    # empty string, and the reversed pair was then built from the emptied record,
    # raising SchemaError mid-stream instead of being filtered out.
    pairs = [
        make(1, src="[gunshot]", tgt="[disparo]"),
        make(2, src="<i>Are you all coming later?</i>", tgt="- ¿Vosotros venís luego?"),
    ]
    report = FilterReport()
    kept = list(apply_filters(apply_direction(pairs, "both"), {"min_tokens": 1, "dedupe": False}, report))

    assert [p.id for p in kept] == ["t2", "t2::rev"]
    assert report.dropped["empty_after_normalise"] == 2
    # Normalisation is applied to the surviving pair in both directions.
    assert kept[0].src == "Are you all coming later?"
    assert kept[1].tgt == "Are you all coming later?"


def test_normalisation_propagates_into_the_reversed_pair():
    pairs = [make(1, src="<i>Hello there my friend</i>", tgt="- Hola amigo mío querido")]
    kept = list(apply_filters(apply_direction(pairs, "both"), {"min_tokens": 1, "dedupe": False}))
    assert kept[0].src == "Hello there my friend"
    assert kept[1].src == "Hola amigo mío querido"


def test_genre_filters_warn_when_the_corpus_has_no_genre_metadata():
    # Genre predicates pass records without metadata, so on a release with no
    # .ids file they are silent no-ops. The report has to say so, or a config
    # looks like it filtered something it did not.
    report = FilterReport()
    list(apply_filters([make(1), make(2)], {"min_tokens": 1, "dedupe": False,
                                           "include_genres": ["comedy"]}, report))
    assert report.kept == 2
    assert any("no effect" in w for w in report.warnings)


def test_no_genre_warning_when_metadata_is_present():
    report = FilterReport()
    list(apply_filters(
        [make(1, meta={"genres": ["Comedy"]})],
        {"min_tokens": 1, "dedupe": False, "include_genres": ["comedy"]},
        report,
    ))
    assert report.warnings == []


def test_no_genre_warning_when_no_genre_filters_configured():
    report = FilterReport()
    list(apply_filters([make(1)], {"min_tokens": 1, "dedupe": False}, report))
    assert report.warnings == []


def test_dedupe_budget_is_bounded_and_warns():
    # Unbounded, a dedupe set over 105M OpenSubtitles segments runs to tens of GB
    # and kills the process. Past the budget records pass through undeduplicated
    # and the report says so -- partial dedupe rather than a crash.
    pairs = [make(i, src=f"Distinct english sentence number {i} here.") for i in range(200)]
    pairs += [make(1000 + i, src=f"Distinct english sentence number {i} here.") for i in range(200)]
    report = FilterReport()
    kept = list(apply_filters(
        pairs, {"min_tokens": 1, "dedupe": "src", "max_dedupe_keys": 50}, report
    ))
    assert any("budget" in w for w in report.warnings)
    # Within budget the duplicates were still caught.
    assert report.dropped["duplicate_src"] >= 50
    assert len(kept) > 50


def test_dedupe_within_budget_does_not_warn():
    report = FilterReport()
    list(apply_filters(
        [make(1), make(2)], {"min_tokens": 1, "dedupe": "pair", "max_dedupe_keys": 1000}, report
    ))
    assert report.warnings == []
    assert report.dropped["duplicate_pair"] == 1
