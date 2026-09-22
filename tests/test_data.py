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
    assert report.dropped["duplicate"] == 1
    assert report.seen == 2 and report.kept == 1


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
