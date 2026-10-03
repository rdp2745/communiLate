import pytest

from communilate.analysis import (
    hash_sample,
    percentiles,
    profile_corpus,
    profile_to_markdown,
    text_histogram,
)
from communilate.schema import Pair


def mk(i, src="Are you all coming over later today?", tgt="¿Vosotros venís luego a casa?", **kw):
    base = dict(id=f"a{i}", src=src, tgt=tgt, src_lang="eng_Latn", tgt_lang="spa_Latn")
    base.update(kw)
    return Pair(**base)


# -- sampling -----------------------------------------------------------

def test_hash_sample_is_deterministic_and_roughly_proportional():
    ids = [f"rec-{i}" for i in range(5000)]
    first = [hash_sample(i, 0.1) for i in ids]
    assert first == [hash_sample(i, 0.1) for i in ids]      # reproducible
    assert 0.08 < sum(first) / len(first) < 0.12


def test_sample_rate_of_one_takes_everything():
    assert all(hash_sample(f"r{i}", 1.0) for i in range(100))


def test_sampling_is_not_head_of_file():
    # Sampling the first N lines of an OPUS dump profiles one alphabetically
    # early film, not the corpus. Hash-based inclusion must not correlate with
    # position.
    included = [i for i in range(1000) if hash_sample(f"rec-{i}", 0.1)]
    assert included and max(included) > 800


# -- summary statistics -------------------------------------------------

def test_percentiles_and_mean():
    stats = percentiles(list(range(1, 101)))
    assert stats["p50"] == pytest.approx(50, abs=1.5)
    assert stats["mean"] == pytest.approx(50.5)


def test_percentiles_of_empty_input():
    assert percentiles([]) == {}


def test_histogram_handles_a_constant_series():
    assert len(text_histogram([5, 5, 5])) == 1


def test_histogram_of_empty_input():
    assert text_histogram([]) == ["(no data)"]


# -- profiling ----------------------------------------------------------

def test_profile_counts_duplicates_and_copies():
    pairs = [mk(1), mk(2), mk(3, src="Madrid 1987", tgt="Madrid 1987")]
    profile = profile_corpus(pairs)
    assert profile.n_seen == 3 and profile.n_sampled == 3
    assert profile.counts["duplicate_pair"] == 1
    assert profile.counts["untranslated_copy"] == 1


def test_profile_detects_markup_and_advertising():
    pairs = [
        mk(1, src="<i>Hello there friend</i>"),
        mk(2, src="Subtitles by OpenSubtitles.org", tgt="Subtitulos por el equipo"),
    ]
    profile = profile_corpus(pairs)
    assert profile.counts["has_markup"] >= 1
    assert profile.counts["advertising"] == 1


def test_profile_buckets_spanish_dialect_markers():
    pairs = [
        mk(1, tgt="¿Vosotros venís luego a casa?"),      # peninsular
        mk(2, tgt="¿Ustedes vienen luego a casa?"),      # ustedes
        mk(3, tgt="Hace mucho frío hoy aquí."),          # unmarked
    ]
    profile = profile_corpus(pairs)
    assert profile.dialect["peninsular"] == 1
    assert profile.dialect["ustedes"] == 1
    assert profile.dialect["unmarked"] == 1


def test_dialect_markers_are_skipped_when_neither_side_is_spanish():
    pairs = [mk(1, tgt="કેમ છો તમે આજે", tgt_lang="guj_Gujr")]
    profile = profile_corpus(pairs)
    assert profile.dialect == {}


def test_profile_flags_language_mismatch():
    pairs = [mk(1, tgt="the and you that was for him")]
    assert profile_corpus(pairs).counts["tgt_language_mismatch"] == 1


def test_profile_surfaces_the_most_repeated_segments():
    pairs = [mk(i, src="What?", tgt="¿Qué?") for i in range(10)] + [mk(99)]
    profile = profile_corpus(pairs)
    assert profile.top_src_segments[0]["text"] == "What?"
    assert profile.top_src_segments[0]["count"] == 10


def test_profile_aggregates_genre_metadata():
    pairs = [mk(1, meta={"genres": ["Comedy"]}), mk(2, meta={"genres": ["Comedy", "Drama"]})]
    profile = profile_corpus(pairs)
    assert profile.genres["Comedy"] == 2
    assert profile.genres["Drama"] == 1


def test_profile_respects_the_sample_rate():
    pairs = [mk(i, src=f"This is unique sentence number {i} here.") for i in range(2000)]
    profile = profile_corpus(pairs, sample_rate=0.1)
    assert profile.n_seen == 2000
    assert 100 < profile.n_sampled < 300


def test_markdown_report_renders_without_optional_sections():
    profile = profile_corpus([mk(1)])
    text = profile_to_markdown(profile, title="t")
    assert "# t" in text and "## Length" in text and "## Quality signals" in text


def test_profile_of_empty_corpus_does_not_crash():
    profile = profile_corpus([])
    assert profile.n_sampled == 0
    assert profile_to_markdown(profile)


# -- memory bounds ------------------------------------------------------

def test_tracking_is_bounded_and_reports_truncation():
    # At 2M sampled records the unbounded dedupe sets and segment counters ran to
    # several GB. Past the budget they stop growing and the report says so.
    pairs = [mk(i, src=f"Unique english sentence number {i} here.",
                tgt=f"Oración única en español número {i} aquí.") for i in range(500)]
    profile = profile_corpus(pairs, max_tracked=50)
    assert profile.truncated is True
    assert profile.max_tracked == 50
    # Rates and length statistics are unaffected by truncation.
    assert profile.n_sampled == 500
    assert profile.src_tokens["p50"] > 0


def test_untruncated_profile_does_not_claim_truncation():
    profile = profile_corpus([mk(i) for i in range(10)], max_tracked=10_000)
    assert profile.truncated is False


def test_top_segment_share_is_reported():
    # The concentration measure that stays reliable at any sample rate, unlike
    # the duplicate rates.
    pairs = [mk(i, src="What?", tgt="¿Qué?") for i in range(90)]
    pairs += [mk(100 + i, src=f"A longer distinct sentence {i} goes here.") for i in range(10)]
    profile = profile_corpus(pairs)
    assert profile.top_segment_share > 0.5


def test_markdown_warns_that_sampled_duplicate_rates_are_lower_bounds():
    profile = profile_corpus([mk(i) for i in range(50)], sample_rate=0.5)
    text = profile_to_markdown(profile)
    assert "lower bounds" in text


def test_markdown_omits_the_sampling_warning_at_full_rate():
    profile = profile_corpus([mk(i) for i in range(10)], sample_rate=1.0)
    assert "lower bounds" not in profile_to_markdown(profile)


def test_yield_estimate_can_be_sampled():
    from communilate.analysis import estimate_filter_yield

    pairs = [mk(i, src=f"Distinct english sentence number {i} goes here.") for i in range(1000)]
    full = estimate_filter_yield(pairs, {"min_tokens": 1, "dedupe": False})
    sampled = estimate_filter_yield(
        [mk(i, src=f"Distinct english sentence number {i} goes here.") for i in range(1000)],
        {"min_tokens": 1, "dedupe": False},
        sample_rate=0.2,
    )
    assert full.seen == 1000
    assert 100 < sampled.seen < 300


def test_sampled_yield_warns_when_dedupe_is_on():
    from communilate.analysis import estimate_filter_yield

    report = estimate_filter_yield(
        [mk(i) for i in range(100)], {"min_tokens": 1, "dedupe": "pair"}, sample_rate=0.5
    )
    assert any("sample" in w for w in report.warnings)
