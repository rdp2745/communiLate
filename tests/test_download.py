import zipfile

import pytest

from communilate.data.download import (
    DownloadError,
    build_api_url,
    build_direct_url,
    count_lines,
    extract_archive,
    normalise_pair,
    pair_slug,
    parse_api_response,
)


# -- URL construction ---------------------------------------------------

def test_language_pairs_are_alphabetised():
    # OPUS serves en-es, never es-en; asking for the wrong order 404s.
    assert normalise_pair("es", "en") == ("en", "es")
    assert pair_slug("es", "en") == "en-es"
    assert pair_slug("EN", "ES") == "en-es"


def test_direct_url_shape():
    url = build_direct_url("OpenSubtitles", "es", "en", "v2018")
    assert url.endswith("/OPUS-OpenSubtitles/v2018/moses/en-es.txt.zip")


def test_direct_url_tolerates_a_version_without_the_v():
    assert "/v2018/" in build_direct_url("OpenSubtitles", "en", "es", "2018")


def test_api_url_orders_source_and_target():
    url = build_api_url("OpenSubtitles", "es", "en")
    assert "source=en" in url and "target=es" in url
    assert "preprocessing=moses" in url


# -- API response parsing -----------------------------------------------

def test_parse_api_response_picks_the_newest_version():
    payload = {
        "corpora": [
            {"version": "v2016", "url": "http://x/v2016.zip", "size": "100"},
            {"version": "v2024", "url": "http://x/v2024.zip", "size": "200"},
            {"version": "v2018", "url": "http://x/v2018.zip", "size": "150"},
        ]
    }
    release = parse_api_response(payload, "OpenSubtitles", "en-es")
    assert release.version == "v2024"
    assert release.url == "http://x/v2024.zip"


def test_parse_api_response_converts_kb_to_bytes():
    payload = {"corpora": [{"version": "v2018", "url": "http://x.zip", "size": "1024"}]}
    assert parse_api_response(payload, "OpenSubtitles", "en-es").size_bytes == 1024 * 1024


def test_parse_api_response_survives_an_unparseable_size():
    payload = {"corpora": [{"version": "v2018", "url": "http://x.zip", "size": "unknown"}]}
    assert parse_api_response(payload, "OpenSubtitles", "en-es").size_bytes is None


def test_parse_api_response_rejects_an_empty_result():
    with pytest.raises(DownloadError, match="no releases"):
        parse_api_response({"corpora": []}, "OpenSubtitles", "en-es")


def test_parse_api_response_rejects_a_release_without_a_url():
    with pytest.raises(DownloadError, match="no url"):
        parse_api_response({"corpora": [{"version": "v2018"}]}, "OpenSubtitles", "en-es")


# -- extraction and verification ----------------------------------------

def _make_moses_zip(tmp_path, en_lines, es_lines, with_ids=True):
    archive = tmp_path / "en-es.txt.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("OpenSubtitles.en-es.en", "\n".join(en_lines) + "\n")
        zf.writestr("OpenSubtitles.en-es.es", "\n".join(es_lines) + "\n")
        if with_ids:
            zf.writestr("OpenSubtitles.en-es.ids", "\n".join(
                f"en/2009/100000{i}/1.xml.gz\tes/2009/100000{i}/1.xml.gz\t1\t1"
                for i in range(len(en_lines))
            ) + "\n")
    return archive


def test_extract_reports_sides_and_counts(tmp_path):
    archive = _make_moses_zip(tmp_path, ["a", "b", "c"], ["x", "y", "z"])
    extracted = extract_archive(archive, tmp_path / "out", "en-es")
    assert extracted.aligned
    assert extracted.n_pairs == 3
    assert set(extracted.side_files) == {"en", "es"}
    assert extracted.ids_file is not None


def test_extract_rejects_misaligned_sides(tmp_path):
    # Moses format aligns by line number, so unequal counts make every pair
    # after the divergence garbage. Better to fail than to train on it.
    archive = _make_moses_zip(tmp_path, ["a", "b", "c"], ["x", "y"])
    with pytest.raises(DownloadError, match="different line counts"):
        extract_archive(archive, tmp_path / "out", "en-es")


def test_extract_tolerates_a_missing_ids_file(tmp_path):
    archive = _make_moses_zip(tmp_path, ["a", "b"], ["x", "y"], with_ids=False)
    assert extract_archive(archive, tmp_path / "out", "en-es").ids_file is None


def test_extract_rejects_a_non_zip(tmp_path):
    # A truncated download or an HTML error page saved as .zip is the usual cause.
    bogus = tmp_path / "en-es.txt.zip"
    bogus.write_text("<html>404 Not Found</html>", encoding="utf-8")
    with pytest.raises(DownloadError, match="not a zip"):
        extract_archive(bogus, tmp_path / "out", "en-es")


def test_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.en", "a\n")
        zf.writestr("OpenSubtitles.en-es.es", "x\n")
    with pytest.raises(DownloadError, match="unsafe archive member"):
        extract_archive(archive, tmp_path / "out", "en-es")


def test_extract_reports_a_missing_side_clearly(tmp_path):
    archive = tmp_path / "en-es.txt.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("OpenSubtitles.en-es.en", "a\n")
    with pytest.raises(DownloadError, match="no 'es' side"):
        extract_archive(archive, tmp_path / "out", "en-es")


def test_count_lines(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    assert count_lines(path) == 3
