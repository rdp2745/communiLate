"""Download OpenSubtitles (or any OPUS corpus) in Moses format.

OPUS serves a JSON API that resolves a corpus/language-pair/version to a real
download URL. The API is preferred over constructing URLs by hand because the
latest version identifier changes over time; a constructed URL is kept as a
fallback for when the API is unreachable.

Two details that cause most of the confusion with OPUS downloads:

* **Language pairs are alphabetically ordered.** There is no ``es-en`` release;
  it is ``en-es``. Asking for the wrong order 404s, so the order is normalised
  here rather than left to the caller.
* **The Moses archive is a zip of two parallel plain-text files** plus an
  ``.ids`` file, aligned by line number. If the two sides have different line
  counts the download is corrupt and every pair after the divergence is
  garbage, so the line counts are checked after extraction rather than trusted.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

OPUS_API = "https://opus.nlpl.eu/opusapi/"
OPUS_OBJECT_STORE = "https://object.pouta.csc.fi"

# Fallback when the API cannot be reached. v2018 is the long-standing release
# most papers cite; v2024 exists for some pairs. The API is authoritative --
# this is only so a blocked API does not stop the download entirely.
FALLBACK_VERSION = "v2018"

USER_AGENT = "communilate/0.1 (research; +https://github.com/rdp2745/communiLate)"


class DownloadError(RuntimeError):
    """Raised when a corpus cannot be resolved, fetched or verified."""


def normalise_pair(lang_a: str, lang_b: str) -> tuple[str, str]:
    """Return the two language codes in OPUS's alphabetical order."""
    return tuple(sorted((lang_a.lower(), lang_b.lower())))  # type: ignore[return-value]


def pair_slug(lang_a: str, lang_b: str) -> str:
    """``("es", "en") -> "en-es"`` -- the form OPUS actually serves."""
    first, second = normalise_pair(lang_a, lang_b)
    return f"{first}-{second}"


def build_direct_url(
    corpus: str, lang_a: str, lang_b: str, version: str = FALLBACK_VERSION
) -> str:
    """Construct the object-store URL for a Moses download."""
    version = version if version.startswith("v") else f"v{version}"
    return (
        f"{OPUS_OBJECT_STORE}/OPUS-{corpus}/{version}/moses/"
        f"{pair_slug(lang_a, lang_b)}.txt.zip"
    )


def build_api_url(
    corpus: str, lang_a: str, lang_b: str, version: str = "latest"
) -> str:
    first, second = normalise_pair(lang_a, lang_b)
    query = urllib.parse.urlencode(
        {
            "corpus": corpus,
            "source": first,
            "target": second,
            "preprocessing": "moses",
            "version": version,
        }
    )
    return f"{OPUS_API}?{query}"


@dataclass
class CorpusRelease:
    """A resolved, downloadable OPUS release."""

    corpus: str
    pair: str
    version: str
    url: str
    size_bytes: int | None = None
    n_alignments: int | None = None
    resolved_via: str = "api"

    @property
    def size_mb(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 1_000_000


def parse_api_response(payload: dict, corpus: str, pair: str) -> CorpusRelease:
    """Pick the best release out of an OPUS API response.

    The API returns every matching release; when several come back the newest
    version is chosen, which is what ``version=latest`` is usually meant to do
    but does not reliably return on its own.
    """
    entries = payload.get("corpora") or []
    if not entries:
        raise DownloadError(
            f"OPUS API returned no releases for corpus={corpus!r} pair={pair!r}"
        )

    def version_key(entry: dict) -> tuple:
        raw = str(entry.get("version", "")).lstrip("v")
        parts = []
        for chunk in raw.replace("-", ".").split("."):
            parts.append(int(chunk) if chunk.isdigit() else 0)
        return tuple(parts) or (0,)

    best = max(entries, key=version_key)
    url = best.get("url")
    if not url:
        raise DownloadError(f"OPUS API release for {corpus}/{pair} has no url: {best}")

    # The API reports sizes in KB in some responses and bytes in others; the
    # field name differs too, so normalise rather than trusting one shape.
    size = best.get("size")
    size_bytes = None
    if size is not None:
        try:
            size_bytes = int(float(size)) * 1024
        except (TypeError, ValueError):
            size_bytes = None

    return CorpusRelease(
        corpus=corpus,
        pair=pair,
        version=str(best.get("version", "unknown")),
        url=url,
        size_bytes=size_bytes,
        n_alignments=_maybe_int(best.get("alignment_pairs") or best.get("alignments")),
        resolved_via="api",
    )


def _maybe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_release(
    corpus: str = "OpenSubtitles",
    lang_a: str = "en",
    lang_b: str = "es",
    version: str = "latest",
    timeout: int = 60,
) -> CorpusRelease:
    """Resolve a download URL, falling back to a constructed one on API failure."""
    pair = pair_slug(lang_a, lang_b)
    api_url = build_api_url(corpus, lang_a, lang_b, version)

    try:
        request = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_api_response(payload, corpus, pair)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
        fallback_version = FALLBACK_VERSION if version == "latest" else version
        print(
            f"OPUS API unreachable ({exc}); falling back to a constructed URL for "
            f"{fallback_version}. If this 404s, check which versions exist at "
            f"https://opus.nlpl.eu/{corpus}/corpus/version/{corpus}"
        )
        return CorpusRelease(
            corpus=corpus,
            pair=pair,
            version=fallback_version,
            url=build_direct_url(corpus, lang_a, lang_b, fallback_version),
            resolved_via="fallback",
        )


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}GB"


def download_file(
    url: str,
    dest: Path,
    resume: bool = True,
    timeout: int = 120,
    chunk_size: int = 1 << 20,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download ``url`` to ``dest``, resuming a partial file when possible.

    Resume matters here: the OpenSubtitles en-es Moses archive is roughly a
    gigabyte, and restarting from zero on a dropped connection is a real cost.
    A server that ignores the Range header is handled by restarting cleanly
    rather than appending to the partial file, which would corrupt it silently.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")

    already = partial.stat().st_size if (resume and partial.exists()) else 0
    headers = {"User-Agent": USER_AGENT}
    if already:
        headers["Range"] = f"bytes={already}-"
        print(f"resuming from {_format_bytes(already)}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and already:  # range beyond EOF: already complete
            partial.replace(dest)
            return dest
        raise DownloadError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"cannot reach {url}: {exc.reason}") from exc

    with response:
        # A 200 to a ranged request means the server ignored Range; appending
        # would interleave a fresh copy onto the partial one.
        if already and response.status != 206:
            print("server ignored the range request -- restarting the download")
            already = 0
            mode = "wb"
        else:
            mode = "ab" if already else "wb"

        remaining = response.headers.get("Content-Length")
        total = (already + int(remaining)) if remaining else None

        downloaded = already
        with partial.open(mode) as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

    if total is not None and downloaded != total:
        raise DownloadError(
            f"truncated download: got {downloaded} bytes, expected {total}. "
            f"Re-run to resume from the partial file at {partial}"
        )

    partial.replace(dest)
    return dest


def _default_progress(downloaded: int, total: int | None) -> None:
    if total:
        pct = 100.0 * downloaded / total
        print(f"\r  {_format_bytes(downloaded)} / {_format_bytes(total)} ({pct:.1f}%)", end="")
    else:
        print(f"\r  {_format_bytes(downloaded)}", end="")


@dataclass
class ExtractedCorpus:
    """Where the extracted sides landed, and how many lines each has."""

    directory: Path
    side_files: dict[str, Path]
    ids_file: Path | None
    line_counts: dict[str, int]

    @property
    def n_pairs(self) -> int:
        return min(self.line_counts.values()) if self.line_counts else 0

    @property
    def aligned(self) -> bool:
        return len(set(self.line_counts.values())) <= 1


def count_lines(path: Path) -> int:
    """Count lines without holding the file in memory."""
    total = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            total += chunk.count(b"\n")
    return total


def extract_archive(archive: Path, dest: Path, pair: str) -> ExtractedCorpus:
    """Extract a Moses zip and verify the two sides have equal line counts."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(archive):
        raise DownloadError(
            f"{archive} is not a zip file. A truncated or error-page download is "
            f"the usual cause -- delete it and retry."
        )

    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            # Refuse absolute paths and traversal before writing anything.
            name = Path(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise DownloadError(f"refusing unsafe archive member {member.filename!r}")
        zf.extractall(dest)

    langs = pair.split("-")
    side_files: dict[str, Path] = {}
    for lang in langs:
        matches = sorted(dest.rglob(f"*.{lang}"))
        if not matches:
            raise DownloadError(
                f"no '{lang}' side found after extracting {archive} into {dest}; "
                f"contents: {[p.name for p in dest.rglob('*') if p.is_file()][:20]}"
            )
        side_files[lang] = max(matches, key=lambda p: p.stat().st_size)

    # v2018 ships the ids file inside the Moses zip; later releases may ship it
    # gzipped, or not at all.
    ids_matches = sorted(dest.rglob("*.ids")) + sorted(dest.rglob("*.ids.gz"))
    ids_file = ids_matches[0] if ids_matches else None

    line_counts = {lang: count_lines(path) for lang, path in side_files.items()}
    corpus = ExtractedCorpus(dest, side_files, ids_file, line_counts)

    if not corpus.aligned:
        raise DownloadError(
            f"the two sides have different line counts {line_counts} -- the Moses "
            f"format aligns by line number, so this archive is unusable. Re-download."
        )
    return corpus


def download_corpus(
    corpus: str = "OpenSubtitles",
    lang_a: str = "en",
    lang_b: str = "es",
    version: str = "latest",
    dest_dir: Path | str = "data/raw/opensubtitles_en_es",
    keep_archive: bool = False,
    show_progress: bool = True,
) -> ExtractedCorpus:
    """Resolve, download, extract and verify an OPUS Moses release."""
    dest_dir = Path(dest_dir)
    release = resolve_release(corpus, lang_a, lang_b, version)

    print(f"{release.corpus} {release.pair} {release.version} ({release.resolved_via})")
    print(f"  {release.url}")
    if release.size_mb:
        print(f"  ~{release.size_mb:.0f} MB compressed")

    archive = dest_dir / f"{release.pair}.txt.zip"
    if archive.exists():
        print(f"archive already present: {archive}")
    else:
        download_file(
            release.url, archive, progress=_default_progress if show_progress else None
        )
        print()

    extracted = extract_archive(archive, dest_dir, release.pair)
    print(f"extracted {extracted.n_pairs:,} aligned segments")
    for lang, path in extracted.side_files.items():
        print(f"  {lang}: {path.name} ({_format_bytes(path.stat().st_size)})")
    print(f"  ids: {extracted.ids_file.name if extracted.ids_file else 'MISSING'}")
    if extracted.ids_file is None:
        print(
            "\n  No .ids file in this release, so there are no IMDb ids and genre\n"
            "  filtering is unavailable. Genre filters in a config will silently pass\n"
            "  every record rather than erroring -- remove them, or try fetching the\n"
            f"  ids separately:\n"
            f"    {OPUS_OBJECT_STORE}/OPUS-{release.corpus}/{release.version}/ids/{release.pair}.ids.gz\n"
            "  (best effort: not every release publishes one)"
        )

    if not keep_archive:
        archive.unlink(missing_ok=True)
        print(f"removed {archive.name}; pass --keep-archive to retain it")

    return extracted
