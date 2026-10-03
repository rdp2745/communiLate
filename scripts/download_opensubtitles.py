#!/usr/bin/env python3
"""Download the OpenSubtitles English-Spanish corpus from OPUS.

Standalone wrapper around :mod:`communilate.data.download`, for running before
the package is installed:

    python3 scripts/download_opensubtitles.py --out data/raw/opensubtitles_en_es

Equivalent to ``communilate download`` once the package is on the path.

Expect roughly a gigabyte compressed for en-es. The download resumes if it is
interrupted, so re-running after a dropped connection picks up where it stopped
rather than starting over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from communilate.data.download import DownloadError, download_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default="OpenSubtitles")
    parser.add_argument("--source", default="en", help="ISO-639-1 code")
    parser.add_argument("--target", default="es", help="ISO-639-1 code")
    parser.add_argument(
        "--version",
        default="latest",
        help="OPUS release; 'latest' asks the API, otherwise e.g. v2018",
    )
    parser.add_argument("--out", default="data/raw/opensubtitles_en_es")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    try:
        extracted = download_corpus(
            corpus=args.corpus,
            lang_a=args.source,
            lang_b=args.target,
            version=args.version,
            dest_dir=args.out,
            keep_archive=args.keep_archive,
        )
    except DownloadError as exc:
        print(f"\ndownload failed: {exc}", file=sys.stderr)
        return 1

    print(f"\n{extracted.n_pairs:,} aligned segments in {extracted.directory}")
    print("\nnext, profile it before deciding how hard to filter:")
    print(
        f"  communilate analyze -c base --data-path {args.out} "
        f"--set data.loader=opensubtitles --sample 0.02"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
