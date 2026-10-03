"""Command line entry point.

Every command takes ``--config`` and optional ``--set key.path=value`` overrides,
so a run is fully described by a config file plus whatever you changed on the
command line. ``--data-path`` is a shortcut for ``--set data.path=...``, which is
the override you will reach for most: swapping corpora is meant to be a
one-argument change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def _display_path(path: Path) -> str:
    """Show a path relative to the repo when possible, absolute otherwise.

    Output directories are frequently outside the repo (a scratch dir, a mounted
    drive), and relative_to raises rather than falling back.
    """
    from .config import REPO_ROOT

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def cmd_download(args) -> int:
    """Fetch an OPUS corpus in Moses format and verify the two sides align."""
    from .data.download import download_corpus

    extracted = download_corpus(
        corpus=args.corpus,
        lang_a=args.source,
        lang_b=args.target,
        version=args.version,
        dest_dir=args.out,
        keep_archive=args.keep_archive,
    )
    print(f"\nnext: profile it before cleaning\n  communilate analyze -c base --data-path {args.out} "
          f"--set data.loader=opensubtitles --sample 0.02")
    return 0 if extracted.aligned else 1


def cmd_analyze(args) -> int:
    """Profile a corpus: lengths, duplication, language sanity, dialect markers.

    Worth running before any cleaning decision -- the filter thresholds in a
    config are guesses until this says what the corpus actually looks like.
    """
    from .analysis import estimate_filter_yield, profile_corpus, save_profile
    from .data import load_split

    cfg = _load(args)
    filter_spec = cfg.get("data.filters")
    filter_spec = filter_spec.to_dict() if hasattr(filter_spec, "to_dict") else (filter_spec or {})

    profile = profile_corpus(
        load_split(cfg, args.split), sample_rate=args.sample, top_n=args.top
    )
    if profile.n_sampled == 0:
        print(f"no records sampled from split {args.split!r} at rate {args.sample}")
        return 1

    if not args.skip_filter_estimate:
        # A second pass: the profiler consumes its iterator, and re-reading is
        # cheaper than buffering a corpus this size in memory.
        report = estimate_filter_yield(load_split(cfg, args.split), filter_spec)
        profile.filter_yield = {
            "seen": report.seen,
            "kept": report.kept,
            "yield_pct": report.yield_pct,
            "dropped": dict(report.dropped),
            "text": report.as_text(),
        }

    out_dir = Path(args.out or cfg.get("eval.output_dir", "runs/analysis")) / "profile"
    json_path, md_path = save_profile(
        profile, out_dir, title=f"{cfg.get('experiment.name', 'corpus')} / {args.split}"
    )
    print(profile_markdown_summary(profile))
    print(f"\nfull profile: {md_path}\n              {json_path}")
    return 0


def profile_markdown_summary(profile) -> str:
    """A short terminal summary; the full tables go to the report file."""
    from .analysis import _pct

    lines = [
        "",
        f"sampled {profile.n_sampled:,} of {profile.n_seen:,} records",
        f"source tokens  median {profile.src_tokens.get('p50', 0):.0f}  "
        f"p95 {profile.src_tokens.get('p95', 0):.0f}",
        f"length ratio   median {profile.length_ratio.get('p50', 0):.2f}  "
        f"p95 {profile.length_ratio.get('p95', 0):.2f}",
        "",
    ]
    for key in (
        "duplicate_pair", "duplicate_src", "untranslated_copy", "has_markup",
        "advertising", "under_4_tokens", "length_ratio_over_2.5",
        "src_language_mismatch", "tgt_language_mismatch",
    ):
        if key in profile.rates:
            lines.append(f"  {key:26} {_pct(profile.rates[key])}")

    if profile.dialect:
        total = profile.n_sampled or 1
        lines.append("")
        lines.append("  Spanish dialect markers:")
        for key, count in profile.dialect.items():
            lines.append(f"    {key:24} {_pct(count / total)}")

    if profile.filter_yield:
        lines += ["", f"  configured filters keep {profile.filter_yield['yield_pct']:.2f}%"]
    return "\n".join(lines)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", "-c", required=True, help="config name or path (e.g. exp1_register)"
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="override data.path -- the folder to pull the corpus from",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config key, e.g. --set lora.r=32 --set training.epochs=1",
    )


def _load(args) -> "object":
    overrides = list(args.overrides)
    if args.data_path:
        overrides.append(f"data.path={args.data_path}")
    return load_config(args.config, overrides)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_inspect(args) -> int:
    """Print the resolved config and count records, without loading any model."""
    import json

    from .data import load_split

    cfg = _load(args)
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))

    if args.count:
        for split in ("train", "dev", "test"):
            try:
                n = sum(1 for _ in load_split(cfg, split))
                print(f"{split}: {n} records")
            except Exception as exc:  # noqa: BLE001
                print(f"{split}: unavailable ({exc})")
    return 0


def cmd_prepare(args) -> int:
    """Read a raw corpus, filter it, and write clean JSONL to data/processed.

    Run once per corpus. Training then reads the output through the jsonl_dir
    loader, so the expensive filtering does not repeat every epoch.
    """
    from .config import REPO_ROOT
    from .data import load_split
    from .data.filters import FilterReport, apply_filters
    from .schema import write_jsonl

    cfg = _load(args)
    out_dir = Path(args.out or cfg.get("data.processed_path", "data/processed/prepared"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    filter_spec = cfg.get("data.filters")
    filter_spec = filter_spec.to_dict() if hasattr(filter_spec, "to_dict") else (filter_spec or {})

    total = 0
    for split in ("train", "dev", "test"):
        report = FilterReport()
        pairs = apply_filters(load_split(cfg, split), filter_spec, report)
        n = write_jsonl(pairs, out_dir / f"{split}.jsonl")
        total += n
        print(f"\n[{split}] -> {out_dir / f'{split}.jsonl'}")
        print(report.as_text())

    print(f"\nwrote {total} records to {out_dir}")
    print(f"point a config at it with:  --data-path {_display_path(out_dir)}")
    return 0


def cmd_make_regional(args) -> int:
    """Derive Peninsular / Latin American variants for Experiment 2.

    Reads a processed dataset, keeps only the records that actually carry
    plural-you marking, and writes two parallel sets differing solely on that
    axis. Records the rules cannot convert cleanly are dropped, not guessed.
    """
    from .config import REPO_ROOT
    from .data import load_split
    from .data.regional_rules import ES_419, ES_ES, detect_region, to_latin_american
    from .schema import write_jsonl

    cfg = _load(args)
    out_dir = Path(args.out or "data/processed/regional")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    strict = not args.lenient
    stats = {"seen": 0, "no_plural_you": 0, "already_latam": 0, "unconvertible": 0, "written": 0}

    for split in ("train", "dev", "test"):
        peninsular, latam = [], []
        for pair in load_split(cfg, split):
            stats["seen"] += 1
            spanish_side = pair.tgt if pair.tgt_lang.startswith("spa") else pair.src
            region = detect_region(spanish_side)

            if region is None:
                stats["no_plural_you"] += 1
                continue
            if region == ES_419:
                stats["already_latam"] += 1
                continue

            converted = to_latin_american(spanish_side, strict=strict)
            if not converted.changed or not converted.is_clean:
                stats["unconvertible"] += 1
                continue

            es_pair = Pair_with(pair, spanish_side, ES_ES)
            la_pair = Pair_with(pair, converted.text, ES_419, suffix="::419")
            peninsular.append(es_pair)
            latam.append(la_pair)
            stats["written"] += 1

        write_jsonl(peninsular, out_dir / "es-ES" / f"{split}.jsonl")
        write_jsonl(latam, out_dir / "es-419" / f"{split}.jsonl")
        print(f"[{split}] {len(peninsular)} pairs per variant")

    print("\n" + "\n".join(f"{k}: {v}" for k, v in stats.items()))
    print(f"\nwrote {out_dir}/es-ES and {out_dir}/es-419")
    print("train two adapters with:")
    shown = _display_path(out_dir)
    print(f"  train -c exp2_regional --data-path {shown}/es-ES")
    print(f"  train -c exp2_regional --data-path {shown}/es-419")
    return 0


def Pair_with(pair, spanish_text: str, region: str, suffix: str = ""):
    """Copy a pair with its Spanish side replaced and its region tagged."""
    from .schema import Pair

    spanish_is_target = pair.tgt_lang.startswith("spa")
    return Pair(
        id=pair.id + suffix,
        src=pair.src if spanish_is_target else spanish_text,
        tgt=spanish_text if spanish_is_target else pair.tgt,
        src_lang=pair.src_lang,
        tgt_lang=pair.tgt_lang,
        register=pair.register,
        region=region,
        source_corpus=pair.source_corpus,
        split=pair.split,
        meta={**pair.meta, "regional_variant": region},
    )


def cmd_train(args) -> int:
    from .train_lora import train

    cfg = _load(args)
    train(cfg)
    return 0


def cmd_evaluate(args) -> int:
    from .eval import evaluate

    cfg = _load(args)
    evaluate(cfg, adapter=args.adapter, split=args.split)
    return 0


def cmd_translate(args) -> int:
    """Translate ad-hoc text, optionally showing frozen and adapted side by side."""
    from .translate import Translator

    cfg = _load(args)
    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print("no input text given", file=sys.stderr)
        return 1

    if args.compare:
        baseline = Translator(cfg, adapter="none").translate(texts)
        adapted = Translator(cfg, adapter=args.adapter).translate(texts)
        for source, base, adapt in zip(texts, baseline, adapted):
            marker = " " if base.hypothesis.strip() == adapt.hypothesis.strip() else "*"
            print(f"\n{source}")
            print(f"  baseline: {base.hypothesis}")
            print(f"{marker} adapted:  {adapt.hypothesis}")
    else:
        for translation in Translator(cfg, adapter=args.adapter).translate(texts):
            print(translation.hypothesis)
    return 0


def cmd_build_memory(args) -> int:
    from .data import load_split
    from .retrieval import TranslationMemory

    cfg = _load(args)
    pairs = list(load_split(cfg, "train"))
    path = TranslationMemory(cfg).build(pairs).save(
        args.out or cfg.get("retrieval.index_path", "runs/memory")
    )
    print(f"indexed {len(pairs)} pairs -> {path}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="communilate",
        description="Heritage-adapted translation: LoRA + retrieval over NLLB-200",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="print the resolved config and dataset counts")
    _add_common(p)
    p.add_argument("--count", action="store_true", help="also count records per split")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("prepare", help="filter a raw corpus into clean JSONL")
    _add_common(p)
    p.add_argument("--out", default=None, help="output directory")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("make-regional", help="derive es-ES / es-419 variants (Experiment 2)")
    _add_common(p)
    p.add_argument("--out", default=None)
    p.add_argument(
        "--lenient",
        action="store_true",
        help="apply suffix rules to unrecognised forms instead of dropping them",
    )
    p.set_defaults(func=cmd_make_regional)

    p = sub.add_parser("train", help="LoRA fine-tune")
    _add_common(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="frozen baseline vs. adapted, with a report")
    _add_common(p)
    p.add_argument("--adapter", default=None, help="adapter path, or 'none' for baseline only")
    p.add_argument("--split", default="test")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("translate", help="translate text from arguments or stdin")
    _add_common(p)
    p.add_argument("text", nargs="*")
    p.add_argument("--adapter", default=None)
    p.add_argument("--compare", action="store_true", help="show frozen and adapted side by side")
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser("download", help="fetch an OPUS corpus (Moses format)")
    p.add_argument("--corpus", default="OpenSubtitles")
    p.add_argument("--source", default="en", help="ISO-639-1 code, e.g. en")
    p.add_argument("--target", default="es", help="ISO-639-1 code, e.g. es")
    p.add_argument("--version", default="latest")
    p.add_argument("--out", default="data/raw/opensubtitles_en_es")
    p.add_argument("--keep-archive", action="store_true", help="do not delete the zip")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("analyze", help="profile a corpus before cleaning it")
    _add_common(p)
    p.add_argument("--split", default="train")
    p.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="fraction of records to profile (hash-based, reproducible). Use ~0.02 on a full OpenSubtitles dump.",
    )
    p.add_argument("--top", type=int, default=15, help="how many repeated segments to list")
    p.add_argument("--out", default=None)
    p.add_argument("--skip-filter-estimate", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("build-memory", help="build the retrieval translation memory")
    _add_common(p)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_build_memory)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
