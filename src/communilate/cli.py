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
