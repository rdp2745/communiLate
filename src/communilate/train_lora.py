"""LoRA fine-tuning of NLLB-200.

Corpus-agnostic by construction: the trainer only ever sees
:class:`~communilate.schema.Pair` records handed over by the configured loader,
so training on OpenSubtitles, on a rule-transformed regional set, or on
hand-transcribed Gujarati recordings is the same code with a different config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Config, save_resolved
from .data import load_split
from .model import load_for_training
from .schema import Pair


def _records_to_dataset(pairs: list[Pair]):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "src": [p.src for p in pairs],
            "tgt": [p.tgt for p in pairs],
            "src_lang": [p.src_lang for p in pairs],
            "tgt_lang": [p.tgt_lang for p in pairs],
        }
    )


def _build_tokenize_fn(tokenizer, cfg: Config):
    max_source = int(cfg.get("model.max_source_length", 128))
    max_target = int(cfg.get("model.max_target_length", 128))

    def tokenize(batch: dict[str, list]) -> dict[str, Any]:
        # src_lang must be set per batch: a both-direction dataset mixes
        # eng->spa and spa->eng rows, and NLLB prefixes the source language
        # token during encoding, so a stale setting silently mislabels half
        # the corpus.
        langs = set(batch["src_lang"])
        if len(langs) != 1:
            raise ValueError(
                "a tokenisation batch mixed source languages; set "
                "training.group_by_direction=true so batches are homogeneous"
            )
        tokenizer.src_lang = next(iter(langs))
        tokenizer.tgt_lang = batch["tgt_lang"][0]

        model_inputs = tokenizer(
            batch["src"], max_length=max_source, truncation=True
        )
        labels = tokenizer(
            text_target=batch["tgt"], max_length=max_target, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


def _sort_by_direction(pairs: list[Pair]) -> list[Pair]:
    """Group rows by language pair so tokenisation batches stay homogeneous."""
    return sorted(pairs, key=lambda p: (p.src_lang, p.tgt_lang))


def train(cfg: Config) -> Path:
    """Run LoRA fine-tuning and return the adapter output directory."""
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    output_dir = Path(cfg["training.output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved(cfg, output_dir / "resolved_config.json")

    model, tokenizer = load_for_training(cfg)

    train_pairs = _sort_by_direction(list(load_split(cfg, "train")))
    if not train_pairs:
        raise RuntimeError(
            f"loader {cfg['data.loader']!r} returned no training records for "
            f"data.path={cfg.get('data.path')!r}"
        )
    dev_pairs = _sort_by_direction(list(load_split(cfg, "dev")))
    print(f"train={len(train_pairs)} dev={len(dev_pairs)} -> {output_dir}")

    tokenize = _build_tokenize_fn(tokenizer, cfg)
    batch_size = int(cfg.get("training.tokenize_batch_size", 256))
    train_ds = _records_to_dataset(train_pairs).map(
        tokenize, batched=True, batch_size=batch_size, remove_columns=["src", "tgt", "src_lang", "tgt_lang"]
    )
    eval_ds = (
        _records_to_dataset(dev_pairs).map(
            tokenize, batched=True, batch_size=batch_size, remove_columns=["src", "tgt", "src_lang", "tgt_lang"]
        )
        if dev_pairs
        else None
    )

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(cfg["training.learning_rate"]),
        per_device_train_batch_size=int(cfg["training.batch_size"]),
        per_device_eval_batch_size=int(cfg.get("training.eval_batch_size", cfg["training.batch_size"])),
        gradient_accumulation_steps=int(cfg.get("training.gradient_accumulation_steps", 1)),
        num_train_epochs=float(cfg["training.epochs"]),
        warmup_ratio=float(cfg.get("training.warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("training.weight_decay", 0.0)),
        logging_steps=int(cfg.get("training.logging_steps", 50)),
        save_steps=int(cfg.get("training.save_steps", 500)),
        save_total_limit=int(cfg.get("training.save_total_limit", 2)),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("training.eval_steps", 500)),
        fp16=bool(cfg.get("training.fp16", False)),
        bf16=bool(cfg.get("training.bf16", False)),
        seed=int(cfg.get("training.seed", 42)),
        report_to=list(cfg.get("training.report_to", [])),
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"adapter saved to {adapter_dir}")
    return adapter_dir
