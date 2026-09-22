"""Model and adapter loading.

All torch/transformers/peft imports are deliberately function-local: the config,
schema and data layers must stay importable (and testable) on a machine with no
ML stack installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Config


def target_lang_id(tokenizer, lang_code: str) -> int:
    """Resolve the forced-BOS token id for a target language.

    ``tokenizer.lang_code_to_id`` was removed in recent transformers releases;
    ``convert_tokens_to_ids`` works across versions. NLLB will not reliably
    generate into the requested language without this id passed to ``generate``.
    """
    token_id = tokenizer.convert_tokens_to_ids(lang_code)
    unk = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk:
        raise ValueError(
            f"tokenizer does not know language code {lang_code!r}; NLLB expects "
            f"FLORES-200 codes such as eng_Latn, spa_Latn, guj_Gujr"
        )
    return token_id


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg["model.base_model"],
        src_lang=cfg["model.src_lang"],
        tgt_lang=cfg["model.tgt_lang"],
    )


def load_base_model(cfg: Config):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    dtype_name = cfg.get("model.dtype", "float32")
    dtype = getattr(torch, dtype_name)
    return AutoModelForSeq2SeqLM.from_pretrained(cfg["model.base_model"], torch_dtype=dtype)


def build_lora_config(cfg: Config):
    """LoRA config for NLLB.

    NLLB-200 is an M2M100 architecture, so the attention projections are named
    ``q_proj``/``k_proj``/``v_proj``/``out_proj``. Adding ``fc1``/``fc2`` brings
    the feed-forward blocks in too -- worth trying only if attention-only
    underfits.
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora.r"]),
        lora_alpha=int(cfg["lora.alpha"]),
        lora_dropout=float(cfg["lora.dropout"]),
        target_modules=list(cfg["lora.target_modules"]),
        bias=cfg.get("lora.bias", "none"),
    )


def resolve_adapter_path(cfg: Config, adapter: str | None) -> Path | None:
    """Pick the adapter to load: explicit argument, else the config's default.

    ``adapter="none"`` forces the frozen baseline even when the config names an
    adapter -- this is what makes the frozen-vs-adapted comparison a single flag
    rather than two config files.
    """
    if adapter == "none":
        return None
    raw = adapter or cfg.get("inference.adapter_path")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_for_inference(cfg: Config, adapter: str | None = None) -> tuple[Any, Any, Path | None]:
    """Load ``(model, tokenizer, adapter_path)`` ready for generation."""
    import torch

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)

    adapter_path = resolve_adapter_path(cfg, adapter)
    if adapter_path is not None:
        from peft import PeftModel

        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter not found at {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload() if cfg.get("inference.merge_adapter", False) else model

    device = cfg.get("inference.device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, adapter_path


def load_for_training(cfg: Config):
    """Load a base model wrapped in a fresh LoRA adapter."""
    from peft import get_peft_model

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)

    if cfg.get("lora.enabled", True):
        model = get_peft_model(model, build_lora_config(cfg))
        model.print_trainable_parameters()
    return model, tokenizer
