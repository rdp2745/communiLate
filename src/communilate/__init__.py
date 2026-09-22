"""communiLate: heritage-adapted machine translation.

Frozen NLLB-200 as a formal baseline, LoRA adapters for register and dialect
adaptation, and a translation-memory + confidence gate on top.

The layering is deliberate: :mod:`~communilate.config`, :mod:`~communilate.schema`
and :mod:`~communilate.data` import nothing from torch, so the data pipeline can
be developed and tested without an ML stack installed.
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .schema import Pair, read_jsonl, write_jsonl

__all__ = ["Config", "load_config", "Pair", "read_jsonl", "write_jsonl", "__version__"]
