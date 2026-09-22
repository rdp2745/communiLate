"""Config loading.

Everything the pipeline needs lives in a YAML file under ``configs/``. A config
may ``extends:`` another config, which is deep-merged underneath it, so the
experiment files only have to state what differs from ``base.yaml``.

Nothing in here imports torch/transformers -- configs are cheap to load and
cheap to test.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a config is malformed or missing a required key."""


class Config(Mapping):
    """Read-only dict wrapper with dotted-path access.

    ``cfg["model.base_model"]`` and ``cfg.get("retrieval.enabled", False)`` both
    work, which keeps call sites from having to chain ``[]`` lookups and guess
    which intermediate dicts exist.
    """

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    # -- Mapping protocol ------------------------------------------------
    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self._resolve(key)
        except KeyError:
            return False
        return True

    def __getitem__(self, key: str) -> Any:
        try:
            value = self._resolve(key)
        except KeyError:
            raise KeyError(f"missing config key {key!r} (config: {self.source})") from None
        return Config(value, self.source) if isinstance(value, dict) else value

    def _resolve(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(dotted)
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def require(self, *keys: str) -> None:
        """Fail loudly, listing every missing key at once."""
        missing = [k for k in keys if k not in self]
        if missing:
            raise ConfigError(
                f"config {self.source} is missing required keys: {', '.join(missing)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self.source}, keys={sorted(self._data)})"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``; ``override`` wins.

    Lists are replaced wholesale rather than concatenated -- appending would
    make it impossible for an experiment to *shrink* a list it inherited.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(raw: str) -> Any:
    """Parse a CLI override value as YAML so ``true``/``3``/``[a,b]`` work."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(data: dict, overrides: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` strings on top of a loaded config."""
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key.path=value")
        dotted, raw = item.split("=", 1)
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {dotted!r} traverses a non-mapping")
        node[parts[-1]] = _coerce(raw.strip())
    return out


def _interpolate(data: Any, root: dict) -> Any:
    """Expand ``${a.b}`` references against the fully merged config.

    Used mainly so ``training.output_dir`` can be written once in base.yaml as
    ``runs/${experiment.name}`` and stay correct for every experiment.
    """
    if isinstance(data, dict):
        return {k: _interpolate(v, root) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate(v, root) for v in data]
    if not isinstance(data, str) or "${" not in data:
        return data

    out = data
    for _ in range(5):  # bounded, so a cyclic reference can't hang the load
        start = out.find("${")
        if start == -1:
            break
        end = out.find("}", start)
        if end == -1:
            break
        dotted = out[start + 2 : end]
        node: Any = root
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"cannot interpolate ${{{dotted}}}: no such key")
            node = node[part]
        out = out[:start] + str(node) + out[end + 1 :]
    return out


def _resolve_path(name: str | os.PathLike) -> Path:
    """Accept a bare name ('exp1_register'), a filename, or a full path."""
    candidate = Path(name)
    if candidate.exists():
        return candidate.resolve()
    for stem in (candidate, candidate.with_suffix(".yaml"), candidate.with_suffix(".yml")):
        probe = CONFIG_DIR / stem
        if probe.exists():
            return probe.resolve()
    raise ConfigError(f"no config found for {name!r} (looked in {CONFIG_DIR})")


def _load_raw(name: str | os.PathLike, _seen: tuple[Path, ...] = ()) -> tuple[dict, Path]:
    """Read a config and merge its ``extends`` chain, without interpolating.

    Interpolation is deliberately deferred to the outermost load. Expanding
    ``${experiment.name}`` while reading the parent would bake in the parent's
    value, so every experiment inheriting ``runs/${experiment.name}`` would
    write to ``runs/base``.
    """
    path = _resolve_path(name)
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"circular extends chain: {chain}")

    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping at the top level")

    parent_name = data.pop("extends", None)
    if parent_name:
        parent_data, _ = _load_raw(parent_name, _seen=(*_seen, path))
        data = deep_merge(parent_data, data)
    return data, path


def load_config(name: str | os.PathLike, overrides: Iterable[str] = ()) -> Config:
    """Load a config, resolving ``extends``, CLI overrides and interpolation.

    Order matters: overrides are applied before interpolation so that
    ``--set experiment.name=foo`` also redirects ``runs/${experiment.name}``.
    """
    data, path = _load_raw(name)
    if overrides:
        data = apply_overrides(data, overrides)
    data = _interpolate(data, data)
    return Config(data, source=path)


def save_resolved(cfg: Config, dest: str | os.PathLike) -> Path:
    """Write the fully-merged config next to a run's outputs.

    Worth doing on every run: six weeks later the only reliable record of what
    produced a checkpoint is the config that was actually resolved at the time,
    not the file in ``configs/`` that has been edited since.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, ensure_ascii=False, sort_keys=True)
    return dest
