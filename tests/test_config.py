import pytest

from communilate.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    load_config,
)


def test_deep_merge_child_wins_and_parent_survives():
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
    assert merged == {"a": {"b": 1, "c": 3}}


def test_deep_merge_replaces_lists_rather_than_appending():
    # Appending would make it impossible for a child config to shrink a list.
    merged = deep_merge({"x": [1, 2, 3]}, {"x": [9]})
    assert merged["x"] == [9]


def test_overrides_are_yaml_typed():
    out = apply_overrides({}, ["lora.r=32", "training.fp16=true", "eval.metrics=[chrf]"])
    assert out["lora"]["r"] == 32
    assert out["training"]["fp16"] is True
    assert out["eval"]["metrics"] == ["chrf"]


def test_override_without_equals_is_rejected():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["lora.r"])


@pytest.mark.parametrize(
    "name", ["base", "exp1_register", "exp2_regional", "exp3_gujarati"]
)
def test_shipped_configs_load_and_have_required_keys(name):
    cfg = load_config(name)
    cfg.require(
        "model.base_model",
        "model.src_lang",
        "model.tgt_lang",
        "data.loader",
        "data.path",
        "lora.r",
        "training.output_dir",
    )


def test_interpolation_uses_child_name_not_parent():
    # Regression: interpolating during the parent load baked in the parent's
    # experiment.name, sending every experiment's output to runs/base.
    assert load_config("exp1_register")["training.output_dir"] == "runs/exp1_register"
    assert load_config("exp3_gujarati")["training.output_dir"] == "runs/exp3_gujarati"


def test_override_applies_before_interpolation():
    cfg = load_config("exp1_register", ["experiment.name=smoke"])
    assert cfg["training.output_dir"] == "runs/smoke"


def test_dotted_access_and_missing_key_behaviour():
    cfg = load_config("base")
    assert isinstance(cfg["lora.r"], int)
    assert cfg.get("nope.nope", "fallback") == "fallback"
    with pytest.raises(KeyError):
        cfg["nope.nope"]
