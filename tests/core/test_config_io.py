import pytest
import yaml
from pydantic import ValidationError

from owm_envs.core.config_io import YamlModel


class Inner(YamlModel):
    gain: float = 1.5
    limits: tuple[float, float] = (0.0, 10.0)


class Outer(YamlModel):
    name: str = "demo"
    count: int = 3
    inner: Inner = Inner()


def test_roundtrips_through_yaml_text():
    original = Outer(name="run-a", count=7, inner=Inner(gain=2.5))
    assert Outer.model_validate(yaml.safe_load(original.to_yaml())) == original


def test_roundtrips_through_a_file(tmp_path):
    original = Outer(name="run-b", inner=Inner(gain=9.0, limits=(1.0, 2.0)))
    path = tmp_path / "cfg.yaml"
    original.to_yaml(path)
    assert Outer.from_yaml(path) == original


def test_to_yaml_returns_text_even_when_writing(tmp_path):
    text = Outer().to_yaml(tmp_path / "cfg.yaml")
    assert "name: demo" in text
    assert (tmp_path / "cfg.yaml").exists()


def test_yaml_is_human_readable_block_style():
    text = Outer().to_yaml()
    # Block style, not inline flow -- the file is meant to be read and hand-edited.
    assert "{" not in text
    assert "inner:" in text


def test_models_are_frozen():
    cfg = Outer()
    with pytest.raises(ValidationError):
        cfg.count = 9


def test_unknown_keys_are_rejected():
    # A typo'd key in a hand-edited YAML must fail loudly, not be silently dropped.
    with pytest.raises(ValidationError):
        Outer.model_validate({"name": "x", "kount": 3})


def test_wrong_types_are_rejected():
    with pytest.raises(ValidationError):
        Outer.model_validate({"count": "not-an-int"})


def test_wrong_tuple_length_is_rejected():
    with pytest.raises(ValidationError):
        Inner.model_validate({"limits": [1.0, 2.0, 3.0]})


def test_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Outer.from_yaml(tmp_path / "absent.yaml")
