import pytest
import yaml
from pydantic import ValidationError

from owm_envs.core.models import ConfigModel


class Inner(ConfigModel):
    gain: float = 1.5
    limits: tuple[float, float] = (0.0, 10.0)


class Outer(ConfigModel):
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


def test_roundtrips_through_toml(tmp_path):
    original = Outer(name="run-a", count=7, inner=Inner(gain=2.5))
    path = tmp_path / "cfg.toml"
    original.to_toml(path)
    assert Outer.from_toml(path) == original


def test_toml_and_yaml_produce_equal_models(tmp_path):
    original = Outer(name="x", count=3, inner=Inner(gain=1.0))
    original.to_toml(tmp_path / "c.toml")
    original.to_yaml(tmp_path / "c.yaml")
    assert Outer.from_toml(tmp_path / "c.toml") == Outer.from_yaml(tmp_path / "c.yaml")


def test_load_dispatches_on_suffix(tmp_path):
    original = Outer()
    original.to_toml(tmp_path / "c.toml")
    original.to_yaml(tmp_path / "c.yaml")
    assert Outer.load(tmp_path / "c.toml") == original
    assert Outer.load(tmp_path / "c.yaml") == original


def test_load_rejects_an_unknown_suffix(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="json"):
        Outer.load(path)


def test_toml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('name = "x"\nkount = 3\n')
    with pytest.raises(ValidationError):
        Outer.from_toml(path)


def test_toml_nests_submodels_as_tables(tmp_path):
    text = Outer(inner=Inner(gain=9.0)).to_toml()
    assert "[inner]" in text, "nested models must serialise as TOML tables"


def test_toml_omits_untouched_none_default(tmp_path):
    # A field whose default is None and was never set shouldn't appear in the
    # dumped text at all -- Pydantic supplies the (None) default back on load.
    class WithOptional(ConfigModel):
        value: str | None = None

    original = WithOptional()
    text = original.to_toml()
    assert "value" not in text
    path = tmp_path / "optional.toml"
    original.to_toml(path)
    reloaded = WithOptional.from_toml(path)
    assert reloaded == original
    assert reloaded.value is None


def test_toml_roundtrips_explicit_none_over_a_non_none_default(tmp_path):
    # TOML can't write `null`, so a naive dump would either crash on a None
    # value or (if it just skips None keys) silently resurrect the field's
    # non-None default on load. Neither is acceptable: a caller who
    # deliberately set a field to None must get None back, not the default.
    class WithFallback(ConfigModel):
        value: str | None = "fallback"

    original = WithFallback(value=None)
    path = tmp_path / "explicit_none.toml"
    original.to_toml(path)
    reloaded = WithFallback.from_toml(path)
    assert reloaded == original
    assert reloaded.value is None
