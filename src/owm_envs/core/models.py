"""YAML/TOML-round-trippable configuration base.

Configs are Pydantic models so a specific experiment's settings can be both a
committed, versioned input (configs/*.yaml) and an as-run artifact written next
to the dataset it produced. Round-tripping and validation both matter: a
hand-edited YAML with a typo'd key or a wrong-length tuple must fail loudly at
load rather than silently fall back to a default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Self

import tomli_w
import yaml
from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Frozen Pydantic model with YAML and TOML load/dump."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise to YAML text. Writes to `path` as well when given."""
        text = yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        payload: Any = yaml.safe_load(p.read_text()) or {}
        return cls.model_validate(payload)

    def to_toml(self, path: str | Path | None = None) -> str:
        """Serialise to TOML text. Writes to `path` as well when given.

        TOML has no `null`, so a field currently holding `None` can't be
        written directly. Fields still at their (None) default are simply
        omitted -- Pydantic supplies the same None back on load. But a field
        whose default is *not* None and was explicitly set to None must not
        be omitted the same way: that would silently resurrect the non-None
        default on load. Those explicit Nones are instead recorded by dotted
        field path under `__explicit_nulls__` and re-applied in `from_toml`.
        """
        payload = self.model_dump(mode="json", exclude_none=True)
        explicit_nulls = _explicit_null_paths(self)
        if explicit_nulls:
            payload["__explicit_nulls__"] = explicit_nulls
        text = tomli_w.dumps(payload)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_toml(cls, path: str | Path) -> Self:
        """Load and validate from a TOML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
        for dotted_path in raw.pop("__explicit_nulls__", []):
            _set_dotted(raw, dotted_path, None)
        return cls.model_validate(raw)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load from `path`, dispatching on its suffix."""
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".toml":
            return cls.from_toml(p)
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(p)
        raise ValueError(f"unsupported config file suffix: {suffix!r}")


def _explicit_null_paths(model: BaseModel, prefix: str = "") -> list[str]:
    """Dotted paths of fields holding None that were explicitly set, not
    merely defaulted -- these must survive a TOML round-trip as None even
    when their field default is something else."""
    paths: list[str] = []
    for name, value in model:
        dotted_path = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            paths.extend(_explicit_null_paths(value, prefix=f"{dotted_path}."))
        elif value is None and name in model.model_fields_set:
            paths.append(dotted_path)
    return paths


def _set_dotted(d: dict, dotted_path: str, value: Any) -> None:
    """Set `value` at `dotted_path` in `d`, creating intermediate tables
    along the way (needed when the whole intermediate table was itself
    omitted, e.g. an explicitly-None submodel)."""
    *parents, leaf = dotted_path.split(".")
    cur = d
    for key in parents:
        cur = cur.setdefault(key, {})
    cur[leaf] = value
