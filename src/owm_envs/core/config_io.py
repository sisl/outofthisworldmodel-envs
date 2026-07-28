"""YAML-round-trippable configuration base.

Configs are Pydantic models so a specific experiment's settings can be both a
committed, versioned input (configs/*.yaml) and an as-run artifact written next
to the dataset it produced. Round-tripping and validation both matter: a
hand-edited YAML with a typo'd key or a wrong-length tuple must fail loudly at
load rather than silently fall back to a default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict


class YamlModel(BaseModel):
    """Frozen Pydantic model with YAML load/dump."""

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
