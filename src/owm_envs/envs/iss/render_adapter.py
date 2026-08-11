"""iss's render adapter: a view row already is a posable frame.

`make_render_adapter` and the function it returns are both module-level (not
a lambda or a closure) because render worker processes rebuild adapters from
(env_name, cfg) via ENV_REGISTRY and have to pickle them across the process
boundary to do it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...render.inputs import RenderInputs

if TYPE_CHECKING:
    import numpy as np

    from .config import ISSConfig


def make_render_adapter(cfg: ISSConfig):
    del cfg  # iss needs no per-config state: the view row already is the pose.
    return _adapt


def _adapt(state: np.ndarray, action: np.ndarray | None = None) -> RenderInputs:
    return RenderInputs.from_view(state, action)
