"""Environment registration for owm_envs.

ENV_REGISTRY is the package's own name -> EnvSpec map, consumed by the CLI
and the rollout drivers; the gym registration below serves external
gym.make() callers. Both are populated here so adding an environment is one
entry in one file.

Building an EnvSpec touches every env's config/dynamics/vector-env classes,
which import jax -- so ENV_REGISTRY is assembled lazily on first access
(module __getattr__, PEP 562) rather than at import time. That keeps
`import owm_envs` (which imports this package) as cheap as the plain
gym.make() registration below, matching
`tests/drivers/test_vector_env_driver.py::test_module_does_not_import_jax`,
which requires importing an owm_envs submodule not to pull in jax as a side
effect of the package's own __init__ chain.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from gymnasium.envs.registration import register

if TYPE_CHECKING:
    import jax.numpy as jnp

    from .common.config import BaseTaskConfig
    from .common.layout import StateLayout


@dataclass(frozen=True)
class EnvSpec:
    name: str
    gym_id: str
    config_cls: type[BaseTaskConfig]
    layout: StateLayout
    make_dynamics: Callable[[BaseTaskConfig], Any]
    make_vector_env: Callable[[int, BaseTaskConfig], Any]
    view: Callable[[jnp.ndarray], jnp.ndarray]


def _build_env_registry() -> dict[str, EnvSpec]:
    from .common.layout import ISS_LAYOUT
    from .iss.config import ISSConfig
    from .iss.dynamics import ISSDynamics
    from .iss.vector_env import ISSVectorEnv

    return {
        "iss": EnvSpec(
            name="iss",
            gym_id="ISS-Docking-v0",
            config_cls=ISSConfig,
            layout=ISS_LAYOUT,
            make_dynamics=ISSDynamics,
            make_vector_env=lambda num_envs, cfg: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
            view=ISS_LAYOUT.slice_view,
        ),
    }


_env_registry_cache: dict[str, EnvSpec] | None = None
_REGISTRY_LOCK = threading.Lock()


def __getattr__(name: str) -> Any:
    global _env_registry_cache
    if name == "ENV_REGISTRY":
        if _env_registry_cache is None:
            with _REGISTRY_LOCK:
                if _env_registry_cache is None:
                    _env_registry_cache = _build_env_registry()
        return _env_registry_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


register(
    id="ISS-Docking-v0",
    entry_point="owm_envs.envs.iss.env:ISSEnv",
)
