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
    import numpy as np

    from ..render.inputs import RenderInputs
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
    # Whether the video path can render this env at all -- which since the
    # RenderInputs seam means exactly that it has a `make_render_adapter`, the
    # thing that reads its rows in its own element order. Deliberately given
    # no default: an env that gets this wrong renders silently wrong video
    # rather than failing, so registering one has to be the moment somebody
    # answers the question. `tests/envs/test_registry.py` holds the two fields
    # in step, because the CLI's `--render` guard reads this one while the
    # render workers reach for the other.
    renderable: bool
    # Factory for this env's observation function: given a config, returns a
    # state -> observation callable deciding what a recorded observation
    # CONTAINS. A factory rather than a plain callable because the choice is a
    # config field (iss-numerical's `observation.mode`), unlike `view` above,
    # which is one fixed derivation per env.
    #
    # None means the identity -- the env records its state as its observation,
    # which is what iss and iss-hcw do and the only behaviour that existed
    # before this field. It is orthogonal to `view`: `view` is what the task
    # layer (policies, reward, events, the goal-error block) reads out of the
    # STATE and is unaffected by what gets recorded.
    make_observe: Callable[[BaseTaskConfig], Callable[[jnp.ndarray], jnp.ndarray]] | None = None
    # How this env's equations of motion are described in a published
    # dataset's card, as the object of "under ...". Every env in the suite
    # flies the same task against the same station, so this phrase and the
    # state layout are the whole of what a card has to say differently about
    # one env versus another.
    card_summary: str = "rigid-body free-flyer dynamics"
    # Factory for this env's render adapter: given a config, returns a
    # (state, action) -> RenderInputs callable that poses a frame from the
    # env's own state layout. Module-level by construction (never a lambda
    # or closure) -- render worker processes rebuild the adapter from
    # (env_name, cfg) via ENV_REGISTRY and have to pickle it across the
    # process boundary to do it. None for an env whose adapter has not landed,
    # which is precisely an env whose `renderable` is False.
    make_render_adapter: (
        Callable[[BaseTaskConfig], Callable[[np.ndarray, np.ndarray | None], RenderInputs]] | None
    ) = None


def _build_env_registry() -> dict[str, EnvSpec]:
    from .common.layout import ISS_LAYOUT
    from .iss.config import ISSConfig
    from .iss.dynamics import ISSDynamics
    from .iss.render_adapter import make_render_adapter as iss_make_render_adapter
    from .iss.vector_env import ISSVectorEnv
    from .iss_hcw.config import HCW_LAYOUT, HCWConfig
    from .iss_hcw.dynamics import HCWDynamics
    from .iss_hcw.render_adapter import make_render_adapter as hcw_make_render_adapter
    from .iss_hcw.vector_env import HCWVectorEnv

    return {
        "iss": EnvSpec(
            name="iss",
            gym_id="ISS-Docking-v0",
            config_cls=ISSConfig,
            layout=ISS_LAYOUT,
            make_dynamics=ISSDynamics,
            make_vector_env=lambda num_envs, cfg: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
            view=ISS_LAYOUT.slice_view,
            renderable=True,
            make_render_adapter=iss_make_render_adapter,
        ),
        "iss-hcw": EnvSpec(
            name="iss-hcw",
            gym_id="ISS-HCW-Docking-v0",
            config_cls=HCWConfig,
            layout=HCW_LAYOUT,
            make_dynamics=HCWDynamics,
            make_vector_env=lambda num_envs, cfg: HCWVectorEnv(num_envs=num_envs, cfg=cfg),
            view=HCW_LAYOUT.slice_view,
            renderable=True,
            # No internal comma: the card reads "under {card_summary} and
            # against the station's collision hull", which a comma clause
            # turns into a garden path.
            card_summary=(
                "Clohessy-Wiltshire relative dynamics about a Keplerian chief "
                "with gravity-gradient attitude torque"
            ),
            make_render_adapter=hcw_make_render_adapter,
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

register(
    id="ISS-HCW-Docking-v0",
    entry_point="owm_envs.envs.iss_hcw.env:HCWEnv",
)
