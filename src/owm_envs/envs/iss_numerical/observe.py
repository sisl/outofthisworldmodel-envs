"""What a recorded observation of this env contains.

The state is always the same 21D one (`NUM_LAYOUT`) and the task layer always
reads the same canonical view of it (`relative_view`); `observation.mode`
decides only which frame that state is REPORTED in, which is a question about
the dataset a run produces rather than about the simulation. The four modes
exist because this env is the first in the suite whose state is absolute, so
it is the first that can offer the choice at all -- a model trained on
`relative` sees what `iss_hcw` shows it and nothing about where in the orbit
the pair is, while the absolute modes hand it an ECI position and let it learn
the orbital motion too.

  absolute (21)         the state itself: [epoch | chief r,v | chaser r,v |
                        q_bi | omega_b].
  chaser_absolute (21)  the chaser first and the chief measured FROM it:
                        [epoch | chaser r,v | (chief - chaser) r,v | q_bi |
                        omega_b]. Still ECI axes -- the difference is which
                        vehicle the origin sits on.
  chief_absolute (21)   the chief in ECI with the chaser as the world-frame
                        offset the task is actually flown in: [epoch |
                        chief r,v | rel_pos | rel_vel | q_bi | omega_b].
  relative (15)         [epoch | relative_view], q_bw and world-relative rates
                        included -- element for element the layout `iss_hcw`
                        carries in-state (`HCW_LAYOUT`), so the same consumer
                        reads either env.

The arithmetic here runs at the state's own f64 width and narrows to f32
once, at the end. The order is what matters for the modes that difference two
positions: an ECI radius is ~6.8e6 m, where f32's grain is ~0.5 m, so
narrowing first would leave a third of a metre of quantization on a relative
position the dock gate scores at 0.1 m. The f32 result is what the datasets
store, and at a ~100 m standoff its grain is ~1e-5 m.

That is a claim about this module, not a claim that every channel is f64 to
the last bit: `relative_view` reaches the world<->ECI rotation and the
quaternion conversions through astrojax, which are f32-pinned regardless of
this package's x64 flag. Its own docstring quantifies what that costs
(~1e-5 m at a 100 m standoff, 3.3e-10 rad/s on the rates) -- the same floor
the task layer already reads the view through, and the same order as f32's
grain at the boundary here.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .config import NumericalConfig, ObservationMode
from .dynamics import relative_view


def _absolute(state: jnp.ndarray) -> jnp.ndarray:
    return state


def _chaser_absolute(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([state[0:2], state[8:14], state[2:8] - state[8:14], state[14:21]])


def _chief_absolute(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([state[0:8], relative_view(state)[0:6], state[14:21]])


def _relative(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([state[0:2], relative_view(state)])


_MODES: dict[ObservationMode, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "absolute": _absolute,
    "chaser_absolute": _chaser_absolute,
    "chief_absolute": _chief_absolute,
    "relative": _relative,
}


def make_observe(cfg: NumericalConfig) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """(21,) state -> the observation `cfg.observation.mode` asks for.

    The result is `OBS_MODE_DIM[mode]` wide and f32; see the module docstring
    on why the narrowing happens here rather than at the input.
    """
    mode_fn = _MODES[cfg.observation.mode]

    def observe(state: jnp.ndarray) -> jnp.ndarray:
        return mode_fn(jnp.asarray(state, jnp.float64)).astype(jnp.float32)

    return observe
