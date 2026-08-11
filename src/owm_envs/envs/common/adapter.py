"""Gymnasium adapter pieces that are the same for every env in the suite.

An env's observation space depends on the state it carries, so each package
builds its own. Its action space and its render rate do not: both are read off
`BaseTaskConfig` fields every env has, so they live here and every adapter --
single and vector -- imports the one copy.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .config import BaseTaskConfig


def render_fps(cfg: BaseTaskConfig) -> int:
    """Playback rate for one frame per simulation step, as a positive integer.

    That rate is 1/dt, but Gymnasium's render_fps has to be a usable frame
    rate: consumers divide by it or hand it to a video encoder. Any dt of 2 s
    or more rounds to zero -- exactly 2.0 included, since Python rounds a tie
    to even -- so the result is floored at 1.
    """
    return max(1, round(1.0 / cfg.dt))


def action_space(cfg: BaseTaskConfig) -> spaces.Box:
    """The 6D `[force (3), torque (3)]` box at this config's actuator limits."""
    high = np.array(
        [cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
        dtype=np.float32,
    )
    return spaces.Box(low=-high, high=high, dtype=np.float32)
