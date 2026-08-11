"""ISS environment configuration: today's frozen-kinematics env.

All sections live in envs.common.config; this env adds nothing beyond them,
so the class is a bare subclass -- which keeps field order, and therefore
every serialized as-run config, byte-identical to before the split.
"""

from __future__ import annotations

from ..common.config import BaseTaskConfig


class ISSConfig(BaseTaskConfig):
    pass
