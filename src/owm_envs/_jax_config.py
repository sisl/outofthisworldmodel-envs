"""Enables 64-bit JAX for every jax-consuming path in this package.

float64 must be real: the [jd, seconds-of-day] epoch prefix is carried in
f64 because an f32 seconds-of-day biases the advance by hundreds of
seconds per orbit at small dt (see `envs/common/epoch_state.py`),
`envs/common/zonal_gravity.py` evaluates the zonal harmonics in f64, and
the iss-numerical orbit propagator integrates in f64. All f32 paths pin
their dtypes explicitly, so flipping this changes nothing for them --
enforced by the golden-rollout tests.

Imported by the two jax chokepoints (`core`, `envs.common`) rather than
the package root, which must stay importable without jax so the driver
seam can host non-JAX backends.
"""

import jax

jax.config.update("jax_enable_x64", True)
