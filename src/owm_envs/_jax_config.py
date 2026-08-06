"""Enables 64-bit JAX for every jax-consuming path in this package.

float64 must be real: the [jd, seconds-of-day] epoch advance silently
loses its low-order term in f32, and the numerical orbit propagator
integrates in f64. All f32 paths pin their dtypes explicitly, so flipping
this changes nothing for them -- enforced by the golden-rollout tests.

Imported by the two jax chokepoints (`core`, `envs.common`) rather than
the package root, which must stay importable without jax so the driver
seam can host non-JAX backends.
"""

import jax

jax.config.update("jax_enable_x64", True)
