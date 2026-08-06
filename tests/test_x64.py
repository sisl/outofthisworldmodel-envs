import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp

import owm_envs
import owm_envs.envs.common  # noqa: F401 -- the import itself flips the flag

SRC_ROOT = Path(owm_envs.__file__).parent


def test_x64_enabled():
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1, jnp.float64).dtype == jnp.float64


def test_core_import_alone_flips_the_flag():
    # The two jax chokepoints carry the flip independently: a module that
    # reaches jax through core without touching envs.common must still get
    # x64, so check core in a subprocess that never imports envs.common.
    result = subprocess.run(
        [sys.executable, "-c", "import owm_envs.core, jax; print(jax.config.jax_enable_x64)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "True", result.stderr


def test_f32_stays_f32_when_pinned():
    a = jnp.zeros(3, jnp.float32)
    assert (a + 1.0).dtype == jnp.float32  # weak python scalar
    assert jnp.cos(a).dtype == jnp.float32


def test_package_import_still_leaves_jax_unloaded():
    # Mirrors tests/drivers/test_vector_env_driver.py::test_module_does_not
    # _import_jax: the package root must stay importable without jax, which
    # is why the flip lives in the core / envs.common chokepoints instead.
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import owm_envs; print('jax' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr


def _candidate_modules() -> list[str]:
    """Every module in the package whose source mentions importing jax,
    discovered from the tree rather than hard-coded so a newly added one is
    covered without anyone remembering to list it here."""
    modules = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name == "_jax_config.py" or "import jax" not in path.read_text():
            continue
        parts = path.relative_to(SRC_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(("owm_envs",) + parts))
    return modules


def _probe(module: str) -> tuple[str, str]:
    """Import `module` alone in a fresh interpreter and report whether jax
    ended up loaded and, if so, whether x64 was on."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module}\n"
            "loaded = 'jax' in sys.modules\n"
            "import jax\n"
            "print('x64' if loaded and jax.config.jax_enable_x64 else "
            "('no-jax' if not loaded else 'f32'))",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # An optional extra (the render stack) is not installed here.
        return "unimportable", result.stderr
    return result.stdout.strip(), result.stderr


def test_every_jax_module_gets_x64_when_imported_alone():
    # The chokepoints only work because every jax-consuming module reaches
    # one of them before it traces anything. That holds today by inspection,
    # which is not a guarantee: a new module importing jax without ever
    # touching core or envs.common would trace in f32 while the rest of the
    # package runs in f64, and the mismatch would surface as a baffling
    # dtype error far from its cause. Import each one alone, in its own
    # interpreter, and demand the flag be set.
    modules = _candidate_modules()
    assert modules, "sanity check: no jax-consuming modules were discovered"

    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(_probe, modules))

    # "no-jax" is a legitimate verdict: the driver seam and the lazy env
    # registry mention jax only in prose or inside deferred imports, and
    # have nothing to configure.
    offenders = [
        f"{module}: {verdict}\n{err}"
        for module, (verdict, err) in zip(modules, verdicts)
        if verdict == "f32"
    ]
    assert not offenders, "modules tracing without x64:\n" + "\n".join(offenders)

    checked = sum(1 for verdict, _ in verdicts if verdict == "x64")
    assert checked >= 10, f"expected the sweep to really exercise the package, got {checked}"
