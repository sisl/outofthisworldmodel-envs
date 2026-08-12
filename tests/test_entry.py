"""The console script's one job before the CLI: settle which GPU JAX gets.

The settings it applies are read once, when JAX initialises its backend, and
that happens during import -- astrojax reaches a backend as `core.quaternion`
imports it. So the only place they can still land is ahead of that import,
which is what the entry point exists for.
"""

import os
import subprocess
import sys

import pytest

import owm_envs._entry as entry
from owm_envs._entry import requested_index


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["generate", "--gpu-index", "1"], 1),
        (["generate", "--gpu-index=1"], 1),
        (["generate", "--out", "run"], None),
        # Malformed is Typer's to report, so it is not half-handled here.
        (["generate", "--gpu-index", "nope"], None),
        # Typer takes the last of a repeated flag, so the pin has to as well,
        # or the rollout would run on one card while the renderer used another.
        (["generate", "--gpu-index", "0", "--gpu-index", "1"], 1),
        # A repeat Typer will reject outright pins nothing, rather than
        # leaving the earlier value applied to a run that never starts.
        (["generate", "--gpu-index", "0", "--gpu-index", "nope"], None),
        # A trailing flag with no value is a usage error, not an index.
        (["generate", "--gpu-index"], None),
        # Past `--` everything is positional to Typer, however it is spelled.
        (["generate", "--", "--gpu-index", "1"], None),
        (["generate", "--gpu-index", "1", "--", "--gpu-index", "2"], 1),
    ],
)
def test_requested_index_reads_the_flag(argv, expected):
    assert requested_index(argv) == expected


def test_the_environment_names_the_gpu_when_the_flag_does_not(monkeypatch):
    """Same rule the renderer resolves by, so a run that names its card through
    the environment pins both halves of itself rather than one."""
    seen = {}

    def fake_pin(index):
        seen["index"] = index
        raise SystemExit(0)

    monkeypatch.setattr(entry, "pin_gpu", fake_pin)
    monkeypatch.setenv("OWM_ENVS_GPU_INDEX", "1")
    monkeypatch.setattr(sys, "argv", ["owm-envs", "generate"])
    with pytest.raises(SystemExit):
        entry.main()
    assert seen["index"] == 1


def test_a_malformed_environment_index_is_a_usage_error_not_a_traceback(monkeypatch):
    """Typer is not up yet to report it, and every command pays this cost --
    `list` and `push` take no GPU flag but still cross this path."""
    monkeypatch.setenv("OWM_ENVS_GPU_INDEX", "left-hand-one")
    monkeypatch.setattr(sys, "argv", ["owm-envs", "list"])
    with pytest.raises(SystemExit) as raised:
        entry.main()
    assert "OWM_ENVS_GPU_INDEX" in str(raised.value)


def test_main_pins_before_anything_reaches_a_jax_backend():
    """The ordering invariant, enforced end to end in a process of its own.

    `pin_gpu` refuses once a backend exists, so a `main()` that gets as far as
    Typer at all is itself proof the settings landed first. Import state is
    per-process and every other test here has already reached astrojax, which
    is what the subprocess is for: a reordering inside `main`, or a top-level
    import of `cli`, fails here and nowhere else. It is the regression this
    whole entry point exists to prevent, and it is invisible in-process.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, os\n"
            "sys.argv = ['owm-envs', 'generate', '--gpu-index', '1', '--help']\n"
            "from owm_envs._entry import main\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "print('VISIBLE=' + os.environ.get('CUDA_VISIBLE_DEVICES', ''))\n"
            "print('PREALLOC=' + os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', ''))\n",
        ],
        capture_output=True,
        text=True,
        # On the CPU backend, so the check costs no GPU and cannot be
        # perturbed by whatever else is using the cards.
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr
    assert "VISIBLE=1" in result.stdout, result.stdout
    assert "PREALLOC=false" in result.stdout, result.stdout
