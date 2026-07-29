import subprocess
import sys


def test_import_owm_envs_alone_registers_the_env():
    # Runs in a fresh interpreter so no other test's `import owm_envs.envs`
    # can register the environment as a side effect and mask a regression
    # here -- this must pass with `import owm_envs` as the only owm_envs
    # import.
    code = (
        "import owm_envs\n"
        "import gymnasium as gym\n"
        "env = gym.make('ISS-Docking-v0')\n"
        "env.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
