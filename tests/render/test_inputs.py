import subprocess
import sys

import numpy as np

from owm_envs.render.inputs import Lighting, RenderInputs


def test_from_view_slices_position_and_quaternion():
    view = np.arange(13.0)
    inputs = RenderInputs.from_view(view)
    np.testing.assert_array_equal(inputs.position_world, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(inputs.quaternion_bw, [6.0, 7.0, 8.0, 9.0])
    assert inputs.action is None
    assert inputs.lighting is None


def test_from_view_carries_action_and_lighting_through():
    view = np.arange(13.0)
    action = np.zeros(6)
    lighting = Lighting(
        sun_direction_world=np.array([1.0, 0.0, 0.0]),
        illumination=1.0,
        chief_distance_m=6.8e6,
        moon_vector_world=np.zeros(3),
    )
    inputs = RenderInputs.from_view(view, action=action, lighting=lighting)
    assert inputs.action is action
    assert inputs.lighting is lighting


def test_module_imports_without_pygfx():
    # render/inputs.py sits on the numpy boundary: worker processes that only
    # need to pose a frame must not have to bring pygfx along to do it. Check
    # in a subprocess so this is a real assertion about what importing the
    # module does, not just about what happens to already be loaded here.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import owm_envs.render.inputs; print('pygfx' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr


def test_module_imports_without_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import owm_envs.render.inputs; print('jax' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr
