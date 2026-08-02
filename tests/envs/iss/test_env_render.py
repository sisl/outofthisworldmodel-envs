import numpy as np
import pytest

from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.env import ISSEnv


def test_render_modes_are_declared():
    assert "rgb_array" in ISSEnv.metadata["render_modes"]


def test_render_without_a_render_mode_returns_none():
    env = ISSEnv()
    env.reset(seed=0)
    assert env.render() is None


def test_render_before_reset_raises():
    env = ISSEnv(render_mode="rgb_array")
    with pytest.raises(RuntimeError, match="reset"):
        env.render()


def test_unknown_render_mode_raises():
    with pytest.raises(ValueError, match="render_mode"):
        ISSEnv(render_mode="ascii")


class _FakeRenderer:
    """Stands in for `ISSRenderer`, recording the `t_offset_s` it is called with."""

    def __init__(self):
        self.t_offset_s_calls: list[float] = []

    def render(self, state, action=None, view="DRAGON_ISO", t_offset_s=0.0):
        self.t_offset_s_calls.append(t_offset_s)
        return np.zeros((4, 4, 3), dtype=np.uint8)


def test_render_passes_the_elapsed_epoch_offset_to_the_renderer():
    from owm_envs.envs.iss.orbit import OrbitConfig

    cfg = ISSConfig(orbit=OrbitConfig(enabled=True))
    env = ISSEnv(cfg, render_mode="rgb_array")
    env.reset(seed=0, options={"epoch_offset_s": 7.0})
    fake = _FakeRenderer()
    env._renderer = fake  # bypass _make_renderer(); no pygfx needed

    for _ in range(2):
        env.step(np.zeros(6, dtype=np.float32))
    env.render()

    assert fake.t_offset_s_calls[-1] == pytest.approx(7.0 + 2 * cfg.dt)


class TestWithRenderExtra:
    """Only runs when the optional render extra is installed."""

    @pytest.fixture(autouse=True)
    def _require_render(self):
        pytest.importorskip("pygfx", reason="rendering is an optional extra")
        pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

    def test_render_returns_an_rgb_frame(self):
        env = ISSEnv(ISSConfig(), render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        env.close()

    def test_frame_changes_as_the_episode_advances(self):
        env = ISSEnv(ISSConfig(), render_mode="rgb_array")
        env.reset(seed=0)
        first = env.render()
        for _ in range(20):
            env.step(np.full(6, 5000.0, dtype=np.float32))
        assert not np.array_equal(first, env.render())
        env.close()

    def test_check_env_still_passes_with_rendering_enabled(self):
        from gymnasium.utils.env_checker import check_env

        check_env(ISSEnv(render_mode="rgb_array"))
