"""True-state recording through pack_episodes and TrajectoryBatch."""
import numpy as np
import pytest

from owm_envs.drivers.types import TrajectoryBatch, pack_episodes


def _episode(length: int, obs_dim: int = 13, act_dim: int = 6) -> dict:
    rng = np.random.default_rng(length)
    return {
        "obs": rng.normal(size=(length, obs_dim)).astype(np.float32),
        "true_state": rng.normal(size=(length, 13)).astype(np.float32),
        "act": rng.normal(size=(length, act_dim)).astype(np.float32),
        "rew": rng.normal(size=(length,)).astype(np.float32),
        "terminated": True,
        "truncated": False,
        "policy_id": 0,
        "dock_target": np.zeros(7, dtype=np.float32),
    }


def test_pack_records_true_state():
    batch = pack_episodes(
        [_episode(4), _episode(2)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_state=True,
    )
    assert batch.true_state is not None
    assert batch.true_state.shape == (2, 4, 13)
    assert np.all(batch.true_state[1, 2:] == 0)  # zero padding


def test_pack_without_flag_leaves_none():
    batch = pack_episodes(
        [_episode(3)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
    )
    assert batch.true_state is None


def test_pack_ignores_policy_id_when_not_recording():
    """Both drivers leave `policy_id` None when they are not recording one, so
    packing must never read it under `records_policy_ids=False`."""
    episode = _episode(3)
    episode["policy_id"] = None
    batch = pack_episodes(
        [episode], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
    )
    assert batch.policy_ids is None


def test_pack_preserves_true_state_values():
    episodes = [_episode(4), _episode(2)]
    batch = pack_episodes(
        episodes, obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_state=True,
    )
    for i, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        np.testing.assert_array_equal(
            batch.true_state[i, :length], episode["true_state"]
        )


@pytest.mark.parametrize("bad_shape", [(13,), (1, 13)])
def test_pack_rejects_broadcastable_true_state(bad_shape):
    """A (13,) or (1, 13) true_state would broadcast over every timestep and
    produce a batch that validates but holds duplicated state."""
    episode = _episode(4)
    episode["true_state"] = np.zeros(bad_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="true_state"):
        pack_episodes(
            [episode], obs_dim=13, act_dim=6,
            records_policy_ids=False, records_dock_targets=True,
            records_true_state=True,
        )


def test_validate_rejects_wrong_episode_count():
    batch = pack_episodes(
        [_episode(3)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_state=True,
    )
    bad = TrajectoryBatch(
        observations=batch.observations, actions=batch.actions,
        rewards=batch.rewards, lengths=batch.lengths,
        terminated=batch.terminated, truncated=batch.truncated,
        policy_ids=None, dock_targets=batch.dock_targets,
        true_state=np.zeros((2, 3, 13), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        bad.validate()


def _rebuild(batch: TrajectoryBatch, true_state: np.ndarray) -> TrajectoryBatch:
    return TrajectoryBatch(
        observations=batch.observations, actions=batch.actions,
        rewards=batch.rewards, lengths=batch.lengths,
        terminated=batch.terminated, truncated=batch.truncated,
        policy_ids=None, dock_targets=batch.dock_targets,
        true_state=true_state,
    )


@pytest.mark.parametrize("shape", [(2, 5, 13), (2, 4, 7)])
def test_validate_rejects_wrong_true_state_shape(shape):
    batch = pack_episodes(
        [_episode(4), _episode(2)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_state=True,
    )
    bad = _rebuild(batch, np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="true_state"):
        bad.validate()


def test_validate_rejects_nonzero_true_state_padding():
    batch = pack_episodes(
        [_episode(4), _episode(2)], obs_dim=13, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_state=True,
    )
    dirty = batch.true_state.copy()
    dirty[1, 3] = 1.0
    with pytest.raises(ValueError, match="padding"):
        _rebuild(batch, dirty).validate()


def _scan_batch(noise: str, goal_error: bool):
    from owm_envs.drivers.scan_driver import ScanDriver
    from owm_envs.drivers.types import RolloutSpec
    from owm_envs.envs.iss.config import ISSConfig, ObservationConfig
    from owm_envs.envs.iss.policies import PolicyConfig
    from owm_envs.envs.iss.sensing import PRESETS

    cfg = ISSConfig(
        sensor_noise=PRESETS[noise],
        observation=ObservationConfig(goal_error=goal_error),
    )
    driver = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="random"), num_envs=2)
    return driver.generate(RolloutSpec(num_episodes=2, max_steps=20, seed=0))


def test_scan_truth_equals_obs_without_noise():
    batch = _scan_batch("off", goal_error=False)
    assert batch.true_state is not None
    np.testing.assert_array_equal(batch.true_state, batch.observations)


def test_scan_truth_differs_under_noise():
    batch = _scan_batch("noncooperative", goal_error=False)
    real = batch.lengths[0]
    measured = batch.observations[0, :real, :13]
    true = batch.true_state[0, :real]
    assert not np.allclose(measured, true)
    # Quaternions stay unit under both channels.
    np.testing.assert_allclose(np.linalg.norm(true[:, 6:10], axis=1), 1.0, atol=1e-5)


def test_scan_truth_is_13_dim_with_goal_error():
    batch = _scan_batch("off", goal_error=True)
    assert batch.observations.shape[-1] == 25
    assert batch.true_state.shape[-1] == 13
    np.testing.assert_array_equal(batch.true_state, batch.observations[:, :, :13])


def test_scan_transitions_mode_records_truth():
    """min_transitions is a separate packing path from episodes mode (it
    accumulates across chunks), so it needs its own proof that truth survives."""
    from owm_envs.drivers.scan_driver import ScanDriver
    from owm_envs.drivers.types import RolloutSpec
    from owm_envs.envs.iss.config import ISSConfig
    from owm_envs.envs.iss.policies import PolicyConfig
    from owm_envs.envs.iss.sensing import PRESETS

    cfg = ISSConfig(max_steps=10, sensor_noise=PRESETS["noncooperative"])
    driver = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="dock"), num_envs=2)
    batch = driver.generate(RolloutSpec(max_steps=10, seed=0, min_transitions=60))

    assert batch.true_state is not None
    assert batch.true_state.shape == batch.observations.shape[:2] + (13,)
    for i in range(batch.num_episodes):
        length = int(batch.lengths[i])
        true = batch.true_state[i, :length]
        assert not np.allclose(true, 0.0)
        assert not np.allclose(true, batch.observations[i, :length, :13])


def test_writer_emits_state_vector(tmp_path):
    pytest.importorskip("lerobot", reason="lerobot is an optional extra")
    import pandas as pd

    from owm_envs.datasets.lerobot_writer import write_lerobot_split

    batch = _scan_batch("cooperative", goal_error=False)
    root = write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=20)
    files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])

    assert "state_vector" in df.columns
    sv = np.stack(df["state_vector"].to_numpy())
    ov = np.stack(df["observation_vector"].to_numpy())
    assert sv.shape[-1] == 13
    assert not np.allclose(sv, ov[:, :13])


def test_writer_omits_state_vector_for_a_legacy_batch(tmp_path):
    """A batch predating the truth channel must still write, with no column
    of NaN stand-ins pretending to be a state."""
    pytest.importorskip("lerobot", reason="lerobot is an optional extra")
    import pandas as pd

    from owm_envs.datasets.lerobot_writer import write_lerobot_split

    batch = _rebuild(_scan_batch("off", goal_error=False), None)
    root = write_lerobot_split(tmp_path / "train", "iss/train", batch, fps=20)
    files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])

    assert "state_vector" not in df.columns


class _StateCapturingRenderer:
    def __init__(self):
        self.states = []

    def render_views(self, state, action=None, views=("DRAGON_FPV",)):
        self.states.append(np.asarray(state))
        return {view: np.zeros((4, 4, 3), dtype=np.uint8) for view in views}


class _TinyRenderConfig:
    image_width = 4
    image_height = 4


def test_video_poses_truth_not_the_noisy_observation():
    from owm_envs.datasets.video import render_episode_frames

    batch = _scan_batch("noncooperative", goal_error=False)
    renderer = _StateCapturingRenderer()
    render_episode_frames(batch, 0, _TinyRenderConfig(), renderer=renderer)

    length = int(batch.lengths[0])
    np.testing.assert_array_equal(
        np.stack(renderer.states), batch.true_state[0, :length]
    )


def test_video_renders_truth_for_goal_error_batch():
    pytest.importorskip("pygfx", reason="rendering is an optional extra")
    pytest.importorskip("trimesh", reason="GLB loading needs trimesh")

    from owm_envs.datasets.video import render_episode_frames
    from owm_envs.render.iss_scene import RenderConfig

    batch = _scan_batch("noncooperative", goal_error=True)  # 25-dim obs
    clips = render_episode_frames(batch, 0, RenderConfig(image_width=64, image_height=64))
    assert [clip.shape[1:] for clip in clips.values()] == [(64, 64, 3)]


def test_scan_truth_stays_aligned_across_autoresets():
    """Truth must be dynamics-consistent for EVERY episode, including the
    second and third a lane produces after an in-scan autoreset.

    One lane, three episodes, so lane 0 crosses two autoreset boundaries.
    Replaying the recorded actions from each recorded true state has to
    reproduce the next one: a truth channel shifted by a timestep, or one
    that leaked the post-reset state across an episode boundary, would still
    differ from the noisy observations and so survive the noise test above.
    """
    import jax.numpy as jnp

    from owm_envs.drivers.scan_driver import ScanDriver
    from owm_envs.drivers.types import RolloutSpec
    from owm_envs.envs.iss.config import ISSConfig
    from owm_envs.envs.iss.dynamics import ISSDynamics
    from owm_envs.envs.iss.policies import PolicyConfig
    from owm_envs.envs.iss.sensing import PRESETS

    cfg = ISSConfig(sensor_noise=PRESETS["noncooperative"])
    driver = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="random"), num_envs=1)
    batch = driver.generate(RolloutSpec(num_episodes=3, max_steps=10, seed=0))
    dynamics = ISSDynamics(cfg)

    assert batch.num_episodes == 3
    for i in range(batch.num_episodes):
        length = int(batch.lengths[i])
        assert length > 1
        true = batch.true_state[i, :length]
        for t in range(length - 1):
            stepped, _ = dynamics.step(
                jnp.asarray(true[t]),
                jnp.asarray(batch.actions[i, t]),
                jnp.asarray(batch.dock_targets[i]),
            )
            np.testing.assert_allclose(np.asarray(stepped), true[t + 1], rtol=1e-4, atol=1e-4)


def _vector_driver(cfg, policy_type="random", num_envs=2, goal_error=False):
    from owm_envs.drivers.vector_env_driver import VectorEnvDriver
    from owm_envs.envs.iss.policies import PolicyConfig
    from owm_envs.envs.iss.policy_source import ISSPolicySource
    from owm_envs.envs.iss.vector_env import ISSVectorEnv

    # The goal-error block is appended by the policy source on this path, so
    # the env itself is always built without it (see the driver-equivalence
    # tests) -- otherwise the block would be added twice.
    source_cfg = cfg.model_copy(
        update={"observation": cfg.observation.model_copy(update={"goal_error": goal_error})}
    )
    return VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
        policy_source=ISSPolicySource(source_cfg, PolicyConfig(type=policy_type)),
    )


def _vector_batch(noise: str, goal_error: bool):
    from owm_envs.drivers.types import RolloutSpec
    from owm_envs.envs.iss.config import ISSConfig
    from owm_envs.envs.iss.sensing import PRESETS

    cfg = ISSConfig(sensor_noise=PRESETS[noise])
    driver = _vector_driver(cfg, goal_error=goal_error)
    return driver.generate(RolloutSpec(num_episodes=2, max_steps=20, seed=0))


def test_vector_truth_equals_obs_without_noise():
    batch = _vector_batch("off", goal_error=False)
    assert batch.true_state is not None
    np.testing.assert_array_equal(batch.true_state, batch.observations)


def test_vector_truth_differs_under_noise():
    batch = _vector_batch("noncooperative", goal_error=False)
    real = batch.lengths[0]
    measured = batch.observations[0, :real, :13]
    true = batch.true_state[0, :real]
    assert not np.allclose(measured, true)
    np.testing.assert_allclose(np.linalg.norm(true[:, 6:10], axis=1), 1.0, atol=1e-5)


def test_vector_truth_is_13_dim_with_goal_error():
    batch = _vector_batch("off", goal_error=True)
    assert batch.observations.shape[-1] == 25
    assert batch.true_state.shape[-1] == 13
    np.testing.assert_array_equal(batch.true_state, batch.observations[:, :, :13])


@pytest.mark.parametrize(
    ("env_max_steps", "spec_max_steps"),
    [
        # The env's own limit binds, so every episode after the first starts
        # from a NEXT_STEP autoreset -- the driver's awaiting-reset path.
        (10, 20),
        # The requested horizon binds, so no lane is ever env-terminated: each
        # lane freezes at the horizon and the cohort is reset together -- the
        # driver's freeze + whole-vector reset path.
        (7200, 10),
    ],
)
def test_vector_truth_stays_aligned_across_resets(env_max_steps, spec_max_steps):
    """Truth must be dynamics-consistent for EVERY episode a lane produces,
    across both of the driver's episode-boundary paths.

    Replaying each recorded action from its recorded true state has to
    reproduce the next one: truth shifted by a timestep, or carrying a
    neighbouring episode's state across a reset, would still differ from the
    noisy observations and so survive the noise test above.
    """
    import jax.numpy as jnp

    from owm_envs.drivers.types import RolloutSpec
    from owm_envs.envs.iss.config import ISSConfig
    from owm_envs.envs.iss.dynamics import ISSDynamics
    from owm_envs.envs.iss.sensing import PRESETS

    cfg = ISSConfig(max_steps=env_max_steps, sensor_noise=PRESETS["noncooperative"])
    driver = _vector_driver(cfg, num_envs=1)
    batch = driver.generate(RolloutSpec(num_episodes=3, max_steps=spec_max_steps, seed=0))
    dynamics = ISSDynamics(cfg)

    assert batch.num_episodes == 3
    for i in range(batch.num_episodes):
        length = int(batch.lengths[i])
        assert length > 1
        true = batch.true_state[i, :length]
        # Every episode's truth must begin on the start sphere: a lane that
        # carried the previous episode's final state across a reset would
        # essentially never land there by chance.
        np.testing.assert_allclose(
            np.linalg.norm(true[0, 0:3]), cfg.physics.start_radius_m, rtol=1e-4
        )
        for t in range(length - 1):
            stepped, _ = dynamics.step(jnp.asarray(true[t]), jnp.asarray(batch.actions[i, t]))
            np.testing.assert_allclose(np.asarray(stepped), true[t + 1], rtol=1e-4, atol=1e-4)


def test_vector_transitions_mode_records_truth():
    from owm_envs.drivers.types import RolloutSpec
    from owm_envs.envs.iss.config import ISSConfig
    from owm_envs.envs.iss.sensing import PRESETS

    cfg = ISSConfig(max_steps=10, sensor_noise=PRESETS["noncooperative"])
    batch = _vector_driver(cfg, policy_type="dock").generate(
        RolloutSpec(max_steps=10, seed=0, min_transitions=60)
    )

    assert batch.true_state is not None
    assert batch.true_state.shape == batch.observations.shape[:2] + (13,)
    for i in range(batch.num_episodes):
        length = int(batch.lengths[i])
        true = batch.true_state[i, :length]
        assert not np.allclose(true, 0.0)
        assert not np.allclose(true, batch.observations[i, :length, :13])
