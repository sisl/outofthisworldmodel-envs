import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver, supports_fused_rollout
from owm_envs.drivers.types import RolloutSpec
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import DockConfig, PhysicsConfig
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.dynamics import ISSDynamics

FREE_FLIGHT_PHYSICS = dict(collision_boxes_path=None)
FREE_FLIGHT_DOCK = dict(enabled=False)


def make_driver(num_envs=2, policy_type="dock", physics=None, dock=None):
    cfg = ISSConfig(
        physics=PhysicsConfig(**{**FREE_FLIGHT_PHYSICS, **(physics or {})}),
        dock=DockConfig(**{**FREE_FLIGHT_DOCK, **(dock or {})}),
    )
    return ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type=policy_type), num_envs=num_envs)


def test_iss_dynamics_advertises_fused_rollout_support():
    assert supports_fused_rollout(ISSDynamics(ISSConfig())) is True


def test_an_opaque_backend_does_not_advertise_support():
    class OpaqueBackend:
        supports_fused_rollout = False

    assert supports_fused_rollout(OpaqueBackend()) is False


def test_generates_the_requested_number_of_episodes():
    batch = make_driver().generate(RolloutSpec(num_episodes=4, max_steps=20, seed=0))
    batch.validate()
    assert batch.num_episodes == 4


def test_output_shapes_and_dtypes():
    batch = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=15, seed=0))
    assert batch.observations.shape[0] == 3
    assert batch.observations.shape[2] == 13
    assert batch.actions.shape[2] == 6
    assert batch.observations.dtype == np.float32
    assert batch.lengths.dtype == np.int32


def test_free_flight_episodes_truncate_at_max_steps():
    # 12 steps -> 13 observations, matching VectorEnvDriver's convention.
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))
    assert np.all(batch.lengths == 13)
    assert np.all(batch.truncated)
    assert not np.any(batch.terminated)


def test_collision_terminates_episodes_early():
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
            start_radius_range_m=(100.0, 100.0),
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    assert np.all(batch.lengths < 50)


def test_leaving_the_domain_terminates_episodes_early():
    # A runaway episode must end at the domain edge rather than burn the whole
    # horizon. max_range_m below the start sphere puts every reset out of
    # bounds, so the escape is immediate and does not depend on where the
    # random policy happens to fly.
    cfg = ISSConfig(
        max_range_m=50.0,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(100.0, 100.0)),
        dock=DockConfig(enabled=False),
    )
    driver = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="random"), num_envs=2)
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=200, seed=0))
    batch.validate()

    assert np.all(batch.terminated)
    assert not np.any(batch.truncated)
    # One step -> two observations, far short of the 200-step horizon.
    assert np.all(batch.lengths == 2)
    # Sensor noise is off by default, so the recorded terminal observation is
    # the out-of-bounds state that ended the episode.
    terminal = batch.observations[np.arange(batch.num_episodes), batch.lengths - 1]
    assert np.all(np.linalg.norm(terminal[:, 0:3], axis=1) > 50.0)


def test_max_range_none_lets_a_far_episode_run_to_truncation():
    # The start sphere sits beyond the 1000 m default deliberately: starting
    # inside it, this would pass even if None silently fell back to that
    # default rather than removing the bound.
    cfg = ISSConfig(
        max_range_m=None,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(2000.0, 2000.0)),
        dock=DockConfig(enabled=False),
    )
    driver = ScanDriver(cfg=cfg, policy_cfg=PolicyConfig(type="random"), num_envs=2)
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))
    assert np.all(batch.truncated)
    assert not np.any(batch.terminated)


def test_is_deterministic_in_the_seed():
    a = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    b = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    np.testing.assert_array_equal(a.observations, b.observations)


def test_union_policy_records_policy_ids():
    batch = make_driver(policy_type="union").generate(
        RolloutSpec(num_episodes=8, max_steps=10, seed=0)
    )
    assert batch.policy_ids is not None
    assert set(np.unique(batch.policy_ids)).issubset({0, 1, 2})


def test_rejects_a_non_positive_episode_count():
    with pytest.raises(ValueError):
        make_driver().generate(RolloutSpec(num_episodes=0, max_steps=10, seed=0))


@pytest.mark.parametrize("num_envs", [0, -1])
def test_rejects_a_non_positive_lane_count(num_envs):
    with pytest.raises(ValueError, match=str(num_envs)):
        make_driver(num_envs=num_envs)


def test_lane_count_error_reports_the_raw_input():
    # -0.5 converts to 0 via int(); the message must still name -0.5, the
    # value actually received, not the post-conversion value.
    with pytest.raises(ValueError, match=r"-0\.5"):
        make_driver(num_envs=-0.5)


def test_segmentation_spreads_truncated_episodes_across_every_lane(monkeypatch):
    # 10 episodes over 8 lanes, free flight so every lane truncates at the
    # same global step. Segmenting lane-major (all of lane 0's episodes,
    # then lane 1's, ...) and stopping once the requested count is reached
    # would take 2 episodes each from lanes 0-4 and none from lanes 5-7;
    # time-major segmentation touches every lane instead.
    captured: dict[str, list[dict]] = {}
    original = ScanDriver._segment_episodes

    def spy(emitted, spec, records_policy_ids):
        episodes = original(emitted, spec, records_policy_ids)
        captured["episodes"] = episodes
        return episodes

    monkeypatch.setattr(ScanDriver, "_segment_episodes", staticmethod(spy))

    batch = make_driver(num_envs=8).generate(RolloutSpec(num_episodes=10, max_steps=5, seed=0))
    batch.validate()
    assert batch.num_episodes == 10

    lanes = {episode["lane"] for episode in captured["episodes"][:10]}
    assert lanes == set(range(8))


def test_episodes_reset_independently_when_max_steps_is_below_the_env_horizon():
    # ScanDriver autoresets in-scan on every `done` (see per_env_step), so
    # each episode starts from an independent reset even when spec.max_steps
    # (5) is far below the env's own horizon -- the segmenter cuts a fresh
    # episode at every `done` rather than letting the scan run on and
    # chaining what should be separate episodes together.
    cfg = ISSConfig(physics=PhysicsConfig(collision_boxes_path=None), dock=DockConfig(enabled=False))
    batch = make_driver(num_envs=1, policy_type="dock").generate(
        RolloutSpec(num_episodes=3, max_steps=5, seed=0)
    )
    batch.validate()
    assert np.all(batch.truncated)

    for i in range(batch.num_episodes):
        low, high = cfg.physics.start_radius_range_m
        radius = np.linalg.norm(batch.observations[i, 0, 0:3])
        assert low * (1 - 1e-4) <= radius <= high * (1 + 1e-4)

    for i in range(1, batch.num_episodes):
        previous_terminal = batch.observations[i - 1, batch.lengths[i - 1] - 1]
        this_initial = batch.observations[i, 0]
        assert not np.allclose(this_initial, previous_terminal)


def test_num_episodes_mode_keeps_lane_quota_not_fastest_finishers():
    # A box over the upper half-space makes lanes that start with z > 0
    # collide on their first step, while lanes starting below truncate at
    # max_steps. With spec seed 4, lanes 0-2 start below (slow) and lane 3
    # above (fast, recycling a new episode every couple of steps).
    # Keeping the first N episodes to finish would return lane 3's
    # collisions and silently skew any policy mixture toward whatever
    # terminates fastest; num_episodes mode must instead keep each lane's
    # first episodes by deterministic quota -- here lanes 0 and 1 --
    # regardless of how long they take.
    driver = make_driver(
        num_envs=4,
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 150.0], "size": [400.0, 400.0, 300.0]}],
            start_radius_range_m=(100.0, 100.0),
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=5, seed=4))
    batch.validate()
    assert batch.lengths.tolist() == [6, 6]
    assert not np.any(batch.terminated)
    assert np.all(batch.truncated)
    # Lane-major order: episode 0 is lane 0's start, episode 1 is lane 1's.
    np.testing.assert_allclose(batch.observations[0, 0, 2], -85.0, atol=1.0)
    np.testing.assert_allclose(batch.observations[1, 0, 2], -36.0, atol=1.0)


def test_accepts_an_explicit_env_spec():
    # Passing the registry's iss spec must be equivalent to the default, which
    # is what makes the spec the only thing a second environment has to supply.
    cfg = ISSConfig(
        physics=PhysicsConfig(**FREE_FLIGHT_PHYSICS), dock=DockConfig(**FREE_FLIGHT_DOCK)
    )
    driver = ScanDriver(
        cfg=cfg,
        policy_cfg=PolicyConfig(type="dock"),
        num_envs=2,
        env_spec=ENV_REGISTRY["iss"],
    )
    spec = RolloutSpec(num_episodes=2, max_steps=10, seed=0)
    batch = driver.generate(spec)
    batch.validate()
    assert batch.num_episodes == 2
    np.testing.assert_array_equal(
        batch.observations, make_driver(num_envs=2).generate(spec).observations
    )
