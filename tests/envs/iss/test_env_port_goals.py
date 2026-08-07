"""Per-episode dock ports in the Gymnasium env.

`DockConfig.ports` gives an episode a goal drawn at reset instead of the one
pose in `DockConfig`. Two things have to hold for that to be usable: a config
that names no ports must behave exactly as it did before the field existed,
and when ports are named the drawn one must govern every consumer of the goal
at once -- the observation's goal block, the reward's position target and the
`docked` gate -- so an episode cannot be flown to one port and scored against
another.

The start sphere is collapsed to radius zero in several tests below. That puts
the chaser at the ISS origin, which is not a place a chaser can be, but it is
the one start position that does not depend on the seed: the distance from it
to a target is just that target's own distance from the origin, which makes
the dock gate a statement about which target is in force and nothing else.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig, dock_target
from owm_envs.envs.iss.docking_ports import PORT_NAMES, PORTS_BY_NAME, port_pose
from owm_envs.envs.iss.env import ISSEnv
from owm_envs.envs.iss.goal import dock_goal_error
from owm_envs.envs.iss.vector_env import ISSVectorEnv

ZERO_ACTION = np.zeros(6, dtype=np.float32)


def port_target(name: str) -> np.ndarray:
    position, quaternion = port_pose(PORTS_BY_NAME[name])
    return np.concatenate([position, quaternion]).astype(np.float32)


def at_origin(**dock: object) -> ISSConfig:
    """A config whose reset puts the chaser exactly at the ISS origin."""
    return ISSConfig(
        max_steps=100,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(0.0, 0.0)),
        dock=DockConfig(**dock),
    )


def test_no_ports_is_the_default_and_keeps_the_single_dock_pose():
    cfg = ISSConfig()
    assert cfg.dock.ports == ()

    env = ISSEnv(ISSConfig(observation={"goal_error": True}))
    obs, info = env.reset(seed=7)
    # No port was drawn, so nothing claims one was.
    assert "dock_port" not in info and "dock_port_index" not in info
    np.testing.assert_allclose(
        obs[13:],
        np.asarray(dock_goal_error(jnp.asarray(obs[:13]), jnp.asarray(dock_target(env.cfg)))),
        atol=1e-6,
    )


def test_a_port_set_leaves_the_seeded_initial_state_alone():
    # The port is drawn after the dynamics seed, so the same seed places the
    # chaser identically with and without ports -- adding a port set does not
    # reshuffle the trajectories a run would otherwise have produced.
    plain = ISSEnv(ISSConfig())
    ported = ISSEnv(ISSConfig(dock=DockConfig(ports=("all",))))
    for seed in (0, 1, 42):
        _, plain_info = plain.reset(seed=seed)
        _, ported_info = ported.reset(seed=seed)
        np.testing.assert_array_equal(plain_info["state"], ported_info["state"])


def test_the_same_seed_draws_the_same_port():
    cfg = ISSConfig(dock=DockConfig(ports=("all",)))
    for seed in (0, 3, 11):
        _, first = ISSEnv(cfg).reset(seed=seed)
        _, second = ISSEnv(cfg).reset(seed=seed)
        assert first["dock_port"] == second["dock_port"]
        assert first["dock_port_index"] == second["dock_port_index"]


def test_successive_episodes_cover_the_whole_port_set():
    env = ISSEnv(ISSConfig(dock=DockConfig(ports=("all",))))
    _, info = env.reset(seed=0)
    drawn = [info["dock_port"]]
    for _ in range(60):
        _, info = env.reset()
        drawn.append(info["dock_port"])
    assert set(drawn) == set(PORT_NAMES)


def test_info_names_and_indexes_the_same_port_on_reset_and_step():
    cfg = ISSConfig(dock=DockConfig(ports=("zvezda_aft", "poisk_zenith")))
    env = ISSEnv(cfg)
    _, info = env.reset(seed=5)
    assert cfg.dock.ports[info["dock_port_index"]].name == info["dock_port"]
    _, _, _, _, step_info = env.step(ZERO_ACTION)
    # The goal is the episode's, not the step's: it does not move mid-episode.
    assert step_info["dock_port"] == info["dock_port"]
    assert step_info["dock_port_index"] == info["dock_port_index"]


def test_goal_block_measures_against_the_drawn_port():
    cfg = ISSConfig(observation={"goal_error": True}, dock=DockConfig(ports=("all",)))
    env = ISSEnv(cfg)
    for seed in range(8):
        obs, info = env.reset(seed=seed)
        target = port_target(info["dock_port"])
        np.testing.assert_allclose(
            obs[13:],
            np.asarray(dock_goal_error(jnp.asarray(obs[:13]), jnp.asarray(target))),
            atol=1e-6,
        )
        # Only for the seeds whose port is not PMA-2 does this say anything,
        # and PMA-2's own pose differs from cfg.dock's anyway (the shipped
        # pose does not sit on the port's centreline).
        assert not np.allclose(
            obs[13:16], np.asarray(obs[:3]) - np.asarray(dock_target(cfg)[0:3]), atol=1e-3
        )


def test_dock_success_gates_on_the_drawn_port_not_the_config_pose():
    # From the origin the goal is 38.6 m away at zvezda_aft and 24.6 m away at
    # the shipped cfg.dock pose, so a 30 m gate separates them: the config
    # pose is inside it, the drawn port is not.
    gate = dict(
        enabled=True, max_distance_m=30.0, max_velocity_m_s=10.0,
        max_attitude_error_deg=None, max_body_rate_rad_s=None,
    )
    ported = ISSEnv(at_origin(ports=("zvezda_aft",), **gate))
    ported.reset(seed=0)
    _, _, terminated, _, info = ported.step(ZERO_ACTION)
    assert terminated is False and info["success"] is False

    plain = ISSEnv(at_origin(**gate))
    plain.reset(seed=0)
    _, _, terminated, _, info = plain.step(ZERO_ACTION)
    assert terminated is True and info["success"] is True

    # And widening the gate past the drawn port's own distance does dock it,
    # so the port is a reachable goal and not merely an unreachable one.
    wide = ISSEnv(at_origin(ports=("zvezda_aft",), **{**gate, "max_distance_m": 39.0}))
    wide.reset(seed=0)
    _, _, terminated, _, info = wide.step(ZERO_ACTION)
    assert terminated is True and info["success"] is True


def test_reward_position_term_targets_the_drawn_port():
    # At rest at the origin with no action, every reward term but position is
    # zero, so the reward is exactly -|goal|^2 for whichever goal is in force.
    ported = ISSEnv(at_origin(ports=("zvezda_aft",), enabled=False))
    ported.reset(seed=0)
    _, reward, _, _, _ = ported.step(ZERO_ACTION)
    expected = -float(np.sum(port_target("zvezda_aft")[0:3] ** 2))
    assert reward == pytest.approx(expected, rel=1e-5)

    plain = ISSEnv(at_origin(enabled=False))
    plain.reset(seed=0)
    _, plain_reward, _, _, _ = plain.step(ZERO_ACTION)
    assert plain_reward == pytest.approx(
        -float(np.sum(np.asarray(ISSConfig().dock.position) ** 2)), rel=1e-5
    )
    assert reward != pytest.approx(plain_reward)


def test_reward_goal_position_still_outranks_a_drawn_port():
    # An explicit reward target is an instruction to shape toward some other
    # point entirely; a per-episode port does not revoke it.
    cfg = ISSConfig(
        max_steps=100,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(0.0, 0.0)),
        dock=DockConfig(ports=("zvezda_aft",), enabled=False),
        reward_goal_position=(0.0, 0.0, 0.0),
    )
    env = ISSEnv(cfg)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(ZERO_ACTION)
    assert reward == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("suffix", [".yaml", ".toml"])
def test_a_port_set_round_trips_through_a_config_file_with_its_poses(tmp_path, suffix):
    # The as-run record has to carry the poses a run flew to, not just names
    # whose meaning depends on which version of the table reads them back.
    cfg = ISSConfig(dock=DockConfig(ports=("poisk_zenith", "rassvet_nadir")))
    path = tmp_path / f"env_config{suffix}"
    cfg.to_yaml(path) if suffix == ".yaml" else cfg.to_toml(path)
    restored = ISSConfig.load(path)
    assert [p.name for p in restored.dock.ports] == ["poisk_zenith", "rassvet_nadir"]
    for port in restored.dock.ports:
        np.testing.assert_allclose(
            np.concatenate([port.position, port.quaternion]), port_target(port.name), atol=1e-6
        )
    # Same draws either side of the round trip.
    _, before = ISSEnv(cfg).reset(seed=4)
    _, after = ISSEnv(restored).reset(seed=4)
    assert before["dock_port"] == after["dock_port"]


def test_unknown_port_names_are_rejected_at_load():
    with pytest.raises(ValueError, match="unknown docking port"):
        DockConfig(ports=("harmony_fwd_pma2", "not_a_port"))
    with pytest.raises(ValueError, match="duplicate docking port"):
        DockConfig(ports=("poisk_zenith", "poisk_zenith"))


def test_vector_env_refuses_a_port_set_rather_than_ignoring_it():
    with pytest.raises(ValueError, match="does not support per-episode dock ports"):
        ISSVectorEnv(2, ISSConfig(dock=DockConfig(ports=("all",))))
