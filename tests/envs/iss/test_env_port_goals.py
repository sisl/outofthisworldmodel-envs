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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from owm_envs.envs.common.config import DockConfig, PhysicsConfig, dock_target
from owm_envs.envs.common.docking_ports import PORT_NAMES, PORTS_BY_NAME, port_pose
from owm_envs.envs.common.goal import dock_goal_error
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.env import ISSEnv
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


def test_env_checker_accepts_a_port_set():
    # Including the extra info keys and the reset-to-reset variation in the
    # goal block, which the checker exercises by resetting twice.
    check_env(
        ISSEnv(ISSConfig(observation={"goal_error": True}, dock=DockConfig(ports=("all",)))),
        skip_render_check=True,
    )


def test_no_ports_goal_block_keeps_the_constant_folded_arithmetic():
    """The guard on `ISSEnv._build_jit_dock_goal_error`'s two forms.

    With no ports the target never changes and is compiled into the goal-error
    computation as a constant. Passing it as a runtime argument instead -- the
    obvious cleanup, since it is what a port set already does -- is not
    bit-identical: XLA folds the constant at a different precision, and the
    block moves by an ulp. That is the whole reason the two forms exist, so
    this pins both halves: the emitted block IS the constant-folded one, and
    the two forms really do differ.

    If the second assertion ever fails because JAX made them agree, the
    special case has become dead weight and `_build_jit_dock_goal_error` can
    collapse to the one-line runtime-argument form.
    """
    cfg = ISSConfig(observation={"goal_error": True})
    env = ISSEnv(cfg)
    pose = jnp.asarray(dock_target(cfg))
    folded = jax.jit(lambda measured: dock_goal_error(measured, pose))
    passed_in = jax.jit(dock_goal_error)

    differs_somewhere = False
    for seed in range(8):
        obs, _ = env.reset(seed=seed)
        measured = jnp.asarray(obs[:13])
        # Exact, not allclose: this is the assertion the refactor would break.
        np.testing.assert_array_equal(obs[13:], np.asarray(folded(measured)))
        if not np.array_equal(
            np.asarray(folded(measured)), np.asarray(passed_in(measured, pose))
        ):
            differs_somewhere = True

    assert differs_somewhere, (
        "the constant-folded and runtime-argument forms of dock_goal_error now "
        "agree bit for bit; ISSEnv._build_jit_dock_goal_error no longer needs "
        "to keep them apart"
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
    # The pose is the one the port table derives for that name.
    assert info["goal_pose"].shape == (7,) and info["goal_pose"].dtype == np.float32
    np.testing.assert_allclose(info["goal_pose"], port_target(info["dock_port"]), atol=1e-6)

    _, _, _, _, step_info = env.step(ZERO_ACTION)
    # The goal is the episode's, not the step's: it does not move mid-episode.
    assert step_info["dock_port"] == info["dock_port"]
    assert step_info["dock_port_index"] == info["dock_port_index"]
    np.testing.assert_array_equal(step_info["goal_pose"], info["goal_pose"])


def test_goal_pose_is_present_without_ports_and_is_the_config_pose():
    # The one key a consumer can always read: a fixed-dock run has no port to
    # name, but it still has a goal, and this is where it is.
    env = ISSEnv(ISSConfig())
    _, info = env.reset(seed=5)
    assert "dock_port" not in info
    np.testing.assert_array_equal(info["goal_pose"], dock_target(env.cfg))
    _, _, _, _, step_info = env.step(ZERO_ACTION)
    np.testing.assert_array_equal(step_info["goal_pose"], dock_target(env.cfg))


def test_goal_pose_follows_the_drawn_port_across_episodes():
    env = ISSEnv(ISSConfig(dock=DockConfig(ports=("all",))))
    for seed in range(8):
        _, info = env.reset(seed=seed)
        np.testing.assert_allclose(
            info["goal_pose"], port_target(info["dock_port"]), atol=1e-6
        )


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
    # At rest at the origin with no action the velocity and body-rate terms are
    # exactly zero, so the reward is the position and attitude pair alone --
    # and both are measured against whichever goal pose is in force. Seed 0
    # leaves the chaser 38.596 m and 176.08 deg from zvezda_aft:
    #     position  -0.5 * (sqrt(38.596**2 + 1**2) - 1) / 225      = -0.0835755
    #     gate       0.1 + 0.9 / (1 + (38.596 / 25)**2)            =  0.366001
    #     attitude  -0.2 * (sqrt(3.073179**2 + 0.05**2) - 0.05)/pi = -0.1924874
    #     total     -0.0835755 + 0.366001 * -0.1924874             = -0.1540261
    ported = ISSEnv(at_origin(ports=("zvezda_aft",), enabled=False))
    ported.reset(seed=0)
    _, reward, _, _, _ = ported.step(ZERO_ACTION)
    assert reward == pytest.approx(-0.1540261, rel=1e-5)

    # The shipped pose is 24.628 m and 161.98 deg from that same state, so the
    # same three lines give -0.0525523, 0.556741 and -0.1768217.
    plain = ISSEnv(at_origin(enabled=False))
    plain.reset(seed=0)
    _, plain_reward, _, _, _ = plain.step(ZERO_ACTION)
    assert plain_reward == pytest.approx(-0.1509963, rel=1e-5)
    assert reward != pytest.approx(plain_reward)


def test_reward_goal_position_still_outranks_a_drawn_port():
    # An explicit reward target is an instruction to shape toward some other
    # point entirely; a per-episode port does not revoke it. It moves the
    # POSITION target only, though, so what is left at the origin is the
    # attitude term against the drawn port's quaternion, ungated because the
    # range to (0, 0, 0) is zero:
    #     -0.2 * (sqrt(3.073179**2 + 0.05**2) - 0.05) / pi = -0.1924874
    cfg = ISSConfig(
        max_steps=100,
        physics=PhysicsConfig(collision_boxes_path=None, start_radius_range_m=(0.0, 0.0)),
        dock=DockConfig(ports=("zvezda_aft",), enabled=False),
        reward_goal_position=(0.0, 0.0, 0.0),
    )
    env = ISSEnv(cfg)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(ZERO_ACTION)
    assert reward == pytest.approx(-0.1924874, rel=1e-5)


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


def expected_norms(state: np.ndarray, target: np.ndarray) -> dict[str, float]:
    block = np.asarray(dock_goal_error(jnp.asarray(state), jnp.asarray(target)))
    return {
        "pos_m": float(np.linalg.norm(block[0:3])),
        "vel_mps": float(np.linalg.norm(block[3:6])),
        "att_rad": float(np.linalg.norm(block[6:9])),
        "rate_radps": float(np.linalg.norm(block[9:12])),
    }


def test_true_goal_error_is_measured_from_the_true_state_not_the_noisy_one():
    # The point of the diagnostic: with a noisy sensor the observation's goal
    # block and this must disagree, and it is the true state this follows.
    cfg = ISSConfig(
        observation={"goal_error": True},
        sensor_noise=PRESETS["cooperative"],
        dock=DockConfig(ports=("all",)),
    )
    env = ISSEnv(cfg)
    obs, info = env.reset(seed=2)
    target = port_target(info["dock_port"])

    expected = expected_norms(info["state"], target)
    for label, value in expected.items():
        assert info["goal_error_true"][label] == pytest.approx(value, rel=1e-5, abs=1e-6)

    # Not the noisy block, which the same observation also carries.
    noisy_pos = float(np.linalg.norm(obs[13:16]))
    assert info["goal_error_true"]["pos_m"] != pytest.approx(noisy_pos, rel=1e-9)
    assert obs[:13] is not info["state"] and not np.array_equal(obs[:13], info["state"])

    # And again after a step, against the same episode goal.
    _, _, _, _, step_info = env.step(ZERO_ACTION)
    expected = expected_norms(step_info["state"], target)
    for label, value in expected.items():
        assert step_info["goal_error_true"][label] == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_true_goal_error_follows_the_drawn_port():
    cfg = ISSConfig(dock=DockConfig(ports=("all",)))
    env = ISSEnv(cfg)
    for seed in range(6):
        _, info = env.reset(seed=seed)
        drawn = expected_norms(info["state"], port_target(info["dock_port"]))
        assert info["goal_error_true"]["pos_m"] == pytest.approx(drawn["pos_m"], rel=1e-5)
        # A different port would give a different distance, so this is a
        # statement about which target is in force, not just arithmetic.
        other = "zvezda_aft" if info["dock_port"] != "zvezda_aft" else "rassvet_nadir"
        assert info["goal_error_true"]["pos_m"] != pytest.approx(
            expected_norms(info["state"], port_target(other))["pos_m"], rel=1e-3
        )


def test_true_goal_error_falls_back_to_the_config_pose_without_ports():
    env = ISSEnv(ISSConfig())
    _, info = env.reset(seed=1)
    expected = expected_norms(info["state"], np.asarray(dock_target(env.cfg)))
    for label, value in expected.items():
        assert info["goal_error_true"][label] == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_true_goal_error_reaches_zero_at_the_goal_pose():
    # Docked at the port, at rest, correctly oriented: every one of the four
    # magnitudes is zero. Pins that the attitude term is the angle to the
    # port's own quaternion, not to the config pose's.
    cfg = ISSConfig(dock=DockConfig(ports=("zvezda_aft",), enabled=False))
    env = ISSEnv(cfg)
    env.reset(seed=0)
    target = port_target("zvezda_aft")
    env._state = jnp.asarray(
        np.concatenate([target[0:3], np.zeros(3), target[3:7], np.zeros(3)]),
        dtype=jnp.float32,
    )
    for label, value in env._goal_error_true().items():
        assert value == pytest.approx(0.0, abs=1e-5), label


def test_unknown_port_names_are_rejected_at_load():
    with pytest.raises(ValueError, match="unknown docking port"):
        DockConfig(ports=("harmony_fwd_pma2", "not_a_port"))
    with pytest.raises(ValueError, match="duplicate docking port"):
        DockConfig(ports=("poisk_zenith", "poisk_zenith"))


def test_vector_env_draws_ports_per_lane_rather_than_refusing():
    # The vector adapter honours the same port set the single env does, one
    # independent draw per lane; tests/envs/iss/test_vector_env.py covers the
    # redraw-at-autoreset and options behaviour in depth.
    venv = ISSVectorEnv(4, ISSConfig(dock=DockConfig(ports=("all",))))
    _, info = venv.reset(seed=0)
    assert info["dock_port_index"].shape == (4,)
    for lane, name in enumerate(info["dock_port"]):
        np.testing.assert_allclose(
            info["goal_pose"][lane], port_target(name), atol=1e-6
        )


def test_reset_options_target_a_single_named_port():
    env = ISSEnv(at_origin(ports=("harmony_fwd_pma2", "zvezda_aft", "poisk_zenith")))
    for _ in range(3):
        _, info = env.reset(seed=0, options={"dock_port": "poisk_zenith"})
        assert info["dock_port"] == "poisk_zenith"
        assert info["dock_port_index"] == 2
        np.testing.assert_allclose(info["goal_pose"], port_target("poisk_zenith"), atol=1e-6)


def test_reset_options_sample_uniformly_among_several_names():
    env = ISSEnv(at_origin(ports=("all",)))
    subset = ("zvezda_aft", "rassvet_nadir")
    seen = set()
    for seed in range(20):
        _, info = env.reset(seed=seed, options={"dock_port": subset})
        assert info["dock_port"] in subset
        seen.add(info["dock_port"])
    assert seen == set(subset)


def test_reset_options_reject_names_outside_the_configured_set():
    env = ISSEnv(at_origin(ports=("harmony_fwd_pma2", "zvezda_aft")))
    # poisk_zenith is a real port, but not one this env was configured with.
    with pytest.raises(ValueError, match="unknown dock_port"):
        env.reset(seed=0, options={"dock_port": "poisk_zenith"})
    with pytest.raises(ValueError, match="duplicate dock_port"):
        env.reset(seed=0, options={"dock_port": ("zvezda_aft", "zvezda_aft")})
    with pytest.raises(ValueError, match="empty set"):
        env.reset(seed=0, options={"dock_port": ()})


def test_reset_options_reject_a_name_when_no_ports_are_configured():
    env = ISSEnv(at_origin())
    with pytest.raises(ValueError, match="configure dock.ports or pass dock_pose"):
        env.reset(seed=0, options={"dock_port": "zvezda_aft"})


def test_reset_options_accept_an_explicit_pose_without_configured_ports():
    env = ISSEnv(at_origin())
    pose = np.array([5.0, -10.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, info = env.reset(seed=0, options={"dock_pose": pose})
    # The pose is the goal, but it is not a named port.
    np.testing.assert_allclose(info["goal_pose"], pose, atol=1e-6)
    assert "dock_port" not in info and "dock_port_index" not in info
    # Every consumer follows it: drive the true state onto the pose and the
    # true goal error collapses to zero.
    env._state = jnp.asarray(
        np.concatenate([pose[0:3], np.zeros(3), pose[3:7], np.zeros(3)]),
        dtype=jnp.float32,
    )
    for label, value in env._goal_error_true().items():
        assert value == pytest.approx(0.0, abs=1e-5), label


def test_reset_options_pose_governs_the_goal_error_observation():
    env = ISSEnv(ISSConfig(observation={"goal_error": True}))
    pose = np.array([5.0, -10.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, _ = env.reset(seed=3, options={"dock_pose": pose})
    np.testing.assert_allclose(
        obs[13:],
        np.asarray(dock_goal_error(jnp.asarray(obs[:13]), jnp.asarray(pose))),
        atol=1e-6,
    )


def test_a_naked_reset_forgets_the_previous_overrides():
    env = ISSEnv(ISSConfig(observation={"goal_error": True}))
    pose = np.array([5.0, -10.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.reset(seed=3, options={"dock_pose": pose})
    obs, info = env.reset(seed=3)
    # Back on the configured single pose -- including the byte-identical
    # constant-folded observation block a no-ports config promises.
    np.testing.assert_allclose(info["goal_pose"], np.asarray(dock_target(env.cfg)), atol=1e-6)
    assert "dock_port" not in info
    fresh_obs, _ = ISSEnv(env.cfg).reset(seed=3)
    np.testing.assert_array_equal(obs, fresh_obs)


def test_reset_options_reject_bad_pose_shapes_and_mixed_keys():
    env = ISSEnv(at_origin(ports=("all",)))
    with pytest.raises(ValueError, match="7 values"):
        env.reset(seed=0, options={"dock_pose": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not both"):
        env.reset(
            seed=0,
            options={"dock_port": "zvezda_aft", "dock_pose": port_target("zvezda_aft")},
        )


def test_reset_options_reject_unknown_keys_rather_than_falling_through():
    # A typo'd key must not silently become a naked reset against a random
    # goal -- this is a goal-selection API.
    env = ISSEnv(at_origin(ports=("all",)))
    with pytest.raises(ValueError, match="unknown reset option"):
        env.reset(seed=0, options={"dock_prt": "zvezda_aft"})


def test_reset_options_reject_malformed_dock_port_types():
    env = ISSEnv(at_origin(ports=("all",)))
    for bad in (123, {"zvezda_aft": 1}, ("zvezda_aft", 3)):
        with pytest.raises(ValueError, match="port name or a list/tuple"):
            env.reset(seed=0, options={"dock_port": bad})


def test_a_named_override_does_not_leak_into_the_next_naked_reset():
    cfg = at_origin(ports=("all",))
    env = ISSEnv(cfg)
    env.reset(seed=11, options={"dock_port": "rassvet_nadir"})
    _, after_override = env.reset(seed=11)
    _, fresh = ISSEnv(cfg).reset(seed=11)
    # The naked reset after an override draws exactly what a fresh env draws.
    assert after_override["dock_port"] == fresh["dock_port"]
    np.testing.assert_array_equal(after_override["goal_pose"], fresh["goal_pose"])
