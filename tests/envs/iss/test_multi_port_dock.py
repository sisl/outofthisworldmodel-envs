"""Multi-port dock targets: sampling, target resolution, and success scoring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from owm_envs.envs.common.config import dock_target
from owm_envs.envs.common.docking_ports import PORT_NAMES, PORTS_BY_NAME, port_pose
from owm_envs.envs.common.events import Events
from owm_envs.envs.common.goal import make_augment
from owm_envs.envs.common.policies import (
    DOCK_SLOT,
    EXTRAS_DIM,
    DockParams,
    PolicyConfig,
    dock_target_selector,
    make_policy,
)
from owm_envs.envs.common.reward import docking_reward
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.dynamics import ISSDynamics

CFG = ISSConfig()
GOAL_CFG = ISSConfig(observation={"goal_error": True})
NO_EVENTS = Events(collision=jnp.array(False), docked=jnp.array(False), escaped=jnp.array(False))
# The union switch's dock branch (extras[0] == 2), so make_augment takes the
# dock arm and the port slot is the thing under test.
_UNION_DOCK_BRANCH = 2.0


def test_extras_carry_a_port_slot():
    assert EXTRAS_DIM["dock"] == 1
    assert EXTRAS_DIM["union"] == 7
    assert DOCK_SLOT == {"dock": 0, "union": 6}


@pytest.mark.parametrize("kind", ["dock", "union"])
def test_sampling_reaches_every_configured_port(kind):
    policy_cfg = PolicyConfig(type=kind, dock=DockParams(ports=("all",)))
    _, extras_fn = make_policy(CFG, policy_cfg)
    slot = DOCK_SLOT[kind]
    drawn = {int(extras_fn(jax.random.PRNGKey(seed))[slot]) for seed in range(200)}
    assert drawn == set(range(len(PORT_NAMES)))


def test_ports_default_to_the_config_dock_pose():
    policy_cfg = PolicyConfig(type="dock")
    select = dock_target_selector(CFG, policy_cfg)
    target = np.asarray(select(jnp.zeros((EXTRAS_DIM["dock"],))))
    np.testing.assert_allclose(target[0:3], CFG.dock.position, atol=1e-6)
    np.testing.assert_allclose(target[3:7], CFG.dock.quaternion, atol=1e-6)


def test_dock_success_follows_the_assigned_port():
    policy_cfg = PolicyConfig(type="dock", dock=DockParams(ports=("all",)))
    select = dock_target_selector(CFG, policy_cfg)
    dynamics = ISSDynamics(CFG)

    for index, name in enumerate(PORT_NAMES):
        target = select(jnp.asarray([float(index)]))
        at_goal = jnp.concatenate([target[0:3], jnp.zeros(3), target[3:7], jnp.zeros(3)])
        _, events = dynamics.step(at_goal, jnp.zeros(6), target)
        assert bool(events.docked), name
        # Not a success against the shipped pose for any port, including
        # PMA-2's: the derived pose sits on the adapter's own centreline,
        # 0.84 m off the shipped one, which is well outside the 0.1 m gate.
        _, default_events = dynamics.step(at_goal, jnp.zeros(6))
        assert not bool(default_events.docked), name


@pytest.mark.parametrize("kind", ["random", "orbit"])
def test_policies_that_never_dock_keep_the_config_dock_pose(kind):
    # random and orbit have no DOCK_SLOT entry, so there is no drawn port to
    # resolve. Falling back to the port table's first row would move their
    # success gate 0.837 m off DockConfig the moment a port set is configured,
    # for episodes that were never flying to a port at all.
    policy_cfg = PolicyConfig(type=kind, dock=DockParams(ports=("all",)))
    select = dock_target_selector(CFG, policy_cfg)
    target = np.asarray(select(jnp.zeros((EXTRAS_DIM[kind],))))
    np.testing.assert_array_equal(target, dock_target(CFG))


def test_reward_peaks_at_the_assigned_port_not_the_config_pose():
    policy_cfg = PolicyConfig(type="dock", dock=DockParams(ports=("all",)))
    select = dock_target_selector(CFG, policy_cfg)

    for index, name in enumerate(PORT_NAMES):
        target = select(jnp.asarray([float(index)]))
        at_port = jnp.concatenate([target[0:3], jnp.zeros(3), target[3:7], jnp.zeros(3)])
        at_port_reward = float(
            docking_reward(at_port, jnp.zeros(6), NO_EVENTS, CFG, target[0:3])
        )
        # Reward is a sum of negative-weighted quadratics, so "peak" is the
        # maximum: at the assigned port the position term is zero, and the
        # same state scored against DockConfig's pose is strictly worse.
        against_config = float(docking_reward(at_port, jnp.zeros(6), NO_EVENTS, CFG))
        assert at_port_reward == pytest.approx(0.0, abs=1e-3), name
        assert against_config < at_port_reward - 0.1, name


def test_reward_without_a_port_set_is_unchanged():
    # The scan driver now always passes a dock position; with no ports
    # configured that position IS DockConfig's, so a fixed state/action probe
    # must reproduce the value the reward produced before the argument existed.
    policy_cfg = PolicyConfig(type="dock")
    select = dock_target_selector(CFG, policy_cfg)
    state = jnp.asarray(
        [12.0, -18.0, 4.0, 0.3, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0, 0.02, -0.01, 0.03],
        dtype=jnp.float32,
    )
    action = jnp.asarray([5.0, -3.0, 1.0, 0.4, 0.2, -0.1], dtype=jnp.float32)
    target = select(jnp.zeros((EXTRAS_DIM["dock"],)))
    assert float(docking_reward(state, action, NO_EVENTS, CFG, target[0:3])) == float(
        docking_reward(state, action, NO_EVENTS, CFG)
    )


def test_union_goal_block_follows_the_port_in_the_extras_slot():
    # End to end through make_augment: the recorded goal-error block of a
    # union episode on its dock branch must be measured against the port the
    # extras' DOCK_SLOT names, not against DockConfig's pose.
    policy_cfg = PolicyConfig(type="union", dock=DockParams(ports=("all",)))
    augment = make_augment(GOAL_CFG, policy_cfg)
    measured = jnp.asarray(
        [30.0, -12.0, 6.0, 0.1, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )

    for index, name in enumerate(PORT_NAMES):
        extras = np.zeros((EXTRAS_DIM["union"],), dtype=np.float32)
        extras[0] = _UNION_DOCK_BRANCH
        extras[DOCK_SLOT["union"]] = float(index)
        observation = np.asarray(augment(measured, jnp.asarray(extras)))

        position, _ = port_pose(PORTS_BY_NAME[name])
        assert observation.shape == (25,)
        np.testing.assert_allclose(
            observation[13:16], np.asarray(measured[0:3]) - position, atol=1e-4, err_msg=name
        )


def test_shipped_dock_pose_is_off_the_pma2_centreline():
    # The shipped DockConfig sits on the Harmony barrel centreline, but the
    # adapter dog-legs nadir-ward over its last 2 m, so its docking ring is
    # 0.734 m lower. Recorded here because a target that far off-axis is a
    # deliberate inheritance, not a rounding difference.
    position, _ = port_pose(PORTS_BY_NAME["harmony_fwd_pma2"])
    offset = np.asarray(CFG.dock.position) - position
    assert np.linalg.norm(offset) == pytest.approx(0.837, abs=0.01)
    assert offset[1] == pytest.approx(0.0, abs=1e-6)
