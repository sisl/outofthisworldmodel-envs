import numpy as np
import pytest

from owm_envs.drivers.scan_driver import ScanDriver
from owm_envs.drivers.types import RolloutSpec, pack_episodes
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import DockConfig
from owm_envs.envs.common.outcome import classify_batch
from owm_envs.envs.common.policies import PolicyConfig
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig

DOCK_POSE = np.asarray([0.225, -24.5, -2.5, 0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float32)
OTHER_DOCK_POSE = np.asarray([100.0, -50.0, 10.0, 0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float32)


def episode(
    states: np.ndarray, terminated: bool, truncated: bool, dock_target: np.ndarray = DOCK_POSE
) -> dict:
    length = states.shape[0]
    return {
        "obs": states.astype(np.float32),
        "act": np.zeros((length, 6), dtype=np.float32),
        "rew": np.zeros((length,), dtype=np.float32),
        "true_state": states.astype(np.float32),
        "dock_target": dock_target,
        "terminated": terminated,
        "truncated": truncated,
        "policy_id": 0,
    }


def state(pos, vel=(0.0, 0.0, 0.0), quat=(0.7071068, -0.7071068, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
    return np.asarray([*pos, *vel, *quat, *omega], dtype=np.float32)


def batch_of(*episodes):
    return pack_episodes(
        list(episodes), obs_dim=13, act_dim=6,
        records_policy_ids=True, records_dock_targets=True,
        records_true_state=True, state_dim=13,
    )


CFG = ISSConfig(dock=DockConfig(position=tuple(DOCK_POSE[0:3]), quaternion=tuple(DOCK_POSE[3:7])))
SPEC = ENV_REGISTRY["iss"]


def test_an_episode_ending_at_the_gate_is_docked():
    approach = np.stack([state((0.225, -24.6, -2.5)), state(tuple(DOCK_POSE[0:3]))])
    [outcome] = classify_batch(batch_of(episode(approach, True, False)), CFG, SPEC)
    assert outcome.docked is True
    assert outcome.collided is False
    assert outcome.truncated is False
    assert outcome.position_error_m == pytest.approx(0.0, abs=1e-4)


def test_an_episode_ending_at_a_point_inside_the_hull_is_a_collision():
    into_station = np.stack([state((0.0, 0.0, 30.0)), state((0.0, 0.0, 0.0))])
    [outcome] = classify_batch(batch_of(episode(into_station, True, False)), CFG, SPEC)
    assert outcome.collided is True
    assert outcome.docked is False


def test_a_step_that_tunnels_through_the_hull_is_a_collision():
    # Both endpoints sit clear of every box -- (0, -2, 20) is above the box
    # centred at (0, -2, 2) with half-extents (5, 1, 1), (0, -2, -20) is below
    # it -- but the straight-line path between them threads through the box's
    # 2 m-thick slab. A checker that dropped the true_state[length - 2] lookup
    # and tested only the terminal point would call this collision-free.
    tunneling = np.stack([state((0.0, -2.0, 20.0)), state((0.0, -2.0, -20.0))])
    [outcome] = classify_batch(batch_of(episode(tunneling, True, False)), CFG, SPEC)
    assert outcome.collided is True
    assert outcome.docked is False


def test_an_episode_past_the_domain_bound_has_escaped():
    outbound = np.stack([state((0.0, 0.0, 700.0)), state((0.0, 0.0, 900.0))])
    [outcome] = classify_batch(batch_of(episode(outbound, True, False)), CFG, SPEC)
    assert outcome.escaped is True
    assert outcome.docked is False


def test_a_truncated_episode_raises_no_event():
    drifting = np.stack([state((0.0, -200.0, 0.0)), state((0.0, -201.0, 0.0))])
    [outcome] = classify_batch(batch_of(episode(drifting, False, True)), CFG, SPEC)
    assert outcome.truncated is True
    assert not (outcome.docked or outcome.collided or outcome.escaped)


def test_outcome_reports_the_terminal_errors():
    approach = np.stack([
        state((0.225, -30.0, -2.5)),
        state((0.225, -27.5, -2.5), vel=(0.0, 0.3, 0.0), omega=(0.001, 0.0, 0.0)),
    ])
    [outcome] = classify_batch(batch_of(episode(approach, False, True)), CFG, SPEC)
    assert outcome.position_error_m == pytest.approx(3.0, abs=1e-4)
    assert outcome.velocity_error_m_s == pytest.approx(0.3, abs=1e-4)
    assert outcome.attitude_error_rad == pytest.approx(0.0, abs=1e-5)
    assert outcome.body_rate_rad_s == pytest.approx(0.001, abs=1e-6)


def test_a_single_observation_episode_raises_no_event():
    # No step was taken, so there is no swept segment to test for collision.
    single = state((0.0, -100.0, 0.0))[None, :]
    [outcome] = classify_batch(batch_of(episode(single, False, True)), CFG, SPEC)
    assert not (outcome.docked or outcome.collided or outcome.escaped)
    assert outcome.steps == 0


def test_each_episode_is_scored_against_its_own_target():
    at_dock = np.stack([state((0.225, -24.6, -2.5)), state(tuple(DOCK_POSE[0:3]))])
    far = np.stack([state((0.0, -200.0, 0.0)), state((0.0, -201.0, 0.0))])
    outcomes = classify_batch(
        batch_of(episode(at_dock, True, False), episode(far, False, True)), CFG, SPEC
    )
    assert [o.docked for o in outcomes] == [True, False]


def test_position_error_is_measured_against_the_episodes_own_target():
    # Sitting exactly at OTHER_DOCK_POSE reads as zero error only if the
    # episode's own dock_target is used; scored against CFG's dock pose (a
    # different point) instead, this position would show a large error.
    at_other_dock = np.stack([
        state(tuple(OTHER_DOCK_POSE[0:3])),
        state(tuple(OTHER_DOCK_POSE[0:3])),
    ])
    [outcome] = classify_batch(
        batch_of(episode(at_other_dock, False, True, dock_target=OTHER_DOCK_POSE)), CFG, SPEC
    )
    assert outcome.position_error_m == pytest.approx(0.0, abs=1e-4)


NUMERICAL = ENV_REGISTRY["iss-numerical"]


def numerical_state(chaser_position, chief_position=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A 21-wide NUM_LAYOUT row, only its two ECI positions meaningful."""
    row = np.zeros(21, dtype=np.float32)
    row[2:5] = chief_position
    row[8:11] = chaser_position
    return row


def numerical_batch_of(states: np.ndarray) -> object:
    length = states.shape[0]
    return pack_episodes(
        [
            {
                "obs": states,
                "act": np.zeros((length, 6), dtype=np.float32),
                "rew": np.zeros((length,), dtype=np.float32),
                "true_state": states,
                "dock_target": DOCK_POSE,
                "terminated": True,
                "truncated": False,
            }
        ],
        obs_dim=21, act_dim=6,
        records_policy_ids=False, records_dock_targets=True,
        records_true_state=True, state_dim=21,
    )


def test_classification_matches_the_reward_the_dynamics_paid():
    # Reward is the independent witness on the outcome: `docking_reward` reads
    # the SAME `Events` from the SAME `dynamics.step` call that ended the
    # episode, and every shaped term is normalised to |r| <= ~1, so a
    # final-step reward past +1e3 can only be the +10000 dock bonus and one
    # past -1e5 only the -1e6 collision penalty. Classification and reward
    # therefore cannot disagree unless the classification is wrong.
    #
    # iss-numerical specifically, and not iss: this env's state holds absolute
    # ECI positions at ~6.8e6 m, which float32 storage quantizes to a 0.5 m
    # grid per axis, and up to 0.5 * sqrt(3) = 0.87 m on the NORM the dock gate
    # tests -- five to nearly nine times that 0.1 m gate -- so an outcome
    # recovered from the stored state rather than recorded during the rollout
    # is unreliable in both directions here. iss stores the canonical relative
    # view itself at ~100 m, where the same recovery resolves the gate fine and
    # this test would pass either way.
    #
    # Sensor noise and `observe="measurement"` because they are what the dock
    # policy actually flies under, and they decide whether this fixture is
    # discriminating: noiseless episodes settle far inside the gate, where
    # even a quantized position stays inside it, while a policy flying on
    # measurements settles ~0.05-0.1 m out, right where 0.5 m of quantization
    # decides the answer. On this seed the stored terminal position error of
    # the two docked episodes reads 0.178 m and 0.366 m against a 0.1 m gate --
    # both of them genuinely inside it.
    cfg = NUMERICAL.config_cls(sensor_noise=PRESETS["cooperative"])
    batch = ScanDriver(
        cfg=cfg,
        policy_cfg=PolicyConfig(type="dock", observe="measurement"),
        num_envs=4,
        env_spec=NUMERICAL,
    ).generate(RolloutSpec(num_episodes=4, max_steps=cfg.max_steps, seed=4))

    outcomes = classify_batch(batch, cfg, NUMERICAL)
    # rewards[e, lengths[e] - 1] is the zero pad beside the terminal
    # observation; the reward the final step actually paid is one back.
    paid = np.array(
        [batch.rewards[e, int(batch.lengths[e]) - 2] for e in range(batch.num_episodes)]
    )

    assert [o.docked for o in outcomes] == (paid > 1e3).tolist()
    assert [o.collided for o in outcomes] == (paid < -1e5).tolist()
    # Non-vacuous: this seed produces both outcomes, so neither assertion above
    # is comparing two all-False lists.
    assert any(o.docked for o in outcomes)
    assert any(o.collided for o in outcomes)


def test_an_absolute_state_batch_without_recorded_events_is_refused():
    # Re-derivation cannot answer the dock gate from a float32 ECI position,
    # so classifying such a batch is refused rather than answered wrongly.
    row = numerical_state((6.8e6, 0.0, 0.0), chief_position=(6.8e6, 0.0, 0.0))
    with pytest.raises(ValueError, match="no terminal_events"):
        classify_batch(numerical_batch_of(np.stack([row, row])), NUMERICAL.config_cls(), NUMERICAL)


def test_the_refusal_measures_the_chief_block_too():
    # The view is a DIFFERENCE of the chaser and chief blocks, so either
    # operand being large is enough to coarsen it. A chaser sitting near the
    # origin says nothing on its own: with the chief out at an ECI radius, the
    # difference still lands on that block's 0.5 m grid.
    row = numerical_state((100.0, 0.0, 0.0), chief_position=(6.8e6, 0.0, 0.0))
    with pytest.raises(ValueError, match="no terminal_events"):
        classify_batch(numerical_batch_of(np.stack([row, row])), NUMERICAL.config_cls(), NUMERICAL)


def test_a_grid_only_the_three_axis_norm_outgrows_is_refused_too():
    # 6e5 m sits in the binade whose float32 grain is 0.0625 m: finer than the
    # 0.1 m gate per axis, so a per-axis comparison would allow it, but the
    # gate tests the NORM of three axes and that reaches 0.108 m. This is the
    # one range where the two comparisons disagree.
    assert float(np.spacing(np.float32(6.0e5))) == pytest.approx(0.0625)
    row = numerical_state((6.0e5, 0.0, 0.0))
    with pytest.raises(ValueError, match="no terminal_events"):
        classify_batch(numerical_batch_of(np.stack([row, row])), NUMERICAL.config_cls(), NUMERICAL)


def test_recorded_events_are_preferred_to_re_deriving_them():
    # The recorded flags win even where re-derivation would be resolvable and
    # would say something else: this terminal state sits 200 m from the dock,
    # which no gate calls docked.
    far = np.stack([state((0.0, -200.0, 0.0)), state((0.0, -201.0, 0.0))])
    row = episode(far, True, False)
    row["terminal_events"] = np.array([False, True, False])
    batch = pack_episodes(
        [row], obs_dim=13, act_dim=6,
        records_policy_ids=True, records_dock_targets=True,
        records_true_state=True, state_dim=13,
        records_terminal_events=True,
    )
    [outcome] = classify_batch(batch, CFG, SPEC)
    assert outcome.docked is True
    assert outcome.position_error_m == pytest.approx(176.5, abs=0.1)
