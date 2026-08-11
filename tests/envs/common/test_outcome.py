import numpy as np
import pytest

from owm_envs.drivers.types import pack_episodes
from owm_envs.envs import ENV_REGISTRY
from owm_envs.envs.common.config import DockConfig
from owm_envs.envs.common.outcome import classify_batch
from owm_envs.envs.iss.config import ISSConfig

DOCK_POSE = np.asarray([0.225, -24.5, -2.5, 0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float32)


def episode(states: np.ndarray, terminated: bool, truncated: bool) -> dict:
    length = states.shape[0]
    return {
        "obs": states.astype(np.float32),
        "act": np.zeros((length, 6), dtype=np.float32),
        "rew": np.zeros((length,), dtype=np.float32),
        "true_state": states.astype(np.float32),
        "dock_target": DOCK_POSE,
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


def test_an_episode_ending_inside_the_hull_is_a_collision():
    into_station = np.stack([state((0.0, 0.0, 30.0)), state((0.0, 0.0, 0.0))])
    [outcome] = classify_batch(batch_of(episode(into_station, True, False)), CFG, SPEC)
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
