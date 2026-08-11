import pytest
from pydantic import ValidationError

from owm_envs.envs.common.config import (
    BaseTaskConfig,
    RewardShapingConfig,
    RewardWeights,
)


def test_reward_weights_defaults_sum_to_one_shaped_unit():
    # The shaped terms are each normalised to ~1 at the edge of the operating
    # envelope, so the weights summing to 1 is what bounds a step's shaped
    # cost at ~1 -- and that bound is what puts the collision penalty two
    # orders above a whole rollout.
    w = RewardWeights()
    shaped = abs(w.position) + abs(w.velocity) + abs(w.attitude) + abs(w.body_rate)
    assert shaped == pytest.approx(1.0)


def test_reward_weights_penalties_are_negative_and_the_dock_bonus_is_not():
    w = RewardWeights()
    assert w.position < 0 and w.velocity < 0
    assert w.attitude < 0 and w.body_rate < 0
    assert w.collision < 0
    assert w.escape < 0
    assert w.dock_success > 0


def test_collision_dominates_a_full_horizon_of_shaped_cost():
    w = RewardWeights()
    shaped = abs(w.position) + abs(w.velocity) + abs(w.attitude) + abs(w.body_rate)
    worst_rollout = shaped * BaseTaskConfig().max_steps
    assert abs(w.collision) > 50.0 * worst_rollout


def test_dock_bonus_cannot_outweigh_a_collision():
    # A coin-flip approach must never be the profitable move.
    w = RewardWeights()
    assert 0.5 * w.dock_success + 0.5 * w.collision < 0.0


def test_removed_reward_weights_are_rejected():
    with pytest.raises(ValidationError):
        RewardWeights(control_effort=-0.05)
    with pytest.raises(ValidationError):
        RewardWeights(angular_velocity=-0.1)


def test_reward_shaping_rejects_a_nonpositive_scale():
    with pytest.raises(ValidationError):
        RewardShapingConfig(position_scale_m=0.0)
    with pytest.raises(ValidationError):
        RewardShapingConfig(attitude_delta_rad=-1.0)


def test_rotation_gate_far_is_a_fraction():
    assert RewardShapingConfig(rotation_gate_far=0.0).rotation_gate_far == 0.0
    assert RewardShapingConfig(rotation_gate_far=1.0).rotation_gate_far == 1.0
    with pytest.raises(ValidationError):
        RewardShapingConfig(rotation_gate_far=1.5)
    with pytest.raises(ValidationError):
        RewardShapingConfig(rotation_gate_far=-0.1)


def test_base_task_config_carries_reward_shaping():
    assert BaseTaskConfig().reward_shaping.position_scale_m == 225.0
