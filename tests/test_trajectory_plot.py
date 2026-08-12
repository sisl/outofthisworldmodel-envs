"""The frame handling behind scripts/plot_trajectories_3d.py.

The scripts directory is not a package; the script is imported by path so the
shape-juggling it does on lerobot's nested cells is covered by the suite rather
than only by looking at the page it draws.
"""
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from astrojax.constants import GM_EARTH

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "plot_trajectories_3d.py"
sys.path.insert(0, str(SCRIPT.parent))

from plot_trajectories_3d import (  # noqa: E402
    classify_outcomes,
    dock_positions,
    policy_groups,
    view_cells,
)

from owm_envs.envs import ENV_REGISTRY  # noqa: E402
from owm_envs.envs.iss_numerical.config import NumericalConfig  # noqa: E402
from owm_envs.envs.iss_numerical.dynamics import chaser_state_from_view  # noqa: E402


def test_dock_positions_unpacks_the_nested_lerobot_cell():
    # dock_target is declared to lerobot at shape (1, 7), so a cell reads back
    # nested. Getting the flattening wrong yields a plausible-looking 3-vector
    # of the wrong numbers -- a dock marker drawn in the wrong place, silently.
    pose = np.array([[1.5, -2.0, 3.25, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    frame = pd.DataFrame({"episode_index": [0, 0], "dock_target": [pose, pose]})
    assert dock_positions(frame).tolist() == [[1.5, -2.0, 3.25]]


def test_dock_positions_drops_episodes_with_no_target():
    # A driver that could not supply a target writes all-NaN, which must not
    # become a marker at the origin.
    unknown = np.full((1, 7), np.nan, dtype=np.float32)
    known = np.array([[4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    frame = pd.DataFrame({"episode_index": [0, 1], "dock_target": [unknown, known]})
    assert dock_positions(frame).tolist() == [[4.0, 0.0, 0.0]]


def _frame(rows):
    return pd.DataFrame(rows)


def _state(pos, vel=(0.0, 0.0, 0.0)):
    return np.array([*pos, *vel, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


META = {"max_range_m": 500.0, "dock_max_distance_m": 0.1,
        "dock_max_velocity_m_s": 0.5, "split_policy": {}}
TARGET = np.array([[0.0, -24.5, -2.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def test_outcomes_separate_the_four_ways_an_episode_ends():
    # Four episodes, one per ending: inside the dock gates, terminated mid-
    # station (collision), terminated on the domain boundary, and truncated.
    frame = _frame({
        "episode_index": [0, 1, 2, 3],
        "frame_index": [10, 10, 10, 10],
        "terminated": [True, True, True, False],
        "state_vector": [
            _state((0.0, -24.45, -2.5), vel=(0.0, 0.1, 0.0)),
            _state((5.0, -10.0, 0.0)),
            _state((500.0, 0.0, 0.0)),
            _state((80.0, 0.0, 0.0)),
        ],
        "dock_target": [TARGET] * 4,
    })
    outcomes = classify_outcomes(frame, META, "state_vector")
    assert outcomes.to_dict() == {0: "docked", 1: "collision", 2: "escaped", 3: "timeout"}


def test_dock_gate_needs_low_speed_not_just_proximity():
    # At the port but above the velocity gate: that termination was a
    # collision with the station, not a successful dock.
    frame = _frame({
        "episode_index": [0],
        "frame_index": [10],
        "terminated": [True],
        "state_vector": [_state((0.0, -24.45, -2.5), vel=(0.0, 2.0, 0.0))],
        "dock_target": [TARGET],
    })
    assert classify_outcomes(frame, META, "state_vector").to_dict() == {0: "collision"}


def test_union_split_groups_episodes_by_recorded_policy_id():
    frame = _frame({
        "episode_index": [0, 1, 2],
        "policy_id": [np.array([2]), np.array([0]), np.array([2])],
    })
    assert policy_groups(frame, "union") == {"dock": [0, 2], "random": [1]}


def test_non_union_split_is_one_group_named_by_its_policy():
    frame = _frame({"episode_index": [0, 1], "policy_id": [np.array([0])] * 2})
    assert policy_groups(frame, "dock") == {"dock": [0, 1]}


def test_view_cells_map_a_numerical_state_to_the_relative_view():
    # iss-numerical's state_vector holds [epoch | chief ECI | chaser ECI |
    # q_bi | omega]; reading its first elements as a relative position would
    # draw the epoch and the chief's ECI radius as a path. The cells have to
    # come back as the 13D world-frame relative view.
    view = np.array([10.0, -24.0, 5.0, 0.1, -0.05, 0.02,
                     1.0, 0.0, 0.0, 0.0, 0.001, -0.002, 0.0005])
    sma = 6.795e6
    chief = jnp.asarray([sma, 0.0, 0.0, 0.0, float(np.sqrt(GM_EARTH / sma)), 0.0],
                        jnp.float64)
    chaser = chaser_state_from_view(chief, jnp.asarray(view, jnp.float64))
    state = np.asarray(jnp.concatenate([jnp.asarray([2460000.5, 0.0]), chief, chaser]))

    cells = view_cells(pd.Series([state]), "state_vector",
                       ENV_REGISTRY["iss-numerical"], NumericalConfig())
    np.testing.assert_allclose(cells[0], view, atol=1e-4)


def test_view_cells_strip_the_epoch_prefix_of_a_relative_observation():
    obs = np.arange(27.0, dtype=np.float32)
    cells = view_cells(pd.Series([obs]), "observation_vector",
                       ENV_REGISTRY["iss-numerical"], NumericalConfig())
    np.testing.assert_array_equal(cells[0], obs[2:15])


def test_view_cells_pass_vectors_through_without_an_env_spec():
    # A run directory without configs still plots, reading the vectors as the
    # 13D-first layout every run predating the card was written in.
    state = _state((1.0, 2.0, 3.0))
    cells = view_cells(pd.Series([state]), "state_vector", None, None)
    np.testing.assert_array_equal(cells[0], state)


@pytest.mark.parametrize("flag", ["--max-points", "--max-episodes"])
def test_non_positive_limits_are_rejected_at_the_command_line(flag, tmp_path):
    # Zero episodes or zero points draws an empty page, and a negative stride
    # fails deep in the thinning arithmetic instead.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), flag, "0"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must be >= 1" in result.stderr
