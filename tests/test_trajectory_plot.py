"""The frame handling behind scripts/plot_trajectories_3d.py.

The scripts directory is not a package; the script is imported by path so the
shape-juggling it does on lerobot's nested cells is covered by the suite rather
than only by looking at the page it draws.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "plot_trajectories_3d.py"
sys.path.insert(0, str(SCRIPT.parent))

from plot_trajectories_3d import dock_positions  # noqa: E402


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
