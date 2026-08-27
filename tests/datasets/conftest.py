import json

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.envs.registration import load_env_creator

from owm_envs.datasets.trajectory import Trajectory
from owm_envs.envs import ENV_REGISTRY


def fly_zero_action(steps: int, seed: int = 0, port: str = "harmony_fwd_pma2") -> Trajectory:
    """A `Trajectory` from `steps` zero-action steps of the real iss-numerical env.

    Real rows rather than random numbers: the render adapter and the plot both
    read the epoch and the chief orbit out of the state, and a made-up state
    would exercise neither.
    """
    spec = ENV_REGISTRY["iss-numerical"]
    cfg = spec.config_cls(max_steps=steps)
    # `model_copy(update=...)` skips field validators, so a bare port name
    # would reach `port_goals` unresolved; go through the constructor so
    # `DockConfig._resolve_ports` turns it into a pinned `DockPort`.
    dock = type(cfg.dock)(**{**cfg.dock.model_dump(), "ports": [port]})
    cfg = cfg.model_copy(update={"dock": dock})
    env = load_env_creator(gym.spec(spec.gym_id).entry_point)(cfg)
    obs, info = env.reset(seed=seed)
    limits = np.array([cfg.control.limit_force_n] * 3 + [cfg.control.limit_torque_nm] * 3,
                      dtype=np.float32)
    states = [np.asarray(info["state"], dtype=np.float32)]
    measured = [np.asarray(info["measured_state"], dtype=np.float32)]
    observations = [np.asarray(obs, dtype=np.float32)]
    actions, rewards, collisions = [], [], []
    for _ in range(steps):
        action = np.zeros(6, dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
        measured.append(np.asarray(info["measured_state"], dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        collisions.append(bool(info.get("collision", False)))
        if term or trunc:
            break
    env.close()
    state = np.stack(states)
    rel_view = np.asarray(jax.vmap(spec.view)(jnp.asarray(state, jnp.float64)))
    action_norm = np.stack(actions)
    goal = np.asarray(info["goal_pose"], dtype=np.float64)
    return Trajectory(
        epoch=state[:, 0:2].astype(np.float64),
        state=state,
        rel_view=rel_view,
        measured_state=np.stack(measured),
        observation=np.stack(observations),
        action_norm=action_norm,
        action_phys=action_norm * limits,
        reward=np.asarray(rewards, dtype=np.float64),
        collision=np.asarray(collisions, dtype=bool),
        dock_target=goal,
        meta={
            "method": "test",
            "port": port,
            "seed": seed,
            "env": "iss-numerical",
            "env_config": json.loads(cfg.model_dump_json()),
            "dt": cfg.dt,
            "rate_hz": 20,
            "action_repeat": 1,
            "steps": len(actions),
            "outcome": "truncated",
            "ever_collided": bool(np.any(collisions)),
            "min_range_m": float(np.min(np.linalg.norm(rel_view[:, 0:3] - goal[0:3], axis=1))),
            "start_fingerprint": "",
            "lighting": "unknown",
            "produced_by": "tests/datasets/conftest.py",
        },
    )


@pytest.fixture(scope="session")
def short_trajectory() -> Trajectory:
    return fly_zero_action(steps=6)
