"""Keep the whole test suite off the network, and off the GPU.

Scene construction resolves Earth textures with `allow_download=True`, so on
a machine without the full-resolution maps on disk an unguarded run would
pull them from the asset dataset on the Hub. Any test that builds a scene
reaches that path -- `tests/envs/iss` constructs render-mode environments
too -- so the guard lives at the root rather than beside the render tests.
Every test must be satisfiable by the committed fallback maps.

The suite computes on the CPU backend for two reasons. The golden rollouts
assert bit identity, and a backend is free to lower the same arithmetic
differently, so a GPU host would compare this package's results against
another backend's rounding rather than against a change in this package --
which is the only thing those tests exist to catch. And a test process that
reaches a GPU takes 75% of it, which on a shared host is a card taken from
whoever else is on it for the length of a run.

Set before anything imports jax, because the choice is read when the backend
initialises, which happens during import -- see `owm_envs._entry` for the same
constraint on the CLI side. `setdefault`, so a run that means to exercise the
GPU path can still say `JAX_PLATFORMS=` and be believed.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from gymnasium.envs.registration import load_env_creator  # noqa: E402

from owm_envs.datasets.trajectory import Trajectory, start_fingerprint  # noqa: E402
from owm_envs.envs import ENV_REGISTRY  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(**kwargs):
        raise OSError(f"tests must not download: {kwargs.get('filename')}")

    monkeypatch.setattr("owm_envs.render.asset_hub.hf_hub_download", refuse)


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
    states = [np.asarray(info["state"], dtype=np.float64)]
    measured = [np.asarray(info["measured_state"], dtype=np.float64)]
    observations = [np.asarray(obs, dtype=np.float32)]
    actions, rewards, collisions = [], [], []
    for _ in range(steps):
        action = np.zeros(6, dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        states.append(np.asarray(info["state"], dtype=np.float64))
        measured.append(np.asarray(info["measured_state"], dtype=np.float64))
        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        collisions.append(bool(info.get("collision", False)))
        if term or trunc:
            break
    env.close()
    state = np.stack(states)
    rel_view = np.asarray(jax.vmap(spec.view)(jnp.asarray(state, jnp.float64)),
                          dtype=np.float64)
    action_norm = np.stack(actions)
    goal = np.asarray(info["goal_pose"], dtype=np.float64)
    return Trajectory(
        epoch=state[:, 0:2],
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
            "start_fingerprint": start_fingerprint(state[0]),
            "lighting": "unknown",
            "produced_by": "tests/conftest.py",
        },
    )


@pytest.fixture(scope="session")
def short_trajectory() -> Trajectory:
    return fly_zero_action(steps=6)
