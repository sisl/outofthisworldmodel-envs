"""Environment registration for owm_envs."""

from gymnasium.envs.registration import register

register(
    id="ISS-Docking-v0",
    entry_point="owm_envs.envs.iss.env:ISSEnv",
)
