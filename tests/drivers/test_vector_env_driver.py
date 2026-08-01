import subprocess
import sys

import numpy as np
import pytest

from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver, _lane_info
from owm_envs.envs.iss.config import DockConfig, ISSConfig, PhysicsConfig
from owm_envs.envs.iss.policies import PolicyConfig
from owm_envs.envs.iss.policy_source import ISSPolicySource
from owm_envs.envs.iss.sensing import PRESETS
from owm_envs.envs.iss.vector_env import ISSVectorEnv

FREE_FLIGHT_PHYSICS = dict(collision_boxes_path=None)
FREE_FLIGHT_DOCK = dict(enabled=False)


def free_flight_cfg(physics=None, dock=None) -> ISSConfig:
    return ISSConfig(
        physics=PhysicsConfig(**{**FREE_FLIGHT_PHYSICS, **(physics or {})}),
        dock=DockConfig(**{**FREE_FLIGHT_DOCK, **(dock or {})}),
    )


def make_driver(num_envs=2, policy_type="dock", physics=None, dock=None):
    cfg = free_flight_cfg(physics=physics, dock=dock)
    policy_cfg = PolicyConfig(type=policy_type)
    return VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=num_envs, cfg=cfg),
        policy_source=ISSPolicySource(cfg, policy_cfg),
    )


def test_lane_info_extracts_nested_vector_infos():
    info = {
        "state": np.arange(6).reshape(2, 3),
        "metrics": {"distance": np.array([1.5, 2.5])},
    }
    lane = _lane_info(info, 1)
    np.testing.assert_array_equal(lane["state"], np.array([3, 4, 5]))
    assert lane["metrics"] == {"distance": 2.5}


class _ConstantPolicySource:
    """A trivial PolicySource with no JAX and no ISS types anywhere in it --
    the extensibility claim the driver seam exists to make real."""

    records_policy_ids = False

    def __init__(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=np.float32)

    def new_episode(self, seed: int):
        del seed
        return None

    def act(self, observation, episode_state, step, info):
        del observation, episode_state, step, info
        return self._action.copy()

    def augment_observation(self, observation, episode_state):
        del episode_state
        return observation

    def policy_id(self, episode_state):
        del episode_state
        return 0


def test_generates_the_requested_number_of_episodes():
    batch = make_driver().generate(RolloutSpec(num_episodes=4, max_steps=20, seed=0))
    batch.validate()
    assert batch.num_episodes == 4


def test_output_shapes_and_dtypes():
    # Each episode stores N + 1 observations (seed state plus each post-step
    # state, including the terminal one) against N + 1 actions (N real, one
    # zero pad), so a max_steps=15 horizon yields length-16 episodes.
    batch = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=15, seed=0))
    assert batch.observations.shape == (3, 16, 13)
    assert batch.actions.shape == (3, 16, 6)
    assert batch.rewards.shape == (3, 16)
    assert batch.observations.dtype == np.float32
    assert batch.actions.dtype == np.float32
    assert batch.lengths.dtype == np.int32


def test_free_flight_episodes_run_to_max_steps_and_truncate():
    # 12 real steps plus the seed observation is a length-13 episode.
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))
    assert np.all(batch.lengths == 13)
    assert np.all(batch.truncated)
    assert not np.any(batch.terminated)


def test_collision_terminates_episodes_early():
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    assert not np.any(batch.truncated)
    assert np.all(batch.lengths < 50)


def test_terminal_state_is_stored_with_a_padded_final_action():
    # The terminal (collision) observation must be captured -- it's the
    # event a world model needs to learn -- with a zero action padding the
    # final slot so observations and actions stay equal length.
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=50, seed=0))
    batch.validate()
    assert np.all(batch.terminated)
    for i, length in enumerate(batch.lengths):
        initial = batch.observations[i, 0]
        terminal = batch.observations[i, length - 1]
        assert not np.allclose(terminal, 0.0)
        assert not np.allclose(terminal, initial)
        np.testing.assert_array_equal(batch.actions[i, length - 1], 0.0)


def test_is_deterministic_in_the_seed():
    a = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    b = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=7))
    c = make_driver().generate(RolloutSpec(num_episodes=3, max_steps=10, seed=8))
    np.testing.assert_array_equal(a.observations, b.observations)
    assert not np.allclose(a.observations, c.observations)


def test_padding_past_episode_length_is_zero():
    driver = make_driver(
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}]
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=40, seed=0))
    for i, length in enumerate(batch.lengths):
        assert np.all(batch.observations[i, length:] == 0.0)
        assert np.all(batch.actions[i, length:] == 0.0)


def test_union_policy_records_policy_ids():
    batch = make_driver(policy_type="union").generate(
        RolloutSpec(num_episodes=8, max_steps=10, seed=0)
    )
    assert batch.policy_ids is not None
    assert batch.policy_ids.shape == (8,)
    assert set(np.unique(batch.policy_ids)).issubset({0, 1, 2})


def test_non_mixture_policy_leaves_policy_ids_none():
    batch = make_driver(policy_type="dock").generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.policy_ids is None


def test_min_transitions_mode_is_not_yet_implemented():
    with pytest.raises(NotImplementedError, match="min_transitions"):
        make_driver().generate(RolloutSpec(max_steps=10, seed=0, min_transitions=5))


def test_actions_stay_within_the_control_limits():
    cfg = free_flight_cfg()
    batch = make_driver().generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert np.all(np.abs(batch.actions[..., 0:3]) <= cfg.control.limit_force_n + 1e-3)
    assert np.all(np.abs(batch.actions[..., 3:6]) <= cfg.control.limit_torque_nm + 1e-3)


def test_rejects_a_non_positive_episode_count():
    with pytest.raises(ValueError):
        make_driver().generate(RolloutSpec(num_episodes=0, max_steps=10, seed=0))


def test_episodes_reset_independently_when_max_steps_is_below_the_env_horizon():
    # spec.max_steps (5) is far below the env's own horizon (cfg.max_steps
    # defaults to 2000), and free flight never terminates early, so every
    # episode must be cut short by the driver itself. A single lane makes
    # each episode in `finished` unambiguously that lane's Nth episode, so
    # "does episode N start where episode N-1 ended" is a direct check, not
    # a structural approximation.
    cfg = free_flight_cfg()
    batch = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=1, cfg=cfg),
        policy_source=ISSPolicySource(cfg, PolicyConfig(type="dock")),
    ).generate(RolloutSpec(num_episodes=3, max_steps=5, seed=0))
    batch.validate()
    assert np.all(batch.truncated)

    for i in range(batch.num_episodes):
        # Every reset places the chaser on the start sphere -- a strong,
        # cheap check that this episode really began from env.reset() and
        # not mid-flight.
        radius = np.linalg.norm(batch.observations[i, 0, 0:3])
        np.testing.assert_allclose(radius, cfg.physics.start_radius_m, rtol=1e-4)

    for i in range(1, batch.num_episodes):
        previous_terminal = batch.observations[i - 1, batch.lengths[i - 1] - 1]
        this_initial = batch.observations[i, 0]
        assert not np.allclose(this_initial, previous_terminal)


def test_module_does_not_import_jax():
    # The whole point of the driver seam: a future Basilisk or brahe backend
    # must not have to bring JAX along just to get dataset generation. Check
    # in a subprocess so this is a real assertion about what importing the
    # module does, not just about what happens to already be loaded here.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import owm_envs.drivers.vector_env_driver; print('jax' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr


class _FakeVectorEnv:
    """Minimal Gymnasium-VectorEnv-shaped stand-in that just records whether
    `close()` was called -- no JAX, no ISS type, anywhere in it, so it can't
    trip test_module_does_not_import_jax."""

    num_envs = 1

    class _Space:
        shape = (1,)
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)

    single_observation_space = _Space()
    single_action_space = _Space()

    def __init__(
        self,
        fail_on_step: bool = False,
        fail_on_close: bool = False,
        step_exception: BaseException | None = None,
    ):
        self._fail_on_step = fail_on_step
        self._fail_on_close = fail_on_close
        self._step_exception = step_exception
        self.closed = False
        self.close_calls = 0

    def reset(self, seed=None):
        del seed
        return np.zeros((self.num_envs, 1), dtype=np.float32), {}

    def step(self, actions):
        del actions
        if self._step_exception is not None:
            raise self._step_exception
        if self._fail_on_step:
            raise RuntimeError("boom")
        obs = np.zeros((self.num_envs, 1), dtype=np.float32)
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminations = np.zeros((self.num_envs,), dtype=bool)
        truncations = np.ones((self.num_envs,), dtype=bool)
        return obs, rewards, terminations, truncations, {}

    def close(self):
        self.close_calls += 1
        self.closed = True
        if self._fail_on_close:
            raise RuntimeError("close failed")


def test_env_is_closed_after_a_successful_generate():
    env = _FakeVectorEnv()
    driver = VectorEnvDriver(
        env_factory=lambda: env, policy_source=_ConstantPolicySource(np.array([0.0]))
    )
    driver.generate(RolloutSpec(num_episodes=1, max_steps=1, seed=0))
    assert env.closed is True
    assert env.close_calls == 1


def test_env_is_closed_when_the_rollout_raises():
    env = _FakeVectorEnv(fail_on_step=True)
    driver = VectorEnvDriver(
        env_factory=lambda: env, policy_source=_ConstantPolicySource(np.array([0.0]))
    )
    with pytest.raises(RuntimeError, match="boom"):
        driver.generate(RolloutSpec(num_episodes=1, max_steps=1, seed=0))
    assert env.closed is True
    assert env.close_calls == 1


def test_a_close_failure_does_not_mask_the_original_rollout_error():
    # The rollout error is the one the caller needs -- a close() failure
    # while unwinding from it must not replace or hide it.
    env = _FakeVectorEnv(fail_on_step=True, fail_on_close=True)
    driver = VectorEnvDriver(
        env_factory=lambda: env, policy_source=_ConstantPolicySource(np.array([0.0]))
    )
    with pytest.raises(RuntimeError, match="boom"):
        driver.generate(RolloutSpec(num_episodes=1, max_steps=1, seed=0))
    assert env.close_calls == 1


class _DirectBaseException(BaseException):
    """Doesn't subclass Exception -- guards against the cleanup handler
    regressing from `except BaseException` to `except Exception`, which
    would let a close() failure mask this instead of it propagating."""


def test_a_close_failure_does_not_mask_a_non_exception_rollout_error():
    env = _FakeVectorEnv(step_exception=_DirectBaseException("interrupted"), fail_on_close=True)
    driver = VectorEnvDriver(
        env_factory=lambda: env, policy_source=_ConstantPolicySource(np.array([0.0]))
    )
    with pytest.raises(_DirectBaseException, match="interrupted"):
        driver.generate(RolloutSpec(num_episodes=1, max_steps=1, seed=0))
    assert env.close_calls == 1


def test_a_close_failure_after_success_propagates():
    # With nothing to mask, a close() failure on the success path is a real
    # error and must not be silently swallowed.
    env = _FakeVectorEnv(fail_on_close=True)
    driver = VectorEnvDriver(
        env_factory=lambda: env, policy_source=_ConstantPolicySource(np.array([0.0]))
    )
    with pytest.raises(RuntimeError, match="close failed"):
        driver.generate(RolloutSpec(num_episodes=1, max_steps=1, seed=0))
    assert env.close_calls == 1


def test_driver_works_with_a_non_iss_non_jax_policy_source():
    cfg = free_flight_cfg()
    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    driver = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
        policy_source=_ConstantPolicySource(action),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=5, seed=0))
    batch.validate()
    assert batch.num_episodes == 2
    assert batch.policy_ids is None
    for i, length in enumerate(batch.lengths):
        # Every real action (all but the final zero-padded slot) is the
        # constant action the fake policy source always returns.
        np.testing.assert_array_equal(
            batch.actions[i, : length - 1], np.tile(action, (length - 1, 1))
        )


class _AutoresetFakeVectorEnv:
    """Two-lane fake env that implements Gymnasium NEXT_STEP autoreset
    faithfully: whichever lane terminated on the previous step() call gets
    its submitted action ignored on the next one, and that call's next_obs
    for the lane is a fresh reset observation instead. Observations encode
    (lane, episode index, step-within-episode) as a single float so a test
    can decode, from the observation alone and with no lane argument on
    `act()`, exactly which lane/episode/step a policy call was for. No JAX,
    no ISS type, anywhere in it -- this must stay importable by
    test_module_does_not_import_jax."""

    num_envs = 2

    class _Space:
        shape = (1,)
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)

    single_observation_space = _Space()
    single_action_space = _Space()

    def __init__(self, terminate_every: list[int]):
        self._terminate_every = terminate_every
        self._episode_id = [0] * self.num_envs
        self._step_in_episode = [0] * self.num_envs
        self._pending_reset = [False] * self.num_envs
        self.step_calls = 0
        # pending_history[i]: the pending_reset flags as they stood right
        # before the i-th call to step() -- i.e. for each lane, whether the
        # action submitted to that call was doomed to be discarded by
        # autoreset.
        self.pending_history: list[list[bool]] = []

    def _obs(self):
        return np.array(
            [
                [lane * 100_000 + self._episode_id[lane] * 100 + self._step_in_episode[lane]]
                for lane in range(self.num_envs)
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None):
        del seed
        self._step_in_episode = [0] * self.num_envs
        self._pending_reset = [False] * self.num_envs
        return self._obs(), {}

    def step(self, actions):
        del actions
        self.pending_history.append(list(self._pending_reset))
        self.step_calls += 1
        terminations = np.zeros((self.num_envs,), dtype=bool)
        truncations = np.zeros((self.num_envs,), dtype=bool)
        for lane in range(self.num_envs):
            if self._pending_reset[lane]:
                # The autoreset step: ignore the submitted action, hand back
                # a fresh episode's start state.
                self._pending_reset[lane] = False
                self._episode_id[lane] += 1
                self._step_in_episode[lane] = 0
                continue
            self._step_in_episode[lane] += 1
            if self._step_in_episode[lane] >= self._terminate_every[lane]:
                terminations[lane] = True
                self._pending_reset[lane] = True
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        return self._obs(), rewards, terminations, truncations, {}

    def close(self):
        pass


class _CountingPolicySource:
    """Records every `act()` call as (iteration, lane, episode_id,
    step_in_episode), decoded from the observation encoding above -- no
    lane argument exists on `act()`, so this is the only way to see which
    lane a call was really for."""

    records_policy_ids = False

    def __init__(self, env: _AutoresetFakeVectorEnv):
        self._env = env
        self.calls: list[tuple[int, int, int, int]] = []

    def new_episode(self, seed):
        del seed
        return None

    def act(self, observation, episode_state, step, info):
        del episode_state, step, info
        # `env.step_calls` at call time is exactly the index this call's
        # action will target when step() next runs -- pending_history will
        # be appended at that same index.
        code = int(observation[0])
        lane, remainder = divmod(code, 100_000)
        episode_id, step_in_episode = divmod(remainder, 100)
        self.calls.append((self._env.step_calls, lane, episode_id, step_in_episode))
        return np.zeros((1,), dtype=np.float32)

    def augment_observation(self, observation, episode_state):
        del episode_state
        return observation

    def policy_id(self, episode_state):
        del episode_state
        return 0


def test_policy_is_never_invoked_for_a_lane_during_its_autoreset_step():
    # Under NEXT_STEP autoreset, the step() call following a lane's
    # termination discards whatever action was submitted for it -- so
    # calling policy_source.act() for that lane is pointless at best. It's
    # actively wrong for a stateful policy source
    # (an OU noise process, an RNN hidden state, anything with an internal
    # counter): that call would consume or mutate state on behalf of the
    # lane's *next* episode using its terminal observation from the
    # *previous* one. A pure output-array check can't see this -- the
    # discarded action never reaches the recorded trajectory -- so this
    # asserts directly on the call pattern instead.
    env = _AutoresetFakeVectorEnv(terminate_every=[2, 3])
    policy_source = _CountingPolicySource(env)
    driver = VectorEnvDriver(env_factory=lambda: env, policy_source=policy_source)

    batch = driver.generate(RolloutSpec(num_episodes=6, max_steps=100, seed=0))
    batch.validate()

    assert policy_source.calls, "sanity check: the policy source was never invoked"
    for iteration, lane, episode_id, step_in_episode in policy_source.calls:
        assert not env.pending_history[iteration][lane], (
            f"act() was called for lane {lane} (episode {episode_id}, "
            f"step {step_in_episode}) on iteration {iteration}, which was "
            f"that lane's autoreset step -- the action is discarded and "
            f"the call used the wrong episode's state"
        )


def test_vector_driver_state_policy_actions_match_clean_run():
    # observe="state" with noise on: actions must be computed from the true
    # state, so a noisy run's actions match a clean run's actions per episode.
    cfg_noisy = ISSConfig(max_steps=12, sensor_noise=PRESETS["cooperative"])
    cfg_clean = ISSConfig(max_steps=12)

    def batch(cfg):
        driver = VectorEnvDriver(
            env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
            policy_source=ISSPolicySource(cfg, PolicyConfig(type="dock", observe="state")),
        )
        return driver.generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))

    a, b = batch(cfg_clean), batch(cfg_noisy)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert not np.array_equal(a.observations, b.observations)


def test_vector_driver_measurement_policy_consumes_the_observation():
    cfg = ISSConfig(max_steps=12, sensor_noise=PRESETS["noncooperative"])

    def batch(observe):
        driver = VectorEnvDriver(
            env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
            policy_source=ISSPolicySource(cfg, PolicyConfig(type="dock", observe=observe)),
        )
        return driver.generate(RolloutSpec(num_episodes=2, max_steps=12, seed=0))

    assert not np.array_equal(batch("state").actions, batch("measurement").actions)


def test_vector_state_policy_survives_autoreset_with_noise():
    # 4 episodes over 2 lanes forces autoreset; random policy consumes act keys.
    clean_cfg = ISSConfig(max_steps=10)
    noisy_cfg = ISSConfig(max_steps=10, sensor_noise=PRESETS["cooperative"])

    def batch(cfg):
        driver = VectorEnvDriver(
            env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
            policy_source=ISSPolicySource(cfg, PolicyConfig(type="random", observe="state")),
        )
        return driver.generate(RolloutSpec(num_episodes=4, max_steps=10, seed=0))

    clean, noisy = batch(clean_cfg), batch(noisy_cfg)
    np.testing.assert_array_equal(clean.actions, noisy.actions)
    np.testing.assert_array_equal(clean.lengths, noisy.lengths)
    assert not np.array_equal(clean.observations, noisy.observations)
