import subprocess
import sys

import numpy as np
import pytest

from owm_envs.drivers.types import RolloutSpec
from owm_envs.drivers.vector_env_driver import VectorEnvDriver, _lane_info
from owm_envs.envs.common.config import DockConfig, PhysicsConfig, dock_target
from owm_envs.envs.common.docking_ports import dock_targets
from owm_envs.envs.common.policies import DockParams, PolicyConfig
from owm_envs.envs.common.policy_source import ISSPolicySource
from owm_envs.envs.common.sensing import PRESETS
from owm_envs.envs.iss.config import ISSConfig
from owm_envs.envs.iss.vector_env import ISSVectorEnv

FREE_FLIGHT_PHYSICS = dict(collision_boxes_path=None)
FREE_FLIGHT_DOCK = dict(enabled=False)


def free_flight_cfg(physics=None, dock=None) -> ISSConfig:
    return ISSConfig(
        physics=PhysicsConfig(**{**FREE_FLIGHT_PHYSICS, **(physics or {})}),
        dock=DockConfig(**{**FREE_FLIGHT_DOCK, **(dock or {})}),
    )


def make_driver(num_envs=2, policy_type="dock", physics=None, dock=None, ports=()):
    cfg = free_flight_cfg(physics=physics, dock=dock)
    policy_cfg = PolicyConfig(type=policy_type, dock=DockParams(ports=ports))
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
    records_dock_targets = False

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

    def dock_target(self, episode_state):
        del episode_state
        return np.zeros(7, dtype=np.float32)


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
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
            start_radius_range_m=(100.0, 100.0),
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
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
            start_radius_range_m=(100.0, 100.0),
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
            collision_boxes_path=[{"center": [0.0, 0.0, 0.0], "size": [400.0, 400.0, 400.0]}],
            start_radius_range_m=(100.0, 100.0),
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


def test_port_set_records_the_assigned_dock_target_per_episode():
    batch = make_driver(policy_type="dock", ports=("all",)).generate(
        RolloutSpec(num_episodes=8, max_steps=10, seed=0)
    )
    assert batch.dock_targets is not None
    assert batch.dock_targets.shape == (8, 7)
    table = dock_targets(("all",))
    for episode in range(batch.num_episodes):
        matches = np.isclose(table, batch.dock_targets[episode], atol=1e-5).all(axis=1)
        assert matches.sum() == 1, batch.dock_targets[episode]


def test_no_port_set_records_the_config_dock_row():
    cfg = free_flight_cfg()
    batch = make_driver(policy_type="dock").generate(RolloutSpec(num_episodes=2, max_steps=10, seed=0))
    assert batch.dock_targets is not None
    for episode in range(batch.num_episodes):
        np.testing.assert_allclose(batch.dock_targets[episode], dock_target(cfg), atol=1e-6)


def test_vector_transitions_target_is_met_with_whole_episodes():
    cfg = ISSConfig(max_steps=10)
    driver = VectorEnvDriver(
        env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
        policy_source=ISSPolicySource(cfg, PolicyConfig(type="dock")),
    )
    batch = driver.generate(RolloutSpec(max_steps=10, seed=0, min_transitions=50))
    assert batch.total_transitions >= 50
    # first-crossing: dropping the last episode must dip below the target
    assert batch.total_transitions - (int(batch.lengths[-1]) - 1) < 50


def test_vector_transitions_mode_is_deterministic():
    cfg = ISSConfig(max_steps=10)

    def batch():
        driver = VectorEnvDriver(
            env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
            policy_source=ISSPolicySource(cfg, PolicyConfig(type="dock")),
        )
        return driver.generate(RolloutSpec(max_steps=10, seed=0, min_transitions=50))

    a, b = batch(), batch()
    np.testing.assert_array_equal(a.observations, b.observations)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.lengths, b.lengths)


def test_vector_transitions_mode_is_independent_of_episodes_mode_at_the_same_seed():
    # numpy's default_rng([seed, 0]) is byte-identical to default_rng(seed),
    # so without a salt, transitions-mode would draw the exact same stream as
    # episodes-mode at the same spec.seed -- a transitions-mode dataset
    # silently containing the episodes-mode dataset as its exact prefix.
    cfg = ISSConfig(max_steps=10)

    def make():
        return VectorEnvDriver(
            env_factory=lambda: ISSVectorEnv(num_envs=2, cfg=cfg),
            policy_source=ISSPolicySource(cfg, PolicyConfig(type="dock")),
        )

    episodes_batch = make().generate(RolloutSpec(max_steps=10, seed=0, num_episodes=2))
    transitions_batch = make().generate(RolloutSpec(max_steps=10, seed=0, min_transitions=1))
    first_len_a = int(episodes_batch.lengths[0])
    first_len_b = int(transitions_batch.lengths[0])
    obs_a = episodes_batch.observations[0, :first_len_a]
    obs_b = transitions_batch.observations[0, :first_len_b]
    assert not (
        obs_a.shape == obs_b.shape and np.array_equal(obs_a, obs_b)
    )


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
    # defaults to 7200), and free flight never terminates early, so every
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
        # Every reset places the chaser inside the start shell -- a strong,
        # cheap check that this episode really began from env.reset() and
        # not mid-flight.
        low, high = cfg.physics.start_radius_range_m
        radius = np.linalg.norm(batch.observations[i, 0, 0:3])
        assert low * (1 - 1e-4) <= radius <= high * (1 + 1e-4)

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


def test_truth_is_none_for_a_backend_that_supplies_no_state():
    # A foreign Gymnasium backend has no reason to publish the true dynamics
    # state in its info dict; the driver must record no truth channel rather
    # than fail or invent one.
    driver = VectorEnvDriver(
        env_factory=_FakeVectorEnv, policy_source=_ConstantPolicySource(np.array([0.0]))
    )
    batch = driver.generate(RolloutSpec(num_episodes=1, max_steps=1, seed=0))
    batch.validate()
    assert batch.true_state is None


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
    records_dock_targets = False

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

    def dock_target(self, episode_state):
        del episode_state
        return np.zeros(7, dtype=np.float32)


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


class _CodedStateVectorEnv:
    """Two-lane fake env whose observation AND info["state"] both encode
    (lane, episode, step-within-episode), truth being the observation plus a
    fixed offset -- so every recorded truth row can be matched to the exact
    observation it must sit against, with no physics involved.

    Lane 0 terminates every `terminate_every` steps while lane 1 never does,
    so a rollout horizon between the two desynchronizes the cohort: lane 1
    freezes at the horizon while lane 0 keeps cycling through env-driven
    autoresets, and the two then go through the same whole-vector reset from
    different states. No JAX, no ISS type -- this must stay importable by
    test_module_does_not_import_jax.
    """

    num_envs = 2
    state_offset = 0.5

    class _ActionSpace:
        shape = (1,)
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)

    single_action_space = _ActionSpace()

    def __init__(self, terminate_every: int = 3):
        self._terminate_every = terminate_every
        self._episode = [0] * self.num_envs
        self._step = [0] * self.num_envs
        self._pending_reset = [False] * self.num_envs

    def _obs_and_info(self):
        codes = np.array(
            [
                [lane * 100_000 + self._episode[lane] * 100 + self._step[lane]] * 13
                for lane in range(self.num_envs)
            ],
            dtype=np.float32,
        )
        return codes, {"state": codes + self.state_offset}

    def reset(self, seed=None):
        del seed
        for lane in range(self.num_envs):
            self._episode[lane] += 1
            self._step[lane] = 0
        self._pending_reset = [False] * self.num_envs
        return self._obs_and_info()

    def step(self, actions):
        del actions
        terminations = np.zeros((self.num_envs,), dtype=bool)
        for lane in range(self.num_envs):
            if self._pending_reset[lane]:
                self._pending_reset[lane] = False
                self._episode[lane] += 1
                self._step[lane] = 0
                continue
            self._step[lane] += 1
            if lane == 0 and self._step[lane] >= self._terminate_every:
                terminations[lane] = True
                self._pending_reset[lane] = True
        obs, info = self._obs_and_info()
        return (
            obs,
            np.zeros((self.num_envs,), dtype=np.float32),
            terminations,
            np.zeros((self.num_envs,), dtype=bool),
            info,
        )

    def close(self):
        pass


def test_truth_stays_aligned_when_lanes_desynchronize():
    # The one episode-boundary path the ISS truth tests cannot reach: a cohort
    # where one lane sits FROZEN at the rollout horizon while the other is
    # still finishing env-terminated episodes, both then going through the
    # same whole-vector reset. A truth entry left stale on the frozen lane, or
    # rebuilt for only one of them, shows up here and nowhere else.
    env = _CodedStateVectorEnv(terminate_every=3)
    driver = VectorEnvDriver(
        env_factory=lambda: env, policy_source=_ConstantPolicySource(np.zeros(1))
    )
    batch = driver.generate(RolloutSpec(num_episodes=5, max_steps=5, seed=0))
    batch.validate()

    # Lane 0's episodes are 3 steps (4 observations), lane 1's run to the
    # 5-step horizon (6) -- proof the cohort really did desynchronize rather
    # than the lanes staying in lockstep, in which case this test would be
    # checking nothing the others don't.
    assert sorted(batch.lengths.tolist()) == [4, 4, 4, 6, 6]

    assert batch.true_state is not None
    for i in range(batch.num_episodes):
        length = int(batch.lengths[i])
        np.testing.assert_array_equal(
            batch.true_state[i, :length],
            batch.observations[i, :length] + _CodedStateVectorEnv.state_offset,
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


def test_num_episodes_mode_keeps_lane_quota_not_fastest_finishers():
    # A box over the upper half-space makes lanes that start with z > 0
    # collide on their first step, while lanes starting below truncate at
    # max_steps. With spec seed 4, lanes 0-2 start below (slow) and lane 3
    # above (fast, recycling a new episode every couple of steps).
    # Keeping the first N episodes to finish would return lane 3's
    # collisions and silently skew any policy mixture toward whatever
    # terminates fastest; num_episodes mode must instead keep each lane's
    # first episodes by deterministic quota -- here lanes 0 and 1 --
    # regardless of how long they take.
    driver = make_driver(
        num_envs=4,
        physics=dict(
            collision_boxes_path=[{"center": [0.0, 0.0, 150.0], "size": [400.0, 400.0, 300.0]}],
            start_radius_range_m=(100.0, 100.0),
        ),
        dock=dict(enabled=False),
    )
    batch = driver.generate(RolloutSpec(num_episodes=2, max_steps=5, seed=4))
    batch.validate()
    assert batch.lengths.tolist() == [6, 6]
    assert not np.any(batch.terminated)
    assert np.all(batch.truncated)
    # Lane-major order: episode 0 is lane 0's start, episode 1 is lane 1's.
    np.testing.assert_allclose(batch.observations[0, 0, 2], -85.0, atol=1.0)
    np.testing.assert_allclose(batch.observations[1, 0, 2], -36.0, atol=1.0)
