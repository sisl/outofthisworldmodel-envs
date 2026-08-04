"""Scripted JAX policies for ISS dataset generation.

Each builder returns (policy_fn, extras_fn):
  policy_fn(state, key, extras) -> action (6,)
  extras_fn(key)                -> per-episode hyperparameters, sampled at reset
extras_fn is None when the policy needs no per-episode randomisation.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field, field_validator

from ...core.models import ConfigModel
from ...core.quaternion import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_body_z_to,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
)
from .config import ISSConfig, dock_target
from .docking_ports import PORTS_BY_NAME, port_pose, resolve_port_names

# How far a pinned pose may sit from the one the current PORTS table derives
# before the config is rejected. Poses are metres and unit quaternions, and
# both sides come from the same `port_pose` computation, so anything above
# float round-off means the table actually moved.
PINNED_POSE_TOLERANCE = 1e-5

PolicyFn = Callable[[jnp.ndarray, jax.Array, jnp.ndarray], jnp.ndarray]
ExtrasFn = Callable[[jax.Array], jnp.ndarray]

# Width of each policy's extras vector. The rollout driver in plan 2 needs this
# to allocate the per-env extras buffer before the first reset.
#
# The dock slot holds the index of the port this episode is flying to, sampled
# at reset. It is present even for a single-port config so that the layout does
# not depend on how many ports were configured.
EXTRAS_DIM: dict[str, int] = {"random": 0, "orbit": 5, "dock": 1, "union": 7}

# Where the dock port index sits in each policy's extras vector.
DOCK_SLOT: dict[str, int] = {"dock": 0, "union": 6}

# Physically meaningful span for a commanded orbit radius around the station:
# millimetres to 1000 km. See OrbitParams._validate_radius_range_m.
_MIN_RADIUS_M = 1e-3
_MAX_RADIUS_M = 1e6


class OrbitParams(ConfigModel):
    radius_range_m: tuple[float, float] = (60.0, 500.0)
    # Rate is a fraction of the fastest orbit the thrusters can actually hold
    # at the sampled radius, not an absolute rad/s. Holding radius R at rate w
    # needs a sustained centripetal force m*w^2*R, so the feasible rate falls
    # as 1/sqrt(R) and no single absolute range serves 60-500 m: a floor low
    # enough for 500 m makes 60 m orbits crawl, and one fast enough for 60 m is
    # unreachable past ~111 m. Expressing it as a fraction makes feasibility
    # intrinsic -- further-out orbits are automatically slower.
    speed_fraction_range: tuple[float, float] = (0.4, 1.0)
    # Fraction of the per-axis force limit committed to that centripetal
    # force. The remainder is the PD's headroom to correct errors with; at 1.0
    # the whole budget goes to holding the circle and the fly-in transient
    # saturates.
    thrust_utilization: float = Field(default=0.6, gt=0.0, le=1.0)
    kp_position: float = 1080.0
    kd_velocity: float = 1500.0
    kp_attitude: float = 54_000.0
    kd_attitude: float = 47_000.0

    @field_validator("speed_fraction_range")
    @classmethod
    def _validate_speed_fraction_range(
        cls, v: tuple[float, float]
    ) -> tuple[float, float]:
        lo, hi = v
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError(
                f"speed_fraction_range must satisfy 0 < lo <= hi <= 1, got {v}"
            )
        return v

    @field_validator("radius_range_m")
    @classmethod
    def _validate_radius_range_m(cls, v: tuple[float, float]) -> tuple[float, float]:
        # The radius divides the rate bound, so a zero or negative value would
        # yield an infinite or NaN omega rather than a load-time error.
        lo, hi = v
        if not 0.0 < lo <= hi or not math.isfinite(hi):
            raise ValueError(
                f"radius_range_m must satisfy 0 < lo <= hi < inf, got {v}"
            )
        # Bounded to a physically meaningful span rather than to float32's
        # representable range: 1 mm to 1000 km around the station brackets
        # anything relative proximity operations can mean, and keeps every
        # intermediate in the rate bound far from overflow. Defending the
        # arithmetic against a 1e38 m orbit instead would be defending a
        # scenario 20 orders of magnitude past the observable universe.
        if not _MIN_RADIUS_M <= lo <= hi <= _MAX_RADIUS_M:
            raise ValueError(
                f"radius_range_m must lie within "
                f"[{_MIN_RADIUS_M:g}, {_MAX_RADIUS_M:g}] m, got {v}"
            )
        return v


class DockPort(ConfigModel):
    """A port a dock episode may be assigned, carrying the pose it resolves to.

    The pose is stored, not just the name, so a versioned or as-run config
    reproduces the run it describes even after the `PORTS` table is revised.
    A name still resolves against the table at load (see
    `DockParams._resolve_ports`); this is what that resolution produces.
    """

    name: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


def _pinned_from_table(name: str) -> DockPort:
    position, quaternion = port_pose(PORTS_BY_NAME[name])
    return DockPort(
        name=name,
        position=tuple(float(v) for v in position),
        quaternion=tuple(float(v) for v in quaternion),
    )


def _verify_against_table(port: DockPort) -> None:
    """Reject a pinned pose that disagrees with the one `PORTS` derives today.

    A name the table no longer knows is left alone: its pinned pose governs,
    which is how a config outlives a table revision. But a name the table
    still knows and now places somewhere else is a genuine conflict between
    two claims about the same port, and silently preferring either one would
    make the config lie about what it ran.
    """
    expected = _pinned_from_table(port.name)
    for field, pinned, table in (
        ("position", port.position, expected.position),
        ("quaternion", port.quaternion, expected.quaternion),
    ):
        if max(abs(a - b) for a, b in zip(pinned, table)) > PINNED_POSE_TOLERANCE:
            raise ValueError(
                f"pinned {field} for docking port '{port.name}' disagrees with the "
                f"current port table: config has {tuple(pinned)}, table derives "
                f"{tuple(table)}. Re-pin the config against the current table, or "
                f"rename the entry if it is meant to outlive the table's version."
            )


class DockParams(ConfigModel):
    kp_position: float = 1080.0
    kd_velocity: float = 7200.0
    kp_attitude: float = 54_000.0
    kd_attitude: float = 132_000.0
    # Ports an episode may be assigned, drawn uniformly at reset. Empty keeps
    # the single pose in ISSConfig.dock, which is what a config that predates
    # multi-port support means.
    #
    # An entry may be written as a bare port name or as a pinned
    # {name, position, quaternion}; both normalise to the pinned form at load,
    # so this is always a tuple of DockPort afterwards and every serialised
    # config carries the poses it used. The keyword "all" expands to every
    # port in docking_ports.PORTS, here at config-load time, so the as-run
    # record names the ports a run actually used rather than a keyword whose
    # meaning could change with the table.
    ports: tuple[DockPort, ...] = ()

    @field_validator("ports", mode="before")
    @classmethod
    def _resolve_ports(cls, v: object) -> object:
        if not v:
            return ()
        entries = list(v)
        if all(isinstance(entry, str) for entry in entries):
            # Unchanged path for a name-only config: `resolve_port_names` owns
            # the "all" expansion and the unknown-name and duplicate errors,
            # exactly as before.
            return tuple(_pinned_from_table(name) for name in resolve_port_names(tuple(entries)))

        resolved: list[DockPort] = []
        for entry in entries:
            if isinstance(entry, str):
                if entry == "all":
                    raise ValueError(
                        "the keyword 'all' stands for the whole port list and cannot "
                        "be mixed with other entries"
                    )
                resolve_port_names((entry,))
                resolved.append(_pinned_from_table(entry))
                continue
            port = entry if isinstance(entry, DockPort) else DockPort.model_validate(entry)
            if port.name in PORTS_BY_NAME:
                _verify_against_table(port)
            resolved.append(port)

        names = [port.name for port in resolved]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate docking port(s) {duplicates}; name each port at most once"
            )
        return tuple(resolved)


class PolicyConfig(ConfigModel):
    """Policy settings. A ConfigModel like ISSConfig so the policy that shaped
    a dataset is part of its as-run record, not just the environment physics."""

    # Which policy make_policy builds. Defaults to "random". Invalid values
    # are rejected here, at config-load time, rather than inside make_policy
    # at rollout time.
    type: Literal["random", "orbit", "dock", "union"] = "random"
    # What the scripted policies consume during data generation. Defaults to
    # "measurement": policies act on the same noisy measurement the dataset
    # records, so the recorded action-outcome pairs carry the aleatoric
    # uncertainty of acting on an observed rather than true state -- the
    # uncertainty a world model should learn. "state" gives noise-independent
    # trajectories, for controlled ablations where clean/noisy runs must
    # share trajectories.
    observe: Literal["state", "measurement"] = "measurement"
    orbit: OrbitParams = Field(default_factory=OrbitParams)
    dock: DockParams = Field(default_factory=DockParams)
    # Positional order: [random, orbit, dock]. Normalised at build time.
    union_weights: tuple[float, float, float] = (0.3, 0.35, 0.35)

    @field_validator("union_weights")
    @classmethod
    def _validate_union_weights(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        for w in v:
            if w < 0.0:
                raise ValueError(f"union_weights must be non-negative, got {w}")
        if sum(v) <= 0.0:
            raise ValueError(f"union_weights must sum to > 0, got {v}")
        return v


def _safe_norm(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.maximum(jnp.linalg.norm(v), eps)


def _attitude_torque(
    q_bw: jnp.ndarray,
    q_target: jnp.ndarray,
    omega_b: jnp.ndarray,
    *,
    inertia_diag: jnp.ndarray,
    kp_attitude: float,
    kd_attitude: float,
) -> jnp.ndarray:
    q_err = quat_multiply(quat_conjugate(q_bw), q_target)
    att_err_body = axis_angle_from_quat(q_err)
    return (
        inertia_diag
        * (kp_attitude * att_err_body - kd_attitude * omega_b)
        / jnp.max(inertia_diag)
    )


def _build_random(cfg: ISSConfig) -> PolicyFn:
    low = jnp.array(
        [-cfg.control.limit_force_n] * 3 + [-cfg.control.limit_torque_nm] * 3,
        dtype=jnp.float32,
    )
    high = -low

    def policy_fn(state, key, extras):
        del state, extras
        return jax.random.uniform(key, shape=(6,), minval=low, maxval=high)

    return policy_fn


def orbit_reference(pos_w, orbit_extras, dt):
    """Commanded orbit reference ONE TIMESTEP AHEAD of the current projection.

    Both the orbit control law (`_build_orbit`'s policy_fn) and the
    goal-error block (`goal._orbit_goal_error`) target this same state, so
    the recorded goal is exactly what the controller chases. Returns
    (p_des, v_des, q_des): position on the commanded circle after advancing
    the current planar projection by omega*dt about the axis, the
    tangential velocity there, and the center-pointing attitude there.

    `orbit_extras` layout: (axis_world[3], radius, omega) -- 5 floats.
    """
    axis = orbit_extras[0:3]
    radius, omega = orbit_extras[3], orbit_extras[4]
    pos_planar = pos_w - jnp.dot(pos_w, axis) * axis
    r_hat = pos_planar / _safe_norm(pos_planar)
    # Advance r_hat by the commanded motion (Rodrigues rotation about `axis`)
    # so the target is the NEXT reference state, one dt ahead.
    angle = omega * dt
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    r_hat_next = (r_hat * cos_a + jnp.cross(axis, r_hat) * sin_a
                  + axis * jnp.dot(axis, r_hat) * (1.0 - cos_a))
    r_hat_next = r_hat_next / _safe_norm(r_hat_next)
    p_des = radius * r_hat_next
    v_des = omega * radius * jnp.cross(axis, r_hat_next)
    q_des = quat_from_body_z_to(-r_hat_next)
    return p_des, v_des, q_des


def _build_orbit(cfg: ISSConfig, params: OrbitParams) -> tuple[PolicyFn, ExtrasFn]:
    # This "orbit" is not a passively stable relative orbit -- it is a
    # circular trajectory at a commanded angular speed around the target,
    # held by continuous control. That's useful for training precisely
    # because it forces trajectories that use control actuation to maintain
    # both centre-pointing attitude and speed, exercising more of the action
    # space and showing the consequences of actions, which helps a policy
    # learn what its actions do.
    inertia_diag = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float32)
    radius_range = jnp.asarray(params.radius_range_m, dtype=jnp.float32)
    fraction_range = jnp.asarray(params.speed_fraction_range, dtype=jnp.float32)
    mass = jnp.asarray(cfg.physics.mass, dtype=jnp.float32)
    damping = jnp.asarray(cfg.physics.linear_damping, dtype=jnp.float32)
    # Largest steady-state force the sampler may commit to holding the circle.
    force_budget = jnp.asarray(
        params.thrust_utilization * cfg.control.limit_force_n, dtype=jnp.float32
    )

    def extras_fn(key: jax.Array) -> jnp.ndarray:
        key_axis, key_radius, key_fraction = jax.random.split(key, 3)
        axis = jax.random.normal(key_axis, (3,), dtype=jnp.float32)
        axis = axis / _safe_norm(axis)
        radius = jax.random.uniform(
            key_radius, (), minval=radius_range[0], maxval=radius_range[1], dtype=jnp.float32
        )
        fraction = jax.random.uniform(
            key_fraction, (), minval=fraction_range[0], maxval=fraction_range[1],
            dtype=jnp.float32,
        )
        # Holding the circle costs m*w^2*R radially (centripetal) and, when
        # linear damping is on, m*c*w*R tangentially to cancel the drag. The
        # two are perpendicular, so the steady-state force magnitude is
        #   |F|^2 = (m*w^2*R)^2 + (m*c*w*R)^2 = (m*R)^2 * w^2 * (w^2 + c^2).
        # Bounding that by force_budget leaves a quadratic in w^2, whose
        # positive root is taken here as
        #   w_max = q * sqrt(2 / (c^2 + hypot(c^2, 2q))),  q = budget/(m*R),
        # rather than the algebraically equal sqrt((-c^2 + sqrt(c^4+4q^2))/2).
        # That form subtracts near-equal quantities once c^4 >> 4q^2 and
        # rounds a positive feasible rate to zero in float32; this one only
        # ever adds. Factoring q out and using hypot also keeps every
        # intermediate in range -- neither q^2 nor c^4 is ever formed, so no
        # accepted radius or damping can overflow its way to a NaN rate.
        # Collapses to sqrt(budget/(m*R)) when c = 0.
        ratio = force_budget / (mass * radius)
        omega_max = ratio * jnp.sqrt(
            2.0 / (damping**2 + jnp.hypot(damping**2, 2.0 * ratio))
        )
        omega = fraction * omega_max
        return jnp.concatenate([axis, radius[None], omega[None]], axis=0)

    def policy_fn(state, key, extras):
        del key
        pos_w, vel_w = state[0:3], state[3:6]
        q_bw = quat_normalize(state[6:10])
        omega_b = state[10:13]
        radius, omega = extras[3], extras[4]

        p_des, v_des, q_des = orbit_reference(pos_w, extras, cfg.dt)

        # The PD reference is the radial projection of the CURRENT position, so
        # it commands zero force exactly when the chaser is perfectly on the
        # circle -- the moment it most needs centripetal force, which free-body
        # dynamics never supply. Without this term the law has to manufacture
        # that force from a standing radial error, settling 5-13% wide of the
        # commanded radius. Feed it forward and the PD corrects only error.
        # `damping * v_des` cancels the drag the reference motion incurs, the
        # other half of the steady-state force the rate bound budgets for.
        feedforward = (
            -mass * omega**2 * radius * (p_des / _safe_norm(p_des))
            + mass * damping * v_des
        )
        force_world = (
            feedforward
            - params.kp_position * (pos_w - p_des)
            - params.kd_velocity * (vel_w - v_des)
        )
        force_body = quat_to_rotmat(q_bw).T @ force_world

        torque_body = _attitude_torque(
            q_bw, q_des, omega_b,
            inertia_diag=inertia_diag,
            kp_attitude=params.kp_attitude, kd_attitude=params.kd_attitude,
        )
        return jnp.concatenate([force_body, torque_body], axis=0)

    return policy_fn, extras_fn


def dock_target_table(cfg: ISSConfig, params: DockParams) -> jnp.ndarray:
    """(K, 7) [position, quaternion] rows a dock policy may be assigned.

    One row, the pose in `ISSConfig.dock`, when no ports are configured.
    Otherwise the rows are the configured entries' own poses, in config order
    -- read off the entries rather than re-resolved from the table, so a
    config that pins a port the table no longer knows still flies to it.
    """
    if not params.ports:
        return jnp.asarray(dock_target(cfg), dtype=jnp.float32)[None, :]
    return jnp.asarray(
        [[*port.position, *port.quaternion] for port in params.ports], dtype=jnp.float32
    )


def dock_target_selector(cfg: ISSConfig, policy_cfg: PolicyConfig) -> Callable:
    """extras -> the (7,) target this episode is docking to.

    The dynamics, the goal-error block and the control law all resolve the
    episode's target through this one function, so a multi-port run cannot end
    up regulating to one port while scoring success against another. Policies
    that never dock -- random and orbit, which have no DOCK_SLOT entry --
    resolve to the `ISSConfig.dock` pose whatever port set is configured, so
    `Events.docked` keeps meaning for them exactly what it means with no ports
    at all. Resolving them to the port table's first row instead would move
    their success gate onto a port nothing in the episode was flying to.
    """
    slot = DOCK_SLOT.get(policy_cfg.type)
    if slot is None:
        pose = jnp.asarray(dock_target(cfg), dtype=jnp.float32)
        return lambda extras: pose
    targets = dock_target_table(cfg, policy_cfg.dock)
    return lambda extras: targets[extras[slot].astype(jnp.int32)]


def _build_dock(cfg: ISSConfig, params: DockParams) -> tuple[PolicyFn, ExtrasFn]:
    inertia_diag = jnp.asarray(cfg.physics.inertia_diag, dtype=jnp.float32)
    targets = dock_target_table(cfg, params)
    count = int(targets.shape[0])

    def extras_fn(key: jax.Array) -> jnp.ndarray:
        return jax.random.randint(key, (1,), 0, count).astype(jnp.float32)

    def policy_fn(state, key, extras):
        del key
        pos_w, vel_w = state[0:3], state[3:6]
        q_bw = quat_normalize(state[6:10])
        omega_b = state[10:13]

        target = targets[extras[0].astype(jnp.int32)]
        dock_pos, dock_quat = target[0:3], quat_normalize(target[3:7])

        force_world = -params.kp_position * (pos_w - dock_pos) - params.kd_velocity * vel_w
        force_body = quat_to_rotmat(q_bw).T @ force_world

        torque_body = _attitude_torque(
            q_bw, dock_quat, omega_b,
            inertia_diag=inertia_diag,
            kp_attitude=params.kp_attitude, kd_attitude=params.kd_attitude,
        )
        return jnp.concatenate([force_body, torque_body], axis=0)

    return policy_fn, extras_fn


def _build_union(cfg: ISSConfig, policy_cfg: PolicyConfig) -> tuple[PolicyFn, ExtrasFn]:
    weights_raw = jnp.asarray(policy_cfg.union_weights, dtype=jnp.float32)
    total = float(jnp.sum(weights_raw))
    if total <= 0.0:
        raise ValueError("union_weights must sum to > 0")
    weights = weights_raw / total

    random_fn = _build_random(cfg)
    orbit_fn, orbit_extras_fn = _build_orbit(cfg, policy_cfg.orbit)
    dock_fn, dock_extras_fn = _build_dock(cfg, policy_cfg.dock)

    def extras_fn(key: jax.Array) -> jnp.ndarray:
        # Layout: [policy_idx, orbit_axis(3), orbit_radius, orbit_omega,
        # dock_port_idx] = 7 floats. Every slot is drawn every episode even
        # though only one branch reads them, so the port an episode would have
        # docked to is recorded whichever sub-policy actually ran.
        key_idx, key_orbit, key_dock = jax.random.split(key, 3)
        policy_idx = jax.random.choice(key_idx, 3, p=weights).astype(jnp.float32)
        return jnp.concatenate(
            [policy_idx[None], orbit_extras_fn(key_orbit), dock_extras_fn(key_dock)], axis=0
        )

    def policy_fn(state, key, extras):
        empty = jnp.zeros((0,), dtype=jnp.float32)
        return jax.lax.switch(
            extras[0].astype(jnp.int32),
            [
                lambda: random_fn(state, key, empty),
                lambda: orbit_fn(state, key, extras[1:6]),
                lambda: dock_fn(state, key, extras[6:7]),
            ],
        )

    return policy_fn, extras_fn


def make_policy(
    cfg: ISSConfig,
    policy_cfg: PolicyConfig,
    policy_type: str | None = None,
) -> tuple[PolicyFn, ExtrasFn | None]:
    """Build a scripted policy. `policy_type` overrides `policy_cfg.type` when
    given (e.g. a CLI --policy flag); otherwise the config value is used."""
    if policy_type is None:
        policy_type = policy_cfg.type
    if policy_type == "random":
        return _build_random(cfg), None
    if policy_type == "orbit":
        return _build_orbit(cfg, policy_cfg.orbit)
    if policy_type == "dock":
        return _build_dock(cfg, policy_cfg.dock)
    if policy_type == "union":
        return _build_union(cfg, policy_cfg)
    raise ValueError(
        f"Unknown ISS policy type '{policy_type}'. "
        "Must be one of: random, orbit, dock, union."
    )
