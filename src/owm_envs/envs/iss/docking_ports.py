"""The station's docking and berthing targets, in the environment world frame.

The render asset is the ISS between February and May 2015 (see
`render/iss_scene`), with the PMM berthed at Unity nadir. The flown station
uses eight visiting-vehicle ports; Unity nadir (a Cygnus berth today) is
occupied by the PMM in this asset, so the seven that are free here are the
targets, listed in `PORTS` order -- the index recorded per episode:

    entry                 flown-station port           typical visiting vehicle
    harmony_fwd_pma2      Harmony fwd, PMA-2/IDA-2     Crew or Cargo Dragon
    harmony_zenith_cbm    Harmony zenith, PMA-3/IDA-3  Crew or Cargo Dragon
    harmony_nadir_cbm     Harmony nadir CBM            HTV-X, Cygnus
    zvezda_aft            Zvezda aft                   Progress
    poisk_zenith          Poisk (MRM-2) zenith         Soyuz, Progress
    pirs_nadir            Prichal nadir on the flown   Soyuz, Progress
                          station; Pirs here
    rassvet_nadir         Rassvet (MRM-1) nadir        Soyuz, Progress

Two of those are not the flown hardware. Harmony zenith carries no PMA/IDA in
a 2015 asset, so its mating plane is the bare CBM and sits about 2.5 m inboard
of the real docking plane. Pirs was deorbited in 2021 and that side of Zvezda
now carries Nauka and the Prichal node, so `pirs_nadir` stands in for Prichal
nadir at a different standoff. Tranquility's outboard end held PMA-3 until
2017 and is modelled here, but its +x corridor runs into the P1 truss
radiators (0.09 m of clearance against a 2.25 m chaser), so it is not a
reachable target and is not listed.

Every interface point is measured from the shipped asset under exactly the
transform `ISSScene._load_iss_group` applies: rotate +90 deg about X (asset +Y
-> world +Z), then translate by `-ISS_RECENTRE_OFFSET`. Blender's glTF
importer applies the same +Y-up -> +Z-up rotation, so for anything measured in
the .blend file this is p_world = p_blender - ISS_RECENTRE_OFFSET.

`interface` is where a port's mating plane meets its own centreline, found by
restricting to vertices within a radius of that centreline (so antennas,
radiators and solar arrays cannot set the plane) and taking the extreme along
the approach axis. `normal` is the outward unit vector along the approach
corridor.

`standoff_m` is how far out along that normal the goal pose sits, and
`clearance_m` the resulting distance to the nearest surface of the collision
hull in `resources/collision_boxes.yaml`. A goal has to be somewhere the
chaser can actually hold station, so each standoff is the shortest that keeps
at least 1.25 m under the 2.25 m chaser radius -- the margin the shipped
`DockConfig.position` already uses at PMA-2.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from ...core.models import ConfigModel
from ...core.quaternion import quat_from_rotmat

Mechanism = Literal["IDSS", "CBM", "SSVP"]


class DockingPort(ConfigModel):
    name: str
    module: str
    mechanism: Mechanism
    interface: tuple[float, float, float]
    normal: tuple[float, float, float]
    standoff_m: float
    clearance_m: float
    notes: str = ""


PORTS: tuple[DockingPort, ...] = (
    DockingPort(
        name="harmony_fwd_pma2",
        module="26 Harmony Node 2 / 04 PMA-2",
        mechanism="IDSS",
        interface=(0.194, -20.840, -3.336),
        normal=(0.0, -1.0, 0.0),
        standoff_m=3.66,
        clearance_m=3.50,
        notes=(
            "The port the shipped DockConfig targets. Its docking-ring centreline is at "
            "z = -3.336, 0.734 m below the Harmony barrel centreline (z = -2.602): the "
            "adapter dog-legs nadir-ward over its last 2 m, which the shipped pose does "
            "not account for."
        ),
    ),
    DockingPort(
        name="harmony_zenith_cbm",
        module="26 Harmony Node 2",
        mechanism="CBM",
        interface=(0.233, -14.939, -0.218),
        normal=(0.0, 0.0, 1.0),
        standoff_m=5.0,
        clearance_m=3.78,
        notes="Bare CBM here; PMA-3/IDA-3 occupies this face on the flown station.",
    ),
    DockingPort(
        name="harmony_nadir_cbm",
        module="26 Harmony Node 2",
        mechanism="CBM",
        interface=(0.233, -14.939, -4.985),
        normal=(0.0, 0.0, -1.0),
        standoff_m=5.0,
        clearance_m=3.77,
        notes="Free in the source asset. Corridor points at Earth.",
    ),
    DockingPort(
        name="zvezda_aft",
        module="05 Zvezda Service Module",
        mechanism="SSVP",
        interface=(0.233, 32.550, -1.870),
        normal=(0.0, 1.0, 0.0),
        standoff_m=6.0,
        clearance_m=3.66,
        notes=(
            "Far aft end, 63 m from the Harmony forward goal and on the opposite approach "
            "axis. The Russian segment centreline is z = -1.870, 0.732 m above the US "
            "segment's."
        ),
    ),
    DockingPort(
        name="poisk_zenith",
        module="34 Poisk MRM-2",
        mechanism="SSVP",
        interface=(0.233, 20.129, 3.555),
        normal=(0.0, 0.0, 1.0),
        standoff_m=5.0,
        clearance_m=3.55,
        notes="Zenith of the Zvezda transfer compartment.",
    ),
    DockingPort(
        name="pirs_nadir",
        module="13 Pirs Docking Compartment",
        mechanism="SSVP",
        interface=(0.233, 20.129, -7.290),
        normal=(0.0, 0.0, -1.0),
        standoff_m=5.5,
        clearance_m=3.79,
        notes="Coaxial with poisk_zenith in this asset.",
    ),
    DockingPort(
        name="rassvet_nadir",
        module="39 Rassvet MRM-1",
        mechanism="SSVP",
        interface=(0.240, 7.161, -9.267),
        normal=(0.0, 0.0, -1.0),
        standoff_m=5.5,
        clearance_m=3.77,
        notes="Deepest nadir point on the station, below the Zarya nadir port.",
    ),
)

PORTS_BY_NAME: dict[str, DockingPort] = {p.name: p for p in PORTS}
PORT_NAMES: tuple[str, ...] = tuple(PORTS_BY_NAME)

def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("cannot normalise a zero-length vector")
    return v / n


def port_pose(
    port: DockingPort,
    standoff: float | None = None,
    up_hint: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Goal position and q_bw for holding off `port`, defaulting to its own standoff.

    Body +z is laid along the inbound corridor (-normal) so the chaser's nose and
    its onboard camera face the port. Body +y is then set as close to `up_hint`
    as the corridor allows, which fixes the remaining roll.

    The default hint is nadir, which carries roll information only on the two
    corridors that are not themselves vertical: on PMA-2's it reproduces the
    shipped `DockConfig.quaternion` (0.7071068, -0.7071068, 0, 0) exactly. The
    other five ports face zenith or nadir, so the hint lies along the corridor
    and fixes nothing; a vertical corridor is rolled by the first fallback
    hint, forward (0, -1, 0), instead. The fallbacks are the rule for those
    five, not an edge case.
    """
    normal = _unit(np.asarray(port.normal, dtype=np.float64))
    body_z = -normal

    up = np.asarray(up_hint, dtype=np.float64)
    up = up - np.dot(up, body_z) * body_z
    if np.linalg.norm(up) < 1e-6:
        for fallback in ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)):
            up = np.asarray(fallback) - np.dot(fallback, body_z) * body_z
            if np.linalg.norm(up) >= 1e-6:
                break
    body_y = _unit(up)
    body_x = np.cross(body_y, body_z)

    rotation = np.column_stack([body_x, body_y, body_z])
    quat = np.asarray(quat_from_rotmat(jnp.asarray(rotation, dtype=jnp.float32)), dtype=np.float64)
    reach = port.standoff_m if standoff is None else float(standoff)
    position = np.asarray(port.interface, dtype=np.float64) + reach * normal
    return position, quat


def resolve_port_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Expand the keyword 'all' and reject anything that is not a known port.

    Duplicates are rejected too: naming a port twice would double its share of
    the uniform per-episode draw without saying so.
    """
    if tuple(names) == ("all",):
        return PORT_NAMES
    unknown = [n for n in names if n not in PORTS_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown docking port(s) {unknown}; known ports are {list(PORT_NAMES)} "
            "(or the keyword 'all')"
        )
    if not names:
        raise ValueError("port list is empty")
    duplicates = sorted({n for n in names if list(names).count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate docking port(s) {duplicates}; name each port at most once"
        )
    return tuple(names)


class DockPort(ConfigModel):
    """A port an episode may be assigned, carrying the pose it resolves to.

    The pose is stored, not just the name, so a versioned or as-run config
    reproduces the run it describes even after the `PORTS` table is revised.
    A bare name resolves against the table at load (see
    `resolve_port_entries`); this is what that resolution produces. An entry
    that already carries its pose is trusted as written.
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


def resolve_port_entries(v: object) -> object:
    """Normalise a configured port list to a tuple of pinned `DockPort`.

    The one implementation behind every `ports` field -- the generation-side
    `policies.DockParams.ports` and the environment-side `config.DockConfig
    .ports` -- so a port set means the same thing wherever it is written.

    An entry may be a bare port name or an already-pinned
    {name, position, quaternion}; both normalise to the pinned form, so every
    serialised config carries the poses it used. A pinned entry is taken as
    the ground truth for its pose -- the table is only consulted to give a
    bare name one. The keyword "all" expands to every port in `PORTS` here at
    config-load time, so the as-run record names the ports a run actually
    used rather than a keyword whose meaning could change with the table.
    """
    if not v:
        return ()
    entries = list(v)
    if all(isinstance(entry, str) for entry in entries):
        # Unchanged path for a name-only config: `resolve_port_names` owns
        # the "all" expansion and the unknown-name and duplicate errors.
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
        # A pinned entry is ground truth: whatever pose the config carries is
        # the pose the run flies to, whether or not the table knows the name
        # or places it elsewhere. Only a bare name consults the table, because
        # a name alone has no pose of its own.
        port = entry if isinstance(entry, DockPort) else DockPort.model_validate(entry)
        resolved.append(port)

    names = [port.name for port in resolved]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate docking port(s) {duplicates}; name each port at most once"
        )
    return tuple(resolved)


def port_target_rows(ports: tuple[DockPort, ...]) -> np.ndarray:
    """(K, 7) [position, quaternion] rows for resolved port entries.

    Read off the entries' own pinned poses rather than re-resolved from the
    table, so a config that pins a port the table no longer knows still flies
    to it.
    """
    return np.asarray(
        [[*port.position, *port.quaternion] for port in ports], dtype=np.float32
    ).reshape(len(ports), 7)


def dock_targets(names: tuple[str, ...] | list[str]) -> np.ndarray:
    """(K, 7) array of [position, quaternion] rows for the named ports."""
    rows = []
    for name in resolve_port_names(names):
        position, quat = port_pose(PORTS_BY_NAME[name])
        rows.append(np.concatenate([position, quat]))
    return np.stack(rows).astype(np.float32)
