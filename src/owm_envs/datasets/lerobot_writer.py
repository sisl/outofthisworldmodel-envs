"""LeRobot dataset writer -- the ISOLATED lerobot API surface.

Every lerobot call in this package lives in this file, so a version bump of a
library whose API drifts across releases touches one place.

Targets the lerobot 0.4.x API:
  LeRobotDataset.create(repo_id, fps, root, features, use_videos)
  .add_frame(frame)   # frame dict carries "task"
  .save_episode()
  .finalize()         # writes the parquet footers and meta/episodes

`finalize` is the one call here that is not optional bookkeeping: without it
the split on disk has no episode metadata and is not a loadable dataset. It
takes no arguments and means the same thing in every lerobot the dependency
floor admits (checked against 0.4.4, 0.5.1 and 0.6.1), so it needs no
version gate.

lerobot is an optional extra. The import is function-local so the rest of the
package imports and tests without it.

Schema: `observation_vector` (float32, obs_dim), `state_vector` (float64,
state_dim) and `action` (float32, act_dim) carry the trajectory itself. Six more
features carry outcome metadata that would otherwise be lost once a
TrajectoryBatch is discarded: `reward` (the per-frame reward; zero on an
episode's final frame, whose action slot is a zero pad rather than a real
action), `is_last`, `terminated`, `truncated`, `policy_id`, and
`dock_target`.

`is_last` is per-FRAME and true only on an episode's final frame. It marks
both the terminal observation and the one frame whose action is the zero pad,
which must be dropped when forming (observation, action, next observation)
transitions. `terminated`, `truncated`, `policy_id` and `dock_target` are
per-EPISODE facts written onto every frame of that episode -- deliberate
redundancy so a dataset can be filtered frame-wise (e.g. "give me every frame
of every collision episode") without joining back to episode boundaries.
Combining them, `is_last & terminated` selects exactly the collision and dock
frames, and `is_last & truncated` exactly the horizon cuts.

`dock_target` is the (1, 7) [position, quaternion] pose the episode was flying
to: the port it was assigned under a port set, or the `DockConfig` pose for a
run that configured none. The pose itself rather than an index into the port
table, because the table's indices shift whenever a port is added or removed,
so a stored index stops reproducing the episode as soon as the table changes;
a pose is self-contained and stays comparable across table versions. A batch
whose `dock_targets` is None -- a driver whose policy source cannot supply one
-- writes all-NaN rows, which no real pose can collide with.

`state_vector` is the TRUE dynamics state at that frame -- what the
simulator actually held, before the sensor model corrupted it into
`observation_vector`. Where both exist,
`observation_vector[..., :13] - state_vector` is exactly the realized
sensor-noise draw for that frame, so a noise model can be measured back off
a written dataset rather than trusted from the config that produced it, and
a policy can be trained on the noisy channel while being scored against
truth. The first 13 dims of `observation_vector` are the corresponding
measured state; any dims past them (the goal-error block) have no
`state_vector` counterpart, which is why the two can differ in width.

It is written at float64 where every other feature here is float32, because
it is the one feature whose consumers SUBTRACT its columns rather than read
them. An env carrying absolute ECI positions holds ~6.8e6 m in those columns,
where float32 lands on a 0.5 m grid; the relative view derived from them --
the pose a rendered frame is drawn at, and the quantity a 0.1 m dock gate
tests -- inherits that grid in full, and it moves frame to frame as the
mantissa bits flip, so it reads as jitter rather than as a fixed offset.
Unlike `dock_target`, a batch with no truth channel omits the feature
entirely rather than writing NaN rows: NaN marks a value that is genuinely
missing for one episode, whereas a driver that cannot record truth has no
frames to fill, and a whole column of stand-ins would only invite a
consumer to average over them.

Episode boundaries are recoverable from lerobot's own bookkeeping too --
`meta.episodes["length"]`, and `frame_index` restarting at 0 -- and that
agrees with `is_last` by construction: only real frames are written, so a
lerobot episode's length equals that episode's `TrajectoryBatch.lengths`
entry.

`reward` and `policy_id` use shape (1, 1), not the seemingly more natural
scalar shape (1,). This is not a style choice: lerobot 0.4.4's schema-to-HF
conversion (`get_hf_features_from_features`) special-cases any feature whose
shape is exactly `(1,)` into a plain HuggingFace `datasets.Value`, whose
`encode_example` calls Python's `float()`/`int()` on the stored array. Under
numpy>=2.0 (installed: 2.5.1) those raise `TypeError: only 0-dimensional
arrays can be converted to Python scalars` for any array with `ndim != 0`,
which every value `add_frame` accepts is (its own validator requires an
`np.ndarray` with `.shape` matching the declared feature shape, i.e. `(1,)`,
which always has `ndim == 1`). The boolean features `is_last`, `terminated`
and `truncated` survive at shape `(1,)` only because `bool()` -- unlike
`float()`/`int()` -- does not raise for a single-element array;
`reward`/`policy_id` do not, for either float32 or int64. Shape `(1, 1)`
takes the `Array2D` path instead, which round-trips correctly. It reads back
as a (1, 1) array rather than a bare scalar; callers should `.reshape(())` or
index `[0, 0]`. `dock_target` takes the same `Array2D` path at (1, 7), which
is not a special case at all -- it reads back as a (1, 7) array, and callers
want the row rather than a scalar anyway.

Video features carry one clip per episode per camera view. They are only
declared when the caller passes `frames` -- one mapping per episode in
`batch`, from feature name to that episode's `(L, H, W, 3)` uint8 clip, each
`L` matching that episode's `lengths[i]` exactly. Every episode must offer
the same set of feature names, since the schema is fixed from the first one.
`frames` may be a sequence or any iterable, and the iterable form is what
makes a large split writable at all: clips are pulled one episode at a time,
so a 500k-frame split holds one episode of video in RAM rather than the
~98 GB per view the whole split would take.

The feature names themselves are the caller's: this file writes what it is
given rather than knowing which camera produced it. `datasets/video.py` owns
that mapping, and rendering lives there too -- kept out of this file because
rendering needs `owm_envs.render` (an optional extra of its own) and this
file's only job is staying the sole lerobot call site.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..drivers.types import TrajectoryBatch

# One episode's video: feature name -> that episode's (L, H, W, 3) uint8 clip.
Clips = Mapping[str, np.ndarray]

# "the iterator had nothing left", distinct from a clip that is itself None --
# which is a caller bug worth the AttributeError it earns, not an end of input.
_MISSING = object()


def write_lerobot_split(
    root: str | Path,
    repo_id: str,
    batch: TrajectoryBatch,
    fps: int,
    task_name: str = "iss_docking",
    frames: Sequence[Clips] | Iterable[Clips] | None = None,
) -> Path:
    """Write one TrajectoryBatch as a LeRobotDataset on disk. Returns its root.

    Only real frames are written; padding past each episode's length is skipped.

    `frames`, when given, is one `{feature name: (L, H, W, 3) uint8 clip}`
    mapping per episode in `batch`, aligned 1:1 with that episode's stored
    observations -- `L` must equal `batch.lengths[i]`, or a ValueError names
    the mismatching episode. Episode 0's keys fix the schema; a later episode
    offering different ones is an error rather than a sparser dataset.

    A sequence is checked in full before anything is written. Any other
    iterable is consumed one episode at a time as the episodes are written,
    which is what keeps a large split's video off the heap; its clips can
    therefore only be checked as they arrive, so a mismatch there surfaces
    with the earlier episodes already on disk.
    """
    batch.validate()
    clips: Iterator[Clips] | None = None
    frame_shapes: dict[str, tuple[int, ...]] | None = None
    root = Path(root)
    # Bound before the try so the failure path can tell "the dataset exists and
    # needs finalizing" from "we never got as far as creating one".
    dataset = None
    # The caller's own iterator, held apart from the peeked-and-handed-back
    # view of it below. This is the one that owns the render pool, and so the
    # one that has to be closed however this call ends -- closing the hand-back
    # would not do: a generator that has never been started runs no code when
    # it is closed, which is exactly its state if set-up fails.
    source: Iterator[Clips] | None = (
        iter(frames) if frames is not None and not isinstance(frames, Sequence) else None
    )

    # Everything from here on can be holding that source open -- peeking a clip
    # to declare the video feature is what starts the pool -- so every step of
    # it has to be able to hand the source back. Set-up fails as readily as the
    # write itself: an unwritable root, a split directory already there, a clip
    # whose frames are not (H, W, C).
    try:
        if frames is not None:
            if isinstance(frames, Sequence):
                _validate_frames(frames, batch)
                # Derived from the actual clips, not assumed square: the
                # renderer returns (H, W, 3), and H need not equal W.
                frame_shapes = _shapes_of(frames[0])
                clips = iter(frames)
            else:
                frame_shapes, clips = _peek_frame_shapes(source)

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation_vector": {
                "dtype": "float32",
                "shape": (batch.observations.shape[-1],),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (batch.actions.shape[-1],),
                "names": None,
            },
            # (1, 1), not (1,): see the module docstring for why the scalar
            # shape is unwritable for non-boolean features on this lerobot
            # version.
            "reward": {"dtype": "float32", "shape": (1, 1), "names": None},
            "is_last": {"dtype": "bool", "shape": (1,), "names": None},
            "terminated": {"dtype": "bool", "shape": (1,), "names": None},
            "truncated": {"dtype": "bool", "shape": (1,), "names": None},
            "policy_id": {"dtype": "int64", "shape": (1, 1), "names": None},
            "dock_target": {"dtype": "float32", "shape": (1, 7), "names": None},
        }
        if batch.true_state is not None:
            features["state_vector"] = {
                "dtype": "float64",
                "shape": (batch.true_state.shape[-1],),
                "names": None,
            }
        for key, shape in (frame_shapes or {}).items():
            height, width, channels = shape
            features[key] = {
                "dtype": "video",
                "shape": (int(height), int(width), int(channels)),
                "names": ["height", "width", "channels"],
            }

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            features=features,
            use_videos=frames is not None,
        )

        for episode in range(batch.num_episodes):
            _write_episode(dataset, batch, episode, clips, frame_shapes, task_name)

        # lerobot writes the parquet footers and meta/episodes only here, so a
        # split that is never finalized is not a loadable dataset -- loading it
        # finds no episode metadata and falls through to the Hub. lerobot's own
        # writer finalizes from `__del__` as a safety net, which is why dropping
        # the dataset appears to work; that makes a split's validity a question
        # of when the object was collected, so it is done explicitly.
        # Before the surplus check below, so the split that check leaves on
        # disk is the complete, loadable one its comment describes.
        dataset.finalize()

        # A surplus clip cannot be seen without pulling one past the last
        # episode, and pulling it earlier would hold two clips at once -- the
        # peak this writer exists to avoid. So it is found with the split
        # already on disk, which is not a half-written one: every episode that
        # was written did match its own clip.
        if clips is not None and next(clips, _MISSING) is not _MISSING:
            raise ValueError(
                f"frames has more clips than the batch's {batch.num_episodes} episodes"
            )
    except BaseException:
        # A call that dies part-way leaves the source suspended, and the raised
        # exception's traceback keeps this frame -- and so the source -- alive
        # for as long as the exception is held. Closing it here runs its
        # cleanup at the failure rather than whenever the traceback is dropped.
        #
        # The split this call leaves behind gets finalized for the same reason:
        # unfinalized, it is only readable once lerobot's `__del__` reaches it,
        # so whether a failed run left a loadable split would come down to when
        # the dataset was collected. Finalizing decides it instead. It closes
        # the writers over whatever was saved; it does not roll anything back,
        # and an episode that was only buffered is not among them -- a failure
        # inside `save_episode` itself can still leave that episode's frames
        # written without its metadata, so the split is "what got saved", not
        # a transaction boundary.
        #
        # Suppressed, and only on this path: neither a source nor a writer that
        # also fails on the way down must replace the error that says what
        # actually went wrong. On the path below there is no such error to
        # protect, so a failed shutdown is itself the news and propagates.
        if dataset is not None:
            with contextlib.suppress(Exception):
                dataset.finalize()
        with contextlib.suppress(Exception):
            _close(source)
        raise
    _close(source)

    return root


def _close(source: Iterator[Clips] | None) -> None:
    """Close `source` if it is the kind of iterator that can be closed."""
    if source is not None and hasattr(source, "close"):
        source.close()


def _write_episode(
    dataset,
    batch: TrajectoryBatch,
    episode: int,
    clips: Iterator[Clips] | None,
    frame_shapes: dict[str, tuple[int, ...]] | None,
    task_name: str,
) -> None:
    """Add one episode's real frames to `dataset` and save it.

    This episode's clips are pulled here, and both they and the frame dict
    that holds VIEWS into them (np.asarray of a uint8 slice does not copy) die
    with this call. That is what bounds the writer at one episode of video: a
    caller that kept either alive would hold this episode while the next one
    is produced -- 2.8 GiB rather than 1.4 for a single 7200-step view at
    256x256, and that much again per extra view.
    """
    length = int(batch.lengths[episode])
    clip = _next_clip(clips, episode, length, frame_shapes) if clips is not None else None
    terminated = bool(batch.terminated[episode])
    truncated = bool(batch.truncated[episode])
    policy_id = int(batch.policy_ids[episode]) if batch.policy_ids is not None else 0
    dock_target = (
        np.asarray(batch.dock_targets[episode], dtype=np.float32).reshape(1, 7)
        if batch.dock_targets is not None
        else np.full((1, 7), np.nan, dtype=np.float32)
    )
    for t in range(length):
        frame = {
            "observation_vector": np.asarray(
                batch.observations[episode, t], dtype=np.float32
            ),
            "action": np.asarray(batch.actions[episode, t], dtype=np.float32),
            "reward": np.array([[batch.rewards[episode, t]]], dtype=np.float32),
            "is_last": np.array([t == length - 1]),
            "terminated": np.array([terminated]),
            "truncated": np.array([truncated]),
            "policy_id": np.array([[policy_id]], dtype=np.int64),
            "dock_target": dock_target,
            "task": task_name,
        }
        if batch.true_state is not None:
            frame["state_vector"] = np.asarray(
                batch.true_state[episode, t], dtype=np.float64
            )
        if clip is not None:
            for key, view_clip in clip.items():
                frame[key] = np.asarray(view_clip[t], dtype=np.uint8)
        dataset.add_frame(frame)
    dataset.save_episode()


def _shapes_of(clip: Clips) -> dict[str, tuple[int, ...]]:
    return {key: np.asarray(view_clip).shape[1:] for key, view_clip in clip.items()}


def _peek_frame_shapes(
    clips: Iterator[Clips],
) -> tuple[dict[str, tuple[int, ...]], Iterator[Clips]]:
    """Read the per-view frame shapes off the first episode and put it back.

    The video features have to be declared before the first `add_frame`, and
    their shapes come from real clips rather than from the render config, so
    one episode must be pulled up front.
    """
    head = next(clips, _MISSING)
    if head is _MISSING:
        raise ValueError("frames is empty; expected one clip per episode")
    return _shapes_of(head), _hand_back(head, clips)


def _hand_back(head: Clips, clips: Iterator[Clips]) -> Iterator[Clips]:
    """Yield the already-pulled episode's clips, then the rest.

    `itertools.chain((head,), clips)` would do the same but keeps `head` in
    its argument tuple for the whole write, pinning one episode of video --
    on a 7200-step episode at 256x256 that is 1.4 GiB per view held for
    nothing.

    This owns nothing but `head`; `clips` is closed by the caller, which holds
    it directly for exactly that reason.
    """
    yield head
    del head
    yield from clips


def _next_clip(
    clips: Iterator[Clips],
    episode: int,
    length: int,
    frame_shapes: dict[str, tuple[int, ...]],
) -> Clips:
    """Pull episode `episode`'s clips and check them against that episode.

    Same checks as `_validate_frames`, applied as the clips arrive -- the only
    point at which a lazily produced episode can be checked at all.
    """
    clip = next(clips, _MISSING)
    if clip is _MISSING:
        raise ValueError(f"frames ran out after {episode} clips; batch has more episodes")
    if set(clip) != set(frame_shapes):
        raise ValueError(
            f"episode {episode}: video features {sorted(clip)} do not match "
            f"episode 0's {sorted(frame_shapes)}"
        )
    for key, shape in frame_shapes.items():
        view_clip = np.asarray(clip[key])
        if int(view_clip.shape[0]) != length:
            raise ValueError(
                f"episode {episode}: {key} clip length {int(view_clip.shape[0])} does "
                f"not match batch length {length}"
            )
        if view_clip.shape[1:] != shape:
            raise ValueError(
                f"episode {episode}: {key} frame shape {view_clip.shape[1:]} does not "
                f"match episode 0's frame shape {shape}"
            )
    return clip


def _validate_frames(frames: Sequence[Clips], batch: TrajectoryBatch) -> None:
    """Raise ValueError if `frames` doesn't align 1:1 with `batch`'s episodes.

    A clip shorter or longer than its episode would silently desynchronise
    video from state for every downstream consumer, so this compares length
    per episode, not just episode count.
    """
    if len(frames) != batch.num_episodes:
        raise ValueError(
            f"frames has {len(frames)} episodes, batch has {batch.num_episodes}"
        )
    # The declared features and their shapes come from episode 0, so every
    # other episode has to offer the same views at the same (H, W, C).
    expected = _shapes_of(frames[0])
    clips = iter(frames)
    for episode in range(batch.num_episodes):
        _next_clip(clips, episode, int(batch.lengths[episode]), expected)
