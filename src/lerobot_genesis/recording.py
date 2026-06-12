"""Record :class:`GenesisEnv` rollouts as LeRobot datasets.

A ``Policy`` is any ``observation -> action`` callable; an :class:`EpisodeSink` is any
add/save/finalize target. :class:`LeRobotDatasetSink` is the real sink over ``LeRobotDataset``;
tests drive a fake one, so the recording loop needs neither LeRobot nor a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .env import GenesisEnv

FloatArray = npt.NDArray[np.float32]
Policy = Callable[[dict[str, Any]], FloatArray]


@runtime_checkable
class EpisodeSink(Protocol):
    """Where recorded frames go. The real impl wraps ``LeRobotDataset``; tests use a fake."""

    def add_frame(self, frame: dict[str, Any]) -> None: ...
    def save_episode(self) -> None: ...
    def finalize(self) -> None: ...


def make_frame(obs: dict[str, Any], action: FloatArray, *, camera: str = "front") -> dict[str, Any]:
    """One LeRobot frame from an observation and the action taken, in convention keys."""
    return {
        f"observation.images.{camera}": np.asarray(obs["pixels"], dtype=np.uint8),
        "observation.state": np.asarray(obs["agent_pos"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
    }


def record_episodes(
    env: GenesisEnv,
    policy: Policy,
    sink: EpisodeSink,
    *,
    n_episodes: int,
    camera: str = "front",
    episode_filter: Callable[[], bool] | None = None,
    max_attempts_factor: int = 3,
    finalize: bool = True,
) -> int:
    """Roll ``policy`` in ``env`` until ``n_episodes`` episodes are recorded; return frame count.

    ``episode_filter`` (optional) is called after each episode finishes, before it is written:
    return True to keep it, False to discard and re-roll — demonstration quality curation (e.g.
    drop scripted-teacher episodes whose task metric came out poor, so the learner imitates only
    good behavior). The closure typically reads the caller's driver/env state. At most
    ``n_episodes * max_attempts_factor`` rollouts are attempted, so a too-strict filter
    terminates.

    ``finalize`` (default True) calls ``sink.finalize()`` once at the end — mandatory for
    ``LeRobotDataset`` or the written parquet is left incomplete. Pass ``finalize=False`` when
    recording in multiple calls against one sink (e.g. several environment variants into one
    dataset) and finalize yourself after the last call.
    """
    frames = 0
    kept = 0
    attempts = 0
    while kept < n_episodes and attempts < n_episodes * max_attempts_factor:
        attempts += 1
        obs, _ = env.reset()
        episode: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = np.asarray(policy(obs), dtype=np.float32)
            episode.append(make_frame(obs, action, camera=camera))
            obs, _, terminated, truncated, _ = env.step(action)
        if episode_filter is not None and not episode_filter():
            continue
        for frame in episode:
            sink.add_frame(frame)
        sink.save_episode()
        kept += 1
        frames += len(episode)
    if finalize:
        sink.finalize()
    return frames


class LeRobotDatasetSink:
    """An :class:`EpisodeSink` backed by ``LeRobotDataset`` — one sink writes one dataset.

    ``lerobot`` is imported lazily, so importing this module never requires it. Frames follow the
    canonical ``{dtype, shape, names}`` feature schema and the convention keys
    ``observation.images.<camera>`` / ``observation.state`` / ``action``.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        fps: int,
        state_dim: int,
        action_dim: int,
        image_shape: tuple[int, int, int],
        camera: str = "front",
        task: str = "",
        root: str | Path | None = None,
        use_videos: bool = True,
        robot_type: str = "genesis",
        resume: bool = False,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        h, w, _ = image_shape
        self._task = task
        if resume:
            # reopen an existing dataset and keep appending episodes — the same path
            # `lerobot-record --resume` uses. This is how multi-process recording (e.g. one
            # simulator process per scene variant) accumulates into ONE dataset.
            self._dataset = LeRobotDataset(repo_id, root=root)
            return
        features = {
            f"observation.images.{camera}": {
                "dtype": "video" if use_videos else "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        }
        self._dataset = LeRobotDataset.create(
            repo_id,
            fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
        )

    def add_frame(self, frame: dict[str, Any]) -> None:
        # LeRobot's v3 add_frame requires a per-frame task string; supply it from the sink so the
        # generic make_frame stays format-agnostic.
        self._dataset.add_frame({**frame, "task": frame.get("task", self._task)})

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def finalize(self) -> None:
        self._dataset.finalize()

    @property
    def dataset(self) -> Any:
        """The underlying ``LeRobotDataset`` (e.g. for ``push_to_hub`` or inspection)."""
        return self._dataset
