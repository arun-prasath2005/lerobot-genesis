"""The recording loop, verified with a fake env + fake sink — no LeRobot, no GPU."""

from __future__ import annotations

import numpy as np

from helpers import FakeDriver, FakeSink
from lerobot_genesis import EpisodeSink, GenesisEnv, make_frame, record_episodes


def test_fake_sink_satisfies_the_protocol() -> None:
    assert isinstance(FakeSink(), EpisodeSink)


def test_make_frame_uses_convention_keys() -> None:
    obs = {"pixels": np.zeros((4, 4, 3), np.uint8), "agent_pos": np.zeros(6, np.float32)}
    frame = make_frame(obs, np.ones(6, np.float32), camera="front")
    assert set(frame) == {"observation.images.front", "observation.state", "action"}
    assert frame["observation.images.front"].dtype == np.uint8
    assert frame["action"].dtype == np.float32


def test_records_frames_episodes_and_one_finalize() -> None:
    env = GenesisEnv(FakeDriver(succeed_after=3), max_episode_steps=10)
    sink = FakeSink()

    frames = record_episodes(env, lambda _obs: np.zeros(6, np.float32), sink, n_episodes=2)

    assert frames == len(sink.frames) == 6  # 3 steps/episode (terminates on success), 2 episodes
    assert sink.episodes == 2
    assert sink.finalized == 1  # exactly once, at the very end
    assert all("observation.state" in frame for frame in sink.frames)


def test_finalize_false_leaves_the_sink_open() -> None:
    env = GenesisEnv(FakeDriver(succeed_after=3), max_episode_steps=10)
    sink = FakeSink()

    record_episodes(env, lambda _obs: np.zeros(6, np.float32), sink, n_episodes=1, finalize=False)
    record_episodes(env, lambda _obs: np.zeros(6, np.float32), sink, n_episodes=1, finalize=False)

    assert sink.episodes == 2  # two calls accumulated into ONE dataset...
    assert sink.finalized == 0  # ...which the caller finalizes once, themselves
    sink.finalize()
    assert sink.finalized == 1


def test_episode_filter_discards_and_rerolls() -> None:
    env = GenesisEnv(FakeDriver(succeed_after=3), max_episode_steps=10)
    sink = FakeSink()
    calls = {"n": 0}

    def keep_every_other() -> bool:
        calls["n"] += 1
        return calls["n"] % 2 == 0  # reject odd attempts -> each kept episode costs 2 rollouts

    frames = record_episodes(
        env,
        lambda _obs: np.zeros(6, np.float32),
        sink,
        n_episodes=2,
        episode_filter=keep_every_other,
    )

    assert sink.episodes == 2  # exactly the requested count survive
    assert frames == 6  # only KEPT episodes contribute frames
    assert calls["n"] == 4  # two rejected + two kept
    assert sink.finalized == 1


def test_episode_filter_attempt_cap_terminates() -> None:
    env = GenesisEnv(FakeDriver(succeed_after=3), max_episode_steps=10)
    sink = FakeSink()

    frames = record_episodes(
        env,
        lambda _obs: np.zeros(6, np.float32),
        sink,
        n_episodes=2,
        episode_filter=lambda: False,
        max_attempts_factor=3,
    )

    assert frames == 0 and sink.episodes == 0  # nothing kept...
    assert sink.finalized == 1  # ...but the dataset is still finalized cleanly
