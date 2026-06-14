# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-06-14

### Added
- `record_episodes(..., finalize=True)`: pass `finalize=False` to accumulate several recording calls
  into one sink (e.g. several environment/scene variants into a single dataset), finalizing yourself
  after the last call.
- `LeRobotDatasetSink(..., resume=False)`: pass `resume=True` to reopen an existing dataset and keep
  appending episodes — the same constructor path `lerobot-record --resume` uses. Together with
  `finalize=False`, this enables multi-process recording (one simulator process per scene variant)
  into one `LeRobotDataset`.

## [0.3.0] - 2026-06-11

### Added
- `record_episodes(..., episode_filter=...)`: an optional post-episode predicate for demonstration-
  quality curation — keep an episode or discard and re-roll it (bounded by `max_attempts_factor`), so
  the learner only imitates good behavior.

## [0.2.0] - 2026-06-02

### Added
- Policy evaluation: `GenesisEnvConfig` (`--env.type=genesis`) + a registered Franka reach reference
  task + the `lerobot-genesis-eval` launcher, so a LeRobot policy can be rolled out in a Genesis scene
  via the standard `lerobot-eval` flow. GPU-verified end to end with an ACT policy.
- `lerobot_genesis.groot`: `build_modality` / `write_modality_json` to emit the `meta/modality.json`
  NVIDIA Isaac-GR00T training expects.

### Fixed
- Restore torch's default device to CPU after `gs.init` (Genesis sets it to cuda globally, which broke
  LeRobot's eval rollout); declare `render_fps` in `GenesisEnv.metadata` for eval video export.

## [0.1.0] - 2026-06-02

### Added
- `GenesisEnv` — a `gymnasium.Env` over a Genesis scene, with an injected `SceneDriver` seam.
- `GenesisRobotDriver` — reference driver: load a URDF/MJCF/USD robot, map a normalised action to a
  chosen subset of joints, step physics, read a camera and joint state.
- `record_episodes` + `LeRobotDatasetSink` — record rollouts as a `LeRobotDataset` (v3 loop).
- `GenesisEnvConfig` — registers Genesis with LeRobot as `--env.type=genesis`.
