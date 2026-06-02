# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-02

### Added
- `GenesisEnv` — a `gymnasium.Env` over a Genesis scene, with an injected `SceneDriver` seam.
- `GenesisRobotDriver` — reference driver: load a URDF/MJCF/USD robot, map a normalised action to a
  chosen subset of joints, step physics, read a camera and joint state.
- `record_episodes` + `LeRobotDatasetSink` — record rollouts as a `LeRobotDataset` (v3 loop).
- `GenesisEnvConfig` — registers Genesis with LeRobot as `--env.type=genesis`.
