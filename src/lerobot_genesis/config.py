"""Register Genesis with LeRobot as ``--env.type=genesis``.

The eval/RL seam. For a locally registered config, LeRobot's ``make_env`` calls
``EnvConfig.create_envs``, which runs ``gym.make(gym_id, **gym_kwargs)``. So ``gym_id`` must match a
``gymnasium.register`` id (see :mod:`lerobot_genesis.tasks`), and the env must emit ``pixels`` and
``agent_pos`` observations plus an ``info["is_success"]`` flag.

This module imports ``lerobot`` at load time (the decorator needs it) — import it on demand, not
from the package root.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import ACTION, OBS_IMAGE, OBS_STATE, EnvConfig

from ._constants import GYM_NAMESPACE


@EnvConfig.register_subclass("genesis")
@dataclass
class GenesisEnvConfig(EnvConfig):
    """LeRobot env config for a Genesis scene.

    ``task`` selects the registered gym id (``lerobot_genesis/<task>``); ``features`` and
    ``features_map`` declare the observation and action layout for the policy. Defaults match the
    bundled Franka reference task — override ``task`` and the dims to point at your own Genesis env.
    """

    task: str | None = "FrankaReach-v0"
    fps: int = 30
    episode_length: int = 300
    action_dim: int = 9
    state_dim: int = 9
    observation_height: int = 240
    observation_width: int = 320
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features:
            self.features = {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
                "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
                "pixels": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(self.observation_height, self.observation_width, 3),
                ),
            }
        if not self.features_map:
            # GenesisEnv emits one image under "pixels"; preprocess_observation maps a bare "pixels"
            # array to OBS_IMAGE and "agent_pos" to OBS_STATE.
            self.features_map = {"action": ACTION, "agent_pos": OBS_STATE, "pixels": OBS_IMAGE}

    @property
    def package_name(self) -> str:
        # Override the default "gym_<type>" (= "gym_genesis", an unrelated HF package). Our tasks
        # register under this package; create_envs imports it if the id isn't already registered.
        return GYM_NAMESPACE

    @property
    def gym_kwargs(self) -> dict[str, object]:
        # max_episode_steps drives gym.make's TimeLimit; render_mode is forwarded to the env.
        return {"max_episode_steps": self.episode_length, "render_mode": self.render_mode}
