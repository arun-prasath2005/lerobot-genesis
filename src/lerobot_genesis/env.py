"""A Genesis scene exposed as a Gymnasium environment, in LeRobot's conventions.

LeRobot consumes simulators as plain ``gymnasium.Env`` objects — there is no LeRobot-owned
simulator interface — so the bridge is exactly that. :class:`GenesisEnv` owns the gym contract and
defers all physics to an injected :class:`SceneDriver`, which keeps the contract testable without a
GPU or Genesis present.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]


@runtime_checkable
class SceneDriver(Protocol):
    """The simulator side of :class:`GenesisEnv`, injected so the gym contract stays GPU-free.

    An implementation owns a Genesis scene: it resets the world, applies an action, steps physics,
    and reports an observation. Actions are normalised joint targets in ``[-1, 1]``; ``observe``
    returns ``(rgb, joint_state)`` where ``rgb`` is an ``(H, W, 3)`` ``uint8`` image.
    """

    @property
    def action_dim(self) -> int: ...
    @property
    def state_dim(self) -> int: ...
    @property
    def image_shape(self) -> tuple[int, int, int]: ...

    def reset(self) -> None: ...
    def apply_action(self, action: FloatArray) -> None: ...
    def step(self) -> None: ...
    def observe(self) -> tuple[ImageArray, FloatArray]: ...
    def is_success(self) -> bool: ...


class GenesisEnv(gym.Env[dict[str, Any], FloatArray]):
    """A Genesis scene as a LeRobot-ready ``gym.Env``.

    Observation is ``{"pixels": HWC uint8, "agent_pos": float32 vector}`` and the action is a
    continuous joint-command ``Box``. An episode terminates on success and truncates at
    ``max_episode_steps``. The reward is ``1.0`` on the success step and ``0.0`` otherwise; wrap the
    env to shape it differently.
    """

    # Mirrors gymnasium.Env.metadata (a plain class attribute, not a ClassVar).
    metadata = {"render_modes": ["rgb_array"]}  # noqa: RUF012

    def __init__(
        self,
        driver: SceneDriver,
        *,
        task: str = "",
        task_description: str = "",
        max_episode_steps: int = 300,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self._driver = driver
        self.task = task
        self.task_description = task_description or task  # the instruction handed to VLA policies
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._steps = 0

        h, w, _ = driver.image_shape
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
                "agent_pos": spaces.Box(-np.inf, np.inf, (driver.state_dim,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (driver.action_dim,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._driver.reset()
        self._steps = 0
        return self._observation(), {"is_success": False}

    def step(self, action: FloatArray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._driver.apply_action(np.asarray(action, dtype=np.float32))
        self._driver.step()
        self._steps += 1
        success = bool(self._driver.is_success())
        truncated = self._steps >= self._max_episode_steps
        return self._observation(), float(success), success, truncated, {"is_success": success}

    def render(self) -> ImageArray | None:
        if self.render_mode != "rgb_array":
            return None
        rgb, _ = self._driver.observe()
        return rgb

    def _observation(self) -> dict[str, Any]:
        rgb, state = self._driver.observe()
        return {"pixels": rgb, "agent_pos": np.asarray(state, dtype=np.float32)}
