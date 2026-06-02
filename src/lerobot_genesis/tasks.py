"""Reference Genesis tasks, registered with Gymnasium so ``lerobot-eval`` can run them.

Importing this module registers the reference task ids (it does not build a scene — the factory runs
only at ``gym.make`` time, on a GPU). These are minimal EXAMPLES that make the eval path concrete
and testable; for your own task, register any ``GenesisEnv`` the same way under the
``lerobot_genesis/`` namespace and select it with ``--env.task=<name>``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ._constants import GYM_NAMESPACE
from .driver import GenesisRobotDriver, bundled_franka_mjcf
from .env import GenesisEnv

FRANKA_REACH_ID = f"{GYM_NAMESPACE}/FrankaReach-v0"

# A reachable goal pose (rad; 7 arm joints + 2 fingers) the reach task is scored against.
_FRANKA_GOAL = np.array([0.6, -0.785, 0.0, -1.5, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=np.float32)
# PD gains from the Genesis Franka reference — without them the arm won't track a position target.
_FRANKA_KP = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0)
_FRANKA_KV = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0)


def make_franka_reach(*, render_mode: str = "rgb_array", **_: Any) -> GenesisEnv:
    """Build the Franka reach env: succeed when every joint is within 0.15 rad of the goal pose."""
    driver = GenesisRobotDriver(
        robot=bundled_franka_mjcf(),
        kp=_FRANKA_KP,
        kv=_FRANKA_KV,
        success_fn=lambda d: bool(np.max(np.abs(d.joint_positions() - _FRANKA_GOAL)) < 0.15),
    )
    return GenesisEnv(driver, task="reach the franka target pose", render_mode=render_mode)


gym.register(id=FRANKA_REACH_ID, entry_point="lerobot_genesis.tasks:make_franka_reach")
