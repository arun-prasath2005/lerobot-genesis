"""Reference task registration — uses gymnasium (a core dep), no Genesis/GPU.

Importing lerobot_genesis.tasks must register the gym ids without building a scene (the factory,
which needs Genesis, runs only at gym.make time).
"""

from __future__ import annotations

import gymnasium as gym


def test_importing_tasks_registers_the_franka_reach_id() -> None:
    import lerobot_genesis.tasks as tasks

    assert tasks.FRANKA_REACH_ID == "lerobot_genesis/FrankaReach-v0"
    assert tasks.FRANKA_REACH_ID in gym.registry


def test_registration_does_not_build_a_scene() -> None:
    # The entry point is a string ref; importing must not import Genesis or construct the env.
    import sys

    import lerobot_genesis.tasks  # noqa: F401

    assert "genesis" not in sys.modules  # the heavy sim stays unimported until gym.make
