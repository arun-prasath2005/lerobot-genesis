"""``lerobot-genesis-eval`` — run ``lerobot-eval`` with the Genesis env + tasks registered.

LeRobot doesn't auto-discover third-party env configs, so this thin launcher imports
:mod:`lerobot_genesis.config` (registers ``--env.type=genesis``) and :mod:`lerobot_genesis.tasks`
(registers the reference gym ids), then hands off to LeRobot's own eval entry point. Every
``lerobot-eval`` flag works unchanged. Smoke-test the plumbing with a from-scratch policy (no
checkpoint):

    lerobot-genesis-eval --env.type=genesis --policy.type=act \
        --eval.batch_size=1 --eval.n_episodes=1 --eval.use_async_envs=false --policy.device=cpu
"""

from __future__ import annotations


def main() -> None:
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    import lerobot_genesis.config
    import lerobot_genesis.tasks  # noqa: F401 — registers the reference gym ids

    lerobot_eval_main()


if __name__ == "__main__":
    main()
