from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import gymnasium as gym
from gymnasium.vector import AutoresetMode

from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.mujoco.catch_upper import CatchUpperEnv
from humanoid.envs.mujoco.utils.wrappers import (
    MujocoAsyncVectorEnv,
    MujocoSyncVectorEnv,
)
from humanoid.envs.wrappers.vector.record_video import RecordVideo
from humanoid.utils.dataclasses import recursive_asdict


@dataclass
class MujocoEnvConfig(EnvConfig):
    suite: str = "mujoco"

    env_id: str = "mujoco-catch-upper-lock-legs-v0"
    """env_id points to env_class, env_config combination"""
    num_envs: int = 4
    """Number of environments to create"""
    max_episode_steps: int = 1000
    """Maximum number of steps per episode"""


def make_env(env_cfg: MujocoEnvConfig, video_path: Optional[str] = None):
    env_cls = {
        "mujoco-catch-upper-lock-legs-v0": partial(CatchUpperEnv, hist=False),
        "mujoco-catch-upper-lock-legs-hist-v0": partial(CatchUpperEnv, hist=True),
    }[env_cfg.env_id]

    def thunk():
        env = env_cls(
            render_mode=env_cfg.render_mode,
            options_config=recursive_asdict(env_cfg.options_config),
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_cfg.max_episode_steps)
        return env

    if env_cfg.num_envs == 1:
        venv = MujocoSyncVectorEnv(
            env_fns=[thunk for _ in range(env_cfg.num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    else:
        venv = MujocoAsyncVectorEnv(
            env_fns=[thunk for _ in range(env_cfg.num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
            context="forkserver",
        )
    return venv
