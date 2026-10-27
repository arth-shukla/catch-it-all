from dataclasses import dataclass, field
from typing import Literal, Optional

import dacite

from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.humanoidverse.config.catching.catch_upper import (
    G1CatchUpperLockLegsConfig,
    G1CatchUpperLockLegsHistConfig,
)
from humanoid.envs.humanoidverse.tasks.catching.catch_upper import CatchUpperEnv
from humanoid.envs.humanoidverse.utils.wrappers import HumanoidverseVectorEnv
from humanoid.utils.dataclasses import recursive_asdict
from humanoid.utils.logger import colorize


@dataclass
class HumanoidverseEnvConfig(EnvConfig):
    suite: str = "humanoidverse"

    env_id: str = "humanoidverse-isaacgym-catch_upper_lock_legs-g1_27dof"
    num_envs: int = 4096
    max_episode_steps: int = 1000

    sim_backend: Literal["cpu", "gpu"] = "gpu"

    reward_mode: Literal["dense", "normalized_dense"] = "dense"

    render_mode: Literal["human", "rgb_array", None] = None

    options_config: dict = field(default_factory=dict)


def make_env(env_cfg: HumanoidverseEnvConfig, video_path: Optional[str] = None):

    task_cls, cfg_cls = {
        "humanoidverse-isaacgym-catch_upper_lock_legs-g1_27dof": (
            CatchUpperEnv,
            G1CatchUpperLockLegsConfig,
        ),
        "humanoidverse-isaacgym-catch_upper_lock_legs_hist-g1_27dof": (
            CatchUpperEnv,
            G1CatchUpperLockLegsHistConfig,
        ),
    }[env_cfg.env_id]

    task_cfg: G1CatchUpperLockLegsConfig = dacite.from_dict(
        cfg_cls, env_cfg.options_config
    )
    task_cfg.env.num_envs = env_cfg.num_envs
    task_cfg.env.max_episode_steps = env_cfg.max_episode_steps
    if env_cfg.render_mode == "human":
        task_cfg.env.render_mode = env_cfg.render_mode
        task_cfg.env.headless = False
    else:
        task_cfg.env.render_mode = (
            None if not env_cfg.save_video else env_cfg.render_mode
        )
        task_cfg.env.headless = not env_cfg.save_video or env_cfg.render_mode is None
    task_cfg.rewards.reward_mode = env_cfg.reward_mode

    sim_device = {"cpu": "cpu", "gpu": "cuda"}[env_cfg.sim_backend]

    print(
        colorize("Options config:\n", color="cyan", bold=True),
    )
    print(recursive_asdict(task_cfg), flush=True)

    env = task_cls(task_cfg, sim_device)

    return HumanoidverseVectorEnv(env)
