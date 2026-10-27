from typing import Dict, Optional, Union

import dacite

import gymnasium as gym

from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.wrappers.vector.record_video import RecordVideo


def parse_env_cfg_by_suite(env_cfg: Dict) -> EnvConfig:
    suite = env_cfg["suite"]
    if suite == "humanoidverse":
        from humanoid.envs.humanoidverse.make import HumanoidverseEnvConfig

        return dacite.from_dict(data_class=HumanoidverseEnvConfig, data=env_cfg)
    if suite == "mujoco":
        from humanoid.envs.mujoco.make import MujocoEnvConfig

        return dacite.from_dict(data_class=MujocoEnvConfig, data=env_cfg)
    raise ValueError(f"Unknown {suite=}")


def make_env(
    env_cfg: EnvConfig, video_path: Optional[str] = None
) -> Union[gym.vector.VectorEnv]:

    if isinstance(env_cfg, dict):
        env_cfg = parse_env_cfg_by_suite(env_cfg)

    if env_cfg.suite == "humanoidverse":
        from humanoid.envs.humanoidverse.make import make_env as _make_humanoidverse_env

        venv = _make_humanoidverse_env(env_cfg, video_path)
    elif env_cfg.suite == "mujoco":
        from humanoid.envs.mujoco.make import make_env as _make_mujoco_env

        venv = _make_mujoco_env(env_cfg, video_path)
    else:
        raise ValueError(f"Unknown suite: {env_cfg.suite}")
    if env_cfg.save_video:
        venv = RecordVideo(
            venv,
            video_path,
            save_video=True,
            info_on_video=env_cfg.info_on_video,
            chunk_infos=env_cfg.chunk_infos,
            video_fps=getattr(venv, "control_freq", 50),
        )
    return venv
