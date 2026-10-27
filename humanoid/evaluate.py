import sys


if sys.version_info < (3, 9):
    # NOTE (arth): import first to avoid importing after torch:
    import isaacgym

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from humanoid.agents.make import get_act_fn_from_ckpt
from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.make import make_env, parse_env_cfg_by_suite
from humanoid.utils.array import recursive_to_python, to_tensor
from humanoid.utils.config import parse_cfg
from humanoid.utils.dataclasses import default_field, recursive_asdict
from humanoid.utils.logger import Logger, LoggerConfig
from humanoid.utils.profile import Profiler
from humanoid.utils.random import set_seed
from humanoid.utils.time import NonOverlappingTimeProfiler


@dataclass
class EvalConfig:
    seed: int
    env: Union[Dict, EnvConfig]
    logger: LoggerConfig

    ckpt_path: Optional[Union[str, Path]] = None
    jitted: bool = False
    num_eval_steps: Optional[int] = None
    extra_stat_keys: List[str] = default_field([])

    def __post_init__(self):
        self.env = parse_env_cfg_by_suite(self.env)

        if self.ckpt_path is not None:
            self.ckpt_path = Path(self.ckpt_path)
        if str(self.ckpt_path).endswith(".jit.pt") or self.jitted:
            assert self.env.num_envs == 1, "jitted policies only support 1 env"
            self.jitted = True
        else:
            self.jitted = False

        if self.num_eval_steps is None:
            self.num_eval_steps = self.env.max_episode_steps

        self.logger.exp_cfg = recursive_asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]


def get_humanoid_eval_cfg(cfg: EvalConfig) -> EvalConfig:
    return from_dict(data_class=EvalConfig, data=OmegaConf.to_container(cfg))


def evaluate(cfg: EvalConfig):
    set_seed(cfg.seed)

    # NOTE (arth): for now, we force GPU usage
    device = torch.device("cuda")

    # -------------------------------------------------------------------------------------------------
    # ENV AND LOGGER
    # -------------------------------------------------------------------------------------------------
    envs = make_env(
        cfg.env,
        video_path=cfg.logger.eval_video_path,
    )
    obs, info = envs.reset(seed=cfg.seed)
    cfg.logger.exp_cfg["options_config"] = recursive_asdict(envs.unwrapped.config)
    logger = Logger(cfg.logger)

    # -------------------------------------------------------------------------------------------------
    # POLICY
    # -------------------------------------------------------------------------------------------------
    reset_teacher_on_done = False
    if cfg.ckpt_path is not None:
        if cfg.jitted:
            policy = torch.jit.load(
                cfg.ckpt_path,
                map_location=device,
            )
        else:
            policy, module = get_act_fn_from_ckpt(
                cfg.ckpt_path,
                device,
                ret_actor=True,
            )
            reset_teacher_on_done = hasattr(module, "reset")
    else:
        policy_steps = 0

        def policy(*args):
            nonlocal policy_steps
            policy_steps += 1

            phase_freq = 50
            mode = (policy_steps // phase_freq) % 2 == 0
            phase_step = policy_steps % phase_freq

            targ = envs.action_space.high if mode else envs.action_space.low

            act = torch.from_numpy(targ).to(device) * (phase_step / phase_freq)
            act[:, -6:] = 0.0
            return act

        policy = lambda x: torch.zeros(envs.action_space.shape, device=device)

    # -------------------------------------------------------------------------------------------------
    # OTHER SETUP
    # -------------------------------------------------------------------------------------------------
    # metrics
    rewards = []
    lens = []
    extra_stat_buffers = {key: [] for key in cfg.extra_stat_keys}
    cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    if "success" in info:
        success_once = []  # Track if episode succeeded at least once
        success_at_end = []  # Track if episode ended successfully
        cur_success_once = torch.zeros(envs.num_envs, dtype=torch.bool, device=device)

    # profiler and timer
    profiler = Profiler()
    timer = NonOverlappingTimeProfiler()

    # -------------------------------------------------------------------------------------------------
    # EVALUATION LOOP
    # -------------------------------------------------------------------------------------------------
    with torch.inference_mode():
        with profiler.profile("eval", cfg.num_eval_steps, envs.num_envs):
            for _ in tqdm(range(cfg.num_eval_steps)):
                if isinstance(obs, dict):
                    obs = obs["actor_obs"]
                if cfg.env.suite == "mujoco":
                    action = policy(to_tensor(obs, device)).cpu().numpy()
                else:
                    action = to_tensor(
                        policy(to_tensor(obs, device)), envs.unwrapped.device
                    )
                timer.end("act")

                obs, rew, term, trunc, info = envs.step(action)
                timer.end("sim_sample")

                cur_reward_sum += to_tensor(rew, device)
                cur_episode_length += 1

                if "success" in info:
                    # Update success tracking
                    success = to_tensor(info["success"], device)
                    if "final_info" in info:
                        _final_info_mask = to_tensor(info["_final_info"], device)
                        success[_final_info_mask] = to_tensor(
                            info["final_info"]["success"], device
                        )[_final_info_mask]
                    cur_success_once |= success

                done: torch.Tensor = to_tensor(term | trunc)
                done_ids = done.nonzero(as_tuple=False).flatten()
                for key in cfg.extra_stat_keys:
                    if key in info:
                        extra_stat = info[key]
                        if (
                            isinstance(extra_stat, torch.Tensor)
                            and extra_stat.numel() > 1
                        ):
                            if len(done_ids) == 0:
                                continue
                            extra_stat = extra_stat[done_ids]
                        extra_stat = recursive_to_python(extra_stat)
                        if np.iterable(info[key]):
                            extra_stat_buffers[key].extend(extra_stat)
                        else:
                            extra_stat_buffers[key].append(extra_stat)
                if done.any():
                    if reset_teacher_on_done:
                        module.reset(done)
                    rewards.extend(cur_reward_sum[done].tolist())
                    lens.extend(cur_episode_length[done].tolist())
                    cur_reward_sum[done] = 0
                    cur_episode_length[done] = 0
                    if "success" in info:
                        success_once.extend(cur_success_once[done].tolist())
                        success_at_end.extend(success[done].tolist())
                        cur_success_once[done] = False
                    timer.end("log")

    eval_logs = dict(
        ep_return=np.mean(rewards),
        len=np.mean(lens),
    )
    for key in cfg.extra_stat_keys:
        if len(extra_stat_buffers[key]) > 0:
            eval_logs[key] = statistics.mean(extra_stat_buffers[key])
    if "success" in info:
        eval_logs.update(
            dict(
                success_once=np.mean(success_once),
                success_at_end=np.mean(success_at_end),
            )
        )
    logger.store("eval", **eval_logs)
    profile_stats = profiler.stats["eval"]
    logger.store("system", **profile_stats)
    logger.store("time", **timer.get_time_logs(cfg.num_eval_steps * envs.num_envs))
    logger.log(0)
    logger.print_summary()

    envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_humanoid_eval_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    evaluate(cfg)
