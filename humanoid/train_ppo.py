# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# Modifications made by Arth Shukla on [date]


import sys


if sys.version_info < (3, 9):
    # NOTE (arth): import first to avoid importing after torch:
    import isaacgym

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

from dacite import from_dict
from omegaconf import OmegaConf

import numpy as np
import torch

from humanoid.agents.ppo import PPO
from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.make import make_env, parse_env_cfg_by_suite
from humanoid.utils.array import recursive_to_python, to_tensor
from humanoid.utils.config import parse_cfg
from humanoid.utils.dataclasses import (
    class_factory,
    default_field,
    get_class,
    recursive_asdict,
)
from humanoid.utils.logger import Logger, LoggerConfig
from humanoid.utils.random import set_seed
from humanoid.utils.time import NonOverlappingTimeProfiler


@dataclass
class PPOConfig:
    name: str = "ppo"

    actor_critic_cls: str = "humanoid.agents.ppo.modules.actor_critic.ActorCritic"

    # max_iterations = 1500  # number of policy updates
    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    rollout_steps: int = 24
    """the number of steps to run in each environment per policy rollout"""
    extra_stat_keys: List[str] = default_field([])
    """List of additional info dict keys to track statistics for"""
    early_break_config: Dict[str, float] = field(default_factory=dict)
    """(optional) early break config {log_key: break_value}; if logs[log_key] >= break_value, script will exit"""

    @dataclass
    class PPOPolicyConfig:
        init_noise_std: float = 1.0
        actor_hidden_dims: List[int] = default_field([512, 256, 128])
        critic_hidden_dims: List[int] = default_field([512, 256, 128])
        activation: Literal[
            "elu", "celu", "relu", "selu", "crelu", "lrelu", "tanh", "sigmoid"
        ] = "elu"

    policy: PPOPolicyConfig = class_factory(PPOPolicyConfig)

    @dataclass
    class PPOAlgorithmConfig:
        learning_rate: float = 1e-3
        learning_rate_range: List[float] = default_field([1e-5, 1e-2])
        schedule: Literal["adaptive", "fixed"] = "adaptive"
        desired_kl: float = 0.01

        gamma: float = 0.99
        lam: float = 0.95

        num_learning_epochs: float = 5
        num_mini_batches: float = 4  # mini batch size = num_envs*nsteps / nminibatches

        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True

        entropy_coef: float = 0.01
        clip_param: float = 0.2
        max_grad_norm: float = 1.0

        bootstrap_on_trunc: bool = True

    algorithm: PPOAlgorithmConfig = class_factory(PPOAlgorithmConfig)

    @dataclass
    class PPOFrequencyConfig:
        log_freq: int = 1_000_000
        """log frequency in terms of global_step"""
        save_freq: int = 1_000_000
        """save frequency in terms of global_step"""
        save_backup_ckpts: bool = False
        """save a numbered checkpoint every save_freq (if false, still overrides latest.pt every save_freq)"""

    frequency: PPOFrequencyConfig = class_factory(PPOFrequencyConfig)

    # filled in later
    num_envs: int = field(init=False)
    num_iterations: int = field(init=False)
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)

    def __post_init__(self):
        assert self.name == "ppo", "Wrong algo config"

    def _additional_processing(self):
        self.batch_size = int(self.num_envs * self.rollout_steps)
        self.minibatch_size = self.batch_size // self.algorithm.num_mini_batches
        self.num_iterations = math.ceil(self.total_timesteps / self.batch_size)


@dataclass
class TrainConfig:
    seed: int
    env: Union[Dict, EnvConfig]
    algo: PPOConfig
    logger: LoggerConfig

    # Resume args
    model_ckpt: Optional[str] = None
    wandb_id: Optional[str] = None

    def __post_init__(self):
        self.env = parse_env_cfg_by_suite(self.env)
        self.algo.num_envs = self.env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = recursive_asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]


def get_humanoid_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    # NOTE (arth): for now, we force GPU usage
    device = torch.device("cuda")

    # -------------------------------------------------------------------------------------------------
    # ENV
    # -------------------------------------------------------------------------------------------------
    envs = make_env(cfg.env, video_path=cfg.logger.eval_video_path)
    dict_obs, infos = envs.reset(seed=cfg.seed)
    obs, critic_obs = dict_obs["actor_obs"], dict_obs["critic_obs"]

    num_obs = np.prod(envs.single_observation_space.shape)
    num_privileged_obs = np.prod(
        envs.single_privileged_observation_space.shape
    )  # defaults to observation_space if no priviledged obs
    num_actions = np.prod(envs.single_action_space.shape)

    # -------------------------------------------------------------------------------------------------
    # ACTOR
    # -------------------------------------------------------------------------------------------------
    actor_critic_kwargs = dict(
        num_actor_obs=num_obs,
        num_critic_obs=num_privileged_obs,
        num_actions=num_actions,
        **recursive_asdict(cfg.algo.policy),
    )
    actor_critic = get_class(cfg.algo.actor_critic_cls)(**actor_critic_kwargs).to(
        device
    )
    alg = PPO(actor_critic, device=device, **recursive_asdict(cfg.algo.algorithm))
    alg.init_storage(
        envs.num_envs,
        cfg.algo.rollout_steps,
        [num_obs],
        [num_privileged_obs],
        [num_actions],
    )

    # -------------------------------------------------------------------------------------------------
    # LOGGING AND CHECKPOINTING
    # -------------------------------------------------------------------------------------------------

    def save(save_path):
        torch.save(
            dict(
                policy_type=cfg.algo.name,
                agent=alg.actor_critic.state_dict(),
                optimizer=alg.optimizer.state_dict(),
                actor_critic_kwargs=actor_critic_kwargs,
            ),
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(load_path, weights_only=False)
        alg.actor_critic.load_state_dict(checkpoint["agent"])
        alg.optimizer.load_state_dict(checkpoint["optimizer"])

    cfg.logger.exp_cfg["options_config"] = recursive_asdict(envs.unwrapped.config)
    logger = Logger(logger_cfg=cfg.logger, save_fn=save)

    if cfg.model_ckpt is not None:
        load(cfg.model_ckpt)

    # -------------------------------------------------------------------------------------------------
    # OTHER SETUP
    # -------------------------------------------------------------------------------------------------
    # train progression
    iteration = 0
    global_step = logger.last_log_step
    global_start_step = logger.last_log_step

    def check_freq(freq):
        return (global_step % freq < cfg.algo.batch_size) or (
            iteration == cfg.algo.num_iterations - 1
        )

    # metrics
    rewbuffer = deque(maxlen=100)
    raw_rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    extra_stat_buffers = {key: deque(maxlen=100) for key in cfg.algo.extra_stat_keys}
    cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    cur_raw_rew_sum = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    if "success" in infos:
        success_once_buffer = deque(maxlen=100)
        success_at_end_buffer = deque(maxlen=100)
        cur_success_once = torch.zeros(envs.num_envs, dtype=torch.bool, device=device)

    # ready training
    alg.actor_critic.train()

    # break early checker
    break_early = False

    # timer
    timer = NonOverlappingTimeProfiler()

    # -------------------------------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------------------------------
    for iteration in range(cfg.algo.num_iterations):

        with torch.inference_mode():
            # -------------------------------------------------------------------------------------------------
            # ROLLOUTS
            # -------------------------------------------------------------------------------------------------
            for _ in range(cfg.algo.rollout_steps):
                actions = alg.act(obs, critic_obs)
                dict_obs, rewards, terms, truncs, infos = envs.step(actions)
                obs, critic_obs = dict_obs["actor_obs"], dict_obs["critic_obs"]

                alg.process_env_step(rewards, terms, truncs, infos)

                # Book keeping
                cur_reward_sum += rewards
                cur_raw_rew_sum += infos["raw_reward"]
                cur_episode_length += 1

                if "success" in infos:
                    success = to_tensor(infos["success"], device)
                    if "final_info" in infos:
                        _final_info_mask = to_tensor(infos["_final_info"], device)
                        success[_final_info_mask] = to_tensor(
                            infos["final_info"]["success"][_final_info_mask], device
                        )
                    cur_success_once |= success

                dones: torch.Tensor = terms | truncs
                done_ids = dones.nonzero(as_tuple=False).flatten()
                for key in cfg.algo.extra_stat_keys:
                    if key in infos:
                        extra_stat = infos[key]
                        if (
                            isinstance(extra_stat, torch.Tensor)
                            and extra_stat.numel() > 1
                        ):
                            if len(done_ids) == 0:
                                continue
                            extra_stat = extra_stat[done_ids]
                        extra_stat = recursive_to_python(extra_stat)
                        if np.iterable(infos[key]):
                            extra_stat_buffers[key].extend(extra_stat)
                        else:
                            extra_stat_buffers[key].append(extra_stat)
                if torch.any(dones):
                    rewbuffer.extend(recursive_to_python(cur_reward_sum[dones]))
                    raw_rewbuffer.extend(recursive_to_python(cur_raw_rew_sum[dones]))
                    lenbuffer.extend(recursive_to_python(cur_episode_length[dones]))
                    cur_reward_sum[dones] = 0
                    cur_raw_rew_sum[dones] = 0
                    cur_episode_length[dones] = 0
                    if "success" in infos:
                        success_once_buffer.extend(
                            recursive_to_python(cur_success_once[dones])
                        )
                        success_at_end_buffer.extend(
                            recursive_to_python(success[dones])
                        )
                        cur_success_once[dones] = False

                global_step += obs.size(0)
            timer.end("sim_sample")

            # -------------------------------------------------------------------------------------------------
            # BOOTSTRAP
            # -------------------------------------------------------------------------------------------------
            alg.compute_returns(critic_obs)
            timer.end("bootstrap")

        # -------------------------------------------------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------------------------------------------------
        mean_value_loss, mean_surrogate_loss = alg.update()
        timer.end("train")

        # -------------------------------------------------------------------------------------------------
        # LOG
        # -------------------------------------------------------------------------------------------------
        if check_freq(cfg.algo.frequency.log_freq) and iteration > 0:
            logger.store("time", **timer.get_time_logs(global_step - global_start_step))

            train_stats = {
                "ep_return": statistics.mean(rewbuffer),
                "raw_ep_return": statistics.mean(raw_rewbuffer),
                "len": statistics.mean(lenbuffer),
                "len_frac": statistics.mean(lenbuffer) / cfg.env.max_episode_steps,
            }
            # Add extra statistics to logging
            for key in cfg.algo.extra_stat_keys:
                if len(extra_stat_buffers[key]) > 0:
                    train_stats[key] = statistics.mean(extra_stat_buffers[key])

            logger.store("train", **train_stats)
            logger.store(
                "losses_and_charts",
                learning_rate=alg.learning_rate,
                value_loss=mean_value_loss,
                surrogate_loss=mean_surrogate_loss,
                mean_noise_std=alg.actor_critic.std.mean(),
            )
            logger.log(global_step, print_summary=True)

            for log_key, break_value in cfg.algo.early_break_config.items():
                cat, key = log_key.split("/")
                if logger.data[cat][key] >= break_value:
                    break_early = True
                    break

            timer.end("log")

        # -------------------------------------------------------------------------------------------------
        # SAVE
        # -------------------------------------------------------------------------------------------------
        if check_freq(cfg.algo.frequency.save_freq):
            if cfg.algo.frequency.save_backup_ckpts:
                save(logger.model_path / f"{global_step}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end("save")

        if break_early:
            break

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    # close everyhting once script is done running
    envs.close()
    logger.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_humanoid_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
