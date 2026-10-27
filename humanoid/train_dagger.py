import sys


if sys.version_info < (3, 9):
    # NOTE (arth): import first to avoid importing after torch:
    import isaacgym

import statistics
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from dacite import from_dict
from muon import SingleDeviceMuon as Muon
from omegaconf import OmegaConf

from gymnasium import spaces

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from humanoid.agents.bc import ReplayBuffer
from humanoid.agents.evaluator import Evaluator
from humanoid.agents.make import get_act_fn_from_ckpt
from humanoid.envs.env_cfg import EnvConfig
from humanoid.envs.make import make_env, parse_env_cfg_by_suite
from humanoid.utils.array import (
    DictArray,
    recursive_to_python,
    to_tensor,
)
from humanoid.utils.config import parse_cfg
from humanoid.utils.dataclasses import recursive_asdict
from humanoid.utils.logger import Logger, LoggerConfig
from humanoid.utils.random import set_seed
from humanoid.utils.time import NonOverlappingTimeProfiler


@dataclass
class DaggerConfig:
    name: str = "dagger"

    loss: Literal["kl", "rmse", "mse"] = "kl"
    "loss type for DAgger update"

    collect_steps_per_iter: int = 200
    """number of steps to collect data for before training"""
    updates_per_iter: int = 200
    """number of updates to perform per iteration"""
    loss_lower_bound: float = 0.0
    """if loss less than this, go to next iter"""

    data_path: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None
    """path to the data file to load"""
    data_type: Optional[Literal["ckpt_map", "ckpt", "env"]] = None
    """type of data to load"""

    buffer_size: int = 1_000_000
    """size of the replay buffer"""
    buffer_device: str = "cuda"
    """device to store the replay buffer on"""

    muon_lr: float = 0.02
    """learning rate for muon"""
    adam_lr: float = 1e-3
    """learning rate for muon"""
    scheduler_gamma: Optional[float] = None
    """gamma for the exp lr scheduler"""
    batch_size: int = 512
    """batch size for training"""
    max_grad_norm: Optional[float] = None
    """max grad norm"""

    num_iterations: int = 1000
    """number of iterations to run"""
    extra_stat_keys: List[str] = field(default_factory=list)
    """extra statistics to log"""
    early_break_config: Dict[str, float] = field(default_factory=dict)
    """(optional) early break config {log_key: break_value}; if logs[log_key] >= break_value, script will exit"""

    log_freq: int = 1_000_000
    """log frequency in terms of global_step"""
    save_freq: int = 5_000_000
    """save frequency in terms of global_step"""
    eval_freq: Optional[int] = None
    """evaluation frequency in terms of global_step"""
    save_backup_ckpts: bool = True
    """save a numbered checkpoint every save_freq (if false, overrides latest.pt every save_freq)"""

    def __post_init__(self):
        assert self.name == "dagger", "Wrong algo config"

    def _additional_processing(self):
        if self.data_type == "ckpt_map":
            for k, ckpt_path in self.data_path.items():
                ckpt_path = Path(ckpt_path)
                assert (
                    ckpt_path.exists()
                ), f"Data path {ckpt_path} for key {k} does not exist"
                self.data_path[k] = ckpt_path
        elif self.data_path is not None:
            assert self.data_type not in ["env"]
            self.data_path = Path(self.data_path)
            assert self.data_path.exists(), f"Data path {self.data_path} does not exist"
            if self.data_path.suffix == ".pt":
                data_type = "ckpt"
            else:
                raise NotImplementedError(f"{self.data_path.suffix=} not implemented")
            if self.data_type is not None:
                assert self.data_type == data_type
            self.data_type = data_type
        else:
            assert self.data_type == "env"


@dataclass
class TrainConfig:
    seed: int
    env: Union[Dict, EnvConfig]
    algo: DaggerConfig
    logger: LoggerConfig

    # Resume args
    model_ckpt: Optional[str] = None
    wandb_id: Optional[str] = None

    def __post_init__(self):
        self.env = parse_env_cfg_by_suite(self.env)
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
    envs = make_env(cfg.env)
    dict_obs, infos = envs.reset(seed=cfg.seed)
    obs, teacher_obs = dict_obs["actor_obs"], dict_obs["teacher_obs"]

    # -------------------------------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------------------------------
    if cfg.algo.data_type == "ckpt":
        data_act_fn, teacher = get_act_fn_from_ckpt(
            cfg.algo.data_path, device, ret_actor=True
        )
        if cfg.algo.loss == "kl":
            assert hasattr(teacher, "act_inference")
        reset_teacher_on_done = hasattr(teacher, "reset")
    elif cfg.algo.data_type == "ckpt_map":
        data_act_fn_map = {}
        teacher_map = {}
        reset_teacher_on_done_map = {}
        key_envs_map = {}
        for k, ckpt_path in cfg.algo.data_path.items():
            data_act_fn_map[k], teacher_map[k] = get_act_fn_from_ckpt(
                ckpt_path, device, ret_actor=True
            )
            if cfg.algo.loss == "kl":
                assert hasattr(teacher_map[k], "act_inference")
            reset_teacher_on_done_map[k] = hasattr(teacher_map[k], "reset")
            key_envs_map[k] = infos[k]
        reset_teacher_on_done = any(reset_teacher_on_done_map.values())
    elif cfg.algo.data_type == "env":
        envs.unwrapped.enable_teacher_actions()
    else:
        raise NotImplementedError(f"{cfg.algo.data_type=} not implemented")

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------

    from humanoid.agents.bc import Agent

    actor = Agent(
        envs.single_observation_space,
        envs.single_action_space,
        gaussian=False,  # cfg.algo.loss == "kl",
    ).to(device)

    # --- split params: Muon for 2D weights (excluding final 2D "head"), AdamW for the rest ---
    two_d = [
        (n, p) for (n, p) in actor.named_parameters() if p.requires_grad and p.ndim == 2
    ]
    muon_params = [p for (_, p) in two_d[:-1]]  # treat last 2D matrix as "head"
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [
        p
        for (_, p) in actor.named_parameters()
        if p.requires_grad and id(p) not in muon_ids
    ]

    actor_optimizer_muon = Muon(muon_params, lr=cfg.algo.muon_lr)
    actor_optimizer_adamw = optim.AdamW(adamw_params, lr=cfg.algo.adam_lr)

    if cfg.algo.scheduler_gamma is not None:
        actor_scheduler_muon = optim.lr_scheduler.ExponentialLR(
            actor_optimizer_muon, cfg.algo.scheduler_gamma
        )
        actor_scheduler_adamw = optim.lr_scheduler.ExponentialLR(
            actor_optimizer_adamw, cfg.algo.scheduler_gamma
        )

    print(f"{actor=}")

    # -------------------------------------------------------------------------------------------------
    # MEMORY
    # -------------------------------------------------------------------------------------------------

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        buffer_size=cfg.algo.buffer_size,
        storage_device=torch.device(cfg.algo.buffer_device),
        sample_device=device,
    )

    if isinstance(envs.single_observation_space, spaces.Dict):
        collect_obs_buf = DictArray(
            (cfg.algo.collect_steps_per_iter, envs.num_envs),
            envs.single_observation_space,
            device=device,
        )
    else:
        collect_obs_buf = torch.zeros(
            (
                cfg.algo.collect_steps_per_iter,
                envs.num_envs,
                *envs.single_observation_space.shape,
            ),
            device=device,
            dtype=torch.float,
        )
    teacher_act_buf = torch.zeros(
        (
            cfg.algo.collect_steps_per_iter,
            envs.num_envs,
            *envs.single_action_space.shape,
        ),
        device=device,
        dtype=torch.float,
    )

    # -------------------------------------------------------------------------------------------------
    # LOGGING AND CHECKPOINTING
    # -------------------------------------------------------------------------------------------------

    def save(save_path):
        ckpt = dict(
            policy_type="dagger",
            obs_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            gaussian=False,  # cfg.algo.loss == "kl",
            agent=actor.state_dict(),
            optimizer_muon=actor_optimizer_muon.state_dict(),
            optimizer_adamw=actor_optimizer_adamw.state_dict(),
        )
        if cfg.algo.scheduler_gamma is not None:
            ckpt["scheduler_muon"] = actor_scheduler_muon.state_dict()
            ckpt["scheduler_adamw"] = actor_scheduler_adamw.state_dict()
        torch.save(
            ckpt,
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(str(load_path), map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["agent"])
        actor_optimizer_muon.load_state_dict(checkpoint["optimizer_muon"])
        actor_optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw"])
        assert (cfg.algo.scheduler_gamma is not None) == (
            "scheduler_muon" in checkpoint
        )
        if cfg.algo.scheduler_gamma is not None:
            actor_scheduler_muon.load_state_dict(checkpoint["scheduler_muon"])
            actor_scheduler_adamw.load_state_dict(checkpoint["scheduler_adamw"])

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
        return (
            global_step % freq < (envs.num_envs * cfg.algo.collect_steps_per_iter)
        ) or (iteration == cfg.algo.num_iterations - 1)

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

    if cfg.algo.eval_freq is not None:
        # NOTE (arth): IsaacGym cannot make multiple envs in the same process, so we must make a separate process for evaluation
        evaluator = Evaluator(
            eval_seed=cfg.seed + 1_000_000,
            env_cfg=cfg.env,
            device=device,
            call_fn_str="forward",
        )

    # break early checker
    break_early = False

    # timer
    timer = NonOverlappingTimeProfiler()

    # -------------------------------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------------------------------

    for iteration in range(cfg.algo.num_iterations):

        # -------------------------------------------------------------------------------------------------
        # COLLECT
        # -------------------------------------------------------------------------------------------------
        with torch.inference_mode():
            for collect_step_num in range(cfg.algo.collect_steps_per_iter):
                global_step += envs.num_envs

                collect_obs_buf[collect_step_num] = obs

                if cfg.algo.data_type == "ckpt":
                    teacher_act_buf[collect_step_num] = data_act_fn(teacher_obs)
                elif cfg.algo.data_type == "ckpt_map":
                    _combined_action = torch.empty_like(
                        teacher_act_buf[collect_step_num]
                    )
                    for k, _act_fn in data_act_fn_map.items():
                        _combined_action[infos[k]] = _act_fn(teacher_obs[infos[k]])
                    teacher_act_buf[collect_step_num] = _combined_action
                else:
                    teacher_act_buf[collect_step_num] = infos["teacher_actions"]

                # ALGO LOGIC: put action logic here
                # TRY NOT TO MODIFY: execute the game and log data.
                deterministic = cfg.algo.loss != "kl"
                next_dict_obs, rewards, terminations, truncations, infos = envs.step(
                    actor(obs, deterministic=deterministic)
                )

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

                dones: torch.Tensor = terminations | truncations
                done_ids = dones.nonzero(as_tuple=False).flatten()
                for key in cfg.algo.extra_stat_keys:
                    if key in infos:
                        extra_stat = infos[key]
                        if isinstance(extra_stat, torch.Tensor) and extra_stat.numel() > 1:
                            if len(done_ids) == 0:
                                continue
                            extra_stat = extra_stat[done_ids]
                        extra_stat = recursive_to_python(extra_stat)
                        if np.iterable(infos[key]):
                            extra_stat_buffers[key].extend(extra_stat)
                        else:
                            extra_stat_buffers[key].append(extra_stat)
                if torch.any(dones):
                    if cfg.algo.data_type == "ckpt" and reset_teacher_on_done:
                        teacher.reset(dones)
                    elif cfg.algo.data_type == "ckpt_map" and reset_teacher_on_done:
                        for k, _reset in reset_teacher_on_done_map.items():
                            if _reset:
                                assert torch.all(
                                    infos[k] == key_envs_map[k]
                                ), f"{cfg.algo.data_type} assumes fixed env split between ckpts"
                                teacher_map[k].reset(dones[infos[k]])

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

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_dict_obs["actor_obs"]
                teacher_obs = next_dict_obs["teacher_obs"]

            if isinstance(collect_obs_buf, DictArray):
                rb_obs = collect_obs_buf.reshape(
                    (cfg.algo.collect_steps_per_iter * envs.num_envs,)
                )
            else:
                rb_obs = collect_obs_buf.reshape(
                    cfg.algo.collect_steps_per_iter * envs.num_envs, -1
                )

            expert_actions = teacher_act_buf.reshape(
                cfg.algo.collect_steps_per_iter * envs.num_envs, -1
            )
            rb.add(rb_obs[:], expert_actions)

        timer.end("rollout")

        # -------------------------------------------------------------------------------------------------
        # UPDATE
        # -------------------------------------------------------------------------------------------------
        for _ in range(cfg.algo.updates_per_iter):
            batch = rb.sample(cfg.algo.batch_size)
            if cfg.algo.loss == "mse":
                actor_loss = F.mse_loss(
                    batch.actions, actor(batch.obs, deterministic=True)
                )
            elif cfg.algo.loss == "rmse":
                actor_loss = torch.sqrt(
                    F.mse_loss(batch.actions, actor(batch.obs, deterministic=True))
                    + 1e-6
                )
            elif cfg.algo.loss == "kl":
                actor_loss = torch.sqrt(
                    (
                        F.mse_loss(
                            batch.actions,
                            actor(batch.obs, deterministic=True),
                            reduction="none",
                        )
                        * teacher.std
                    ).mean()
                    + 1e-6
                )
            else:
                raise NotImplementedError(f"{cfg.algo.loss=} not supported")

            actor_optimizer_muon.zero_grad()
            actor_optimizer_adamw.zero_grad()
            actor_loss.backward()
            if cfg.algo.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.algo.max_grad_norm)
            actor_optimizer_muon.step()
            actor_optimizer_adamw.step()
            _loss = actor_loss.item()

            if _loss < cfg.algo.loss_lower_bound:
                break

        if cfg.algo.scheduler_gamma is not None:
            actor_scheduler_muon.step()
            actor_scheduler_adamw.step()
        timer.end("update")

        # -------------------------------------------------------------------------------------------------
        # LOG
        # -------------------------------------------------------------------------------------------------
        if check_freq(cfg.algo.log_freq) and iteration > 0:
            # -------------------------------------------------------------------------------------------------
            # EVAL
            # -------------------------------------------------------------------------------------------------
            if cfg.algo.eval_freq is not None and check_freq(cfg.algo.eval_freq):
                actor.eval()
                eval_stats = evaluator.eval_policy(
                    actor, None  # obs_rms if cfg.algo.obs_rms else None
                )
                logger.store("eval", **eval_stats)
                actor.train()
                timer.end("eval")

            time_logs = timer.get_time_logs(global_step - global_start_step)
            time_logs.update(
                {
                    "rollout+update_time": time_logs["rollout_time"]
                    + time_logs["update_time"],
                    "rollout+update_ratio": time_logs["rollout_ratio"]
                    + time_logs["update_ratio"],
                }
            )
            logger.store("time", **time_logs)

            train_stats = {
                "ep_return": statistics.mean(rewbuffer),
                "raw_ep_return": statistics.mean(raw_rewbuffer),
                "len_frac": statistics.mean(lenbuffer) / cfg.env.max_episode_steps,
            }
            if "success" in infos:
                train_stats.update(
                    success_once=statistics.mean(success_once_buffer),
                    success_at_end=statistics.mean(success_at_end_buffer),
                )
            # Add extra statistics to logging
            for key in cfg.algo.extra_stat_keys:
                if len(extra_stat_buffers[key]) > 0:
                    train_stats[key] = statistics.mean(extra_stat_buffers[key])

            logger.store("train", **train_stats)

            loss_logs = dict(
                bc_loss=_loss,
                lr_muon=actor_optimizer_muon.param_groups[0]["lr"],
                lr_adamw=actor_optimizer_adamw.param_groups[0]["lr"],
            )
            logger.store("losses_and_charts", **loss_logs)

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
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{global_step}_ckpt.pt")
            save(logger.model_path / "latest.pt")
            timer.end("save")

        if break_early:
            break

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    # close everything once script is done running
    logger.close()
    envs.close()
    if cfg.algo.eval_freq is not None:
        evaluator.close()


if __name__ == "__main__":
    PASSED_CONFIG_PATH = sys.argv[1]
    cfg = get_humanoid_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))
    train(cfg)
